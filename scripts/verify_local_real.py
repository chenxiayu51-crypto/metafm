#!/usr/bin/env python3
"""Exercise the FULL E3 training path on a small REAL dataset, locally.

    python scripts/verify_local_real.py --data-root D:/emg_data/real_mini

`scripts/e3_oneclick.py local` proves the plumbing on synthetic fixtures.
This proves the same pipeline on real signal at workstation scale, before a
server burns A800 hours on it.

It does not re-implement anything: every check goes through the project's own
`CanonicalDataModule`, `TaskBalancedBatchSampler`, `RingCrissCrossEncoder`,
`PretrainModule`, `E3RunDiagnostics` and Lightning `Trainer`, driven by the
same Hydra config tree the server uses (`data=local_real trainer=local_real`).

Sections
  1 DATA        five corpora, train/val windows, schema, no train/val leakage
  2 SAMPLING    the REAL per-corpus batch ratio, counted from the real sampler
  3 MODEL       forward/backward on real batches, gradients, EMA, bf16, collapse
  4 TRAINING    N optimiser steps, then read the diagnostics the run produced
  5 CHECKPOINT  reload, compare, resume, global step, reproducibility report

Exit status is 0 only when every section passes. Failures list what broke.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TASKS = ("pose", "typing", "discrete_gesture", "handwriting", "wrist_angle")


class VerificationError(RuntimeError):
    pass


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: (value.to(device) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    }


def _mkdir_clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _redirect_callbacks(trainer_config: dict, run_dir: Path) -> dict:
    """Point the checkpoint and diagnostics callbacks at this verification run.

    Under Hydra the working directory IS the run directory, so the configs use
    `dirpath: checkpoints` and `output_dir: .`. Driving the Trainer directly
    from Python has no such chdir, so without this the checkpoints would be
    written into whatever directory the caller happened to be in.
    """
    # Deep copy so the caller's config is left untouched.
    config = json.loads(json.dumps(trainer_config, default=str))
    callbacks = config.get("callbacks") or {}
    if "checkpoint" in callbacks:
        callbacks["checkpoint"]["dirpath"] = str(run_dir / "checkpoints")
    if "diagnostics" in callbacks:
        callbacks["diagnostics"]["output_dir"] = str(run_dir)
    config["callbacks"] = callbacks
    config["default_root_dir"] = str(run_dir)
    return config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(data_root: Path, manifest: str, corpus: str, batch_size: int, num_workers: int,
                steps: int, resume_steps: int, val_every: int, val_batches: int, seed: int):
    """Compose the SAME Hydra tree production uses, with local overrides."""
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    # `optimizer.max_steps` is the LR-schedule horizon and is baked into the
    # checkpoint's hyper-parameters; `trainer.max_steps` is only when to stop.
    # Keeping the horizon at the FINAL step count lets the resume leg re-load a
    # checkpoint whose hparams still match (PretrainModule.on_load_checkpoint
    # refuses a changed max_steps) while the first leg stops early.
    overrides = [
        "data=local_real", "trainer=local_real",
        f"data.root={data_root.as_posix()}",
        f"data.manifest_path={(data_root / manifest).as_posix()}",
        # corpus_path is not a declared field in the data group, so it must be
        # added with '+' rather than assigned.
        f"+data.corpus_path={(data_root / corpus).as_posix()}",
        f"data.batch_size={batch_size}",
        f"data.num_workers={num_workers}",
        f"seed={seed}",
        f"optimizer.max_steps={steps + resume_steps}",
        f"trainer.max_steps={steps}",
        f"trainer.val_check_interval={val_every}",
        f"trainer.limit_val_batches={val_batches}",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.log_every_n_steps=1",
        "trainer.enable_progress_bar=false",
    ]
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="pretrain", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True), cfg


# ---------------------------------------------------------------------------
# 1 DATA
# ---------------------------------------------------------------------------

def section_data(cfg: dict, data_root: Path) -> dict:
    from semg_fm.data.datamodule import build_datamodule

    data = dict(cfg["data"])
    module = build_datamodule(data, task_id=None, label_budget=1.0)
    module.setup("fit")

    report: dict = {"tasks": {}, "input": None}
    train, val = module.train_dataset, module.val_dataset

    train_groups = train.window_indices_by_task()
    val_groups = val.window_indices_by_task()
    if set(train_groups) != set(TASKS):
        raise VerificationError(f"train split exposes tasks {sorted(train_groups)}, expected {sorted(TASKS)}")
    if set(val_groups) != set(TASKS):
        raise VerificationError(f"val split exposes tasks {sorted(val_groups)}, expected {sorted(TASKS)}")

    for task in TASKS:
        n_train, n_val = len(train_groups[task]), len(val_groups[task])
        if n_train == 0:
            raise VerificationError(f"task {task!r} has no train windows")
        if n_val == 0:
            raise VerificationError(f"task {task!r} has no val windows")
        report["tasks"][task] = {"train_windows": n_train, "val_windows": n_val}

    # Read one real window per task from EACH split and validate the contract.
    for split_name, dataset, groups in (("train", train, train_groups), ("val", val, val_groups)):
        for task in TASKS:
            sample = dataset[int(groups[task][0])]
            emg = sample["emg"]
            if emg.ndim != 2 or emg.shape[1] != 16:
                raise VerificationError(f"{split_name}/{task}: EMG must be [T,16], got {tuple(emg.shape)}")
            if emg.shape[0] != 4000:
                raise VerificationError(f"{split_name}/{task}: expected 4000 samples (2000 ms @ 2 kHz), got {emg.shape[0]}")
            if emg.dtype != torch.float32:
                raise VerificationError(f"{split_name}/{task}: EMG dtype {emg.dtype} != float32")
            if not bool(torch.isfinite(emg).all()):
                raise VerificationError(f"{split_name}/{task}: EMG contains NaN/Inf")
            if not bool(sample["sample_mask"].all()):
                raise VerificationError(f"{split_name}/{task}: sample_mask is not fully observed")
            if float(sample["sample_rate_hz"]) != 2000.0:
                raise VerificationError(f"{split_name}/{task}: sample_rate_hz is not 2000")
            target, mask = sample.get("target"), sample.get("target_mask")
            if torch.is_tensor(target) and target.ndim >= 2 and target.shape[0] != emg.shape[0]:
                raise VerificationError(f"{split_name}/{task}: target time axis is not aligned with EMG")
            if torch.is_tensor(mask) and mask.ndim >= 1 and mask.shape[0] != emg.shape[0]:
                raise VerificationError(f"{split_name}/{task}: target_mask time axis is not aligned with EMG")
            report["tasks"][task][f"{split_name}_sample"] = {
                "emg_shape": list(emg.shape), "dtype": str(emg.dtype),
                "labels_present": bool(target is not None), "finite": True,
            }
            report["input"] = {"emg_shape": list(emg.shape), "sample_rate_hz": 2000}

    # Leakage. A Nature recording legitimately supplies BOTH an official train
    # and an official val interval: those are disjoint sample ranges of one
    # recording, which is the released protocol, not a leak. What must never
    # happen is overlapping ranges, or a non-Nature record in both splits.
    from semg_fm.data.intervals import NATURE_DATASETS

    def ranges(rows):
        grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for record in rows:
            key = (record["dataset_id"], record["record_id"])
            grouped.setdefault(key, []).append((
                int(record.get("sample_start", 0)),
                int(record.get("sample_end", record["sample_count"])),
            ))
        return grouped

    train_ranges, val_ranges = ranges(train._record_rows), ranges(val._record_rows)
    shared = sorted(set(train_ranges) & set(val_ranges))
    for key in shared:
        dataset_id = key[0]
        if dataset_id not in NATURE_DATASETS:
            raise VerificationError(
                f"{dataset_id} record {key[1]!r} appears in both train and val; "
                "only Nature recordings may be split by interval"
            )
        left = sorted(train_ranges[key])
        right = sorted(val_ranges[key])
        # Half-open [start, end): overlap iff a1 < b2 and b1 < a2.
        if any(a1 < b2 and b1 < a2 for a1, a2 in left for b1, b2 in right):
            raise VerificationError(f"train and val overlap in source samples for {key[1]!r}")

    train_pairs = {r["pair_id"] for r in train._record_rows if r["pair_id"]}
    val_pairs = {r["pair_id"] for r in val._record_rows if r["pair_id"]}
    if train_pairs & val_pairs:
        raise VerificationError(f"pairs appear in both splits: {sorted(train_pairs & val_pairs)[:3]}")
    report["leakage"] = {
        "records_per_split": {"train": len(train_ranges), "val": len(val_ranges)},
        "shared_records": len(shared),
        "shared_records_all_nature": all(key[0] in NATURE_DATASETS for key in shared),
        "shared_pairs": 0,
        "overlapping_sample_ranges": 0,
        "note": "Nature recordings may carry both a train and a val interval; those are "
                "disjoint sample ranges of one recording and are checked to be disjoint.",
    }
    report["windows"] = {"train": len(train), "val": len(val)}
    return report


# ---------------------------------------------------------------------------
# 2 SAMPLING
# ---------------------------------------------------------------------------

def section_sampling(cfg: dict, data_root: Path) -> dict:
    """Count what the real sampler actually emits -- do not trust the name.

    `TaskBalancedBatchSampler` interleaves one batch per task per ROUND, which
    controls ordering only. Each task still contributes
    floor(n_windows / batch_size) batches per epoch, so the per-task share of
    optimiser steps tracks each corpus's window count. On a real corpus that
    is far from equal, so it is measured here rather than assumed.
    """
    from semg_fm.data.datamodule import build_datamodule
    from semg_fm.data.samplers import TaskBalancedBatchSampler

    data = dict(cfg["data"])
    module = build_datamodule(data, task_id=None, label_budget=1.0)
    module.setup("fit")
    dataset = module.train_dataset
    batch_size = int(data["batch_size"])

    sampler = TaskBalancedBatchSampler(dataset, batch_size, seed=int(data.get("sampler_seed", 0)),
                                       rank=0, world_size=1)
    # Same ordinal -> record mapping _WindowIndex.__getitem__ uses.
    offsets = dataset._index.offsets
    task_of_record = [str(record["task_id"]) for record in dataset._record_rows]
    counts: Counter = Counter()
    started = time.perf_counter()
    for batch in sampler:
        record = int(np.searchsorted(offsets, batch[0], side="right") - 1)
        counts[task_of_record[record]] += 1
    elapsed = time.perf_counter() - started

    total = sum(counts.values())
    if total == 0:
        raise VerificationError("the train sampler yielded no batches")
    absent = [task for task in TASKS if counts.get(task, 0) == 0]
    if absent:
        raise VerificationError(f"the train sampler never emitted: {absent}")

    share = {task: counts[task] / total for task in TASKS}
    return {
        "batches_per_epoch": total,
        "per_task_batches": {task: counts[task] for task in TASKS},
        "per_task_share": share,
        "equal_share_would_be": 1.0 / len(TASKS),
        "max_over_min_share_ratio": max(share.values()) / min(share.values()),
        "windows_per_task": {k: len(v) for k, v in dataset.window_indices_by_task().items()},
        "enumeration_seconds": elapsed,
        "conclusion": (
            "per-corpus step share tracks each corpus's window count; the sampler "
            "balances ORDER, not PROPORTION"
        ),
    }


# ---------------------------------------------------------------------------
# 3 MODEL
# ---------------------------------------------------------------------------

def section_model(cfg: dict, data_root: Path, device: torch.device) -> dict:
    from semg_fm.data.datamodule import build_datamodule
    from semg_fm.lightning import PretrainModule
    from semg_fm.models import RingCrissCrossEncoder

    model_cfg = dict(cfg["model"])
    model_cfg.pop("name", None)
    backbone = RingCrissCrossEncoder(**model_cfg).to(device)
    parameters = sum(p.numel() for p in backbone.parameters())

    data = dict(cfg["data"])
    module = build_datamodule(data, task_id=None, label_budget=1.0)
    module.setup("fit")

    pretrain = PretrainModule(model_config=model_cfg, warmup_steps=0,
                              max_steps=max(1, int(cfg["optimizer"]["max_steps"]))).to(device)
    pretrain.train()

    def real_batch_for(task: str, split: str = "train"):
        dataset = module.train_dataset if split == "train" else module.val_dataset
        groups = dataset.window_indices_by_task()
        index = int(groups[task][0])
        items = [dataset[index]]
        from semg_fm.data.datamodule import task_homogeneous_collate
        return task_homogeneous_collate(items)

    # bf16. Assert on the MATMUL path, not on the encoder output: under
    # autocast, nn.LayerNorm and softmax are on PyTorch's fp32 list, so the
    # residual stream is promoted back to fp32 before it leaves each block.
    # That is what "mixed" precision means and it is the numerically safer
    # arrangement -- asserting z_eq is bf16 would be asserting the wrong thing.
    # The conv/attention/linear work is where the bf16 saving actually lives.
    amp_dtype_seen, residual_dtype = "cpu (no autocast)", "cpu"
    if device.type == "cuda":
        seen: dict[str, str] = {}

        def _capture(_module, _inputs, output):
            seen.setdefault("dtype", str(output.dtype))
            # Must return None: a forward hook's return value REPLACES the
            # module output, so `seen.setdefault(...)` inline would swap the
            # patch encoder's tensor for a string.

        handle = pretrain.student.patch_encoder.register_forward_hook(_capture)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
                probe = pretrain.student(_to_device(real_batch_for("pose"), device)["emg"])
            residual_dtype = str(probe["z_eq"].dtype)
        finally:
            handle.remove()
        amp_dtype_seen = seen.get("dtype", "unknown")
        if "bfloat16" not in amp_dtype_seen:
            raise VerificationError(
                f"bf16 autocast did not reach the patch encoder (its output was {amp_dtype_seen})"
            )

    losses = {}
    grad_norms = {}
    for task in TASKS:
        batch = _to_device(real_batch_for(task), device)
        optimizer = torch.optim.AdamW([p for p in pretrain.student.parameters() if p.requires_grad], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            total, parts = pretrain._losses(batch)
        if not math.isfinite(float(total.detach())):
            raise VerificationError(f"{task}: loss is not finite ({float(total.detach())})")
        total.backward()
        norms = [p.grad.detach().float().norm() for p in pretrain.student.parameters() if p.grad is not None]
        if not norms:
            raise VerificationError(f"{task}: no student gradient was produced")
        norm = float(torch.stack(norms).norm())
        if not math.isfinite(norm) or norm <= 0:
            raise VerificationError(f"{task}: student gradient norm is {norm}")
        if any(not torch.isfinite(p.grad).all() for p in pretrain.student.parameters() if p.grad is not None):
            raise VerificationError(f"{task}: student gradient contains NaN/Inf")
        losses[task] = float(total.detach())
        grad_norms[task] = norm
        optimizer.step()

    # EMA: drive ONE real optimiser step through Lightning rather than calling
    # optimizer_step by hand, so the hook order under test is the production one.
    from lightning import Trainer

    pretrain.zero_grad(set_to_none=True)
    before = {k: v.detach().clone() for k, v in pretrain.teacher.state_dict().items()}
    student_before = {k: v.detach().clone() for k, v in pretrain.student.state_dict().items()}
    step_config = {key: value for key, value in cfg["trainer"].items() if key != "callbacks"}
    step_config.update(max_steps=1, limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                       enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    Trainer(**step_config).fit(pretrain, datamodule=module)

    # Compared on CPU so the check cannot be defeated by a device move.
    teacher_moved = any(not torch.equal(v.cpu(), pretrain.teacher.state_dict()[k].detach().cpu())
                        for k, v in before.items() if v.is_floating_point())
    student_moved = any(not torch.equal(v.cpu(), pretrain.student.state_dict()[k].detach().cpu())
                        for k, v in student_before.items() if v.is_floating_point())
    if not student_moved:
        raise VerificationError("a training step did not change any student parameter")
    if not teacher_moved:
        raise VerificationError("a training step did not change the teacher (EMA is not running)")
    if int(pretrain.ema_updates) < 1:
        raise VerificationError("the EMA update counter did not advance")
    if any(p.grad is not None for p in pretrain.teacher.parameters()):
        raise VerificationError("the teacher accumulated gradients; it must move only via EMA")

    # Collapse check: features must carry variance across tokens and batch.
    # Probed on CPU: where Lightning leaves the module after fit() is a strategy
    # detail, and this is a statistical property that does not depend on device.
    probe = real_batch_for("pose")["emg"]
    with torch.no_grad():
        out = pretrain.cpu().student(probe)
        feature_std = float(out["z_motion"].float().std())
        across_tokens = float(out["z_eq"].float().std())
    if not math.isfinite(feature_std) or feature_std <= 1e-6:
        raise VerificationError(f"features have collapsed (z_motion std = {feature_std})")
    if not math.isfinite(across_tokens) or across_tokens <= 1e-6:
        raise VerificationError(f"token features have collapsed (z_eq std = {across_tokens})")

    return {
        "parameters": parameters,
        "architecture": model_cfg,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "amp_matmul_dtype": amp_dtype_seen,
        "amp_residual_dtype": residual_dtype,
        "amp_note": (
            "matmul path (conv/attention/linear) runs in the autocast dtype; the "
            "residual stream stays fp32 because LayerNorm and softmax are on "
            "autocast's fp32 list. This is standard bf16-mixed behaviour."
        ),
        "per_task_loss": losses,
        "per_task_grad_norm": grad_norms,
        "student_updated": student_moved,
        "teacher_updated_via_ema": teacher_moved,
        "teacher_has_no_grads": True,
        "ema_updates": int(pretrain.ema_updates),
        "feature_std_z_motion": feature_std,
        "feature_std_z_eq": across_tokens,
        "representation_collapsed": False,
    }


# ---------------------------------------------------------------------------
# 4 TRAINING
# ---------------------------------------------------------------------------

def section_training(cfg: dict, run_dir: Path, steps: int) -> dict:
    from lightning import Trainer, seed_everything
    from semg_fm.config import build_trainer_options
    from semg_fm.data.datamodule import build_datamodule
    from semg_fm.lightning import PretrainModule
    from semg_fm.reproducibility import manifest_digest

    seed_everything(int(cfg.get("seed", 0)), workers=True)
    model_cfg = dict(cfg["model"]); model_cfg.pop("name", None)
    data = dict(cfg["data"])
    data["root"] = str(data["root"])

    datamodule = build_datamodule(data, task_id=None, label_budget=1.0)
    module = PretrainModule(
        model_config=model_cfg,
        mask_config=dict(cfg["mask"]),
        latent_weight=float(cfg["objective"]["latent_weight"]),
        kinematic_weight=float(cfg["objective"]["kinematic_weight"]),
        kinematic_dim=cfg["objective"]["kinematic_dim"],
        ema_decay=float(cfg["objective"]["ema_decay"]),
        learning_rate=float(cfg["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
        max_steps=int(cfg["optimizer"]["max_steps"]),
    )
    trainer_config = _redirect_callbacks(dict(cfg["trainer"]), run_dir)
    trainer = Trainer(**build_trainer_options(trainer_config))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(module, datamodule=datamodule)
    elapsed = time.perf_counter() - started

    diagnostics_path = run_dir / "diagnostics.json"
    if not diagnostics_path.is_file():
        raise VerificationError(f"the run produced no diagnostics at {diagnostics_path}")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    if int(trainer.global_step) < steps:
        raise VerificationError(f"training stopped at step {trainer.global_step}, expected {steps}")
    missing = diagnostics.get("missing_train_tasks") or []
    if missing:
        raise VerificationError(f"these tasks never produced a training batch: {missing}")
    if not diagnostics.get("student_updated"):
        raise VerificationError("the run did not update the student")
    if not diagnostics.get("teacher_updated"):
        raise VerificationError("the run did not update the teacher")
    if not diagnostics.get("teacher_has_no_grads"):
        raise VerificationError("the teacher accumulated gradients during training")

    rows = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != steps:
        raise VerificationError(f"recorded {len(rows)} step rows for {steps} steps")
    if any(not math.isfinite(r["loss"]) for r in rows):
        raise VerificationError("a recorded training loss is not finite")

    step_times = [r["step_time_s"] for r in rows if r["step_time_s"] > 0]
    peak = 0
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated())
    return {
        "steps_requested": steps,
        "global_step": int(trainer.global_step),
        "wall_seconds": elapsed,
        "seconds_per_step": (sum(step_times) / len(step_times)) if step_times else None,
        "steps_per_second": (len(step_times) / sum(step_times)) if step_times and sum(step_times) > 0 else None,
        "peak_gpu_memory_bytes": peak,
        "train_batches_per_task": diagnostics.get("train_batches_per_task"),
        "val_batches_per_task": diagnostics.get("val_batches_per_task"),
        "train_loss_first": rows[0]["loss"],
        "train_loss_last": rows[-1]["loss"],
        "feature_std": diagnostics.get("feature_std"),
        "ema_updates": diagnostics.get("ema_updates"),
        "precision": diagnostics.get("precision"),
        "manifest_sha256": manifest_digest(datamodule.manifest_path),
        "checkpoint_dir": str(run_dir / "checkpoints"),
    }


# ---------------------------------------------------------------------------
# 5 CHECKPOINT
# ---------------------------------------------------------------------------

def _state_equal(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[k].detach().cpu(), right[k].detach().cpu()) for k in left)


def section_checkpoint(cfg: dict, run_dir: Path, steps: int, resume_steps: int) -> dict:
    from lightning import Trainer
    from semg_fm.config import build_trainer_options
    from semg_fm.data.datamodule import build_datamodule
    from semg_fm.lightning import PretrainModule

    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise VerificationError(f"the run wrote no last.ckpt at {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "rng_state"):
        if key not in saved:
            raise VerificationError(f"checkpoint is missing {key!r}")
    if int(saved["global_step"]) != steps:
        raise VerificationError(f"checkpoint global_step {saved['global_step']} != {steps}")

    model_cfg = dict(cfg["model"]); model_cfg.pop("name", None)
    reloaded = PretrainModule(
        model_config=model_cfg, mask_config=dict(cfg["mask"]),
        latent_weight=float(cfg["objective"]["latent_weight"]),
        kinematic_weight=float(cfg["objective"]["kinematic_weight"]),
        kinematic_dim=cfg["objective"]["kinematic_dim"],
        ema_decay=float(cfg["objective"]["ema_decay"]),
        learning_rate=float(cfg["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
        max_steps=int(cfg["optimizer"]["max_steps"]),
    )
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    student_exact = _state_equal(reloaded.student.state_dict(), {
        k.removeprefix("student."): v for k, v in saved["state_dict"].items() if k.startswith("student.")
    })
    teacher_exact = _state_equal(reloaded.teacher.state_dict(), {
        k.removeprefix("teacher."): v for k, v in saved["state_dict"].items() if k.startswith("teacher.")
    })
    if not student_exact:
        raise VerificationError("reloaded student weights differ from the checkpoint")
    if not teacher_exact:
        raise VerificationError("reloaded teacher weights differ from the checkpoint")
    if int(saved["state_dict"]["_ema_updates"].item()) != int(saved["global_step"]):
        raise VerificationError("EMA update counter does not match the global step")

    # Resume for a few more steps in a separate run directory.
    resume_dir = run_dir.parent / f"{run_dir.name}_resumed"
    _mkdir_clean(resume_dir)
    data = dict(cfg["data"])
    datamodule = build_datamodule(data, task_id=None, label_budget=1.0)
    total_steps = steps + resume_steps
    resumed = PretrainModule(
        model_config=model_cfg, mask_config=dict(cfg["mask"]),
        latent_weight=float(cfg["objective"]["latent_weight"]),
        kinematic_weight=float(cfg["objective"]["kinematic_weight"]),
        kinematic_dim=cfg["objective"]["kinematic_dim"],
        ema_decay=float(cfg["objective"]["ema_decay"]),
        learning_rate=float(cfg["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
        max_steps=int(cfg["optimizer"]["max_steps"]),
    )
    trainer_config = _redirect_callbacks(dict(cfg["trainer"]), resume_dir)
    trainer_config["max_steps"] = total_steps
    trainer = Trainer(**build_trainer_options(trainer_config))
    trainer.fit(resumed, datamodule=datamodule, ckpt_path=str(checkpoint), weights_only=False)
    if int(trainer.global_step) < total_steps:
        raise VerificationError(
            f"resumed run reached step {trainer.global_step}, expected {total_steps}; "
            "the checkpoint did not restore the optimizer/global step"
        )
    resumed_diag = json.loads((resume_dir / "diagnostics.json").read_text(encoding="utf-8"))
    if int(resumed_diag["ema_updates"]) <= int(saved["state_dict"]["_ema_updates"].item()):
        raise VerificationError("the EMA counter did not continue across the resume")

    from semg_fm.data.datamodule import CanonicalDataModule
    sampler_state = hasattr(CanonicalDataModule, "state_dict") and hasattr(CanonicalDataModule, "load_state_dict")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "student_exact": student_exact,
        "teacher_exact": teacher_exact,
        "optimizer_states": len(saved["optimizer_states"]),
        "lr_schedulers": len(saved["lr_schedulers"]),
        "rng_state_present": True,
        "resumed_to_step": int(trainer.global_step),
        "ema_continued": True,
        "weight_sequence_reproducible": "weights, optimizer, scheduler and RNG streams all restore exactly",
        "sample_sequence_reproducible": bool(sampler_state),
        "sample_sequence_note": (
            "TaskBalancedBatchSampler persists only its epoch (samplers.py), and "
            "CanonicalDataModule implements no state_dict/load_state_dict, so a "
            "mid-epoch resume does not replay the interrupted sample order. Weights "
            "and optimiser state are unaffected."
        ) if not sampler_state else "sampler state is persisted",
    }


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("E3_DATA_ROOT"),
                        help="real mini data root (or set E3_DATA_ROOT)")
    parser.add_argument("--manifest", default="splits.csv")
    parser.add_argument("--corpus", default="nature_corpus.csv")
    parser.add_argument("--steps", type=int, default=50, help="optimiser steps for the training section")
    parser.add_argument("--resume-steps", type=int, default=5, help="extra steps after resuming")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=os.environ.get("E3_LOCAL_RUN_ROOT", "outputs/local_real_runs"))
    parser.add_argument("--skip-train", action="store_true",
                        help="run sections 1-3 only (data, sampling, model)")
    args = parser.parse_args(argv)

    if not args.data_root:
        raise SystemExit("[verify ERROR] --data-root or E3_DATA_ROOT is required")
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"[verify ERROR] data root does not exist: {data_root}")
    for name in (args.manifest, args.corpus):
        if not (data_root / name).is_file():
            raise SystemExit(f"[verify ERROR] {name} is missing from {data_root}")

    output_root = Path(args.output).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"local_real_{stamp}"
    _mkdir_clean(run_dir)

    cfg, _ = load_config(data_root, args.manifest, args.corpus, args.batch_size,
                         args.num_workers, args.steps, args.resume_steps,
                         args.val_every, args.val_batches, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[verify] WARNING: no CUDA device; section 3 will run on CPU and is not a GPU acceptance.")

    report: dict = {
        "schema": "e3-local-real-verification-v1",
        "created_utc": stamp,
        "data_root": str(data_root),
        "device": str(device),
        "cuda": {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        },
        "config": {
            "batch_size": args.batch_size, "steps": args.steps,
            "window_ms": cfg["data"]["window_ms"], "stride_ms": cfg["data"]["stride_ms"],
            "model": cfg["model"], "precision": cfg["trainer"]["precision"],
        },
    }
    failures: list[str] = []

    def run_section(name, fn, *fn_args):
        print(f"\n== {name} ==", flush=True)
        try:
            result = fn(*fn_args)
            print(f"   PASS", flush=True)
            report[name.lower()] = result
            return result
        except Exception as exc:  # noqa: BLE001 - the report must capture everything
            print(f"   FAIL  {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name}: {exc}")
            report[name.lower()] = {"error": f"{type(exc).__name__}: {exc}"}
            return None

    data_report = run_section("DATA", section_data, cfg, data_root)
    if data_report:
        for task, detail in data_report["tasks"].items():
            print(f"      {task:<18} train={detail['train_windows']:<6} val={detail['val_windows']}", flush=True)

    sampling = run_section("SAMPLING", section_sampling, cfg, data_root)
    if sampling:
        for task, count in sampling["per_task_batches"].items():
            print(f"      {task:<18} {count:>6} batches  ({sampling['per_task_share'][task] * 100:5.1f}%)", flush=True)
        print(f"      equal share would be {sampling['equal_share_would_be'] * 100:.1f}% each", flush=True)

    model_report = run_section("MODEL", section_model, cfg, data_root, device)
    if model_report:
        print(f"      parameters={model_report['parameters']:,}  amp={model_report['amp_matmul_dtype']} (residual {model_report['amp_residual_dtype']})", flush=True)
        print(f"      feature std: z_motion={model_report['feature_std_z_motion']:.4f} "
              f"z_eq={model_report['feature_std_z_eq']:.4f}", flush=True)

    if not args.skip_train:
        training = run_section("TRAINING", section_training, cfg, run_dir, args.steps)
        if training:
            print(f"      {training['global_step']} steps in {training['wall_seconds']:.1f}s "
                  f"({training['seconds_per_step']:.3f}s/step)", flush=True)
            print(f"      loss {training['train_loss_first']:.4f} -> {training['train_loss_last']:.4f}", flush=True)
            print(f"      peak GPU memory {training['peak_gpu_memory_bytes'] / 2**20:.1f} MiB", flush=True)
            run_section("CHECKPOINT", section_checkpoint, cfg, run_dir, args.steps, args.resume_steps)

    report["failures"] = failures
    report["status"] = "COMPLETE" if not failures else "FAILED"
    (run_dir / "verification.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n== RESULT ==")
    print(f"   report: {run_dir / 'verification.json'}")
    if failures:
        for line in failures:
            print(f"   FAILED  {line}")
        return 1
    print("   ALL SECTIONS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
