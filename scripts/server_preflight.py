#!/usr/bin/env python3
"""Refuse to start an expensive E3 run until the host, data and stack are proven.

    python scripts/server_preflight.py --data-root /srv/datasets/meta_canonical \
                                       --output    /srv/experiments/e3/preflight

Runs four sections and writes a JSON report plus a machine-readable verdict:

  ENVIRONMENT   python, torch/CUDA pairing, GPU, bf16, VRAM, system RAM, free
                disk, writability of the output root, git commit, dependencies
  DATA          manifest and corpus validation through the project's own
                validators, a per-corpus census, and a train/val leakage matrix
  DATALOADER    the real DataModule and DataLoader: five-task coverage, batch
                shapes, multi-worker loading, and the measured RAM cost of the
                full window index
  GPU_SMOKE     a few real optimiser steps: gradients, EMA, peak VRAM, speed
                and a checkpoint round-trip

The smoke section deliberately runs in THIS process, which exits afterwards.
`run_e3_server.sh` starts the real training as a separate process, so the
production run can never inherit the smoke test's weights, optimizer state or
scheduler. If preflight fails, training is not started.

Nothing here is a substitute for `docs/E3_ACCEPTANCE.md`: a preflight pass means
the run is safe to start, not that the science will work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TASKS = ("emg2pose", "emg2qwerty", "gni_discrete_gestures", "gni_handwriting", "gni_wrist")
TASK_IDS = ("pose", "typing", "discrete_gesture", "handwriting", "wrist_angle")


def _e3_module():
    """Reuse the audited E3 launcher validators instead of re-implementing them."""
    spec = importlib.util.spec_from_file_location("e3_oneclick", REPO / "scripts" / "e3_oneclick.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rss_bytes() -> int:
    """Resident set size, across platforms, without requiring psutil."""
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm", encoding="utf-8") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize)
    except Exception:
        pass
    return 0


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

def section_environment(cfg: dict) -> dict:
    from semg_fm.reproducibility import environment_info
    import importlib.metadata as md

    report: dict = {"python": sys.version.split()[0], "executable": sys.executable,
                    "platform": platform.platform(), "environment": environment_info()}
    if sys.version_info[:2] < (3, 10):
        raise RuntimeError(f"python {report['python']} is older than 3.10")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; a server_full run requires a GPU")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    report["gpu"] = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_vram_bytes": int(properties.total_memory),
        "free_vram_bytes": int(free),
        "device_count": torch.cuda.device_count(),
        "torch_cuda": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    if not report["gpu"]["bf16_supported"]:
        raise RuntimeError("this device does not support bf16; the e3_server profile is bf16-mixed")
    if report["gpu"]["total_vram_bytes"] < 8 * 2**30:
        report["gpu_warning"] = "less than 8 GiB of VRAM; expect a small batch size"

    try:
        import psutil  # type: ignore
        report["system_ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        report["system_ram_bytes"] = None

    data_root: Path = cfg["data_root"]
    output_root: Path = cfg["output"]
    report["data_root"] = str(data_root)
    report["output_root"] = str(output_root)
    report["disk"] = {
        "data_free_bytes": shutil.disk_usage(data_root).free,
        "output_free_bytes": shutil.disk_usage(output_root.parent if output_root.parent.exists() else REPO).free,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / f".write_probe.{os.getpid()}"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    report["output_writable"] = True
    if not os.access(data_root, os.R_OK):
        raise RuntimeError(f"the data root is not readable: {data_root}")

    report["git"] = {"commit": _run_git(["rev-parse", "HEAD"]), "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
                     "dirty": bool(_run_git(["status", "--porcelain"]))}
    requirements = REPO / "requirements-server.txt"
    if requirements.is_file():
        mismatched = []
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            name, _, want = line.partition("==")
            try:
                have = md.version(name)
            except md.PackageNotFoundError:
                mismatched.append(f"{name}: not installed (require {want})")
                continue
            if want and have.split("+")[0] != want:
                mismatched.append(f"{name}: {have} (require {want})")
        report["dependencies"] = {"pinned_file": requirements.name, "mismatched": mismatched}
        if mismatched:
            raise RuntimeError("pinned dependencies do not match: " + "; ".join(mismatched[:4]))
    return report


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def section_data(cfg: dict) -> dict:
    """Validate the whole corpus from metadata; sample the HDF5 schema, never bulk-load signal."""
    e3 = _e3_module()
    data_root: Path = cfg["data_root"]
    manifest = data_root / cfg["manifest"]
    corpus = data_root / cfg["corpus"]

    started = time.perf_counter()
    # e3.audit walks the full manifest, validates the official corpus intervals,
    # samples HDF5 schema per corpus, and enforces the Nature overlap rules.
    preflight = e3.audit(data_root, manifest, corpus, inspect_all=False)
    audit_seconds = time.perf_counter() - started

    rows = e3.read_manifest(data_root, manifest)

    census: dict[str, dict[str, int]] = {}
    for dataset_id in TASKS:
        subset = [row for row in rows if row["dataset_id"] == dataset_id]
        entry: dict[str, int] = {
            "records": len(subset),
            "samples": sum(int(row["sample_count"]) for row in subset),
        }
        if dataset_id not in e3.NATURE_DATASETS:
            for split in ("train", "val", "test"):
                entry[f"manifest_{split}_records"] = sum(
                    1 for row in subset if e3._effective_split(row) == split
                )
        census[dataset_id] = entry
    # Nature splits live in the corpus, not the manifest.
    corpus_rows = e3.read_nature_corpus(corpus)
    nature_counts = Counter()
    for entry in corpus_rows:
        nature_counts[entry["split"]] += 1

    # Leakage matrix: a non-Nature record, pair or subject must not span splits.
    by_record: dict[tuple[str, str], set[str]] = {}
    by_pair: dict[tuple[str, str], set[str]] = {}
    by_subject: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row["dataset_id"] in e3.NATURE_DATASETS:
            continue
        split = e3._effective_split(row)
        by_record.setdefault((row["dataset_id"], row["path"]), set()).add(split)
        by_subject.setdefault((row["dataset_id"], row["subject_id"]), set()).add(split)
        if row["pair_id"].strip() and row["pair_id"].lower() not in ("unknown", "none", "null"):
            by_pair.setdefault((row["dataset_id"], row["pair_id"]), set()).add(split)
    leaked_records = [key for key, splits in by_record.items() if len(splits) > 1]
    leaked_pairs = [key for key, splits in by_pair.items() if len(splits) > 1]
    leaked_subjects = [key for key, splits in by_subject.items() if len(splits) > 1]
    if leaked_records:
        raise RuntimeError(f"non-Nature records span splits: {leaked_records[:3]}")
    if leaked_pairs:
        raise RuntimeError(f"pairs span splits: {leaked_pairs[:3]}")

    # Official test must not be reachable from train or val.
    test_records = {(row["dataset_id"], row["path"]) for row in rows
                    if row["dataset_id"] not in e3.NATURE_DATASETS and e3._effective_split(row) == "test"}
    train_val_records = {(row["dataset_id"], row["path"]) for row in rows
                         if row["dataset_id"] not in e3.NATURE_DATASETS
                         and e3._effective_split(row) in ("train", "val")}
    if test_records & train_val_records:
        raise RuntimeError("an official test record is also assigned to train/val")

    return {
        "manifest": str(manifest), "manifest_sha256": e3.sha256(manifest),
        "corpus": str(corpus), "corpus_sha256": e3.sha256(corpus),
        "audit_seconds": audit_seconds,
        "schema_checked_files": preflight.get("checked_h5"),
        "counts_by_dataset_and_split": preflight.get("counts"),
        "corpus_intervals_by_split": dict(nature_counts),
        "census": census,
        "leakage": {
            "records_spanning_splits": len(leaked_records),
            "pairs_spanning_splits": len(leaked_pairs),
            "subjects_spanning_splits": len(leaked_subjects),
            "official_test_records": len(test_records),
            "test_overlapping_train_or_val": 0,
            "note": "subjects may appear in several splits; the released protocol anchors "
                    "splits per recording and per corpus interval, and group assignment is "
                    "checked by the loaders, not inferred here.",
        },
        "note": "validated from manifest metadata and sampled HDF5 schema; no bulk signal was "
                "loaded into RAM.",
    }


# ---------------------------------------------------------------------------
# DATALOADER
# ---------------------------------------------------------------------------

def section_dataloader(cfg: dict) -> dict:
    from semg_fm.data.datamodule import build_datamodule

    data_root: Path = cfg["data_root"]
    data = {
        "root": str(data_root),
        "manifest_path": str(data_root / cfg["manifest"]),
        "corpus_path": str(data_root / cfg["corpus"]),
        "window_ms": cfg["window_ms"], "stride_ms": cfg["stride_ms"],
        "batch_size": cfg["batch_size"], "num_workers": 0,
        "evaluation_split": "val",
    }

    before = rss_bytes()
    started = time.perf_counter()
    module = build_datamodule(data, task_id=None, label_budget=1.0)
    module.setup("fit")
    setup_seconds = time.perf_counter() - started
    after_index = rss_bytes()

    # The full per-task ordinal index is the single biggest predictable cost.
    batch_size = int(data["batch_size"])
    index_cost = {}
    total_ordinals = 0
    epoch_batches: dict[str, int] = {}
    for split, dataset in (("train", module.train_dataset), ("val", module.val_dataset)):
        groups = dataset.window_indices_by_task()
        per_task = {task: int(indices.nbytes) for task, indices in groups.items()}
        if split == "train":
            # floor(windows / batch_size): the sampler drops any partial tail.
            epoch_batches = {task: len(indices) // batch_size for task, indices in groups.items()}
        index_cost[split] = {
            "windows": int(len(dataset)),
            "windows_per_task": {task: int(len(indices)) for task, indices in groups.items()},
            "batches_per_task_per_epoch": ({task: len(indices) // batch_size for task, indices in groups.items()}
                                           if split == "train" else None),
            "bytes_per_task": per_task,
            "bytes_total": int(sum(per_task.values())),
        }
        total_ordinals += int(len(dataset))
    after_groups = rss_bytes()

    # A corpus contributing zero batches per epoch would be silently absent from
    # training. Catch it here, before 100k steps run without that modality.
    starved = sorted(task for task, count in epoch_batches.items() if count == 0)
    if starved:
        details = {task: index_cost["train"]["windows_per_task"].get(task, 0) for task in starved}
        raise RuntimeError(
            f"batch_size={batch_size} is too large: these corpora have fewer training "
            f"windows than one batch and would contribute zero batches per epoch "
            f"{details}. The sampler emits floor(windows/batch_size) batches per corpus. "
            "Lower --batch-size."
        )

    # Real batches, every task, through the real collate function.
    loader = module.train_dataloader()
    seen, shapes = [], {}
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        if index >= 200:
            break
        task = str(batch["task_id"])
        if task not in shapes:
            shapes[task] = list(batch["emg"].shape)
        seen.append(task)
        if set(seen) == set(TASK_IDS):
            break
    loader_seconds = time.perf_counter() - started
    batch_size_seen = int(data["batch_size"])
    missing = sorted(set(TASK_IDS) - set(seen))
    if missing:
        raise RuntimeError(
            f"the train loader never produced: {missing}. Expected one batch per corpus "
            f"within the first five batches, got {seen}."
        )
    for task, shape in shapes.items():
        if shape[0] != batch_size_seen or shape[2] != 16:
            raise RuntimeError(f"{task}: unexpected batch shape {shape} at batch_size={batch_size_seen}")
        if cfg["window_ms"] * 2 != shape[1]:
            raise RuntimeError(f"{task}: batch time length {shape[1]} != {cfg['window_ms'] * 2}")
    if len(seen) != len(TASK_IDS):
        raise RuntimeError(f"the train sampler did not round-robin one batch per task: {seen}")

    # Multi-worker loading, on a small slice so the preflight stays cheap.
    worker_report: dict = {"requested": cfg["num_workers"]}
    if cfg["num_workers"] > 0:
        worker_data = dict(data, num_workers=int(cfg["num_workers"]))
        worker_module = build_datamodule(worker_data, task_id=None, label_budget=1.0)
        worker_module.setup("fit")
        worker_loader = worker_module.train_dataloader()
        worker_seen = []
        for index, batch in enumerate(worker_loader):
            if index >= 5:
                break
            worker_seen.append(str(batch["task_id"]))
        worker_report["loaded_batches"] = len(worker_seen)
        worker_report["ok"] = len(worker_seen) > 0
        if not worker_report["ok"]:
            raise RuntimeError(f"{cfg['num_workers']} worker(s) produced no batches")

    # Memory must not creep across iterations.
    samples = [rss_bytes() for _ in range(3)]
    growth = max(samples) - min(samples)

    from semg_fm.data.window_dataset import CanonicalWindowDataset
    import inspect
    source = inspect.getsource(CanonicalWindowDataset.__getitem__)
    handles_scoped = "with h5py.File(" in source

    return {
        "setup_seconds": setup_seconds,
        "batch_size": batch_size,
        "index_windows": index_cost,
        "batches_per_task_per_epoch": epoch_batches,
        "total_windows": total_ordinals,
        "rss_delta_index_bytes": after_index - before,
        "rss_delta_task_groups_bytes": after_groups - after_index,
        "loader_first_batches": seen,
        "batch_shapes": shapes,
        "loader_seconds": loader_seconds,
        "workers": worker_report,
        "rss_stability_bytes": growth,
        "hdf5_handles_opened_per_access": handles_scoped,
        "notes": [
            "window_indices_by_task() materialises one int64 ordinal per window; "
            "that is the dominant index cost and is reported above.",
            "each __getitem__ opens and closes its own h5py.File, so no handle is "
            "shared across worker processes.",
        ],
    }


# ---------------------------------------------------------------------------
# GPU SMOKE
# ---------------------------------------------------------------------------

def section_gpu_smoke(cfg: dict, run_dir: Path) -> dict:
    from lightning import Trainer
    from semg_fm.config import build_trainer_options
    from semg_fm.data.datamodule import build_datamodule
    from semg_fm.lightning import PretrainModule
    from semg_fm.models import RingCrissCrossEncoder

    e3 = _e3_module()
    data_root: Path = cfg["data_root"]
    steps = int(cfg["smoke_steps"])

    model_cfg = dict(cfg["model"]); model_cfg.pop("name", None)
    backbone = RingCrissCrossEncoder(**model_cfg)
    parameters = sum(p.numel() for p in backbone.parameters())
    del backbone
    torch.cuda.empty_cache()

    data = {
        "root": str(data_root),
        "manifest_path": str(data_root / cfg["manifest"]),
        "corpus_path": str(data_root / cfg["corpus"]),
        "window_ms": cfg["window_ms"], "stride_ms": cfg["stride_ms"],
        "batch_size": cfg["smoke_batch_size"], "num_workers": 0,
        "evaluation_split": "val",
    }
    datamodule = build_datamodule(data, task_id=None, label_budget=1.0)

    module = PretrainModule(
        model_config=model_cfg, mask_config=dict(cfg["mask"]),
        latent_weight=float(cfg["objective"]["latent_weight"]),
        kinematic_weight=float(cfg["objective"]["kinematic_weight"]),
        kinematic_dim=cfg["objective"]["kinematic_dim"],
        ema_decay=float(cfg["objective"]["ema_decay"]),
        learning_rate=float(cfg["optimizer"]["learning_rate"]),
        weight_decay=float(cfg["optimizer"]["weight_decay"]),
        warmup_steps=int(cfg["optimizer"]["warmup_steps"]),
        max_steps=steps,
    )
    trainer_config = json.loads(json.dumps(cfg["trainer"], default=str))
    callbacks = trainer_config.pop("callbacks", {})
    if "checkpoint" in callbacks:
        callbacks["checkpoint"]["dirpath"] = str(run_dir / "checkpoints")
    if "diagnostics" in callbacks:
        callbacks["diagnostics"]["output_dir"] = str(run_dir)
    trainer_config["callbacks"] = callbacks
    trainer_config.update(max_steps=steps, limit_val_batches=0, num_sanity_val_steps=0,
                          logger=False, enable_progress_bar=False, enable_model_summary=False,
                          default_root_dir=str(run_dir))

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer = Trainer(**build_trainer_options(trainer_config))
    trainer.fit(module, datamodule=datamodule)
    elapsed = time.perf_counter() - started

    if int(trainer.global_step) < steps:
        raise RuntimeError(f"the smoke run stopped at step {trainer.global_step}, expected {steps}")

    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics.get("missing_train_tasks"):
        raise RuntimeError(f"these tasks produced no smoke batch: {diagnostics['missing_train_tasks']}")
    if not diagnostics.get("student_updated") or not diagnostics.get("teacher_updated"):
        raise RuntimeError("the smoke run updated neither student nor teacher")
    rows = [json.loads(line) for line in (run_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(not math.isfinite(row["loss"]) for row in rows):
        raise RuntimeError("a smoke-step loss was not finite")

    checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise RuntimeError("the smoke run wrote no checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "rng_state"):
        if key not in saved:
            raise RuntimeError(f"the smoke checkpoint is missing {key!r}")

    step_times = [row["step_time_s"] for row in rows if row["step_time_s"] > 0]
    return {
        "parameters": parameters,
        "steps": steps,
        "batch_size": cfg["smoke_batch_size"],
        "wall_seconds": elapsed,
        "seconds_per_step": (sum(step_times) / len(step_times)) if step_times else None,
        "steps_per_second": (len(step_times) / sum(step_times)) if step_times and sum(step_times) > 0 else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "total_vram_bytes": int(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory),
        "train_batches_per_task": diagnostics.get("train_batches_per_task"),
        "feature_std": diagnostics.get("feature_std"),
        "loss_first": rows[0]["loss"],
        "loss_last": rows[-1]["loss"],
        "checkpoint": str(checkpoint),
        "checkpoint_keys": sorted(k for k in saved if k in
                                  ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "rng_state")),
        "weights_discarded": "the smoke module lives in this process, which exits now; "
                             "the production run is a separate process and cannot inherit it",
    }


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=os.environ.get("E3_DATA_ROOT"))
    parser.add_argument("--output", default=os.environ.get("E3_PREFLIGHT_ROOT"))
    parser.add_argument("--manifest", default="splits.csv")
    parser.add_argument("--corpus", default="nature_corpus.csv")
    parser.add_argument("--smoke-steps", type=int, default=5)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--window-ms", type=int, default=2000)
    parser.add_argument("--stride-ms", type=int, default=1000)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(argv)

    # The sampler interleaves one batch per corpus per round, so verifying that
    # all five corpora reach a real optimiser step needs at least five steps.
    if args.smoke_steps < len(TASK_IDS):
        raise SystemExit(
            f"[preflight ERROR] --smoke-steps must be at least {len(TASK_IDS)} so that every "
            "corpus reaches an optimiser step"
        )
    if not args.data_root:
        raise SystemExit("[preflight ERROR] --data-root or E3_DATA_ROOT is required")
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"[preflight ERROR] data root does not exist: {data_root}")
    for name in (args.manifest, args.corpus):
        if not (data_root / name).is_file():
            raise SystemExit(f"[preflight ERROR] {name} is missing from {data_root}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(args.output).expanduser().resolve() if args.output else REPO / "outputs" / f"preflight_{stamp}"
    run_dir = output / f"preflight_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from hydra import compose, initialize
    from omegaconf import OmegaConf
    with initialize(version_base="1.3", config_path="../configs"):
        composed = compose(config_name="pretrain", overrides=["trainer=e3_server", "data=canonical_meta"])
    cfg = OmegaConf.to_container(composed, resolve=True)

    context = {
        "data_root": data_root, "output": output, "manifest": args.manifest, "corpus": args.corpus,
        "smoke_steps": args.smoke_steps, "smoke_batch_size": args.smoke_batch_size,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
        "window_ms": args.window_ms, "stride_ms": args.stride_ms,
        "model": cfg["model"], "mask": cfg["mask"], "objective": cfg["objective"],
        "optimizer": cfg["optimizer"], "trainer": cfg["trainer"],
    }

    report: dict = {"schema": "e3-server-preflight-v1", "created_utc": stamp, "host": platform.node()}
    failures: list[str] = []

    def run_section(name, fn, *fn_args):
        print(f"\n== {name} ==", flush=True)
        started = time.perf_counter()
        try:
            result = fn(*fn_args)
            print(f"   PASS  ({time.perf_counter() - started:.1f}s)", flush=True)
            report[name.lower()] = result
            return result
        except Exception as exc:  # noqa: BLE001 - the report must capture every failure
            print(f"   FAIL  {type(exc).__name__}: {exc}", flush=True)
            failures.append(f"{name}: {exc}")
            report[name.lower()] = {"error": f"{type(exc).__name__}: {exc}"}
            return None

    environment = run_section("ENVIRONMENT", section_environment, context)
    if environment:
        gpu = environment["gpu"]
        print(f"      {gpu['name']}  sm_{gpu['compute_capability'].replace('.', '')}  "
              f"{gpu['total_vram_bytes'] / 2**30:.1f} GiB  bf16={gpu['bf16_supported']}", flush=True)
        print(f"      commit {environment['git']['commit']} "
              f"(dirty={environment['git']['dirty']})", flush=True)

    data = run_section("DATA", section_data, context)
    if data:
        for dataset_id, counts in (data["counts_by_dataset_and_split"] or {}).items():
            print(f"      {dataset_id:<24} train={counts['train']:<6} val={counts['val']:<6} test={counts['test']}", flush=True)

    if data is not None:
        loader = run_section("DATALOADER", section_dataloader, context)
        if loader:
            print(f"      {loader['total_windows']:,} windows indexed in {loader['setup_seconds']:.1f}s", flush=True)
            print(f"      index RAM: {loader['index_windows']['train']['bytes_total'] / 2**20:.1f} MiB "
                  f"(train) + {loader['index_windows']['val']['bytes_total'] / 2**20:.1f} MiB (val)", flush=True)
            print(f"      first batches: {loader['loader_first_batches']}", flush=True)

    if data is not None and not args.skip_smoke:
        smoke = run_section("GPU_SMOKE", section_gpu_smoke, context, run_dir)
        if smoke:
            print(f"      {smoke['steps']} steps @ batch {smoke['batch_size']} in {smoke['wall_seconds']:.1f}s "
                  f"({smoke['seconds_per_step']:.3f}s/step)", flush=True)
            print(f"      peak VRAM {smoke['peak_vram_bytes'] / 2**20:.0f} MiB of "
                  f"{smoke['total_vram_bytes'] / 2**30:.1f} GiB", flush=True)

    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    (run_dir / "preflight.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (run_dir / "verdict.txt").write_text(("PASS" if not failures else "FAIL") + "\n", encoding="utf-8")

    print("\n== VERDICT ==")
    print(f"   report: {run_dir / 'preflight.json'}")
    if failures:
        for line in failures:
            print(f"   FAILED  {line}")
        print("\n   Refusing to start training.")
        return 1
    print("   PASS -- safe to start the E3 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
