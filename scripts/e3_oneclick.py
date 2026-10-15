#!/usr/bin/env python3
"""E3-only five-corpus pretraining launcher for the existing metafmodel repository.

Place this script in metafmodel/scripts/e3_oneclick.py.
Uses the repository's original pretrain.py, RingCrissCrossEncoder, EMA and Lightning.
Only synthetic smoke data is created locally. Production mode NEVER invents splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

TASKS = {
    "emg2pose": "pose",
    "emg2qwerty": "typing",
    "gni_discrete_gestures": "discrete_gesture",
    "gni_handwriting": "handwriting",
    "gni_wrist": "wrist_angle",
}
SPLITS = {"train", "val", "test"}
NATURE_DATASETS = frozenset({"gni_discrete_gestures", "gni_handwriting", "gni_wrist"})
CORPUS_COLUMNS = ("dataset", "split", "start", "end")
REQUIRED = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side",
    "pair_id", "official_split", "sample_count", "duration_s",
    "sample_rate_hz", "source_file", "source_sha256",
]
MARKER = ".E3_SYNTHETIC_DO_NOT_TRAIN"


class E3Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    dig = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(2**20), b""):
            dig.update(block)
    return dig.hexdigest()


def _effective_split(row: dict[str, str]) -> str:
    """Mirror window_dataset.from_manifest: official split wins."""
    official = row["official_split"].strip().lower()
    fallback = row.get("split", "").strip().lower()
    if official in SPLITS:
        return official
    if fallback in SPLITS:
        return fallback
    raise E3Error(f"No usable official/fallback split in {row['path']}")

def read_nature_corpus(path: Path) -> list[dict[str, str]]:
    import math
    if not path.is_file():
        raise E3Error(f"Missing official Nature corpus: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not set(CORPUS_COLUMNS).issubset(reader.fieldnames or []):
            raise E3Error(f"Nature corpus must have columns {CORPUS_COLUMNS}: {path}")
        rows = list(reader)
    if not rows:
        raise E3Error(f"Nature corpus is empty: {path}")
    by_source = defaultdict(list)
    for r in rows:
        name = Path(r["dataset"]).name
        if not name:
            raise E3Error("Blank source basename in Nature corpus")
        try:
            a, b = float(r["start"]), float(r["end"])
        except ValueError as exc:
            raise E3Error(f"Non-numeric corpus boundary: {r}") from exc
        if not math.isfinite(a) or not math.isfinite(b) or a >= b or r["split"] not in SPLITS:
            raise E3Error(f"Invalid half-open Nature interval: {r}")
        by_source[name].append((a, b, r["split"]))
    for name, segments in by_source.items():
        order = sorted(segments)
        if any(left[1] > right[0] for left, right in zip(order, order[1:])):
            raise E3Error(f"Overlapping Nature intervals for {name}")
    return rows


def find_nature_corpus(root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = root / "nature_corpus.csv"
    if not path.is_file():
        raise E3Error(
            f"Nature corpus not found: {path}. Preserve the three official "
            "GNI task corpora in one CSV with dataset,start,end,split, "
            "named nature_corpus.csv under --data-root (or pass --corpus). "
            "Do not infer Nature train/val from filenames."
        )
    read_nature_corpus(path)
    return path


def verify_real_datamodule(root: Path, manifest: Path, corpus: Path, probe_loader: bool = True) -> dict:
    """Run the actual repo Dataset, train sampler, validation sampler, and collate."""
    from semg_fm.data.datamodule import build_datamodule
    config = {
        "root": str(root.resolve()), "manifest_path": str(manifest.resolve()),
        "window_ms": 2000, "stride_ms": 1000, "batch_size": 1,
        "num_workers": 0, "corpus_path": str(corpus.resolve()),
        "evaluation_split": "val",
    }
    module = build_datamodule(config, task_id=None, label_budget=1.0)
    module.setup("fit")
    required = set(TASKS.values())
    inspected = {}
    for split, dataset in (("train", module.train_dataset), ("val", module.val_dataset)):
        groups = dataset.window_indices_by_task()
        actual = set(groups)
        if actual != required:
            raise E3Error(f"Real {split} Dataset has {sorted(actual)}, expected {sorted(required)}")
        for task_id, inds in groups.items():
            if len(inds) == 0:
                raise E3Error(f"No windows for {task_id} in real {split} Dataset")
            sample = dataset[int(inds[0])]
            if tuple(sample["emg"].shape) != (4000, 16):
                raise E3Error(f"Wrong real Dataset {split}/{task_id} EMG: {sample['emg'].shape}")
            if not bool(sample["sample_mask"].all()):
                raise E3Error(f"Unexpected empty sample mask for {split}/{task_id}")
        seen = []
        if probe_loader:
            batches = module.train_dataloader() if split == "train" else module.val_dataloader()
            for index, batch in enumerate(batches):
                if index >= 100:
                    break
                if tuple(batch["emg"].shape) != (1, 4000, 16):
                    raise E3Error(f"Real DataLoader unexpected batch shape: {batch['emg'].shape}")
                seen.append(batch["task_id"])
                if set(seen) == required:
                    break
            if set(seen) != required:
                raise E3Error(
                    f"Real {split} loader did not yield all five tasks in the first "
                    f"{len(seen)} batches: {seen}"
                )
            if split == "train" and len(seen) != len(required):
                raise E3Error(f"Train sampler failed one-batch-per-task round robin: {seen}")
        inspected[split] = {"windows": len(dataset), "first_batches": seen}
    return inspected


def read_manifest(root: Path, manifest: Path) -> list[dict[str, str]]:
    """Match the actual metafmodel split semantics; Nature is interval based."""
    if not manifest.is_file():
        raise E3Error(f"Missing manifest: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = sorted(set(REQUIRED) - set(reader.fieldnames or []))
        if missing:
            raise E3Error(f"Manifest missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise E3Error("Manifest has zero rows")
    root = root.resolve()
    nature_basenames = set()
    split_by_path = defaultdict(set)
    for row in rows:
        p = (root / row["path"]).resolve()
        if not p.is_relative_to(root):
            raise E3Error(f"Path escapes data root: {row['path']}")
        if not p.is_file():
            raise E3Error(f"Missing HDF5: {p}")
        row["_path"] = str(p)
        ds = row["dataset_id"]
        if ds not in TASKS or row["task_id"] != TASKS[ds]:
            raise E3Error(f"Unknown dataset/task combination: {ds}/{row['task_id']}")
        if ds in NATURE_DATASETS:
            # Original selected_records() always requires the corpus, even
            # when official_split is unknown. There is no per-file split here.
            row["_split"] = "corpus"
            basename = Path(row["source_file"]).name
            if basename in nature_basenames:
                raise E3Error(f"Ambiguous Nature source basename: {basename}")
            nature_basenames.add(basename)
        else:
            row["_split"] = _effective_split(row)
            split_by_path[str(p)].add(row["_split"])
        try:
            count = int(row["sample_count"])
            rate = float(row["sample_rate_hz"])
        except ValueError as ex:
            raise E3Error(f"Invalid numeric metadata in {row['path']}") from ex
        if count < 1 or abs(rate - 2000.0) > 0.01:
            raise E3Error(f"Expected 2 kHz and positive samples: {row['path']}")
    bad = [p for p, splits in split_by_path.items() if len(splits) > 1]
    if bad:
        raise E3Error(f"Non-Nature H5 reused across splits: {bad[:3]}")
    return rows

def audit(root: Path, manifest: Path, corpus: Path | None = None,
          window_ms: int = 2000, inspect_per_group: int = 2,
          inspect_all: bool = False) -> dict:
    """Preflight the actual canonical schema and official Nature half-open intervals."""
    import h5py
    import numpy as np

    rows = read_manifest(root, manifest)
    nature = [r for r in rows if r["dataset_id"] in NATURE_DATASETS]
    if nature and corpus is None:
        raise E3Error("Nature records require a real corpus_path")
    corpus_rows = read_nature_corpus(corpus) if nature else []
    by_basename = defaultdict(list)
    for r in corpus_rows:
        by_basename[Path(r["dataset"]).name].append(r)

    count = Counter()
    non_nature_groups = defaultdict(set)
    for r in rows:
        if r["dataset_id"] in NATURE_DATASETS:
            entries = by_basename.get(Path(r["source_file"]).name, [])
            if not entries:
                raise E3Error(f"Nature source missing from corpus: {r['source_file']}")
            for entry in entries:
                if r["official_split"] in SPLITS and entry["split"] != r["official_split"]:
                    raise E3Error(f"Official Nature split conflict: {r['source_file']}")
                count[(r["dataset_id"], entry["split"])] += 1
        else:
            count[(r["dataset_id"], r["_split"])] += 1
            ds = r["dataset_id"]
            non_nature_groups[("file", ds, r["_path"])].add(r["_split"])
            if r["source_sha256"].strip() and len(r["source_sha256"].strip()) == 64:
                non_nature_groups[("source", ds, r["source_sha256"])].add(r["_split"])
            pair = r["pair_id"].strip()
            if pair and pair.lower() not in ("unknown", "none", "null"):
                non_nature_groups[("pair", ds, pair)].add(r["_split"])
    conflicts = [repr(k) for k,v in non_nature_groups.items() if len(v)>1]
    if conflicts:
        raise E3Error(f"Non-Nature source/pair split leakage: {conflicts[:4]}")
    missing = [(ds, split) for ds in TASKS for split in ("train", "val") if count[(ds, split)] == 0]
    if missing:
        raise E3Error(f"Train/Val interval or file missing: {missing}")

    selected = rows if inspect_all else [
        row for ds in TASKS for row in [r for r in rows if r["dataset_id"] == ds][:inspect_per_group]
    ]
    checked = 0
    for r in selected:
        p = Path(r["_path"])
        with h5py.File(p, "r") as h:
            attr_pairs = {
                "dataset_id": r["dataset_id"], "task_id": r["task_id"],
                "subject_id": r["subject_id"], "session_id": r["session_id"],
                "side": r["side"], "pair_id": r["pair_id"],
                "official_split": r["official_split"],
                "source_file": r["source_file"], "schema_version": "1.0",
            }
            for key, value in attr_pairs.items():
                actual = h.attrs.get(key)
                if isinstance(actual, bytes):
                    actual = actual.decode()
                if str(actual) != value:
                    raise E3Error(f"{p}: attr {key} mismatch: {actual!r} != {value!r}")
            if abs(float(h.attrs["sample_rate_hz"]) - 2000.0) > 0.01:
                raise E3Error(f"{p}: Wrong HDF5 sample_rate_hz")
            if "signal/emg" not in h or "signal/time_s" not in h:
                raise E3Error(f"{p}: Missing signal/emg or signal/time_s")
            emg, t = h["signal/emg"], h["signal/time_s"]
            if emg.dtype != np.dtype("float32") or emg.ndim != 2 or emg.shape[1] != 16:
                raise E3Error(f"{p}: EMG must be float32 [T,16]")
            if t.dtype != np.dtype("float64") or t.ndim != 1 or t.shape[0] != emg.shape[0]:
                raise E3Error(f"{p}: time_s must be float64 [T]")
            if emg.shape[0] != int(r["sample_count"]):
                raise E3Error(f"{p}: sample_count mismatch")
            required_window = int(window_ms * 2)
            if len(emg) < required_window:
                raise E3Error(f"{p}: shorter than {window_ms} ms window")
            if not np.isfinite(emg[:required_window]).all():
                raise E3Error(f"{p}: NaN/Inf in first EMG window")
            edge = np.asarray(t[:min(required_window+1, len(t))])
            if not np.isfinite(edge).all() or np.any(np.diff(edge) < 0):
                raise E3Error(f"{p}: invalid relative signal clock")
            if r["dataset_id"] in NATURE_DATASETS:
                if "meta/source_json" not in h:
                    raise E3Error(f"{p}: Nature H5 missing meta/source_json")
                try:
                    meta = json.loads(h["meta/source_json"].asstr()[()])
                    origin = float(meta["source_time_origin_s"])
                except Exception as exc:
                    raise E3Error(f"{p}: malformed source_time_origin_s") from exc
                if not np.isfinite(origin):
                    raise E3Error(f"{p}: non-finite time origin")
                # This bounds check detects an entirely nonoverlapping corpus,
                # while the ORIGINAL intervals.corpus_bounds does exact searchsorted.
                t_start = float(t[0]) + origin
                t_last = float(t[-1]) + origin
                entries = by_basename[Path(r["source_file"]).name]
                if not any(float(e["start"]) <= t_last and float(e["end"]) > t_start for e in entries):
                    raise E3Error(f"{p}: no corpus interval overlaps the source recording")
        checked += 1
    return {
        "schema": "e3-preflight-v3",
        "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest),
        "nature_corpus": str(corpus.resolve()) if corpus else None,
        "nature_corpus_sha256": sha256(corpus) if corpus else None,
        "records": len(rows), "checked_h5": checked,
        "counts": {ds: {sp: count[(ds, sp)] for sp in ("train","val","test")} for ds in TASKS},
        "note": "Nature counts are number of corpus intervals, non-Nature counts are files. Exact windows checked by the original DataModule.",
    }

def create_synthetic(root: Path) -> Path:
    """A real-format MINIMAL canonical fixture; never to be used for research."""
    import h5py
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    marker = root / MARKER
    if marker.exists():
        corpus = root / "nature_corpus.csv"
        if not corpus.is_file():
            raise E3Error("Incomplete synthetic fixture: missing nature_corpus.csv")
        return root / "splits.csv"
    if any(root.iterdir()):
        raise E3Error(f"Refusing to generate synthetic data in nonempty directory: {root}")
    marker.write_text("SYNTHETIC ONLY: DO NOT USE FOR PRODUCTION TRAINING\n", encoding="utf-8")
    rng = np.random.default_rng(2026)
    fs, samples = 2000, 8000  # 4 seconds, sufficient for 2 s / 1 s windows
    records = []
    text_dtype = "S24"
    for ds, task in TASKS.items():
        for split in ("train", "val"):
            for side in (("left", "right") if task == "typing" else ("right",)):
                subject = f"{ds}_{split}_user"
                session = f"{split}_session"
                pair = f"typing_{split}_pair" if task == "typing" else ""
                rel = f"{ds}/{task}/{subject}/{ds}_{session}_{side}.h5"
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                t = np.arange(samples, dtype=np.float64) / fs
                emg = (rng.standard_normal((samples, 16)) * 0.1).astype("float32")
                # Enough nontrivial temporal/channel structure to test gradient propagation
                emg += (0.02 * np.sin(2 * np.pi * 65 * t)[:, None]).astype("float32")
                with h5py.File(path, "w") as h:
                    for key, val in {
                        "schema_version": "1.0", "source_file": rel, "dataset_id": ds, "task_id": task,
                        "subject_id": subject, "session_id": session, "side": side,
                        "pair_id": pair, "official_split": split,
                        "record_id": f"{ds}_{split}_{side}", "sample_rate_hz": fs,
                        "preprocessing_json": "{}",
                    }.items():
                        h.attrs[key] = val
                    meta = h.create_group("meta")
                    meta.create_dataset("source_json",
                                        data=json.dumps({"source_time_origin_s": 0.0, "synthetic": True}),
                                        dtype=h5py.string_dtype(encoding="utf-8"))
                    sig = h.create_group("signal")
                    sig.create_dataset("emg", data=emg, chunks=(2000, 16))
                    sig.create_dataset("time_s", data=t, chunks=(2000,))
                    labels = h.create_group("labels")
                    if task == "pose":
                        labels.create_dataset("joint_angles", data=np.zeros((samples, 20), dtype="float32"))
                        labels.create_dataset("no_ik_failure", data=np.ones(samples, dtype="?"))
                    elif task == "wrist_angle":
                        targets = labels.create_group("targets")
                        targets.create_dataset("angle", data=np.zeros((samples, 1), dtype="float32"))
                    elif task == "discrete_gesture":
                        events = np.array([(b"thumb_up", 0.5)], dtype=[("name", text_dtype), ("time_s", "f8")])
                        labels.create_dataset("events", data=events)
                    else:
                        prompts = np.array([(b"example", 0.0, 4.0)], dtype=[
                            ("text", text_dtype), ("start_s", "f8"), ("end_s", "f8")])
                        labels.create_dataset("prompts", data=prompts)
                        if task == "typing":
                            keys = np.array([(b"a", 0.5, 0.55)], dtype=[
                                ("key", text_dtype), ("press_s", "f8"), ("release_s", "f8")])
                            labels.create_dataset("keystrokes", data=keys)
                records.append({
                    "path": rel, "dataset_id": ds, "task_id": task, "subject_id": subject,
                    "session_id": session, "side": side, "pair_id": pair,
                    "official_split": split, "split": split,
                    "sample_count": samples, "duration_s": 4.0,
                    "sample_rate_hz": fs, "source_file": rel,
                    "source_sha256": hashlib.sha256(rel.encode()).hexdigest(),
                })
    nature_rows = [r for r in records if r["dataset_id"].startswith("gni_")]
    names = [Path(r["source_file"]).name for r in nature_rows]
    if len(names) != len(set(names)):
        raise E3Error("Synthetic Nature source filename collision; fixture is invalid")
    manifest = root / "splits.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REQUIRED + ["split"])
        writer.writeheader()
        writer.writerows(records)
    # A synthetic analogue of the *official* Nature corpus protocol, never
    # confuse it with actual Meta files. Every entry is a half-open [0, 4) s
    # source-time interval with an explicit split.
    corpus = root / "nature_corpus.csv"
    with corpus.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CORPUS_COLUMNS))
        writer.writeheader()
        for r in records:
            if r["dataset_id"] in NATURE_DATASETS:
                writer.writerow({
                    "dataset": r["source_file"], "split": r["official_split"],
                    "start": "0.0", "end": "4.0",
                })
    return manifest


def command(repo: Path, data_root: Path, manifest: Path, corpus: Path, run: Path,
            mode: str, max_steps: int | None, resume: Path | None,
            batch_size: int | None = None) -> list[str]:
    import torch

    if not (repo / "pretrain.py").is_file():
        raise E3Error(f"Missing metafmodel/pretrain.py under {repo}")
    for module in ("hydra", "lightning", "h5py", "pandas", "torch"):
        if importlib.util.find_spec(module) is None:
            raise E3Error(f"Missing dependency '{module}'. Install metafmodel requirements in this Python environment.")
    cuda = bool(torch.cuda.is_available())
    if mode == "server" and not cuda:
        raise E3Error("SERVER mode requires CUDA; not silently falling back to CPU.")
    accelerator = "gpu" if cuda else "cpu"
    precision = "bf16-mixed" if cuda and torch.cuda.is_bf16_supported() else "32-true"
    if mode == "server" and precision != "bf16-mixed":
        raise E3Error("SERVER default requires verified bf16 support; review device and config.")
    local = mode == "local"
    steps = int(max_steps if max_steps is not None else (5 if local else 100000))
    if steps < 1:
        raise E3Error("max_steps must be >= 1")
    overrides = {
        "data.root": str(data_root.resolve()),
        "data.manifest_path": str(manifest.resolve()),
        "+data.corpus_path": str(corpus.resolve()),
        "data.window_ms": 2000,
        "data.stride_ms": 1000,
        "data.batch_size": batch_size if batch_size is not None else (1 if local else 8),
        "data.num_workers": 0 if local else 8,
        "model.d_model": 32 if (local and not cuda) else 128,
        "model.depth": 2 if (local and not cuda) else 6,
        "model.heads": 4 if (local and not cuda) else 8,
        "model.patch_samples": 40,
        "optimizer.learning_rate": 0.0003,
        "optimizer.max_steps": steps,
        "objective.kinematic_dim": None,
        "objective.kinematic_weight": 0.0,
        "trainer.accelerator": accelerator,
        "trainer.devices": 1,
        "trainer.strategy": "auto",
        "trainer.precision": precision,
        "trainer.val_check_interval": min(steps, 5 if local else 2000),
        "trainer.limit_val_batches": 5 if local else 250,
        "trainer.enable_progress_bar": False,
        "+trainer.num_sanity_val_steps": 0,
        "+trainer.log_every_n_steps": 1 if local else 50,
        "trainer.callbacks.checkpoint.save_last": True,
        "trainer.callbacks.checkpoint.save_top_k": 1,
        "trainer.callbacks.checkpoint.monitor": "val/loss",
        "trainer.callbacks.checkpoint.mode": "min",
        "hydra.run.dir": str(run.resolve()),
    }
    argv = [sys.executable, str(repo / "pretrain.py"), "trainer=e3_server"]
    for key, value in overrides.items():
        if value is None:
            val = "null"
        elif isinstance(value, bool):
            val = str(value).lower()
        else:
            val = str(value)
        if isinstance(value, str):
            # Hydra's override parser needs literal quoting for paths with blanks/equals.
            val = json.dumps(val.replace("\\", "/"))
        argv.append(f"{key}={val}")
    if resume:
        if not resume.is_file():
            raise E3Error(f"Requested checkpoint does not exist: {resume}")
        # Hydra string values containing '=' require quotes; Path.as_posix is safe here.
        argv.append("resume_from=" + json.dumps(str(resume.resolve()).replace("\\", "/")))
    return argv


def describe_checkpoint(path: Path, expected_steps: int) -> dict:
    """Assert a checkpoint holds everything a strict resume needs."""
    import torch

    # Trusted: written by this process in this run tree, never an unknown download.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = list(ckpt.get("state_dict", {}))
    student = [k for k in keys if k.startswith("student.")]
    teacher = [k for k in keys if k.startswith("teacher.")]
    step = int(ckpt.get("global_step", -1))
    if not student or not teacher or step < expected_steps:
        raise E3Error(f"Invalid checkpoint: step={step}, student={len(student)}, teacher={len(teacher)}")
    if not ckpt.get("optimizer_states"):
        raise E3Error(f"Checkpoint lacks optimizer state: {path}")
    if not ckpt.get("lr_schedulers"):
        raise E3Error(f"Checkpoint lacks the LR scheduler state: {path}")
    if "rng_state" not in ckpt:
        raise E3Error(f"Checkpoint lacks the RNG state needed for an exact resume: {path}")
    return {"path": str(path.resolve()), "sha256": sha256(path),
            "step": step, "student_tensors": len(student),
            "teacher_tensors": len(teacher), "optimizer_states": len(ckpt["optimizer_states"]),
            "lr_schedulers": len(ckpt["lr_schedulers"]), "rng_state": True}


def validate_checkpoint(run: Path, expected_steps: int) -> dict:
    candidates = list(run.rglob("last.ckpt"))
    if not candidates:
        raise E3Error(f"No last.ckpt saved in {run}")
    last = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    return describe_checkpoint(last, expected_steps)


def main(argv=None):
    parser = argparse.ArgumentParser(description="E3 five-corpus SSL: local / server / audit")
    parser.add_argument("mode", choices=("local", "server", "audit"))
    parser.add_argument("--data-root", default=os.environ.get("E3_DATA_ROOT"))
    parser.add_argument("--manifest", default="splits.csv")
    parser.add_argument("--corpus", default=os.environ.get("E3_NATURE_CORPUS"),
                        help="Official Nature corpus CSV; by default <data-root>/nature_corpus.csv")
    parser.add_argument("--output", default=os.environ.get("E3_RUN_ROOT", "outputs/e3_runs"))
    parser.add_argument("--repo", help="metafmodel repo root; default script's parent parent")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int,
                        help="override the mode default (1 local / 8 server)")
    parser.add_argument("--resume")
    parser.add_argument("--audit-all", action="store_true")
    parser.add_argument("--probe-loader", action="store_true",
                        help="Also read real Train/Val DataLoader batches in audit mode.")
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]

    if args.mode == "local":
        root = (repo / "outputs" / "e3_synthetic_v3").resolve()
        manifest = create_synthetic(root)
        corpus = find_nature_corpus(root, None)
    else:
        if not args.data_root:
            raise E3Error("--data-root or E3_DATA_ROOT is required in server/audit mode")
        root = Path(args.data_root).expanduser().resolve()
        if not root.is_dir():
            raise E3Error(f"Data root does not exist: {root}")
        if (root / MARKER).exists():
            raise E3Error("Synthetic HDF5 fixture is forbidden for production")
        manifest = Path(args.manifest).expanduser()
        manifest = manifest if manifest.is_absolute() else root / manifest
        corpus = find_nature_corpus(root, args.corpus)

    # A production run must never write checkpoints or logs into the published
    # source tree. Checked before the preflight so a misconfigured launch fails
    # fast, but after the synthetic-fixture guard, which is the more severe
    # error when both apply.
    if args.mode == "server":
        run_root = Path(args.output).expanduser().resolve()
        if run_root == repo or repo in run_root.parents:
            raise E3Error(
                f"Server output root must live outside the Git repository, but {run_root} "
                f"is inside {repo}. Export E3_RUN_ROOT (for example /srv/experiments/e3) so "
                "checkpoints and logs are never mixed into the source tree."
            )

    preflight = audit(root, manifest, corpus, inspect_all=(args.audit_all or args.mode == "server"))

    # Run the REAL installed metafmodel Dataset/DataModule, not just H5 schema tests.
    # `local` always probes the loader; the server preflight opts in explicitly so
    # that `verify_server.sh` exercises the same data path a launch would use.
    probe_loader = args.mode == "local" or args.probe_loader
    dataflow = verify_real_datamodule(root, manifest, corpus, probe_loader=probe_loader)
    # In audit mode stdout is reserved for the JSON report so a caller can pipe
    # it straight into a JSON parser; progress goes to stderr instead.
    stream = sys.stderr if args.mode == "audit" else sys.stdout
    print("[E3] Real Train/Val Dataset and five-task windows: PASS", flush=True, file=stream)
    if probe_loader:
        print("[E3] Real Train/Val DataLoader batch coverage: PASS", flush=True, file=stream)

    if args.mode == "audit":
        preflight["real_dataflow"] = dataflow
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(args.output).expanduser().resolve() / f"e3_{args.mode}_{now}"
    if run.exists():
        raise E3Error(f"Refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    info = {
        "mode": args.mode, "created_utc": now, "host": platform.node(),
        "python": sys.version.split()[0], "repository": str(repo),
        "preflight": preflight, "real_dataflow": dataflow, "status": "DATAFLOW_OK",
    }
    status_path = run / "e3_status.json"
    status_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    resume = Path(args.resume).resolve() if args.resume else None
    cli = command(repo, root, manifest, corpus, run, args.mode, args.max_steps, resume,
                  getattr(args, "batch_size", None))
    (run / "command.json").write_text(json.dumps(cli, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[E3] Executing ORIGINAL metafmodel pretrain.py with official Nature intervals.", flush=True)
    log = run / "e3_train.log"
    try:
        with log.open("w", encoding="utf-8") as out:
            p = subprocess.Popen(cli, cwd=repo, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            assert p.stdout is not None
            for line in p.stdout:
                out.write(line)
                out.flush()
                print(line, end="", flush=True)
            code = p.wait()
        info["returncode"] = code
        if code:
            raise E3Error(f"Training failed with exit code {code}; see {log}")
        expected = args.max_steps or (5 if args.mode == "local" else 100000)
        try:
            info["checkpoint"] = validate_checkpoint(run, expected)
        except E3Error:
            # Resuming a run that already reached the step limit does no further
            # work, so Lightning writes no checkpoint into this fresh run
            # directory. That is a clean completion: validate the checkpoint the
            # run was resumed FROM rather than reporting a failure. A run that
            # was not resuming still fails here.
            if not resume:
                raise
            info["checkpoint"] = describe_checkpoint(resume, expected)
            info["checkpoint_source"] = "resume_input"
        info["status"] = "COMPLETE"
    except Exception as exc:
        info["status"] = "TRAINING_FAILED"
        info["error"] = str(exc)
        raise
    finally:
        status_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[E3] COMPLETE:", info["checkpoint"]["path"], flush=True)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except E3Error as exc:
        print(f"[E3 ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
