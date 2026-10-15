#!/usr/bin/env python3
"""Export a small, portable, contract-faithful subset of the real five-corpus data.

    python scripts/export_real_mini.py \
        --data-root /srv/datasets/meta_canonical \
        --output    /srv/datasets/meta_real_mini

Run this on the machine that holds the real corpora, then transfer the result to
a workstation so the full training path can be exercised on REAL signal without
downloading hundreds of gigabytes.

What it guarantees
------------------
* **Official splits are never reassigned.** Only `train` and `val` records are
  eligible; `test` is never exported. `official_split` is copied verbatim.
* **The source clock is preserved.** Truncating a recording re-bases
  `signal/time_s` and moves `meta/source_json["source_time_origin_s"]` by the
  same amount, so `source_time = time_s + origin` is identical to the original at
  every kept sample. The official corpus intervals therefore still map onto the
  same source instants and `intervals.corpus_bounds` keeps working unmodified.
* **The data contract is unchanged.** Sampling rate, channel order, dtypes,
  labels and metadata keep the exact schema the converters wrote. Nothing is
  filtered, rescaled or resampled to make files smaller.
* **One record per source basename.** The loaders require Nature source
  basenames to be unique, so each source recording is written exactly once -- and
  when one source supplies both a train and a val interval, its single window is
  sized to contain both.

Why truncation instead of whole recordings: a single emg2pose session is hundreds
of megabytes. Copying whole files would blow the size target, so each selected
recording is cut to a contiguous window containing the official intervals it was
chosen for.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TASKS = {
    "emg2pose": "pose",
    "emg2qwerty": "typing",
    "gni_discrete_gestures": "discrete_gesture",
    "gni_handwriting": "handwriting",
    "gni_wrist": "wrist_angle",
}
NATURE_DATASETS = frozenset({"gni_discrete_gestures", "gni_handwriting", "gni_wrist"})
NEEDED_SPLITS = ("train", "val")
FORBIDDEN_SPLIT = "test"
CORPUS_COLUMNS = ("dataset", "split", "start", "end")
MANIFEST_COLUMNS = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_count", "duration_s", "sample_rate_hz", "source_file", "source_sha256",
]


class ExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _basename(row: dict[str, str]) -> str:
    return Path(row["source_file"]).name


def read_manifest(root: Path, manifest: Path) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise ExportError(f"missing manifest: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(MANIFEST_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ExportError(f"{manifest} is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ExportError(f"{manifest} has no records")
    resolved_root = root.resolve()
    for row in rows:
        target = (root / row["path"]).resolve()
        if not target.is_relative_to(resolved_root):
            raise ExportError(f"manifest path escapes the data root: {row['path']}")
        if not target.is_file():
            raise ExportError(f"manifest references a missing file: {target}")
    return rows


def read_corpus(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ExportError(f"missing official Nature corpus: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(CORPUS_COLUMNS).issubset(reader.fieldnames or []):
            raise ExportError(f"corpus must have columns {CORPUS_COLUMNS}: {path}")
        rows = list(reader)
    if not rows:
        raise ExportError(f"corpus is empty: {path}")
    for row in rows:
        lo, hi = float(row["start"]), float(row["end"])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ExportError(f"invalid corpus interval: {row}")
        if row["split"] not in (*NEEDED_SPLITS, FORBIDDEN_SPLIT):
            raise ExportError(f"unknown corpus split: {row['split']}")
    return rows


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class Plan:
    """One source recording and the official intervals the export must contain."""

    __slots__ = ("row", "intervals", "splits")

    def __init__(self, row: dict[str, str], intervals: dict[str, list[dict[str, str]]] | None = None,
                 splits: set[str] | None = None):
        self.row = row
        self.intervals = intervals or {}        # split -> chosen corpus entries
        self.splits = set(splits) if splits is not None else {s for s, v in self.intervals.items() if v}


def effective_split(row: dict[str, str]) -> str:
    """Mirror window_dataset.from_manifest: an official split wins over `split`."""
    official = (row.get("official_split") or "").strip().lower()
    fallback = (row.get("split") or "").strip().lower()
    if official in (*NEEDED_SPLITS, FORBIDDEN_SPLIT):
        return official
    if fallback in (*NEEDED_SPLITS, FORBIDDEN_SPLIT):
        return fallback
    return ""


def _spread(items: list, count: int, rng: np.random.Generator) -> list:
    """Pick `count` items, preferring distinct (subject, session) groups."""
    buckets: dict[tuple[str, str], list] = defaultdict(list)
    for row in items:
        buckets[(row["subject_id"], row["session_id"])].append(row)
    keys = sorted(buckets)
    rng.shuffle(keys)
    picked, depth = [], 0
    while len(picked) < count and keys:
        for key in list(keys):
            if len(picked) >= count:
                break
            bucket = buckets[key]
            picked.append(bucket[depth % len(bucket)])
            keys.remove(key)
        depth += 1
    return picked


def plan_nature(rows: list[dict[str, str]], corpus: list[dict[str, str]],
                per_split: int, seed: int) -> list[Plan]:
    by_basename: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in corpus:
        by_basename[Path(entry["dataset"]).name].append(entry)

    entries_for: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        name = _basename(row)
        entries = by_basename.get(name)
        if not entries:
            raise ExportError(
                f"Nature source {name} has no official corpus interval; it cannot be "
                "exported without its official split, and inventing one is not allowed"
            )
        entries_for[name] = {
            split: sorted((e for e in entries if e["split"] == split), key=lambda e: float(e["start"]))
            for split in NEEDED_SPLITS
        }

    rng = np.random.default_rng(seed)
    train_rows = [row for row in rows if entries_for[_basename(row)]["train"]]
    val_rows = [row for row in rows if entries_for[_basename(row)]["val"]]
    if not train_rows:
        raise ExportError("no Nature recording carries an official train interval")
    if not val_rows:
        raise ExportError("no Nature recording carries an official val interval")

    train_pick = _spread(train_rows, per_split, rng)
    train_names = {_basename(row) for row in train_pick}
    # Prefer val sources that are not already serving train: a shared source needs
    # one window covering both intervals, which can exceed the size target.
    val_only = [row for row in val_rows if _basename(row) not in train_names]
    val_pick = _spread(val_only or val_rows, per_split, rng)
    val_names = {_basename(row) for row in val_pick}

    plans: list[Plan] = []
    for name in sorted(train_names | val_names):
        row = next(item for item in rows if _basename(item) == name)
        chosen: dict[str, list[dict[str, str]]] = {}
        if name in train_names:
            chosen["train"] = [entries_for[name]["train"][0]]
        if name in val_names:
            entries = entries_for[name]["val"]
            if "train" in chosen:
                # The window spans both intervals; take the val interval nearest
                # the chosen train interval to keep the span as small as possible.
                train_start, train_end = float(chosen["train"][0]["start"]), float(chosen["train"][0]["end"])
                entries = sorted(entries, key=lambda e: min(
                    abs(float(e["start"]) - train_end), abs(train_start - float(e["end"]))
                ))
            chosen["val"] = [entries[0]]
        plans.append(Plan(row, chosen))
    return plans


def plan_plain(rows: list[dict[str, str]], per_split: int, seed: int) -> list[Plan]:
    """Plan a corpus that ships its own official split in the record metadata."""
    resolved: dict[str, str] = {}
    for row in rows:
        split = effective_split(row)
        if split == FORBIDDEN_SPLIT:
            continue  # never export test; the real corpus legitimately contains it
        if split not in NEEDED_SPLITS:
            raise ExportError(
                f"{row['path']} has official_split={row['official_split']!r} and no usable "
                "'split' column value, so its membership cannot be established. Run "
                "make_splits first; nothing may be relabelled."
            )
        resolved[row["path"]] = split

    rng = np.random.default_rng(seed)
    plans: list[Plan] = []
    for split in NEEDED_SPLITS:
        candidates = [row for row in rows if resolved.get(row["path"]) == split]
        if not candidates:
            raise ExportError(f"no record belongs to official {split}")
        # emg2qwerty writes one record per wrist sharing a pair_id. Both sides
        # must land in the same split or validate_group_assignments rejects the
        # manifest with "pair crosses split boundaries", so group by pair.
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            groups[row["pair_id"] or f"__{row['path']}"].append(row)
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys[:per_split]:
            for row in groups[key]:
                plans.append(Plan(row, {}, {split}))
    return plans


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _clock(source: Path) -> tuple[np.ndarray, float]:
    with h5py.File(source, "r") as handle:
        times = handle["signal/time_s"][:]
        origin = float(json.loads(handle["meta/source_json"].asstr()[()])["source_time_origin_s"])
    return times, origin


def nature_window(plan: Plan, source: Path, segment_seconds: float,
                  margin_seconds: float) -> tuple[int, int, list[dict[str, str]]]:
    times, origin = _clock(source)
    wanted = [entry for entries in plan.intervals.values() for entry in entries]
    lo = min(float(e["start"]) for e in wanted) - origin - margin_seconds
    hi = max(float(e["end"]) for e in wanted) - origin + margin_seconds
    a = int(np.searchsorted(times, lo, side="right"))
    b = int(np.searchsorted(times, hi, side="left"))
    cap = int(segment_seconds * float(plan.row["sample_rate_hz"]))
    if b - a > cap:
        centre = (a + b) // 2
        a = max(0, min(centre - cap // 2, len(times) - cap))
        b = min(len(times), a + cap)
    if b - a < 2:
        raise ExportError(f"{source}: window is too small to export ({b - a} samples)")

    retained = [entry for entry in wanted if _inside(entry, times, origin, a, b)]
    for split in plan.splits:
        if not any(entry["split"] == split for entry in retained):
            raise ExportError(
                f"{source}: the {int(segment_seconds)}s window dropped its official {split} "
                "interval. Raise --segment-seconds or increase --records-per-task-per-split."
            )
    return a, b, retained


def _inside(entry: dict[str, str], times: np.ndarray, origin: float, a: int, b: int) -> bool:
    start = int(np.searchsorted(times, float(entry["start"]) - origin, side="left"))
    stop = int(np.searchsorted(times, float(entry["end"]) - origin, side="left"))
    return stop > start and start >= a and stop <= b


def plain_window(source: Path, segment_seconds: float, rate_hz: float) -> tuple[int, int]:
    with h5py.File(source, "r") as handle:
        total = int(handle["signal/emg"].shape[0])
    return 0, min(total, max(2, int(segment_seconds * rate_hz)))


# ---------------------------------------------------------------------------
# Rewriting one canonical record
# ---------------------------------------------------------------------------

def _shift_structured(values: np.ndarray, time_fields: tuple[str, ...],
                      start_s: float, end_s: float, shift: float) -> np.ndarray:
    """Keep rows whose primary time falls in the window, then re-base the times."""
    if values.size == 0:
        return values
    primary = time_fields[0]
    kept = [
        row for row in values
        if math.isfinite(float(row[primary])) and start_s <= float(row[primary]) < end_s
    ]
    if not kept:
        return values[:0]
    out = np.array(kept, dtype=values.dtype)
    for field in time_fields:
        if field in out.dtype.names:
            out[field] = out[field] - shift
    return out


def _copy_labels(task_id: str, src: h5py.File, dst: h5py.File, a: int, b: int, shift: float,
                 start_s: float, end_s: float) -> None:
    text = h5py.string_dtype("utf-8")
    labels = dst.create_group("labels")
    if "labels/source_json" in src:
        labels.create_dataset("source_json", data=src["labels/source_json"][()], dtype=text)
    if "labels/stage" in src:
        labels.create_dataset("stage", data=src["labels/stage"][()], dtype=text)

    if task_id == "pose":
        labels.create_dataset("joint_angles", data=src["labels/joint_angles"][a:b])
        if "labels/no_ik_failure" in src:
            labels.create_dataset("no_ik_failure", data=src["labels/no_ik_failure"][a:b])
    elif task_id == "wrist_angle":
        group = labels.create_group("targets")
        for name, dataset in src["labels/targets"].items():
            group.create_dataset(str(name), data=dataset[a:b])
    elif task_id == "typing":
        labels.create_dataset("keystrokes", data=_shift_structured(
            src["labels/keystrokes"][:], ("press_s", "release_s"), start_s, end_s, shift))
        labels.create_dataset("prompts", data=_shift_structured(
            src["labels/prompts"][:], ("start_s", "end_s"), start_s, end_s, shift))
    else:
        name = "prompts" if task_id == "handwriting" else "events"
        fields = ("start_s", "end_s") if task_id == "handwriting" else ("time_s",)
        labels.create_dataset(name, data=_shift_structured(
            src[f"labels/{name}"][:], fields, start_s, end_s, shift))


def write_record(row: dict[str, str], source: Path, destination: Path, task_id: str,
                 a: int, b: int, retained: list[dict[str, str]]) -> dict[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ExportError(f"refusing to overwrite an existing record: {destination}")

    with h5py.File(source, "r") as src:
        times = src["signal/time_s"][a:b].astype(np.float64)
        if times.size < 2:
            raise ExportError(f"{source}: window shorter than two samples")
        shift = float(times[0])
        relative = times - shift
        emg = np.ascontiguousarray(src["signal/emg"][a:b], dtype=np.float32)

        meta = json.loads(src["meta/source_json"].asstr()[()])
        old_origin = float(meta["source_time_origin_s"])
        # The one line that keeps the source clock honest: time_s loses `shift`,
        # so the origin must gain exactly `shift`.
        meta["source_time_origin_s"] = old_origin + shift
        meta["mini_export"] = {
            "source_record": row["path"],
            "source_start_s": old_origin + shift,
            "source_end_s": old_origin + float(relative[-1]),
            "official_intervals_retained": [
                {"split": e["split"], "start": float(e["start"]), "end": float(e["end"])} for e in retained
            ],
        }

        with h5py.File(destination, "w") as dst:
            attrs = {key: _text(value) for key, value in src.attrs.items()}
            attrs["sample_rate_hz"] = float(src.attrs["sample_rate_hz"])
            dst.attrs.update(attrs)
            signal = dst.create_group("signal")
            signal.create_dataset("emg", data=emg)
            signal.create_dataset("time_s", data=relative)
            _copy_labels(task_id, src, dst, a, b, shift, float(relative[0]), float(relative[-1]) + 1e-12)
            meta_group = dst.create_group("meta")
            meta_group.create_dataset("source_json", data=json.dumps(meta, sort_keys=True, default=str),
                                      dtype=h5py.string_dtype("utf-8"))
            quality = meta_group.create_group("quality")
            quality.create_dataset("sample_count", data=int(emg.shape[0]))
            quality.create_dataset("duration_s", data=float(relative[-1]))
            quality.create_dataset("nan_count", data=int(np.isnan(emg).sum()))
            quality.create_dataset("channel_rms", data=np.sqrt(np.mean(emg.astype(np.float64) ** 2, axis=0)))

    return {
        "path": row["path"], "dataset_id": row["dataset_id"], "task_id": row["task_id"],
        "subject_id": row["subject_id"], "session_id": row["session_id"], "side": row["side"],
        "pair_id": row["pair_id"], "official_split": row["official_split"],
        "sample_count": str(int(b - a)), "duration_s": f"{float(relative[-1]):.6f}",
        "sample_rate_hz": row["sample_rate_hz"], "source_file": row["source_file"],
        # Identifies the ORIGINAL raw release file, which is not on this machine,
        # so it is carried over rather than recomputed.
        "source_sha256": row["source_sha256"], "split": "",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def export(data_root: Path, output: Path, *, manifest_name: str, corpus_name: str,
           per_split: int, segment_seconds: float, margin_seconds: float,
           seed: int, overwrite: bool) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise ExportError(f"output directory is not empty: {output} (pass --overwrite)")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / manifest_name
    corpus_path = data_root / corpus_name
    rows = read_manifest(data_root, manifest_path)
    corpus = read_corpus(corpus_path)

    plans: list[Plan] = []
    per_task_plan: dict[str, list[Plan]] = {}
    for dataset_id in TASKS:
        task_rows = [row for row in rows if row["dataset_id"] == dataset_id]
        if not task_rows:
            raise ExportError(f"the manifest has no records for {dataset_id}")
        planned = (plan_nature(task_rows, corpus, per_split, seed) if dataset_id in NATURE_DATASETS
                   else plan_plain(task_rows, per_split, seed))
        per_task_plan[dataset_id] = planned
        plans.extend(planned)

    seen_basenames: dict[str, str] = {}
    for plan in plans:
        if plan.row["dataset_id"] not in NATURE_DATASETS:
            continue
        name = _basename(plan.row)
        if name in seen_basenames:
            raise ExportError(f"ambiguous Nature source basename: {name}")
        seen_basenames[name] = plan.row["path"]

    exported: list[dict[str, str]] = []
    per_task: dict[str, dict[str, int]] = {}
    for dataset_id, planned in per_task_plan.items():
        counts = {split: 0 for split in NEEDED_SPLITS}
        for plan in planned:
            row = plan.row
            source = (data_root / row["path"]).resolve()
            if dataset_id in NATURE_DATASETS:
                a, b, retained = nature_window(plan, source, segment_seconds, margin_seconds)
            else:
                a, b, retained = *plain_window(source, segment_seconds, float(row["sample_rate_hz"])), []
            written = write_record(row, source, output / row["path"], TASKS[dataset_id], a, b, retained)
            exported.append(written)
            for split in plan.splits:
                counts[split] += 1
            print(f"[mini] {dataset_id}/{row['official_split'] or '-':>5}: {row['path']} samples={b - a}", flush=True)
        per_task[dataset_id] = counts

    for dataset_id, counts in per_task.items():
        for split in NEEDED_SPLITS:
            if counts[split] == 0:
                raise ExportError(
                    f"{dataset_id} produced no {split} record; the mini could not train that task. "
                    "Increase --records-per-task-per-split."
                )

    selected = {_basename(row) for row in exported if row["dataset_id"] in NATURE_DATASETS}
    kept_corpus = [entry for entry in corpus if Path(entry["dataset"]).name in selected]
    write_corpus(output / "nature_corpus.csv", kept_corpus)
    write_manifest(output / "splits.csv", exported)

    metadata = {
        "schema": "e3-real-mini-v1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_root": str(data_root),
        "source_manifest": {"name": manifest_name, "sha256": sha256_file(manifest_path)},
        "source_corpus": {"name": corpus_name, "sha256": sha256_file(corpus_path)},
        "parameters": {
            "records_per_task_per_split": per_split,
            "segment_seconds": segment_seconds,
            "margin_seconds": margin_seconds,
            "seed": seed,
        },
        "records_per_task": per_task,
        "records_exported": len(exported),
        "note": (
            "Official train/val only; official test was never exported. Each record is a "
            "contiguous window of a real recording; signal/time_s was re-based and "
            "meta/source_json.source_time_origin_s shifted by the same amount, so the source "
            "clock and the official corpus intervals are preserved exactly."
        ),
    }
    (output / "export_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_checksums(output)
    return metadata


def write_corpus(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CORPUS_COLUMNS))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["dataset"], float(item["start"]))):
            writer.writerow({column: row[column] for column in CORPUS_COLUMNS})


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [*MANIFEST_COLUMNS, "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["path"]):
            writer.writerow({column: row.get(column, "") for column in columns})


def write_checksums(output: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (output / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(output: Path) -> dict:
    """Read the mini back through the project's own Dataset, not a re-implementation."""
    from semg_fm.data.datamodule import build_datamodule

    module = build_datamodule({
        "root": str(output.resolve()),
        "manifest_path": str((output / "splits.csv").resolve()),
        "window_ms": 2000, "stride_ms": 1000, "batch_size": 1, "num_workers": 0,
        "corpus_path": str((output / "nature_corpus.csv").resolve()),
        "evaluation_split": "val",
    }, task_id=None, label_budget=1.0)
    module.setup("fit")
    required = set(TASKS.values())
    report: dict[str, dict] = {}
    for split, dataset in (("train", module.train_dataset), ("val", module.val_dataset)):
        groups = dataset.window_indices_by_task()
        if set(groups) != required:
            raise ExportError(f"mini {split} exposes {sorted(groups)}, expected {sorted(required)}")
        detail = {}
        for task_id, indices in groups.items():
            if len(indices) == 0:
                raise ExportError(f"mini {split} has no {task_id} windows")
            sample = dataset[int(indices[0])]
            if tuple(sample["emg"].shape) != (4000, 16):
                raise ExportError(f"mini {split}/{task_id} EMG {tuple(sample['emg'].shape)} != (4000, 16)")
            if not bool(sample["sample_mask"].all()):
                raise ExportError(f"mini {split}/{task_id} produced an empty sample mask")
            detail[task_id] = len(indices)
        report[split] = detail
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="real canonical data root")
    parser.add_argument("--output", required=True, help="destination for the mini dataset")
    parser.add_argument("--manifest", default="splits.csv")
    parser.add_argument("--corpus", default="nature_corpus.csv")
    parser.add_argument("--records-per-task-per-split", type=int, default=2)
    parser.add_argument("--segment-seconds", type=float, default=30.0,
                        help="cap on the exported window; must still contain the official interval")
    parser.add_argument("--margin-seconds", type=float, default=1.0,
                        help="extra context kept around a Nature interval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    if args.records_per_task_per_split < 1:
        raise ExportError("--records-per-task-per-split must be at least 1")
    if args.segment_seconds <= 0:
        raise ExportError("--segment-seconds must be positive")

    data_root = Path(args.data_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not data_root.is_dir():
        raise ExportError(f"data root does not exist: {data_root}")
    if output == data_root or data_root in output.parents:
        raise ExportError(f"output {output} must not live inside the source data root {data_root}")

    metadata = export(
        data_root, output, manifest_name=args.manifest, corpus_name=args.corpus,
        per_split=args.records_per_task_per_split, segment_seconds=args.segment_seconds,
        margin_seconds=args.margin_seconds, seed=args.seed, overwrite=args.overwrite,
    )
    total = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    print(f"\n[mini] exported {metadata['records_exported']} records, {total / 1e6:.1f} MB -> {output}", flush=True)

    if not args.skip_validation:
        report = validate(output)
        print("[mini] validation against the project's own Dataset: PASS", flush=True)
        for split, detail in report.items():
            print(f"       {split}: {detail}")

    name = output.name
    print("\n[mini] transfer it with either:\n")
    print(f"    tar -C {output.parent} -czf {name}.tar.gz {name}")
    print(f"    scp -r <user>@<server>:{output} <local>/real_mini")
    print(f"    rsync -av --checksum <user>@<server>:{output}/ <local>/real_mini/")
    print("\n    verify after transfer:  sha256sum -c checksums.sha256")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"[mini ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
