"""Lazy windowed access to canonical sEMG HDF5 records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest
from .task_labels import parse_target


OFFICIAL_SPLITS = frozenset({"train", "val", "test"})
REQUIRED_ROOT_ATTRIBUTES = (
    "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_rate_hz",
)
MANIFEST_ATTRIBUTE_FIELDS = REQUIRED_ROOT_ATTRIBUTES[:-1]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)


def _canonical_signal_datasets(h5: h5py.File, record: dict[str, Any], path: Path):
    missing_attrs = [name for name in REQUIRED_ROOT_ATTRIBUTES if name not in h5.attrs]
    if missing_attrs:
        raise ValueError(f"{path} lacks required root attributes: {', '.join(missing_attrs)}")
    for name in MANIFEST_ATTRIBUTE_FIELDS:
        if _text(h5.attrs[name]) != record[name]:
            raise ValueError(f"{path} root attribute {name} does not match manifest")
    try:
        sample_rate_hz = float(h5.attrs["sample_rate_hz"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} sample_rate_hz must be numeric") from exc
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"{path} has invalid sample_rate_hz")
    if not np.isclose(sample_rate_hz, record["sample_rate_hz"]):
        raise ValueError(f"{path} root attribute sample_rate_hz does not match manifest")
    emg_dataset = h5.get("signal/emg")
    time_dataset = h5.get("signal/time_s")
    if not isinstance(emg_dataset, h5py.Dataset) or not isinstance(time_dataset, h5py.Dataset):
        raise ValueError(f"{path} requires signal/emg and signal/time_s datasets")
    if emg_dataset.dtype != np.dtype(np.float32):
        raise ValueError(f"{path} signal/emg must have dtype float32")
    if time_dataset.dtype != np.dtype(np.float64):
        raise ValueError(f"{path} signal/time_s must have dtype float64")
    if emg_dataset.ndim != 2 or emg_dataset.shape[1] != 16:
        raise ValueError(f"{path} signal/emg must have shape [T, 16]")
    if time_dataset.ndim != 1 or time_dataset.shape[0] != emg_dataset.shape[0]:
        raise ValueError(f"{path} signal/time_s must have shape [T] aligned with signal/emg")
    return emg_dataset, time_dataset, sample_rate_hz


class _WindowIndex:
    """Enumerate ``(path, start, end)`` windows from per-record geometry.

    The converted corpus holds tens of millions of windows, and one tuple per
    window costs roughly 10 GB.  Only the per-record geometry is stored here;
    an individual entry is recovered by locating the record that owns the
    ordinal and multiplying by the record's stride.
    """

    def __init__(self, paths, window_samples, stride_samples, offsets):
        self.paths = paths
        self.offsets = offsets
        self._window_samples = window_samples
        self._stride_samples = stride_samples

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        ordinal = int(index)
        if ordinal < 0:
            ordinal += len(self)
        if not 0 <= ordinal < len(self):
            raise IndexError(index)
        record = int(np.searchsorted(self.offsets, ordinal, side="right") - 1)
        local = ordinal - int(self.offsets[record])
        start = local * int(self._stride_samples[record])
        return (self.paths[record], start, start + int(self._window_samples[record]))

    def __iter__(self):
        for record, path in enumerate(self.paths):
            window_samples = int(self._window_samples[record])
            stride_samples = int(self._stride_samples[record])
            count = int(self.offsets[record + 1]) - int(self.offsets[record])
            for start in range(0, count * stride_samples, stride_samples):
                yield (path, start, start + window_samples)


class CanonicalWindowDataset(Dataset):
    """Index full, fixed-duration windows without keeping HDF5 files open."""

    def __init__(self, records: list[dict[str, Any]], window_ms: float, stride_ms: float):
        if window_ms <= 0 or stride_ms <= 0:
            raise ValueError("window_ms and stride_ms must be positive")
        self.window_ms = float(window_ms)
        self.stride_ms = float(stride_ms)
        self._records = {Path(record["path"]): record for record in records}
        self._index = self._build_index(records)
        if records and not len(self._index):
            raise ValueError(
                f"no windows fit in the selected records: each recording is shorter than "
                f"one {self.window_ms} ms window, so the dataset would be empty."
            )

    def _build_index(self, records: list[dict[str, Any]]) -> _WindowIndex:
        paths: list[Path] = []
        window_samples: list[int] = []
        stride_samples: list[int] = []
        counts: list[int] = []
        for record in records:
            rate_hz = float(record["sample_rate_hz"])
            window = int(round(rate_hz * self.window_ms / 1000.0))
            stride = int(round(rate_hz * self.stride_ms / 1000.0))
            if window <= 0 or stride <= 0:
                raise ValueError("window_ms and stride_ms must produce at least one sample")
            sample_count = int(record["sample_count"])
            paths.append(Path(record["path"]))
            window_samples.append(window)
            stride_samples.append(stride)
            counts.append((sample_count - window) // stride + 1 if sample_count >= window else 0)
        offsets = np.concatenate(
            ([0], np.cumsum(np.asarray(counts, dtype=np.int64)))
        )
        return _WindowIndex(
            paths,
            np.asarray(window_samples, dtype=np.int64),
            np.asarray(stride_samples, dtype=np.int64),
            offsets.astype(np.int64),
        )

    def window_indices_by_task(self) -> dict[str, np.ndarray]:
        """Group global window ordinals by task without one Python object each.

        The task-balanced sampler shuffles every window of a task, so it needs
        the ordinals themselves; as Python lists they cost about 1 GB per task
        on the converted corpus, so they are assembled per record instead.
        """
        paths = self._index.paths
        offsets = self._index.offsets
        groups: dict[str, list[np.ndarray]] = {}
        for position, path in enumerate(paths):
            task_id = str(self._records[path]["task_id"])
            start, stop = int(offsets[position]), int(offsets[position + 1])
            if stop > start:
                groups.setdefault(task_id, []).append(np.arange(start, stop, dtype=np.int64))
        return {
            task_id: np.concatenate(parts)
            for task_id, parts in sorted(groups.items())
        }

    @classmethod
    def from_manifest(cls, manifest_path, split: str, window_ms: float, stride_ms: float, root=None, task_id: str | None = None):
        if split not in OFFICIAL_SPLITS:
            raise ValueError(f"split must be one of {sorted(OFFICIAL_SPLITS)}")
        manifest_path = Path(manifest_path)
        root = Path(root) if root is not None else manifest_path.parent
        frame = load_manifest(manifest_path, root)
        if task_id is not None and task_id not in set(frame["task_id"].astype(str)):
            raise ValueError(f"no manifest records have task_id {task_id!r}")
        records = []
        skipped_unassigned = 0
        for _, row in frame.iterrows():
            # Prefer a non-empty official split; fall back to the deterministic
            # `split` column written by `make_splits`, so its output CSV can be
            # passed directly as the manifest for later runs.
            official = _text(row["official_split"]).strip()
            if official in OFFICIAL_SPLITS:
                row_split = official
            elif "split" in row:
                row_split = _text(row["split"]).strip()
            else:
                row_split = ""
            if row_split not in OFFICIAL_SPLITS:
                skipped_unassigned += 1
                continue
            if row_split != split:
                continue
            if task_id is not None and _text(row["task_id"]) != task_id:
                continue
            sample_count = int(row["sample_count"])
            # The rate comes from the manifest column and nowhere else.  It
            # cannot be inferred from the clock span: /signal/time_s preserves
            # the release's own clock, so dropped samples make the span exceed
            # (T-1)/rate and the inferred rate come out low -- 90 Hz for the
            # worst record in the converted corpus -- which silently shrinks
            # every window.
            rate_hz = float(row["sample_rate_hz"])
            if not np.isfinite(rate_hz) or rate_hz <= 0:
                raise ValueError(f"invalid sample rate for {row['path']}")
            records.append({
                "path": Path(row["resolved_path"]), "dataset_id": _text(row["dataset_id"]),
                "task_id": _text(row["task_id"]), "record_id": _text(row.get("record_id", Path(row["path"]).stem)),
                "subject_id": _text(row["subject_id"]), "session_id": _text(row["session_id"]),
                "side": _text(row["side"]), "pair_id": _text(row["pair_id"]),
                "official_split": official,
                "sample_count": sample_count, "sample_rate_hz": rate_hz,
            })
        # A converter-written manifest has official_split 'unknown' and no 'split'
        # column, so no split can ever be selected from it.  Without this check
        # that misconfiguration silently yields an empty dataset and a training
        # run that takes zero steps.
        if not records:
            split_column = "split" in frame.columns
            names = set(frame["official_split"].astype(str).str.strip())
            if not split_column and not (names & OFFICIAL_SPLITS):
                raise ValueError(
                    f"{manifest_path} cannot select a split: it has no 'split' column and "
                    "no official_split values in train/val/test. Run make_splits.py and "
                    "pass its output as the manifest, or convert a corpus that ships "
                    "official splits."
                )
        elif skipped_unassigned:
            # Partially assigned manifests (e.g. an unmodified converter output
            # whose 'unknown' rows were never split) would otherwise train on a
            # silent subset of the corpus.
            warnings.warn(
                f"{manifest_path}: skipped {skipped_unassigned} record(s) with no "
                f"'{split}' split assignment (official_split empty/'unknown' and no "
                "'split' column value). Run make_splits.py on the manifest to assign "
                "them; without it they are excluded from training.",
                stacklevel=2,
            )
        return cls(records, window_ms, stride_ms)

    def apply_label_budget(self, budget: float, seed: int) -> None:
        """Keep windows from a deterministic subject-level subset of records.

        Low-label fine-tuning selects whole subject groups (dataset/subject,
        extended by ``pair_id`` so bilateral qwerty records stay together)
        using a stable hash, so the same budget always keeps the same windows
        and no group leaks across the kept/dropped boundary.
        """
        if not 0.0 < budget <= 1.0:
            raise ValueError("label budget must be in (0, 1]")
        if budget >= 1.0:
            return
        # The draw is per subject group, so filtering whole records keeps every
        # window of a kept group and rebuilds the geometry in one pass instead
        # of copying one entry per window.
        kept: list[dict[str, Any]] = []
        for record in self._records.values():
            key = f"{seed}:{record['dataset_id']}:{record['subject_id']}:{record['pair_id']}"
            point = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
            if point < budget:
                kept.append(record)
        if not kept:
            groups = {
                f"{record['dataset_id']}:{record['subject_id']}:{record['pair_id']}"
                for record in self._records.values()
            }
            raise ValueError(
                f"label budget {budget} kept none of the {len(groups)} subject groups. "
                "The budget is a per-group hash draw, not an exact fraction, so a small "
                "budget over few groups can select nothing; the run would otherwise "
                "train for zero steps without reporting an error."
            )
        self._records = {Path(record["path"]): record for record in kept}
        self._index = self._build_index(kept)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, start_index, end_index = self._index[index]
        record = self._records[path]
        with h5py.File(path, "r") as h5:
            emg_dataset, time_dataset, sample_rate_hz = _canonical_signal_datasets(h5, record, path)
            emg = np.asarray(emg_dataset[start_index:end_index])
            # Windows are contiguous half-open clock intervals. A nonterminal window ends
            # exactly at its successor's first timestamp; the final window extrapolates a
            # local half interval from its two final selected timestamps.
            time_with_boundary = np.asarray(time_dataset[start_index:min(end_index + 1, time_dataset.shape[0])])
            time_s = time_with_boundary[: end_index - start_index]
            if emg.shape[0] != end_index - start_index:
                raise ValueError(f"{path} selected EMG slice has an unexpected length")
            if not np.isfinite(emg).all():
                raise ValueError(f"{path} selected EMG slice must be finite numeric values")
            if time_s.ndim != 1 or time_s.shape[0] != emg.shape[0]:
                raise ValueError(f"{path} selected time_s slice must align with EMG")
            if not np.isfinite(time_with_boundary).all():
                raise ValueError(f"{path} selected time_s slice must be finite numeric values")
            if time_with_boundary.shape[0] > 1 and np.any(np.diff(time_with_boundary) < 0):
                raise ValueError(f"{path} selected time_s slice must be monotonic")
            start_s = float(time_s[0])
            if time_with_boundary.shape[0] > time_s.shape[0]:
                end_s = float(time_with_boundary[-1])
            elif time_s.shape[0] > 1:
                end_s = float(time_s[-1] + (time_s[-1] - time_s[-2]) / 2)
            else:
                end_s = float(time_s[-1] + 0.5 / sample_rate_hz)
            item = {
                "emg": torch.as_tensor(emg, dtype=torch.float32),
                "sample_mask": torch.ones(emg.shape[0], dtype=torch.bool),
                "dataset_id": record["dataset_id"], "task_id": record["task_id"],
                "record_id": _text(h5.attrs.get("record_id", record["record_id"])),
                "subject_id": record["subject_id"], "session_id": record["session_id"],
                "side": record["side"], "pair_id": record["pair_id"], "start_s": start_s,
                "sample_rate_hz": sample_rate_hz,
            }
            item.update(parse_target(record["task_id"], h5, start_s, end_s, start_index, end_index))
        return item
