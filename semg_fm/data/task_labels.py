"""Task-specific target extraction from canonical sEMG records."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import torch


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _time_length(h5: h5py.File) -> int:
    if "signal/emg" not in h5:
        raise ValueError("canonical record lacks signal/emg needed to align labels")
    return int(h5["signal/emg"].shape[0])


def _numeric_slice(
    dataset: h5py.Dataset, name: str, start_index: int, end_index: int, length: int, *, allow_bool: bool = False
) -> np.ndarray:
    if dataset.ndim == 0:
        raise ValueError(f"{name} target must be time-aligned, not scalar")
    if not np.issubdtype(dataset.dtype, np.number) and not (allow_bool and np.issubdtype(dataset.dtype, np.bool_)):
        raise ValueError(f"{name} target must be numeric, got {dataset.dtype}")
    if dataset.shape[0] != length:
        raise ValueError(f"{name} target length {dataset.shape[0]} does not align with EMG length {length}")
    return np.asarray(dataset[start_index:end_index])


def _structured_rows(dataset: h5py.Dataset, name: str, required: tuple[str, ...]) -> np.ndarray:
    if dataset.dtype.names is None:
        raise ValueError(f"labels/{name} must be a structured dataset")
    missing = [field for field in required if field not in dataset.dtype.names]
    if missing:
        raise ValueError(f"labels/{name} lacks fields: {', '.join(missing)}")
    return dataset[:]


def _prompt_target(h5: h5py.File, start_s: float, end_s: float) -> dict[str, Any]:
    if "labels/prompts" not in h5:
        raise ValueError("labels/prompts is required")
    prompts = _structured_rows(h5["labels/prompts"], "prompts", ("text", "start_s", "end_s"))
    containing = []
    intersecting = []
    for row in prompts:
        prompt_start, prompt_end = float(row["start_s"]), float(row["end_s"])
        if not np.isfinite((prompt_start, prompt_end)).all() or prompt_start > prompt_end:
            raise ValueError("prompt bounds must be finite and ordered")
        candidate = {
            "target_text": str(_decode(row["text"])),
            "prompt_start_s": prompt_start,
            "prompt_end_s": prompt_end,
        }
        if prompt_start <= start_s and end_s <= prompt_end:
            containing.append(candidate)
        elif prompt_start < end_s and start_s < prompt_end:
            intersecting.append(candidate)
    if containing:
        return containing[0]
    if intersecting:
        return intersecting[0]
    return {"target_text": "", "prompt_start_s": None, "prompt_end_s": None}


def parse_target(task_id: str, h5: h5py.File, start_s: float, end_s: float, start_index: int, end_index: int) -> dict[str, Any]:
    """Read the labels relevant to one time window from an open canonical record."""
    if start_index < 0 or end_index <= start_index:
        raise ValueError("window sample indices must be non-negative and ordered")
    if task_id == "pose":
        if "labels/joint_angles" not in h5:
            raise ValueError("pose record lacks labels/joint_angles")
        values = _numeric_slice(h5["labels/joint_angles"], "joint_angles", start_index, end_index, _time_length(h5))
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("joint_angles must be a rank-2 [T, D] array with D > 0")
        if not np.isfinite(values).all():
            raise ValueError("joint_angles must contain finite values")
        target_mask = np.ones(values.shape[0], dtype=np.bool_)
        if "labels/no_ik_failure" in h5:
            valid = _numeric_slice(
                h5["labels/no_ik_failure"], "no_ik_failure", start_index, end_index, _time_length(h5), allow_bool=True
            )
            if valid.ndim != 1:
                raise ValueError("no_ik_failure must be a time-aligned vector")
            target_mask = valid.astype(bool, copy=False)
        return {"target": torch.as_tensor(values, dtype=torch.float32), "target_mask": torch.as_tensor(target_mask, dtype=torch.bool)}

    if task_id == "wrist_angle":
        if "labels/targets" not in h5 or not isinstance(h5["labels/targets"], h5py.Group):
            raise ValueError("wrist_angle record lacks labels/targets group")
        length = _time_length(h5)
        targets: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for name, dataset in h5["labels/targets"].items():
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"wrist target {name} must be a dataset")
            values = _numeric_slice(dataset, f"wrist target {name}", start_index, end_index, length)
            targets[name] = torch.as_tensor(values, dtype=torch.float32)
            masks[name] = torch.as_tensor(np.isfinite(values), dtype=torch.bool)
        if not targets:
            raise ValueError("wrist_angle labels/targets must contain at least one target")
        return {"target": targets, "target_mask": masks}

    if task_id == "discrete_gesture":
        if "labels/events" not in h5:
            raise ValueError("discrete_gesture record lacks labels/events")
        rows = _structured_rows(h5["labels/events"], "events", ("name", "time_s"))
        events = []
        for row in rows:
            time_s = float(row["time_s"])
            if not np.isfinite(time_s):
                raise ValueError("gesture event time_s must be finite")
            if start_s <= time_s < end_s:
                events.append((str(_decode(row["name"])), time_s - start_s))
        return {"events": events}

    if task_id == "handwriting":
        return _prompt_target(h5, start_s, end_s)

    if task_id == "typing":
        if "labels/keystrokes" not in h5:
            raise ValueError("typing record lacks labels/keystrokes")
        rows = _structured_rows(h5["labels/keystrokes"], "keystrokes", ("key", "press_s", "release_s"))
        keystrokes = []
        for row in rows:
            press_s, release_s = float(row["press_s"]), float(row["release_s"])
            if not np.isfinite((press_s, release_s)).all() or press_s > release_s:
                raise ValueError("keystroke times must be finite and ordered")
            if start_s <= press_s < end_s:
                keystrokes.append((str(_decode(row["key"])), press_s - start_s, release_s - start_s))
        prompt = _prompt_target(h5, start_s, end_s)
        for name in ("prompt_start_s", "prompt_end_s"):
            if prompt[name] is not None:
                prompt[name] -= start_s
        return {"keystrokes": keystrokes, **prompt}

    raise ValueError(f"Unsupported task_id: {task_id}")
