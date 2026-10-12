"""Deterministic, group-aware train/validation/test assignment."""

from __future__ import annotations

import hashlib
import math

import pandas as pd


RATIO_TOLERANCE = 1e-9
_REQUIRED_COLUMNS = {"dataset_id", "subject_id", "pair_id"}
_SPLITS = ("train", "val", "test")


def _validate_ratios(ratios):
    try:
        values = tuple(float(value) for value in ratios)
    except (TypeError, ValueError) as exc:
        raise ValueError("ratios must contain exactly three positive values summing to 1.0") from exc
    if (
        len(values) != 3
        or any(not math.isfinite(value) or value <= 0 for value in values)
        or abs(sum(values) - 1.0) > RATIO_TOLERANCE
    ):
        raise ValueError("ratios must contain exactly three positive values summing to 1.0")
    return values


def _is_present(value) -> bool:
    if value is None:
        return False
    if value is pd.NA:
        return False
    missing = pd.isna(value)
    try:
        if bool(missing):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def assign_group_splits(frame: pd.DataFrame, seed: int, ratios=(0.8, 0.1, 0.1)) -> pd.DataFrame:
    """Return a copy with deterministic group-level ``split`` assignments."""
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    values = _validate_ratios(ratios)
    result = frame.copy()
    parent = list(range(len(result)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    seen = {}
    for position, (_, row) in enumerate(result.iterrows()):
        subject_key = f"subject:{row['dataset_id']}\x1f{row['subject_id']}"
        if subject_key in seen:
            union(position, seen[subject_key])
        else:
            seen[subject_key] = position
        if _is_present(row["pair_id"]):
            # pair_id strings are only unique within a dataset; namespace them
            # so two corpora sharing a pair_id string are not merged.
            pair_key = f"pair:{row['dataset_id']}\x1f{row['pair_id']}"
            if pair_key in seen:
                union(position, seen[pair_key])
            else:
                seen[pair_key] = position

    component_keys = {}
    for position, (_, row) in enumerate(result.iterrows()):
        root = find(position)
        keys = component_keys.setdefault(root, set())
        keys.add(f"subject:{row['dataset_id']}\x1f{row['subject_id']}")
        if _is_present(row["pair_id"]):
            keys.add(f"pair:{row['dataset_id']}\x1f{row['pair_id']}")
    assignments = {}
    cumulative = (values[0], values[0] + values[1], 1.0)
    for keys in component_keys.values():
        key = "\x1e".join(sorted(keys))
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        point = int.from_bytes(digest[:8], "big") / 2**64
        assignments[key] = _SPLITS[next(index for index, bound in enumerate(cumulative) if point < bound)]
    split_values = []
    for position in range(len(result)):
        keys = component_keys[find(position)]
        split_values.append(assignments["\x1e".join(sorted(keys))])
    result["split"] = pd.Series(split_values, index=result.index, dtype="string")
    return result


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Raise ``ValueError`` when a group is assigned to multiple splits."""
    required = _REQUIRED_COLUMNS | {"split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    for key_name, mask in (("pair_id", frame["pair_id"].map(_is_present)),):
        grouped = frame.loc[mask].groupby(["dataset_id", key_name], dropna=False)["split"].nunique(dropna=False)
        if (grouped > 1).any():
            raise ValueError(f"leakage detected for dataset/{key_name}")
    grouped = frame.groupby(["dataset_id", "subject_id"], dropna=False)["split"].nunique(dropna=False)
    if (grouped > 1).any():
        raise ValueError("leakage detected for dataset/subject group")
