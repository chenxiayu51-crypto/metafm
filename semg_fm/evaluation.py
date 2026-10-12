"""Machine-readable evaluation outputs written to the Hydra run directory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import pandas as pd


def write_evaluation_outputs(
    output_dir: str | Path,
    *,
    metrics: Mapping[str, Any],
    per_subject: Sequence[Mapping[str, Any]] | None = None,
    split_manifest_path: str | Path | None = None,
    environment: Mapping[str, Any] | None = None,
    predictions: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Write the design-documented evaluation outputs.

    Produces ``metrics.json``, ``per_subject_metrics.csv`` (when per-subject
    rows exist), ``split_manifest.csv`` (a copy of the consumed split file),
    ``environment.txt``, and ``predictions.h5`` (when predictions were
    collected and requested).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    if per_subject:
        frame = pd.DataFrame(list(per_subject))
        frame.to_csv(output_dir / "per_subject_metrics.csv", index=False, lineterminator="\n")
    if split_manifest_path is not None:
        shutil.copyfile(split_manifest_path, output_dir / "split_manifest.csv")
    if environment:
        lines = [f"{key}={value}" for key, value in sorted(environment.items())]
        (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if predictions:
        _write_predictions(output_dir / "predictions.h5", predictions)


def _write_predictions(path: Path, predictions: Sequence[Mapping[str, Any]]) -> None:
    """Store per-window predictions as variable-length string arrays in HDF5."""
    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        record_ids = [str(row.get("record_id", "")) for row in predictions]
        subject_ids = [str(row.get("subject_id", "unknown")) for row in predictions]
        h5.create_dataset("record_id", data=record_ids, dtype=text)
        h5.create_dataset("subject_id", data=subject_ids, dtype=text)
        if any("hypothesis" in row for row in predictions):
            h5.create_dataset("reference", data=[str(row.get("reference", "")) for row in predictions], dtype=text)
            h5.create_dataset("hypothesis", data=[str(row.get("hypothesis", "")) for row in predictions], dtype=text)
        elif any("predicted" in row for row in predictions):
            encoded = [json.dumps(row.get("predicted", [])) for row in predictions]
            h5.create_dataset("predicted", data=encoded, dtype=text)
        else:
            h5.create_dataset("loss", data=[float(row.get("loss", float("nan"))) for row in predictions])


__all__ = ["write_evaluation_outputs"]
