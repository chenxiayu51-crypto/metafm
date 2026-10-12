"""Validation and path resolution for canonical training manifests."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "path",
    "dataset_id",
    "task_id",
    "subject_id",
    "session_id",
    "side",
    "pair_id",
    "official_split",
    "sample_count",
    "duration_s",
    "sample_rate_hz",
    "source_file",
    "source_sha256",
}


def load_manifest(manifest_path, root):
    """Load a canonical manifest, validate its schema, and resolve record paths."""
    manifest_path = Path(manifest_path)
    root = Path(root)
    # Identifiers and hashes are strings, including values with leading zeros.
    frame = pd.read_csv(manifest_path, keep_default_na=False, dtype=str)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    root_resolved = root.resolve()
    resolved_paths = []
    for value in frame["path"]:
        resolved = (root / value).resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"record path outside root: {value}") from exc
        resolved_paths.append(resolved)
    frame["resolved_path"] = resolved_paths
    absent = [str(path) for path in frame["resolved_path"] if not path.is_file()]
    if absent:
        preview = ", ".join(absent[:3])
        raise FileNotFoundError(f"missing record paths: {preview}")
    return frame
