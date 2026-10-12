"""将 GNI 腕角录制数据转换为 canonical HDF5。"""

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os

if os.name == "nt":
    import msvcrt

    def _try_lock(handle) -> None:
        """Take an OS-released advisory lock, raising OSError while it is held."""
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(handle) -> None:
        """Take an OS-released advisory lock, raising OSError while it is held."""
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

from pathlib import Path
import time
from typing import Any
import uuid

import h5py
import numpy as np
import pandas as pd


SCHEMA_VERSION, SAMPLE_RATE_HZ = "1.0", 2000
EXPECTED_SOURCE_TASK, DATASET_ID, TASK_ID = "wrist", "gni_wrist", "wrist_angle"
MANIFEST_COLUMNS = ["path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "sample_count", "duration_s", "sample_rate_hz", "source_file", "source_sha256"]


@contextmanager
def _replace_after_success(temporary: Path, destination: Path):
    try:
        yield
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _acquire_manifest_lock(path: Path):
    lock = path.with_name(f".{path.name}.lock"); handle = lock.open("a+b")
    try:
        if lock.stat().st_size == 0: handle.write(b"\0"); handle.flush()
        deadline = time.monotonic() + 10.0
        while True:
            try:
                handle.seek(0); _try_lock(handle); return handle
            except OSError:
                if time.monotonic() >= deadline: raise TimeoutError(f"Timed out waiting for manifest lock: {lock}")
                time.sleep(0.05)
    except BaseException:
        handle.close(); raise


def _release_manifest_lock(handle) -> None:
    try:
        handle.seek(0); _unlock(handle)
    finally:
        handle.close()


def _decode(value: Any) -> Any:
    if isinstance(value, bytes): return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    return value


def _labels_provenance(labels: dict[str, Any]) -> dict[str, Any]:
    """Summarize labels for the provenance dump without duplicating arrays.

    Each array in ``labels`` is written as its own HDF5 dataset; expanding it
    into JSON as well would store the same numbers twice as text.
    """

    def summarize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {key: summarize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [summarize(item) for item in value]
        return value

    return summarize(labels)


def _json_default(value: Any) -> Any:
    return _decode(value) if isinstance(value, (bytes, np.generic, np.ndarray)) else str(value)


def _rows(path: Path, key: str) -> list[dict[str, Any]]:
    try:
        frame = pd.read_hdf(path, key=key)
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
        with h5py.File(path, "r") as h5:
            if key not in h5 or not isinstance(h5[key], h5py.Dataset) or h5[key].dtype.names is None: return []
            return [{field: _decode(row[field]) for field in h5[key].dtype.names} for row in h5[key][:]]
    return [{str(column): _decode(value) for column, value in row.items()} for row in frame.to_dict("records")]


def _safe(value: str) -> str:
    return ("".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip(".") or "unknown")


def _record_subject(path: Path, metadata: dict[str, Any]) -> str:
    """Identify a subject from the file name when the release omits metadata.

    The public GNI drop carries only ``data.attrs['task']``; identity lives in
    the name, ``{task}_user_{NNN}_dataset_{MMM}.hdf5``.  Falling back to the
    parent directory gives every subject in a corpus the same id, which would
    collapse leakage-safe subject grouping into a single group.
    """
    declared = str(metadata.get("subject_id", "")).strip()
    if declared:
        return declared
    stem = path.stem
    if "_user_" in stem:
        return f"user_{stem.split('_user_', 1)[1].split('_', 1)[0]}"
    return path.parent.name


def _read_gni(path: str | Path, official_split: str) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "data" not in h5 or not isinstance(h5["data"], h5py.Dataset): raise ValueError(f"{path} lacks documented GNI 'data' dataset")
        source = h5["data"]; metadata = {str(key): _decode(value) for key, value in source.attrs.items()}
        if str(metadata.get("task", "")) != EXPECTED_SOURCE_TASK: raise ValueError(f"{path} has task {metadata.get('task', '')!r}; {EXPECTED_SOURCE_TASK!r} was required")
        fields, data = list(source.dtype.names or ()), source[:]
        if "time" not in fields: raise ValueError(f"{path} GNI data lacks required 'time' field")
        candidates = [field for field in fields if np.asarray(data[field]).ndim == 2 and np.asarray(data[field]).shape[1] == 16]
        emg_field = "emg" if "emg" in candidates else candidates[0] if len(candidates) == 1 else None
        if emg_field is None: raise ValueError(f"{path} must contain exactly one 16-channel GNI EMG field")
        emg, time_s = np.asarray(data[emg_field]), np.asarray(data["time"])
        if time_s.ndim != 1 or len(time_s) != len(emg): raise ValueError("GNI time must be a T-length vector aligned with EMG")
        if len(emg) == 0: raise ValueError("GNI records must contain at least one sample")
        targets: dict[str, np.ndarray] = {}
        target_fields: dict[str, str] = {}
        for field in fields:
            values = np.asarray(data[field])
            if field in {"time", emg_field} or not np.issubdtype(values.dtype, np.number): continue
            if values.shape[0] != len(emg): raise ValueError(f"GNI wrist target {field!r} does not align with EMG")
            if not np.isfinite(values).all(): raise ValueError(f"GNI wrist target {field!r} must be finite")
            target_name = _safe(field)
            if target_name in targets:
                raise ValueError(f"GNI wrist target sanitization collision: {target_fields[target_name]!r} and {field!r} both map to {target_name!r}")
            target_fields[target_name] = field
            targets[target_name] = values
    stages = _rows(path, "stages"); prompt_rows = _rows(path, "prompts"); labels: dict[str, Any] = {"targets": targets}
    if len(stages) == 1 and "stage" in stages[0]: labels["stage"] = stages[0]["stage"]
    return {"dataset_id": DATASET_ID, "task_id": TASK_ID, "subject_id": f"{DATASET_ID}:{_record_subject(path, metadata)}", "session_id": f"{DATASET_ID}:{metadata.get('session_id', path.stem)}", "side": str(metadata.get("side", "dominant")), "pair_id": "", "official_split": official_split, "source_file": path.name, "source_path": str(path.resolve()), "emg": emg, "time_s": time_s, "labels": labels, "source_meta": {"source_dataset": "data", "emg_field": emg_field, "attributes": metadata, "source_time_origin_s": float(time_s[0]), "prompts": prompt_rows, "stages": stages}, "preprocessing": {"converter_transformations": ["time_s shifted to record-relative origin"]}}


def _relative(record: dict[str, Any]) -> Path:
    return Path(_safe(record["dataset_id"])) / _safe(record["task_id"]) / _safe(record["subject_id"].split(":", 1)[-1]) / f"{_safe(record['session_id'].split(':', 1)[-1])}_{_safe(record['side'])}.h5"


def _validate(record: dict[str, Any]) -> None:
    emg, time_s = record["emg"], record["time_s"]
    if emg.ndim != 2 or emg.shape[1] != 16: raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if time_s.ndim != 1 or len(time_s) != len(emg): raise ValueError("time_s must be a length-T vector matching EMG")
    if len(emg) == 0: raise ValueError("GNI records must contain at least one sample")
    if not np.isfinite(emg).all() or not np.isfinite(time_s).all(): raise ValueError("EMG and time_s must be finite")
    if len(time_s) > 1 and np.any(np.diff(time_s) < 0): raise ValueError("time_s must be monotonic")


def _write(path: str | Path, record: dict[str, Any], *, overwrite: bool) -> None:
    _validate(record); path = Path(path)
    if path.exists() and not overwrite: raise FileExistsError(f"Refusing to overwrite existing canonical record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); relative_time = record["time_s"].astype(np.float64, copy=False) - record["time_s"][0]
    if not record["labels"]["targets"]: raise ValueError("Wrist-angle records require non-empty labels['targets']")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with _replace_after_success(temporary, path), h5py.File(temporary, "w") as h5:
        h5.attrs.update({"schema_version": SCHEMA_VERSION, **{key: record[key] for key in ("dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "source_file")}, "sample_rate_hz": SAMPLE_RATE_HZ, "preprocessing_json": json.dumps(record["preprocessing"], default=_json_default, sort_keys=True)})
        signal = h5.create_group("signal"); signal.create_dataset("emg", data=record["emg"].astype(np.float32, copy=False), compression="gzip"); signal.create_dataset("time_s", data=relative_time)
        text = h5py.string_dtype("utf-8"); labels = h5.create_group("labels"); labels.create_dataset("source_json", data=json.dumps(_labels_provenance(record["labels"]), default=_json_default, sort_keys=True), dtype=text)
        targets = labels.create_group("targets")
        for name, values in record["labels"]["targets"].items(): targets.create_dataset(str(name), data=np.asarray(values))
        if "stage" in record["labels"]: labels.create_dataset("stage", data=str(record["labels"]["stage"]), dtype=text)
        meta = h5.create_group("meta"); meta.create_dataset("source_json", data=json.dumps(record["source_meta"], default=_json_default, sort_keys=True), dtype=text)
        quality = meta.create_group("quality"); quality.create_dataset("sample_count", data=len(record["emg"])); quality.create_dataset("duration_s", data=float(relative_time[-1])); quality.create_dataset("nan_count", data=int(np.isnan(record["emg"]).sum())); quality.create_dataset("channel_rms", data=np.sqrt(np.mean(record["emg"].astype(np.float32, copy=False) ** 2, axis=0)))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _manifest(record: dict[str, Any], relative: Path) -> dict[str, Any]:
    return {"path": relative.as_posix(), **{key: record[key] for key in ("dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id", "official_split", "source_file")}, "sample_count": int(len(record["emg"])), "duration_s": float(record["time_s"][-1] - record["time_s"][0]), "sample_rate_hz": SAMPLE_RATE_HZ, "source_sha256": _sha256(record["source_path"])}


def _append(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); lock_handle = _acquire_manifest_lock(path)
    try:
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
        rows = [item for item in rows if item["path"] != row["path"]] + [{column: row[column] for column in MANIFEST_COLUMNS}]; rows.sort(key=lambda item: item["path"])
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with _replace_after_success(temporary, path), temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS); writer.writeheader(); writer.writerows(rows)
    finally:
        _release_manifest_lock(lock_handle)


def _splits(path: str | Path | None) -> dict[str, str]:
    if path is None: return {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "split" not in rows.fieldnames: raise ValueError("Split CSV must include a 'split' column")
        column = next((name for name in ("path", "source_file", "file") if name in rows.fieldnames), None)
        if column is None: raise ValueError("Split CSV must include one of: path, source_file, file")
        return {str(row[column]).replace("\\", "/"): str(row["split"]) for row in rows}


def convert_file(source_path: str | Path, output_root: str | Path, *, official_split: str = "unknown", overwrite: bool = False, source_file: str | None = None) -> dict[str, object]:
    record = _read_gni(source_path, official_split); record["source_file"] = source_file or Path(source_path).name; relative = _relative(record); _write(Path(output_root) / relative, record, overwrite=overwrite); return _manifest(record, relative)


def _source_files(input_root: Path) -> list[Path]:
    """List convertible source files under ``input_root``.

    The released corpora use both ``.h5`` (emg2pose, emg2qwerty) and ``.hdf5``
    (the GNI drop).  Globbing only ``*.h5`` matched nothing on the GNI corpora
    and the run reported success after converting zero records.
    """
    sources = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix in {".h5", ".hdf5"}
    )
    if not sources:
        raise ValueError(f"no .h5 or .hdf5 source files found under {input_root}")
    return sources


def run_conversion(input_root: str | Path, output_root: str | Path, *, split: str | None = None, splits_csv: str | Path | None = None, overwrite: bool = False) -> list[dict[str, object]]:
    input_root, output_root, lookup = Path(input_root), Path(output_root), _splits(splits_csv); results = []
    try:
        output_root.resolve().relative_to(input_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-root must not be nested under input-root")
    for source in _source_files(input_root):
        official_split = lookup.get(source.relative_to(input_root).as_posix(), lookup.get(source.name, "unknown"))
        if split is None or official_split == split:
            row = convert_file(source, output_root, official_split=official_split, overwrite=overwrite, source_file=source.relative_to(input_root).as_posix()); _append(output_root / "manifest.csv", row); results.append(row)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-root", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--split"); parser.add_argument("--splits-csv"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv); rows = run_conversion(args.input_root, args.output_root, split=args.split, splits_csv=args.splits_csv, overwrite=args.overwrite); print(f"Converted {len(rows)} {DATASET_ID} records."); return 0


if __name__ == "__main__": raise SystemExit(main())
