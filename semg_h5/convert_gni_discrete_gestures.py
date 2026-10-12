"""将 GNI 离散手势录制数据转换为 canonical HDF5。"""

import argparse
import csv
from contextlib import contextmanager
from dataclasses import dataclass
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


SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000
EXPECTED_SOURCE_TASK = "discrete_gestures"
DATASET_ID = "gni_discrete_gestures"
TASK_ID = "discrete_gesture"
MANIFEST_COLUMNS = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_count", "duration_s", "sample_rate_hz", "source_file",
    "source_sha256",
]


@dataclass
class CanonicalRecord:
    dataset_id: str
    task_id: str
    subject_id: str
    session_id: str
    side: str
    pair_id: str
    official_split: str
    source_file: str
    source_path: str
    emg: np.ndarray
    time_s: np.ndarray
    labels: dict[str, Any]
    source_meta: dict[str, Any]
    preprocessing: dict[str, Any]


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
    lock = path.with_name(f".{path.name}.lock")
    handle = lock.open("a+b")
    try:
        if lock.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 10.0
        while True:
            try:
                handle.seek(0)
                _try_lock(handle)
                return handle
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for manifest lock: {lock}")
                time.sleep(0.05)
    except BaseException:
        handle.close()
        raise


def _release_manifest_lock(handle) -> None:
    try:
        handle.seek(0)
        _unlock(handle)
    finally:
        handle.close()


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validate_record(record: CanonicalRecord) -> None:
    if record.emg.ndim != 2 or record.emg.shape[1] != 16:
        raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if record.time_s.ndim != 1 or len(record.time_s) != len(record.emg):
        raise ValueError("time_s must be a length-T vector matching EMG")
    if len(record.emg) == 0:
        raise ValueError("GNI records must contain at least one sample")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")


def _structured_table(rows: list[dict[str, Any]], fields: list[tuple[str, Any]]) -> np.ndarray:
    result = np.empty(len(rows), dtype=np.dtype(fields))
    for index, row in enumerate(rows):
        result[index] = tuple(row[field] for field, _ in fields)
    return result


def _write_record(path: str | Path, record: CanonicalRecord, *, overwrite: bool = False) -> Path:
    _validate_record(record)
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    relative_time_s = record.time_s.astype(np.float64, copy=False) - record.time_s[0]
    with _replace_after_success(temporary, output), h5py.File(temporary, "w") as h5:
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION, "dataset_id": record.dataset_id,
            "task_id": record.task_id, "subject_id": record.subject_id,
            "session_id": record.session_id, "side": record.side, "pair_id": record.pair_id,
            "sample_rate_hz": SAMPLE_RATE_HZ, "official_split": record.official_split,
            "source_file": record.source_file,
            "preprocessing_json": json.dumps(record.preprocessing, default=_json_default, sort_keys=True),
        })
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=record.emg.astype(np.float32, copy=False), compression="gzip")
        signal.create_dataset("time_s", data=relative_time_s)
        labels = h5.create_group("labels")
        labels.create_dataset("source_json", data=json.dumps(_labels_provenance(record.labels), default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        labels.create_dataset("events", data=_structured_table(record.labels.get("events", []), [("name", h5py.string_dtype("utf-8")), ("time_s", np.float64)]))
        if "stage" in record.labels:
            labels.create_dataset("stage", data=str(record.labels["stage"]), dtype=h5py.string_dtype("utf-8"))
        meta = h5.create_group("meta")
        meta.create_dataset("source_json", data=json.dumps(record.source_meta, default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=len(record.emg))
        quality.create_dataset("duration_s", data=float(relative_time_s[-1]))
        quality.create_dataset("nan_count", data=int(np.isnan(record.emg).sum()))
        quality.create_dataset("channel_rms", data=np.sqrt(np.mean(record.emg.astype(np.float32, copy=False) ** 2, axis=0)))
    return output


def _read_gni_table(path: Path, key: str) -> list[dict[str, Any]]:
    try:
        frame = pd.read_hdf(path, key=key)
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError):
        with h5py.File(path, "r") as h5:
            if key not in h5 or not isinstance(h5[key], h5py.Dataset) or h5[key].dtype.names is None:
                return []
            return [{field: _decode_value(row[field]) for field in h5[key].dtype.names} for row in h5[key][:]]
    return [{str(column): _decode_value(value) for column, value in row.items()} for row in frame.to_dict("records")]


def _required_value(row: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"{label} row lacks one of {names}")


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


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


def _read_gni(path: str | Path, official_split: str) -> CanonicalRecord:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "data" not in h5 or not isinstance(h5["data"], h5py.Dataset):
            raise ValueError(f"{path} lacks documented GNI 'data' dataset")
        data_dataset = h5["data"]
        metadata = {str(key): _decode_value(value) for key, value in data_dataset.attrs.items()}
        source_task = str(metadata.get("task", ""))
        if source_task != EXPECTED_SOURCE_TASK:
            raise ValueError(f"{path} has task {source_task!r}; {EXPECTED_SOURCE_TASK!r} was required")
        fields = list(data_dataset.dtype.names or ())
        if "time" not in fields:
            raise ValueError(f"{path} GNI data lacks required 'time' field")
        data = data_dataset[:]
        emg_fields = [field for field in fields if np.asarray(data[field]).ndim == 2 and np.asarray(data[field]).shape[1] == 16]
        if "emg" in emg_fields:
            emg_field = "emg"
        elif len(emg_fields) == 1:
            emg_field = emg_fields[0]
        else:
            raise ValueError(f"{path} must contain exactly one 16-channel GNI EMG field")
        emg, time_s = np.asarray(data[emg_field]), np.asarray(data["time"])
        if time_s.ndim != 1 or len(time_s) != len(emg):
            raise ValueError("GNI time must be a T-length vector aligned with EMG")
        if len(emg) == 0:
            raise ValueError("GNI records must contain at least one sample")
        stages = _read_gni_table(path, "stages")
        prompt_rows = _read_gni_table(path, "prompts")
        if not prompt_rows:
            raise ValueError("Discrete gesture records require prompt/event rows")
        origin = _finite_float(time_s[0], "GNI time origin")
        labels: dict[str, Any] = {"events": [{"name": str(_required_value(row, ("name", "gesture"), "gesture")), "time_s": _finite_float(_required_value(row, ("time", "time_s"), "gesture"), "gesture time") - origin} for row in prompt_rows]}
        if len(stages) == 1 and "stage" in stages[0]:
            labels["stage"] = stages[0]["stage"]
        subject_raw = _record_subject(path, metadata)
        session_raw = str(metadata.get("session_id", path.stem))
        return CanonicalRecord(DATASET_ID, TASK_ID, f"{DATASET_ID}:{subject_raw}", f"{DATASET_ID}:{session_raw}", str(metadata.get("side", "dominant")), "", official_split, path.name, str(path.resolve()), emg, time_s, labels, {"source_dataset": "data", "emg_field": emg_field, "attributes": metadata, "source_time_origin_s": origin, "prompts": prompt_rows, "events": prompt_rows, "stages": stages}, {"converter_transformations": ["time_s shifted to record-relative origin"]})


def _safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip(".") or "unknown"


def _output_relative_path(record: CanonicalRecord) -> Path:
    return Path(_safe_component(record.dataset_id)) / _safe_component(record.task_id) / _safe_component(record.subject_id.split(":", 1)[-1]) / f"{_safe_component(record.session_id.split(':', 1)[-1])}_{_safe_component(record.side)}.h5"


def _source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_row(record: CanonicalRecord, relative_path: Path) -> dict[str, Any]:
    return {"path": relative_path.as_posix(), "dataset_id": record.dataset_id, "task_id": record.task_id, "subject_id": record.subject_id, "session_id": record.session_id, "side": record.side, "pair_id": record.pair_id, "official_split": record.official_split, "sample_count": int(len(record.emg)), "duration_s": float(record.time_s[-1] - record.time_s[0]), "sample_rate_hz": SAMPLE_RATE_HZ, "source_file": record.source_file, "source_sha256": _source_sha256(record.source_path)}


def _append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    missing = set(MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Manifest row missing required fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = _acquire_manifest_lock(path)
    try:
        rows: list[dict[str, str]] = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        normalized = {column: row[column] for column in MANIFEST_COLUMNS}
        rows = [existing for existing in rows if existing["path"] != normalized["path"]]
        rows.append(normalized)
        rows.sort(key=lambda item: str(item["path"]))
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with _replace_after_success(temporary, path), temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        _release_manifest_lock(lock_handle)


def _split_lookup(csv_path: str | Path | None) -> dict[str, str]:
    if csv_path is None:
        return {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "split" not in rows.fieldnames:
            raise ValueError("Split CSV must include a 'split' column")
        key_column = next((name for name in ("path", "source_file", "file") if name in rows.fieldnames), None)
        if key_column is None:
            raise ValueError("Split CSV must include one of: path, source_file, file")
        return {str(row[key_column]).replace("\\", "/"): str(row["split"]) for row in rows}


def convert_file(source_path: str | Path, output_root: str | Path, *, official_split: str = "unknown", overwrite: bool = False, source_file: str | None = None) -> dict[str, object]:
    record = _read_gni(source_path, official_split)
    record.source_file = source_file or Path(source_path).name
    relative_path = _output_relative_path(record)
    _write_record(Path(output_root) / relative_path, record, overwrite=overwrite)
    return _manifest_row(record, relative_path)


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
    input_root, output_root = Path(input_root), Path(output_root)
    try:
        output_root.resolve().relative_to(input_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-root must not be nested under input-root")
    lookup = _split_lookup(splits_csv)
    rows: list[dict[str, object]] = []
    for source_path in _source_files(input_root):
        relative_source = source_path.relative_to(input_root).as_posix()
        official_split = lookup.get(relative_source, lookup.get(source_path.name, "unknown"))
        if split is not None and official_split != split:
            continue
        row = convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite, source_file=relative_source)
        _append_manifest_row(output_root / "manifest.csv", row)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", help="Convert only this official split")
    parser.add_argument("--splits-csv", help="CSV mapping source path or file to official split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = run_conversion(args.input_root, args.output_root, split=args.split, splits_csv=args.splits_csv, overwrite=args.overwrite)
    print(f"Converted {len(rows)} {DATASET_ID} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
