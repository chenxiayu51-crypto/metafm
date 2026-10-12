"""将双侧 emg2qwerty HDF5 session 转换为成对关联的 canonical 单腕记录。"""

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


SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000
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
        raise ValueError("emg2qwerty records must contain at least one sample")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")


def _structured_table(rows: list[dict[str, Any]], fields: list[tuple[str, Any]]) -> np.ndarray:
    dtype = np.dtype(fields)
    result = np.empty(len(rows), dtype=dtype)
    for index, row in enumerate(rows):
        result[index] = tuple(row[field] for field, _ in fields)
    return result


def _write_record_contents(temporary: Path, record: CanonicalRecord) -> None:
    _validate_record(record)
    relative_time_s = record.time_s.astype(np.float64, copy=False) - record.time_s[0]
    emg = record.emg.astype(np.float32, copy=False)
    with h5py.File(temporary, "w") as h5:
        h5.attrs.update({
            "schema_version": SCHEMA_VERSION, "dataset_id": record.dataset_id,
            "task_id": record.task_id, "subject_id": record.subject_id,
            "session_id": record.session_id, "side": record.side, "pair_id": record.pair_id,
            "sample_rate_hz": SAMPLE_RATE_HZ, "official_split": record.official_split,
            "source_file": record.source_file,
            "preprocessing_json": json.dumps(record.preprocessing, default=_json_default, sort_keys=True),
        })
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=emg, compression="gzip")
        signal.create_dataset("time_s", data=relative_time_s)
        labels = h5.create_group("labels")
        labels.create_dataset("source_json", data=json.dumps(_labels_provenance(record.labels), default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        text = h5py.string_dtype("utf-8")
        labels.create_dataset("keystrokes", data=_structured_table(record.labels.get("keystrokes", []), [("key", text), ("press_s", np.float64), ("release_s", np.float64)]))
        labels.create_dataset("prompts", data=_structured_table(record.labels.get("prompts", []), [("text", text), ("start_s", np.float64), ("end_s", np.float64)]))
        meta = h5.create_group("meta")
        meta.create_dataset("source_json", data=json.dumps(record.source_meta, default=_json_default, sort_keys=True), dtype=h5py.string_dtype("utf-8"))
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=len(emg))
        quality.create_dataset("duration_s", data=float(relative_time_s[-1]))
        quality.create_dataset("nan_count", data=int(np.isnan(emg).sum()))
        quality.create_dataset("channel_rms", data=np.sqrt(np.mean(emg**2, axis=0)))


def _stage_record(output: Path, record: CanonicalRecord) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_record_contents(temporary, record)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_record(path: str | Path, record: CanonicalRecord, *, overwrite: bool = False) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {output}")
    temporary = _stage_record(output, record)
    with _replace_after_success(temporary, output):
        pass
    return output


def _commit_pair(staged: list[tuple[Path, Path]], *, overwrite: bool) -> None:
    destinations = [destination for _, destination in staged]
    if len(set(destinations)) != len(destinations):
        raise ValueError("qwerty left and right outputs must be distinct")
    existing = [destination for destination in destinations if destination.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {existing[0]}")
    backups: list[tuple[Path, Path]] = []
    committed: list[Path] = []
    try:
        for destination in existing:
            backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.bak")
            os.replace(destination, backup)
            backups.append((destination, backup))
        for temporary, destination in staged:
            os.replace(temporary, destination)
            committed.append(destination)
    except BaseException:
        for destination in committed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups:
            if backup.exists():
                os.replace(backup, destination)
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    return safe.strip(".") or "unknown"


def output_relative_path(record: CanonicalRecord) -> Path:
    return (
        Path(safe_component(record.dataset_id)) / safe_component(record.task_id)
        / safe_component(record.subject_id.split(":", 1)[-1])
        / f"{safe_component(record.session_id.split(':', 1)[-1])}_{safe_component(record.side)}.h5"
    )


def build_manifest_row(record: CanonicalRecord, relative_path: Path) -> dict[str, Any]:
    return {
        "path": relative_path.as_posix(), "dataset_id": record.dataset_id, "task_id": record.task_id,
        "subject_id": record.subject_id, "session_id": record.session_id, "side": record.side,
        "pair_id": record.pair_id, "official_split": record.official_split,
        "sample_count": int(len(record.emg)), "duration_s": float(record.time_s[-1] - record.time_s[0]),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "source_file": record.source_file, "source_sha256": source_sha256(record.source_path),
    }


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_manifest_rows(path: str | Path, new_rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    for row in new_rows:
        missing = set(MANIFEST_COLUMNS) - set(row)
        if missing:
            raise ValueError(f"Manifest row missing required fields: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = _acquire_manifest_lock(path)
    try:
        normalized = [{column: row[column] for column in MANIFEST_COLUMNS} for row in new_rows]
        replaced_paths = {row["path"] for row in normalized}
        rows = [existing for existing in _read_manifest(path) if existing["path"] not in replaced_paths]
        rows.extend(normalized)
        rows.sort(key=lambda item: str(item["path"]))
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with _replace_after_success(temporary, path), temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    finally:
        _release_manifest_lock(lock_handle)


def append_manifest_row(path: str | Path, row: dict[str, Any]) -> None:
    _append_manifest_rows(path, [row])


def _json_rows(value: Any, label_name: str) -> list[dict[str, Any]]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON for {label_name}: {error}") from error
    if isinstance(value, dict):
        value = value.get(label_name, value.get("rows", []))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{label_name} must decode to a list of objects")
    return [{str(key): _decode_value(item) for key, item in row.items()} for row in value]


def _dataset_rows(group: h5py.Group, names: tuple[str, ...]) -> list[dict[str, Any]] | None:
    for name in names:
        if name not in group:
            continue
        dataset = group[name]
        if not isinstance(dataset, h5py.Dataset):
            continue
        if dataset.dtype.names is None:
            return _json_rows(dataset.asstr()[()], name)
        return [{field: _decode_value(row[field]) for field in dataset.dtype.names} for row in dataset[:]]
    return None


def _metadata_rows(group: h5py.Group, metadata: dict[str, Any], names: tuple[str, ...], label_name: str) -> list[dict[str, Any]]:
    rows = _dataset_rows(group, names)
    if rows is not None:
        return rows
    for name in names:
        for candidate in (name, f"{name}_json"):
            if candidate in metadata:
                return _json_rows(metadata[candidate], name)
    raise ValueError(f"emg2qwerty requires {label_name} labels")


def _required_value(row: dict[str, Any], names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"{label} row lacks one of {names}")


def _prompt_text(row: dict[str, Any]) -> str | None:
    """Return the typed text of a prompt row, or None for a non-text segment.

    The released sessions keep the string under ``payload.text`` on rows whose
    ``payload_type`` is ``text_prompt``; a structured ``prompts`` dataset would
    already expose ``text`` directly.  The same attribute also holds recording
    segments (``emg_left``, ``Intro``, ``Metadata``, ...) that carry no text.
    """
    for name in ("text", "prompt"):
        if name in row:
            return str(row[name])
    payload = row.get("payload")
    if isinstance(payload, dict) and payload.get("text") is not None:
        return str(payload["text"])
    return None


def _relative_qwerty_labels(group: h5py.Group, metadata: dict[str, Any], origin_s: float) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    keystrokes = []
    source_keystrokes = _metadata_rows(group, metadata, ("keystrokes", "keylogger"), "keystrokes")
    for row in source_keystrokes:
        # The release spells key down/up as `start`/`end`.
        press_s = float(_required_value(row, ("press_s", "press", "down_s", "down", "start"), "keystroke"))
        release_s = float(_required_value(row, ("release_s", "release", "up_s", "up", "end"), "keystroke"))
        if not np.isfinite((press_s, release_s)).all():
            raise ValueError("keystroke times must be finite")
        if press_s > release_s:
            raise ValueError("keystroke press time must not exceed release time")
        keystrokes.append({
            "key": str(_required_value(row, ("key", "key_name", "name"), "keystroke")),
            "press_s": press_s - origin_s,
            "release_s": release_s - origin_s,
        })
    prompts = []
    source_prompts = _metadata_rows(group, metadata, ("prompts", "prompt"), "prompts")
    for row in source_prompts:
        text = _prompt_text(row)
        if text is None:
            # The same attribute also holds recording segments (emg_left,
            # emg_right, Intro, Metadata, ...) that carry no typed text.  They
            # stay in the source rows for provenance but are not prompts.
            continue
        start_s = float(_required_value(row, ("start_s", "start"), "prompt"))
        end_s = float(_required_value(row, ("end_s", "end"), "prompt"))
        if not np.isfinite((start_s, end_s)).all():
            raise ValueError("prompt times must be finite")
        if start_s > end_s:
            raise ValueError("prompt start time must not exceed end time")
        prompts.append({
            "text": text,
            "start_s": start_s - origin_s,
            "end_s": end_s - origin_s,
        })
    if source_prompts and not prompts:
        raise ValueError(
            "emg2qwerty session has prompt rows but none carry text, so the typing "
            "target would be empty. Expected a 'text'/'prompt' field or a payload_type "
            "of 'text_prompt'."
        )
    return {"keystrokes": keystrokes, "prompts": prompts}, {"keystrokes": source_keystrokes, "prompts": source_prompts}


def read_emg2qwerty(path: str | Path, official_split: str = "unknown", source_file: str | None = None) -> list[CanonicalRecord]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "emg2qwerty/timeseries" not in h5:
            raise ValueError(f"{path} lacks documented emg2qwerty/timeseries dataset")
        group, timeseries = h5["emg2qwerty"], h5["emg2qwerty/timeseries"]
        required = {"emg_left", "emg_right", "time"}
        fields = set(timeseries.dtype.names or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path} missing qwerty timeseries fields: {sorted(missing)}")
        data = timeseries[:]
        left, right, time_s = data["emg_left"], data["emg_right"], data["time"]
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or left.shape[1] != 16:
            raise ValueError("qwerty EMG sides must be aligned [T,16] arrays")
        if time_s.ndim != 1 or len(time_s) != len(left):
            raise ValueError("qwerty time must align with both EMG sides")
        if len(left) == 0:
            raise ValueError("emg2qwerty records must contain at least one sample")
        metadata = {str(key): _decode_value(value) for key, value in group.attrs.items()}
        # The released sessions carry the subject as `user` and the session as
        # `session_name`; `subject_id`/`session_id` are absent.  Falling back to
        # the parent directory would give every subject in the release the same
        # id and collapse leakage-safe subject grouping into one group.
        subject_raw = str(metadata.get("subject_id") or metadata.get("user") or path.parent.name)
        session_raw = str(metadata.get("session_id") or metadata.get("session_name") or path.stem)
        pair_id = f"emg2qwerty:{session_raw}"
        source_time_origin_s = float(time_s[0])
        labels, source_label_rows = _relative_qwerty_labels(group, metadata, source_time_origin_s)
        common = {
            "dataset_id": "emg2qwerty", "task_id": "typing", "subject_id": f"emg2qwerty:{subject_raw}",
            "session_id": pair_id, "pair_id": pair_id, "official_split": official_split,
            "source_file": source_file or path.name, "source_path": str(path.resolve()), "time_s": np.asarray(time_s), "labels": labels,
            "source_meta": {"source_group": "emg2qwerty", "source_timeseries": "emg2qwerty/timeseries", "attributes": metadata, "source_time_origin_s": source_time_origin_s, "source_label_rows": source_label_rows},
            "preprocessing": {"converter_transformations": ["time_s shifted to record-relative origin"]},
        }
        return [CanonicalRecord(side="left", emg=np.asarray(left), **common), CanonicalRecord(side="right", emg=np.asarray(right), **common)]


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


def convert_file(
    source_path: str | Path,
    output_root: str | Path,
    *,
    official_split: str = "unknown",
    overwrite: bool = False,
    source_file: str | None = None,
) -> list[dict[str, object]]:
    records = read_emg2qwerty(source_path, official_split, source_file)
    destinations = [(record, Path(output_root) / output_relative_path(record)) for record in records]
    existing = [destination for _, destination in destinations if destination.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing canonical record: {existing[0]}")
    staged: list[tuple[Path, Path]] = []
    try:
        for record, destination in destinations:
            staged.append((_stage_record(destination, record), destination))
        _commit_pair(staged, overwrite=overwrite)
    except BaseException:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    rows: list[dict[str, object]] = []
    for record, destination in destinations:
        relative_path = output_relative_path(record)
        rows.append(build_manifest_row(record, relative_path))
    return rows


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


def run_conversion(
    input_root: str | Path,
    output_root: str | Path,
    *,
    split: str | None = None,
    splits_csv: str | Path | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
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
        source_rows = convert_file(source_path, output_root, official_split=official_split, overwrite=overwrite, source_file=relative_source)
        _append_manifest_rows(output_root / "manifest.csv", source_rows)
        rows.extend(source_rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", help="Convert only this official split")
    parser.add_argument("--splits-csv", help="CSV mapping source path or file to official split")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows = run_conversion(
        args.input_root,
        args.output_root,
        split=args.split,
        splits_csv=args.splits_csv,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(rows)} canonical wrist records from emg2qwerty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
