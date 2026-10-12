"""独立 GNI 转换器安全防护的回归测试。"""

import importlib.util
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


GNI_CONVERTERS = [
    "convert_gni_discrete_gestures",
    "convert_gni_handwriting",
    "convert_gni_wrist",
]

SEMG_H5_DIRECTORY = Path(__file__).resolve().parents[1] / "semg_h5"


def _load_converter(converter_name):
    script_path = (SEMG_H5_DIRECTORY / f"{converter_name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"standalone_{converter_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_manifest_lock_is_os_released_advisory_lock(converter_name):
    converter = _load_converter(converter_name)
    source = inspect.getsource(converter)

    assert "msvcrt.locking" in source
    assert "O_EXCL" not in source


def _write_gni_source(path, task, *, target_fields=(), sample_count=1, prompt_rows=None):
    fields = [("emg", np.float32, (16,)), ("time", np.float64)] + list(target_fields)
    rows = np.zeros(sample_count, dtype=np.dtype(fields))
    with h5py.File(path, "w") as h5:
        data = h5.create_dataset("data", data=rows)
        data.attrs["task"] = task
        if prompt_rows is None:
            prompt_rows = [(b"fist", 0.0)] if task == "discrete_gestures" else [(b"a", 0.0, 0.0)] if task == "handwriting" else []
        if task == "discrete_gestures" and prompt_rows:
            h5.create_dataset("prompts", data=np.array(prompt_rows, dtype=[("name", "S32"), ("time", np.float64)]))
        if task == "handwriting" and prompt_rows:
            h5.create_dataset("prompts", data=np.array(prompt_rows, dtype=[("prompt", "S32"), ("start", np.float64), ("end", np.float64)]))


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_rejects_empty_record_before_time_endpoint_access(tmp_path, converter_name):
    source = tmp_path / "empty.h5"
    task = {"convert_gni_discrete_gestures": "discrete_gestures", "convert_gni_handwriting": "handwriting", "convert_gni_wrist": "wrist"}[converter_name]
    _write_gni_source(source, task, target_fields=[("flexion_extension", np.float32)] if task == "wrist" else (), sample_count=0)

    converter = _load_converter(converter_name)
    with pytest.raises(ValueError, match="at least one sample"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_rejects_output_root_nested_in_input_root(tmp_path, converter_name):
    input_root = tmp_path / "input"
    input_root.mkdir()
    converter = _load_converter(converter_name)

    with pytest.raises(ValueError, match="output-root.*input-root"):
        converter.run_conversion(input_root, input_root / "converted")


def test_wrist_converter_rejects_sanitized_target_name_collision(tmp_path):
    source = tmp_path / "collision.h5"
    _write_gni_source(source, "wrist", target_fields=[("a/b", np.float32), ("a?b", np.float32)])

    converter = _load_converter("convert_gni_wrist")
    with pytest.raises(ValueError, match="collision"):
        converter.convert_file(source, tmp_path / "out")


def test_wrist_converter_rejects_nonfinite_target_with_field_name(tmp_path):
    source = tmp_path / "nonfinite_target.h5"
    _write_gni_source(source, "wrist", target_fields=[("flexion_extension", np.float32)])
    with h5py.File(source, "r+") as h5:
        h5["data"]["flexion_extension"] = np.nan

    converter = _load_converter("convert_gni_wrist")
    with pytest.raises(ValueError, match="flexion_extension.*finite"):
        converter.convert_file(source, tmp_path / "out")


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_h5_replace_failure_preserves_existing_output_and_cleans_temp(tmp_path, monkeypatch, converter_name):
    source = tmp_path / "source.h5"
    task = {"convert_gni_discrete_gestures": "discrete_gestures", "convert_gni_handwriting": "handwriting", "convert_gni_wrist": "wrist"}[converter_name]
    _write_gni_source(source, task, target_fields=[("flexion_extension", np.float32)] if task == "wrist" else ())
    converter = _load_converter(converter_name)
    output_root = tmp_path / "out"
    row = converter.convert_file(source, output_root)
    output = output_root / row["path"]
    original = output.read_bytes()

    def fail_replace(source_path, destination_path):
        raise OSError("injected replace failure")

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        converter.convert_file(source, output_root, overwrite=True)

    assert output.read_bytes() == original
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_manifest_replace_failure_cleans_temp(tmp_path, monkeypatch, converter_name):
    converter = _load_converter(converter_name)
    manifest = tmp_path / "manifest.csv"
    row = {column: f"value-{column}" for column in converter.MANIFEST_COLUMNS}
    row["path"] = "record.h5"
    append = getattr(converter, "_append_manifest_row", None) or converter._append

    def fail_replace(source_path, destination_path):
        raise OSError("injected manifest replace failure")

    monkeypatch.setattr(converter.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected manifest replace failure"):
        append(manifest, row)

    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_run_conversion_records_relative_source_file_and_time_provenance(tmp_path, converter_name):
    input_root = tmp_path / "input"; source = input_root / "nested" / "source.h5"; source.parent.mkdir(parents=True)
    task = {"convert_gni_discrete_gestures": "discrete_gestures", "convert_gni_handwriting": "handwriting", "convert_gni_wrist": "wrist"}[converter_name]
    _write_gni_source(source, task, target_fields=[("flexion_extension", np.float32)] if task == "wrist" else ())
    converter = _load_converter(converter_name); output_root = tmp_path / "output"
    row = converter.run_conversion(input_root, output_root)[0]
    assert row["source_file"] == "nested/source.h5"
    with h5py.File(output_root / row["path"], "r") as h5:
        source_meta = json.loads(h5["meta/source_json"].asstr()[()])
    assert source_meta["source_time_origin_s"] == 0.0
    assert "stages" in source_meta


@pytest.mark.parametrize(("converter_name", "task"), [("convert_gni_discrete_gestures", "discrete_gestures"), ("convert_gni_handwriting", "handwriting")])
def test_gni_labelled_tasks_require_finite_source_rows(tmp_path, converter_name, task):
    converter = _load_converter(converter_name)
    missing = tmp_path / "missing.h5"; _write_gni_source(missing, task, prompt_rows=[])
    with pytest.raises(ValueError, match="require"):
        converter.convert_file(missing, tmp_path / "out")
    invalid = tmp_path / "invalid.h5"
    rows = [(b"fist", np.nan)] if task == "discrete_gestures" else [(b"a", 1.0, 0.0)]
    _write_gni_source(invalid, task, prompt_rows=rows)
    with pytest.raises(ValueError, match="finite|start"):
        converter.convert_file(invalid, tmp_path / "out")


_GNI_TASK_AND_PREFIX = {
    "convert_gni_discrete_gestures": ("discrete_gestures", "discrete_gestures"),
    "convert_gni_handwriting": ("handwriting", "handwriting"),
    "convert_gni_wrist": ("wrist", "wrist"),
}


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_converter_recovers_subject_identity_from_the_release_file_name(tmp_path, converter_name):
    """The public drop carries only ``data.attrs['task']``; identity is in the name.

    Reading the parent directory instead gave every subject in a corpus the same
    id, which collapsed leakage-safe subject grouping into a single group.
    """
    task, prefix = _GNI_TASK_AND_PREFIX[converter_name]
    source_dir = tmp_path / "GNI_release"
    source_dir.mkdir()
    first = source_dir / f"{prefix}_user_007_dataset_000.hdf5"
    second = source_dir / f"{prefix}_user_008_dataset_000.hdf5"
    targets = [("flexion_extension", np.float32)] if task == "wrist" else ()
    _write_gni_source(first, task, target_fields=targets, sample_count=8)
    _write_gni_source(second, task, target_fields=targets, sample_count=8)
    converter = _load_converter(converter_name)

    row_one = converter.convert_file(first, tmp_path / "out")
    row_two = converter.convert_file(second, tmp_path / "out")

    assert row_one["subject_id"].endswith("user_007")
    assert row_two["subject_id"].endswith("user_008")
    assert row_one["subject_id"] != row_two["subject_id"]


@pytest.mark.parametrize("converter_name", GNI_CONVERTERS)
def test_gni_conversion_discovers_hdf5_sources_and_rejects_an_empty_root(tmp_path, converter_name):
    """The released corpora ship ``*.hdf5``; globbing only ``*.h5`` converted nothing.

    A full-corpus run matched zero files and still exited 0, so the failure was
    invisible until the output directory was inspected.
    """
    task, prefix = _GNI_TASK_AND_PREFIX[converter_name]
    source_dir = tmp_path / "release"
    source_dir.mkdir()
    targets = [("flexion_extension", np.float32)] if task == "wrist" else ()
    _write_gni_source(
        source_dir / f"{prefix}_user_000_dataset_000.hdf5", task,
        target_fields=targets, sample_count=8,
    )
    converter = _load_converter(converter_name)

    rows = converter.run_conversion(source_dir, tmp_path / "out")

    assert len(rows) == 1
    assert rows[0]["source_file"].endswith(".hdf5")

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(ValueError, match="no .h5 or .hdf5"):
        converter.run_conversion(empty_root, tmp_path / "out_empty")
