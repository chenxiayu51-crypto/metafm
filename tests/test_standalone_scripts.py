"""验证各转换脚本能够脱离 ``semg_h5`` 包独立运行。"""

import ast
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEMG_H5_DIRECTORY = REPOSITORY_ROOT / "semg_h5"
CONVERTER_NAMES = [
    "convert_emg2pose.py",
    "convert_emg2qwerty.py",
    "convert_gni_discrete_gestures.py",
    "convert_gni_handwriting.py",
    "convert_gni_wrist.py",
]

CONVERTER_CASES = [
    ("convert_emg2pose.py", "emg2pose_h5", "pose", "labels/joint_angles"),
    ("convert_emg2qwerty.py", "qwerty_h5", "typing", "labels/keystrokes"),
    (
        "convert_gni_discrete_gestures.py",
        "discrete_gesture",
        "discrete_gesture",
        "labels/events",
    ),
    ("convert_gni_handwriting.py", "handwriting", "handwriting", "labels/prompts"),
    (
        "convert_gni_wrist.py",
        "wrist_angle",
        "wrist_angle",
        "labels/targets/wrist_angles",
    ),
]


def test_converter_scripts_are_the_expected_standalone_files_without_package_imports():
    converter_paths = sorted(SEMG_H5_DIRECTORY.glob("*.py"))
    assert [path.name for path in converter_paths] == CONVERTER_NAMES

    violations = []
    for script_path in converter_paths:
        tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.level > 0 or (node.module or "").startswith("semg_h5")
            ):
                violations.append(f"{script_path.name}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(
                alias.name.startswith("semg_h5") for alias in node.names
            ):
                violations.append(f"{script_path.name}:{node.lineno}")

    assert not violations, f"converter scripts must not import semg_h5 packages: {violations}"


def _load_converter(converter_name):
    script_path = (SEMG_H5_DIRECTORY / converter_name).resolve()
    spec = importlib.util.spec_from_file_location(f"standalone_{script_path.stem}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_path(request, fixture_name):
    if fixture_name in {"emg2pose_h5", "qwerty_h5"}:
        return request.getfixturevalue(fixture_name)
    return request.getfixturevalue("gni_h5_by_task")[fixture_name]


def test_converter_scripts_have_absolute_import_specs():
    for converter_name in CONVERTER_NAMES:
        script_path = (SEMG_H5_DIRECTORY / converter_name).resolve()
        spec = importlib.util.spec_from_file_location(f"standalone_{script_path.stem}", script_path)
        assert script_path.is_absolute()
        assert spec is not None and spec.origin == str(script_path)


def test_emg2pose_preserves_pose_labels_and_source_split(tmp_path, emg2pose_h5):
    converter = _load_converter("convert_emg2pose.py")
    manifest_row = converter.convert_file(emg2pose_h5, tmp_path / "out", official_split="val")

    with h5py.File(tmp_path / "out" / manifest_row["path"]) as h5:
        assert h5.attrs["dataset_id"] == "emg2pose"
        assert h5.attrs["task_id"] == "pose"
        assert h5.attrs["official_split"] == "val"
        assert h5["signal/emg"].shape == (4, 16)
        assert h5["labels/joint_angles"].shape == (4, 3)
        assert h5["labels/no_ik_failure"].shape == (4,)


def test_emg2qwerty_emits_one_linked_record_per_wrist(tmp_path, qwerty_h5):
    converter = _load_converter("convert_emg2qwerty.py")
    rows = converter.convert_file(qwerty_h5, tmp_path / "out", official_split="train")

    assert len(rows) == 2
    assert {row["side"] for row in rows} == {"left", "right"}
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5["signal/emg"].shape == (4, 16)
            assert h5["labels/keystrokes"].shape[0] == 1
            assert h5["labels/keystrokes"][0]["press_s"] == pytest.approx(0.0005)


@pytest.mark.parametrize(
    ("converter_name", "task_id", "label_path"),
    [
        ("convert_gni_discrete_gestures.py", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting.py", "handwriting", "labels/prompts"),
        ("convert_gni_wrist.py", "wrist_angle", "labels/targets/wrist_angles"),
    ],
)
def test_gni_converter_preserves_16ch_stream_and_labels(
    tmp_path, gni_h5_by_task, converter_name, task_id, label_path
):
    converter = _load_converter(converter_name)
    row = converter.convert_file(
        gni_h5_by_task[task_id], tmp_path / "out", official_split="test"
    )

    with h5py.File(tmp_path / "out" / row["path"]) as h5:
        assert h5.attrs["task_id"] == task_id
        assert h5["signal/emg"].shape == (4, 16)
        assert label_path in h5


@pytest.mark.parametrize(
    ("converter_name", "fixture_name", "task_id", "label_path"), CONVERTER_CASES
)
def test_converter_script_executes_as_a_clean_subprocess(
    tmp_path, request, converter_name, fixture_name, task_id, label_path
):
    script_path = (SEMG_H5_DIRECTORY / converter_name).resolve()
    input_root = tmp_path / "input"
    input_root.mkdir()
    source_path = _source_path(request, fixture_name)
    shutil.copy2(source_path, input_root / source_path.name)
    output_root = tmp_path / "out"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    with (output_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows
    for manifest_row in manifest_rows:
        with h5py.File(output_root / manifest_row["path"]) as h5:
            assert h5.attrs["task_id"] == task_id
            assert h5["signal/emg"].shape == (4, 16)
            assert label_path in h5


def test_converters_keep_the_manifest_lock_importable_off_windows():
    """A module-scope ``import msvcrt`` made every converter Windows-only.

    The lock must stay OS-released advisory locking on Windows while also
    working on POSIX, and it must never regress to O_EXCL lock files.
    """
    for name in CONVERTER_NAMES:
        source = (SEMG_H5_DIRECTORY / name).read_text(encoding="utf-8")
        module_scope_imports = {
            alias.name
            for node in ast.parse(source).body
            if isinstance(node, ast.Import)
            for alias in node.names
        }

        assert "msvcrt" not in module_scope_imports, f"{name} imports msvcrt at module scope"
        assert "msvcrt.locking" in source, f"{name} lost the Windows advisory lock"
        assert "fcntl.flock" in source, f"{name} has no POSIX advisory lock"
        assert "os.name" in source, f"{name} does not guard the platform import"
        assert "O_EXCL" not in source, f"{name} must not use O_EXCL lock files"
