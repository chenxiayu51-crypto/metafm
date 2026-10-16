"""The raw-corpus inspection path, and where each corpus is looked for.

`--inspect` exists so a wrong raw layout is caught in seconds instead of after
hours of conversion, so its own failure modes have to be right: a malformed
corpus must FAIL (not be skipped), and a missing location must name the corpus.
"""
import importlib.util
import shutil
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prepare_data():
    spec = importlib.util.spec_from_file_location("prepare_data", REPO / "scripts" / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(prepare_data, **overrides):
    base = {"raw_root": None, "raw_emg2pose": None, "raw_emg2qwerty": None,
            "raw_gni_discrete_gestures": None, "raw_gni_handwriting": None, "raw_gni_wrist": None}
    base.update(overrides)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# Locating the corpora
# ---------------------------------------------------------------------------

def test_explicit_flags_win_and_need_no_shared_parent(prepare_data, tmp_path, monkeypatch):
    """The five releases are often unpacked on different mounts entirely."""
    for name in prepare_data.ENVIRONMENT_VARIABLES.values():
        monkeypatch.delenv(name, raising=False)
    elsewhere = tmp_path / "other-mount" / "qwerty"
    resolved = prepare_data.resolve_raw_roots(_args(
        prepare_data, raw_root=str(tmp_path / "root"), raw_emg2qwerty=str(elsewhere)))
    assert resolved["emg2qwerty"] == elsewhere.resolve()
    # The rest still fall back to <raw-root>/<corpus>.
    assert resolved["emg2pose"] == (tmp_path / "root" / "emg2pose").resolve()


def test_environment_variable_is_used_when_no_flag_is_given(prepare_data, tmp_path, monkeypatch):
    target = tmp_path / "env-wrist"
    monkeypatch.setenv("E3_RAW_GNI_WRIST", str(target))
    resolved = prepare_data.resolve_raw_roots(_args(prepare_data, raw_root=str(tmp_path / "root")))
    assert resolved["gni_wrist"] == target.resolve()


def test_missing_location_names_the_corpus(prepare_data, monkeypatch):
    for name in prepare_data.ENVIRONMENT_VARIABLES.values():
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as excinfo:
        prepare_data.resolve_raw_roots(_args(prepare_data))
    message = str(excinfo.value)
    for corpus in ("emg2pose", "emg2qwerty", "gni_wrist"):
        assert corpus in message


# ---------------------------------------------------------------------------
# Probing a corpus with the converter's own reader
# ---------------------------------------------------------------------------

def test_probe_accepts_a_real_emg2pose_release_layout(prepare_data, emg2pose_h5, tmp_path):
    root = tmp_path / "raw_emg2pose"
    root.mkdir()
    shutil.copy2(emg2pose_h5, root / emg2pose_h5.name)
    report = prepare_data.probe_corpus("emg2pose", root)
    assert report["status"] == "PASS"
    assert report["hdf5_files"] == 1
    assert report["parsed"][0]["task_id"] == "pose"
    assert "emg2pose" in report["top_level"]


def test_probe_accepts_each_gni_task(prepare_data, gni_h5_by_task, tmp_path):
    for task_id, source in gni_h5_by_task.items():
        dataset_id = {"discrete_gesture": "gni_discrete_gestures",
                      "handwriting": "gni_handwriting",
                      "wrist_angle": "gni_wrist"}[task_id]
        root = tmp_path / dataset_id
        root.mkdir()
        shutil.copy2(source, root / source.name)
        report = prepare_data.probe_corpus(dataset_id, root)
        assert report["status"] == "PASS", report
        # The GNI readers return dicts, not objects; both must report properly.
        assert report["parsed"][0]["task_id"] == task_id


def test_probe_rejects_a_file_that_is_not_hdf5(prepare_data, tmp_path):
    root = tmp_path / "bogus"
    root.mkdir()
    (root / "not_really.h5").write_bytes(b"this is not an HDF5 file")
    report = prepare_data.probe_corpus("emg2pose", root)
    assert report["status"] == "FAIL"
    assert "HDF5" in report["error"]


def test_probe_rejects_a_directory_with_no_hdf5(prepare_data, tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "readme.txt").write_text("nothing here", encoding="utf-8")
    report = prepare_data.probe_corpus("emg2pose", root)
    assert report["status"] == "FAIL"
    assert "no .h5" in report["error"]


def test_probe_rejects_a_missing_directory(prepare_data, tmp_path):
    report = prepare_data.probe_corpus("emg2pose", tmp_path / "nope")
    assert report["status"] == "FAIL"
    assert "not a directory" in report["error"]


def test_probe_rejects_an_hdf5_with_the_wrong_internal_layout(prepare_data, tmp_path):
    """A valid HDF5 that is not an emg2pose release must still fail."""
    import h5py
    import numpy as np

    root = tmp_path / "wrong_layout"
    root.mkdir()
    with h5py.File(root / "mystery.h5", "w") as handle:
        handle.create_dataset("something_else", data=np.zeros(4, dtype=np.float32))
    report = prepare_data.probe_corpus("emg2pose", root)
    assert report["status"] == "FAIL"
    assert "emg2pose" in report["error"]


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def test_inspect_cli_writes_nothing_and_returns_zero(prepare_data, emg2pose_h5, qwerty_h5,
                                                     gni_h5_by_task, tmp_path, monkeypatch, capsys):
    for name in prepare_data.ENVIRONMENT_VARIABLES.values():
        monkeypatch.delenv(name, raising=False)
    sources = {"emg2pose": (emg2pose_h5, "emg2pose"), "emg2qwerty": (qwerty_h5, "emg2qwerty")}
    mapping = {"discrete_gesture": "gni_discrete_gestures",
               "handwriting": "gni_handwriting", "wrist_angle": "gni_wrist"}
    for task_id, source in gni_h5_by_task.items():
        sources[mapping[task_id]] = (source, mapping[task_id])
    flags = {"emg2pose": "--raw-emg2pose", "emg2qwerty": "--raw-emg2qwerty",
             "gni_discrete_gestures": "--raw-gni-discrete-gestures",
             "gni_handwriting": "--raw-gni-handwriting", "gni_wrist": "--raw-gni-wrist"}

    argv = ["--inspect"]
    for dataset_id, (source, _) in sources.items():
        directory = tmp_path / dataset_id
        directory.mkdir()
        shutil.copy2(source, directory / source.name)
        argv += [flags[dataset_id], str(directory)]

    before = sorted(p.name for p in tmp_path.iterdir())
    assert prepare_data.main(argv) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == before, "--inspect must not write anything"
    output = capsys.readouterr().out
    assert output.count("[PASS]") == 5 and "[FAIL]" not in output


def test_inspect_cli_returns_one_when_a_corpus_is_unreadable(prepare_data, tmp_path, monkeypatch, capsys):
    for name in prepare_data.ENVIRONMENT_VARIABLES.values():
        monkeypatch.delenv(name, raising=False)
    directories = {}
    for dataset_id in prepare_data.CONVERTERS:
        directory = tmp_path / dataset_id
        directory.mkdir()
        (directory / "broken.h5").write_bytes(b"not hdf5")
        directories[dataset_id] = directory
    argv = ["--inspect",
            "--raw-emg2pose", str(directories["emg2pose"]),
            "--raw-emg2qwerty", str(directories["emg2qwerty"]),
            "--raw-gni-discrete-gestures", str(directories["gni_discrete_gestures"]),
            "--raw-gni-handwriting", str(directories["gni_handwriting"]),
            "--raw-gni-wrist", str(directories["gni_wrist"])]
    assert prepare_data.main(argv) == 1
    assert "[FAIL]" in capsys.readouterr().out
