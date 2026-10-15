"""Every failure mode E3 must refuse to start on, must actually refuse.

The point of this file is negative testing: for each broken input the pipeline
has to fail loudly with a diagnosable message. A silent skip, a zero-step run
or a stack trace from deep inside h5py would all be worse than a clear refusal,
because on a server they would burn GPU hours before anyone noticed.
"""
import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def e3():
    spec = importlib.util.spec_from_file_location("e3_oneclick", REPO / "scripts" / "e3_oneclick.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FS = 2000
SAMPLES = 16000
ORIGIN = 100.0
TEXT = h5py.string_dtype("utf-8")
# Each official interval must be longer than one 2000 ms window (4000 samples).
CORPUS = {"train": (100.5, 104.0), "val": (104.5, 107.5)}
DATASETS = {
    "emg2pose": "pose",
    "emg2qwerty": "typing",
    "gni_discrete_gestures": "discrete_gesture",
    "gni_handwriting": "handwriting",
    "gni_wrist": "wrist_angle",
}
NATURE = ("gni_discrete_gestures", "gni_handwriting", "gni_wrist")
MANIFEST_COLUMNS = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_count", "duration_s", "sample_rate_hz", "source_file", "source_sha256",
]


def _write_h5(path: Path, dataset_id, task_id, subject, session, side, pair_id,
              official_split, origin, channels=16, samples=SAMPLES, corrupt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(samples, dtype=np.float64) / FS
    emg = np.zeros((samples, channels), dtype=np.float32)
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "schema_version": "1.0", "dataset_id": dataset_id, "task_id": task_id,
            "subject_id": subject, "session_id": session, "side": side, "pair_id": pair_id,
            "official_split": official_split, "sample_rate_hz": float(FS),
            "source_file": path.name, "preprocessing_json": "{}",
        })
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=emg)
        signal.create_dataset("time_s", data=t)
        labels = h5.create_group("labels")
        labels.create_dataset("source_json", data="{}", dtype=TEXT)
        if task_id == "pose":
            labels.create_dataset("joint_angles", data=np.zeros((samples, 20), dtype=np.float32))
            labels.create_dataset("no_ik_failure", data=np.ones(samples, dtype=np.bool_))
        elif task_id == "typing":
            labels.create_dataset("keystrokes", data=np.array(
                [(b"a", 1.0, 1.05)], dtype=[("key", TEXT), ("press_s", "f8"), ("release_s", "f8")]))
            labels.create_dataset("prompts", data=np.array(
                [(b"ab", 0.5, 2.0)], dtype=[("text", TEXT), ("start_s", "f8"), ("end_s", "f8")]))
        elif task_id == "discrete_gesture":
            labels.create_dataset("events", data=np.array(
                [(b"fist", 1.2)], dtype=[("name", TEXT), ("time_s", "f8")]))
        elif task_id == "handwriting":
            labels.create_dataset("prompts", data=np.array(
                [(b"ab", 0.5, 2.0)], dtype=[("text", TEXT), ("start_s", "f8"), ("end_s", "f8")]))
        else:
            targets = labels.create_group("targets")
            targets.create_dataset("angle", data=np.zeros((samples, 2), dtype=np.float32))
        meta = h5.create_group("meta")
        meta.create_dataset("source_json", data=json.dumps({"source_time_origin_s": origin}), dtype=TEXT)
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=samples)
        quality.create_dataset("duration_s", data=float(t[-1]))
        quality.create_dataset("nan_count", data=0)
        quality.create_dataset("channel_rms", data=np.ones(channels))


def build_root(root: Path) -> Path:
    """A valid five-corpus canonical root, used as the baseline for mutations."""
    root.mkdir(parents=True, exist_ok=True)
    rows, corpus_rows = [], []

    def add(rel, dataset_id, task_id, subject, session, side, pair_id, official_split, split, **kw):
        _write_h5(root / rel, dataset_id, task_id, subject, session, side, pair_id,
                  official_split, 0.0 if dataset_id not in NATURE else ORIGIN, **kw)
        rows.append({
            "path": rel, "dataset_id": dataset_id, "task_id": task_id, "subject_id": subject,
            "session_id": session, "side": side, "pair_id": pair_id, "official_split": official_split,
            "sample_count": str(kw.get("samples", SAMPLES)), "duration_s": "7.99",
            "sample_rate_hz": str(FS), "source_file": Path(rel).name,
            "source_sha256": hashlib.sha256(rel.encode()).hexdigest(), "split": split,
        })

    for split in ("train", "val"):
        rel = f"emg2pose/pose/u_{split}/s_{split}_right.h5"
        add(rel, "emg2pose", "pose", f"u_{split}", f"s_{split}", "right", "", split, split)
    for session, split in (("s1", "train"), ("s2", "val")):
        for side in ("left", "right"):
            rel = f"emg2qwerty/typing/u_{session}/{session}_{side}.h5"
            add(rel, "emg2qwerty", "typing", f"u_{session}", session, side,
                f"emg2qwerty:{session}", split, split)
    for dataset_id in NATURE:
        task_id = DATASETS[dataset_id]
        for index in (1, 2):
            base = f"{dataset_id}_rec_{index}.h5"
            rel = f"{dataset_id}/{task_id}/u_{index}/{base}"
            add(rel, dataset_id, task_id, f"u_{index}", f"rec_{index}", "right", "", "unknown", "")
            for split, (start, end) in CORPUS.items():
                corpus_rows.append({"dataset": base, "split": split, "start": str(start), "end": str(end)})

    with (root / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*MANIFEST_COLUMNS, "split"])
        writer.writeheader()
        writer.writerows(rows)
    with (root / "nature_corpus.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "start", "end"])
        writer.writeheader()
        writer.writerows(corpus_rows)
    return root


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return build_root(tmp_path_factory.mktemp("e3_valid_root") / "canonical")


@pytest.fixture
def root(base, tmp_path):
    """A private copy so each test can break something without affecting others."""
    target = tmp_path / "canonical"
    shutil.copytree(base, target)
    return target


def _manifest_rows(root: Path):
    with (root / "splits.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames)


def _write_manifest(root: Path, rows, columns):
    with (root / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _audit(e3, root):
    return e3.audit(root, root / "splits.csv", root / "nature_corpus.csv", inspect_all=True)


# 1 --------------------------------------------------------------------------
def test_missing_data_root_is_refused(e3, tmp_path):
    with pytest.raises(e3.E3Error, match="Data root does not exist"):
        e3.main(["audit", "--data-root", str(tmp_path / "nope")])


# 2 --------------------------------------------------------------------------
def test_missing_manifest_is_refused(e3, root):
    (root / "splits.csv").unlink()
    with pytest.raises(e3.E3Error, match="Missing manifest"):
        _audit(e3, root)


# 3 --------------------------------------------------------------------------
def test_missing_nature_corpus_is_refused(e3, root):
    (root / "nature_corpus.csv").unlink()
    # audit() is given the corpus path directly, so the reader reports it missing.
    with pytest.raises(e3.E3Error, match="Missing official Nature corpus"):
        _audit(e3, root)
    # The launcher's own discovery path reports the same problem by name.
    with pytest.raises(e3.E3Error, match="Nature corpus not found"):
        e3.find_nature_corpus(root, None)


# 4 --------------------------------------------------------------------------
def test_overlapping_official_intervals_are_refused(e3, root):
    with (root / "nature_corpus.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
    rows.append(dict(rows[0], split="val", start="103.0", end="105.0"))  # overlaps train
    with (root / "nature_corpus.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(e3.E3Error, match="Overlapping Nature intervals"):
        _audit(e3, root)


# 5 --------------------------------------------------------------------------
def test_missing_hdf5_file_is_refused(e3, root):
    (root / "emg2pose/pose/u_train/s_train_right.h5").unlink()
    with pytest.raises(e3.E3Error, match="Missing HDF5"):
        _audit(e3, root)


# 6 --------------------------------------------------------------------------
def test_empty_train_split_for_one_corpus_is_refused(e3, root):
    rows, columns = _manifest_rows(root)
    for row in rows:
        if row["dataset_id"] == "emg2pose" and row["official_split"] == "train":
            row["official_split"] = "val"
            row["split"] = "val"
    _write_manifest(root, rows, columns)
    with pytest.raises(e3.E3Error, match="Train/Val interval or file missing"):
        _audit(e3, root)


# 7 --------------------------------------------------------------------------
def test_empty_val_split_for_one_corpus_is_refused(e3, root):
    rows, columns = _manifest_rows(root)
    for row in rows:
        if row["dataset_id"] == "emg2qwerty" and row["official_split"] == "val":
            row["official_split"] = "train"
            row["split"] = "train"
    _write_manifest(root, rows, columns)
    with pytest.raises(e3.E3Error, match="Train/Val interval or file missing"):
        _audit(e3, root)


# 8 --------------------------------------------------------------------------
def test_nan_in_emg_is_refused(e3, root):
    path = root / "emg2pose/pose/u_train/s_train_right.h5"
    with h5py.File(path, "r+") as h5:
        data = h5["signal/emg"][:]
        data[10, 3] = np.nan
        h5["signal/emg"][...] = data
    with pytest.raises(e3.E3Error, match="NaN/Inf in first EMG window"):
        _audit(e3, root)


# 9 --------------------------------------------------------------------------
def test_wrong_channel_count_is_refused(e3, root):
    rel = "emg2pose/pose/u_train/s_train_right.h5"
    _write_h5(root / rel, "emg2pose", "pose", "u_train", "s_train", "right", "", "train",
              0.0, channels=15)
    with pytest.raises(e3.E3Error, match="EMG must be float32"):
        _audit(e3, root)


# 10 -------------------------------------------------------------------------
def test_checkpoint_config_mismatch_is_refused():
    """The guard that stops a resume from silently changing training semantics."""
    from semg_fm.lightning.pretrain_module import PretrainModule

    geometry = dict(d_model=8, depth=1, heads=2, patch_samples=40, channels=16, dropout=0.0)
    saved = PretrainModule(model_config=dict(geometry), warmup_steps=0, max_steps=10)

    longer = PretrainModule(model_config=dict(geometry), warmup_steps=0, max_steps=20)
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        longer.on_load_checkpoint({"hyper_parameters": dict(saved.hparams)})

    wider = PretrainModule(model_config=dict(geometry, d_model=16), warmup_steps=0, max_steps=10)
    with pytest.raises(ValueError, match="resume configuration mismatch"):
        wider.on_load_checkpoint({"hyper_parameters": dict(saved.hparams)})

    identical = PretrainModule(model_config=dict(geometry), warmup_steps=0, max_steps=10)
    identical.on_load_checkpoint({"hyper_parameters": dict(saved.hparams)})  # must not raise


# 11 -------------------------------------------------------------------------
def test_server_mode_without_cuda_is_refused(e3, root, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(e3.E3Error, match="SERVER mode requires CUDA"):
        e3.command(REPO, root, root / "splits.csv", root / "nature_corpus.csv",
                   root / "run", "server", 5, None)


# 12 -------------------------------------------------------------------------
def test_unwritable_output_root_is_refused(tmp_path):
    """A preflight report path that cannot be created must fail, not be skipped."""
    from importlib import util as importlib_util

    spec = importlib_util.spec_from_file_location("preflight", REPO / "scripts" / "server_preflight.py")
    preflight = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    blocked = tmp_path / "not_a_directory"
    blocked.write_text("this is a file, not a directory", encoding="utf-8")
    with pytest.raises((OSError, NotADirectoryError, RuntimeError)):
        preflight.section_environment({"data_root": tmp_path, "output": blocked / "child"})


# 13 -------------------------------------------------------------------------
def test_synthetic_fixture_is_refused_for_server_mode(e3, root):
    (root / e3.MARKER).write_text("SYNTHETIC ONLY\n", encoding="utf-8")
    with pytest.raises(e3.E3Error, match="Synthetic HDF5 fixture is forbidden"):
        e3.main(["server", "--data-root", str(root), "--output", str(root / "runs"),
                 "--max-steps", "1"])


# 14 -------------------------------------------------------------------------
def test_official_test_split_cannot_enter_training(root):
    """`evaluation_split=train` is forbidden, and test data stays out of train/val."""
    from semg_fm.data.datamodule import build_datamodule

    for split in ("train", "val"):
        with pytest.raises(ValueError, match="evaluation_split must be val or test"):
            build_datamodule({
                "root": str(root), "manifest_path": str(root / "splits.csv"),
                "corpus_path": str(root / "nature_corpus.csv"),
                "window_ms": 2000, "stride_ms": 1000, "batch_size": 1, "num_workers": 0,
                "evaluation_split": "train",
            }, task_id=None, label_budget=1.0)

    # A record marked official test must not appear in either fitted split.
    rows, columns = _manifest_rows(root)
    for row in rows:
        if row["dataset_id"] == "emg2pose":
            row["official_split"] = "test" if row["split"] == "train" else row["split"]
    _write_manifest(root, rows, columns)
    module = build_datamodule({
        "root": str(root), "manifest_path": str(root / "splits.csv"),
        "corpus_path": str(root / "nature_corpus.csv"),
        "window_ms": 2000, "stride_ms": 1000, "batch_size": 1, "num_workers": 0,
        "evaluation_split": "val",
    }, task_id=None, label_budget=1.0)
    module.setup("fit")
    used = {(record["dataset_id"], record["record_id"])
            for record in (*module.train_dataset._record_rows, *module.val_dataset._record_rows)}
    assert ("emg2pose", "s_train_right") not in used


# 15 -------------------------------------------------------------------------
def test_dataloader_worker_error_is_not_swallowed(root):
    """A record that cannot be read must raise, and must not be skipped."""
    from torch.utils.data import DataLoader
    from semg_fm.data.datamodule import task_homogeneous_collate
    from semg_fm.data.window_dataset import CanonicalWindowDataset

    path = root / "gni_wrist/wrist_angle/u_1/gni_wrist_rec_1.h5"
    path.write_bytes(b"not an HDF5 file at all")

    # The unreadable record must surface as an error, never as a silent skip.
    # It may fail while the Nature window index is built (from_manifest reads the
    # source clock) or on the first __getitem__; both are loud and acceptable.
    with pytest.raises(OSError) as excinfo:
        dataset = CanonicalWindowDataset.from_manifest(
            root / "splits.csv", "train", 2000, 1000, root=root,
            corpus_path=str(root / "nature_corpus.csv"))
        loader = DataLoader(dataset, batch_size=1, num_workers=0,
                            collate_fn=task_homogeneous_collate)
        for _ in loader:
            pass
    assert "signature" in str(excinfo.value).lower() or "hdf5" in str(excinfo.value).lower()
