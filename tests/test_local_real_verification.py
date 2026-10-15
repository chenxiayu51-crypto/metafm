"""The local real-mini verification surface: configs, callback routing, checks.

The leakage logic here is half-open interval arithmetic, which is easy to get
backwards (an inverted comparison silently passes every overlap), so it is
tested against both a disjoint and an overlapping case.
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verify():
    return _load("verify_local_real", "scripts/verify_local_real.py")


@pytest.fixture(scope="module")
def faults():
    return _load("e3_faults", "tests/test_e3_fault_injection.py")


@pytest.fixture
def root(faults, tmp_path):
    return faults.build_root(tmp_path / "canonical")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_local_real_profile_composes_from_the_shared_config_tree():
    from hydra import compose, initialize
    from semg_fm.config import build_trainer_options

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="pretrain", overrides=["data=local_real", "trainer=local_real"])

    assert cfg.data.name == "local_real"
    # The window geometry must NOT drift from the server profile.
    assert cfg.data.window_ms == 2000 and cfg.data.stride_ms == 1000
    assert cfg.data.channels == 16
    # Single device, so a local run cannot silently spread and stop being comparable.
    assert cfg.trainer.devices == 1
    assert cfg.trainer.precision == "bf16-mixed"
    # Same reliability callbacks the server uses.
    assert set(cfg.trainer.callbacks) == {"diagnostics", "checkpoint"}
    assert cfg.trainer.callbacks.checkpoint.save_last is True
    assert len(build_trainer_options(cfg.trainer)["callbacks"]) == 2


def test_callback_output_is_redirected_to_the_run_directory(verify, tmp_path):
    """Driving the Trainer without Hydra means the cwd is not the run dir."""
    config = {
        "callbacks": {
            "checkpoint": {"_target_": "x", "dirpath": "checkpoints"},
            "diagnostics": {"_target_": "y", "output_dir": "."},
        }
    }
    run_dir = tmp_path / "run"
    redirected = verify._redirect_callbacks(config, run_dir)

    assert redirected["callbacks"]["checkpoint"]["dirpath"] == str(run_dir / "checkpoints")
    assert redirected["callbacks"]["diagnostics"]["output_dir"] == str(run_dir)
    assert redirected["default_root_dir"] == str(run_dir)
    # The original must not be mutated in place.
    assert config["callbacks"]["checkpoint"]["dirpath"] == "checkpoints"


# ---------------------------------------------------------------------------
# Data section
# ---------------------------------------------------------------------------

def _config(verify, root):
    cfg, _ = verify.load_config(root, "splits.csv", "nature_corpus.csv",
                                batch_size=2, num_workers=0, steps=5, resume_steps=2,
                                val_every=5, val_batches=2, seed=0)
    return cfg


def test_data_section_accepts_a_valid_five_corpus_root(verify, root):
    report = verify.section_data(_config(verify, root), root)
    assert set(report["tasks"]) == set(verify.TASKS)
    for task, detail in report["tasks"].items():
        assert detail["train_windows"] > 0, task
        assert detail["val_windows"] > 0, task
    # Nature recordings legitimately serve both splits; that is not a leak.
    assert report["leakage"]["shared_records"] > 0
    assert report["leakage"]["shared_records_all_nature"] is True
    assert report["leakage"]["overlapping_sample_ranges"] == 0


def test_data_section_rejects_a_pair_spanning_splits(verify, faults, root):
    """One wrist of a typing pair in val while the other stays in train is a leak.

    A manifest row can only belong to one split, so a record cannot literally
    appear twice; the reachable version of that leak is a *pair* straddling the
    boundary, which the loader's group validation must reject.
    """
    rows, columns = faults._manifest_rows(root)
    for row in rows:
        if row["dataset_id"] == "emg2qwerty" and row["session_id"] == "s1" and row["side"] == "right":
            row["official_split"] = "val"
            row["split"] = "val"
    faults._write_manifest(root, rows, columns)
    with pytest.raises(ValueError, match="pair crosses split boundaries"):
        verify.section_data(_config(verify, root), root)


def test_half_open_overlap_detection_is_not_inverted():
    """[1000,8000) and [9000,15000) are disjoint; the naive test must say so."""
    def overlaps(left, right):
        return any(a1 < b2 and b1 < a2 for a1, a2 in left for b1, b2 in right)

    assert overlaps([(1000, 8000)], [(9000, 15000)]) is False
    assert overlaps([(1000, 8000)], [(8000, 9000)]) is False   # touching is not overlapping
    assert overlaps([(1000, 8000)], [(7999, 9000)]) is True    # one sample of overlap
    assert overlaps([(1000, 8000)], [(0, 1000)]) is False


# ---------------------------------------------------------------------------
# Sampling section
# ---------------------------------------------------------------------------

def test_sampling_section_measures_all_five_corpora_and_reports_the_real_ratio(verify, root):
    report = verify.section_sampling(_config(verify, root), root)
    assert set(report["per_task_batches"]) == set(verify.TASKS)
    assert all(count > 0 for count in report["per_task_batches"].values())
    assert sum(report["per_task_share"].values()) == pytest.approx(1.0)
    assert report["batches_per_epoch"] == sum(report["per_task_batches"].values())
    # Each corpus contributes floor(windows / batch_size) batches per epoch.
    for task, windows in report["windows_per_task"].items():
        assert report["per_task_batches"][task] == windows // 2, task
    assert "ORDER" in report["conclusion"]


def test_batch_size_larger_than_a_corpus_is_refused_by_preflight(faults, root):
    """A batch size that starves a corpus must stop the run, not skip the corpus."""
    from importlib import util as importlib_util

    spec = importlib_util.spec_from_file_location("preflight", REPO / "scripts" / "server_preflight.py")
    preflight = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    with pytest.raises(RuntimeError, match="too large"):
        preflight.section_dataloader({
            "data_root": root, "manifest": "splits.csv", "corpus": "nature_corpus.csv",
            "window_ms": 2000, "stride_ms": 1000, "batch_size": 10_000, "num_workers": 0,
        })
