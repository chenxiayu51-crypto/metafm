"""Acceptance tests for the E3 server release surface.

Covers the reliability guarantees the E3 production launch depends on:
per-corpus batch accounting, student/teacher update reporting, the NaN/Inf
stop, the server output-root guard, and the fact that Nature rows are kept out
of whole-file split assignment.
"""
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from lightning import Trainer
from torch.utils.data import DataLoader

from semg_fm.lightning.pretrain_module import PretrainModule

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e3():
    return _load_script("e3_oneclick")


@pytest.fixture(scope="module")
def prepare_data():
    return _load_script("prepare_data")


def _tiny_module(**kwargs) -> PretrainModule:
    return PretrainModule(
        model_config=dict(d_model=8, depth=1, heads=2, patch_samples=40),
        warmup_steps=0, max_steps=1, **kwargs
    )


def _batch(task_id: str, samples: int = 2):
    return dict(
        emg=torch.randn(samples, 400, 16),
        sample_mask=torch.ones(samples, 400, dtype=torch.bool),
        task_id=task_id,
    )


def _train(tmp_path, batches, callbacks):
    module = _tiny_module()
    trainer = Trainer(
        accelerator="cpu", max_steps=len(batches), logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
    )
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
    return trainer


# ---------------------------------------------------------------------------
# Trainer profile
# ---------------------------------------------------------------------------

def test_e3_server_profile_registers_diagnostics_and_checkpoint():
    from hydra import compose, initialize
    from semg_fm.config import build_trainer_options

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="pretrain", overrides=["trainer=e3_server"])

    callbacks = cfg.trainer.callbacks
    assert set(callbacks) == {"diagnostics", "checkpoint"}
    assert callbacks.diagnostics._target_ == "semg_fm.e3_diagnostics.E3RunDiagnostics"
    checkpoint = callbacks.checkpoint
    assert checkpoint.save_last is True and checkpoint.save_top_k == 1
    assert checkpoint.monitor == "val/loss" and checkpoint.mode == "min"
    # A crash must not be able to cost more than this window of work.
    assert checkpoint.every_n_train_steps == 1000
    assert cfg.trainer.precision == "bf16-mixed"

    options = build_trainer_options(cfg.trainer)
    assert len(options["callbacks"]) == 2


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_reports_per_task_counts_and_update_status(tmp_path):
    from semg_fm.e3_diagnostics import E3RunDiagnostics

    _train(tmp_path, [_batch("wrist_angle"), _batch("typing")], [E3RunDiagnostics(tmp_path)])
    report = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))

    assert report["train_batches_per_task"] == {"typing": 1, "wrist_angle": 1}
    assert report["train_batch_total"] == 2
    assert report["missing_train_tasks"] == ["discrete_gesture", "handwriting", "pose"]
    assert report["student_updated"] is True
    assert report["teacher_updated"] is True
    assert report["teacher_has_no_grads"] is True
    assert report["ema_updates"] == 2
    assert report["feature_std"]["mean"] > 0

    # Per-step rows stream to disk rather than accumulating in memory.
    rows = [json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in rows] == ["wrist_angle", "typing"]
    assert all(row["student_gradient_norm"] > 0 for row in rows)


def test_diagnostics_stops_on_nonfinite_loss_and_records_the_step(tmp_path, monkeypatch):
    from semg_fm.e3_diagnostics import E3RunDiagnostics

    module = _tiny_module()
    reference = next(module.student.parameters()).reshape(-1)[:0].sum()

    def poisoned(_batch):
        total = reference + float("nan")
        return total, {"latent": total, "ring_equivariance_error": reference.detach(), "kinematic": reference}

    monkeypatch.setattr(module, "_losses", poisoned)
    callback = E3RunDiagnostics(tmp_path)
    trainer = Trainer(
        accelerator="cpu", max_steps=1, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, callbacks=[callback],
    )
    with pytest.raises(RuntimeError, match="non-finite training state"):
        trainer.fit(module, train_dataloaders=DataLoader([_batch("pose")], batch_size=None))

    payload = json.loads((tmp_path / "e3_error.json").read_text(encoding="utf-8"))
    assert payload["error"] == "non_finite_training_state"
    assert payload["non_finite"] == ["loss"]
    assert payload["global_step"] == 1
    assert payload["task_id"] == "pose"


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------

def test_checkpoint_contract_requires_scheduler_and_rng(tmp_path):
    e3 = _load_script("e3_oneclick")
    checkpoint = tmp_path / "last.ckpt"
    torch.save({
        "state_dict": {"student.w": torch.zeros(1), "teacher.w": torch.zeros(1)},
        "global_step": 5, "optimizer_states": [{}], "lr_schedulers": [{}], "rng_state": {},
    }, checkpoint)
    described = e3.describe_checkpoint(checkpoint, 5)
    assert described["student_tensors"] == 1 and described["teacher_tensors"] == 1
    assert described["lr_schedulers"] == 1 and described["rng_state"] is True

    for missing in ("lr_schedulers", "rng_state", "optimizer_states"):
        payload = {
            "state_dict": {"student.w": torch.zeros(1), "teacher.w": torch.zeros(1)},
            "global_step": 5, "optimizer_states": [{}], "lr_schedulers": [{}], "rng_state": {},
        }
        payload.pop(missing)
        torch.save(payload, checkpoint)
        with pytest.raises(e3.E3Error, match="lacks"):
            e3.describe_checkpoint(checkpoint, 5)


# ---------------------------------------------------------------------------
# Server guards
# ---------------------------------------------------------------------------

def test_server_mode_refuses_output_root_inside_repository(e3, tmp_path):
    data_root = tmp_path / "canonical"
    data_root.mkdir()
    # A parsable official corpus, so the run reaches the output-root guard
    # instead of stopping at the missing-corpus error first.
    (data_root / "nature_corpus.csv").write_text(
        "dataset,split,start,end\ngni_wrist_user_1.h5,train,0.0,1.0\n", encoding="utf-8"
    )
    with pytest.raises(e3.E3Error, match="outside the Git repository"):
        e3.main(["server", "--data-root", str(data_root), "--output", str(REPO)])


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

NON_NATURE = [
    ("emg2pose", "pose", "train"),
    ("emg2qwerty", "typing", "val"),
]
NATURE = [
    ("gni_discrete_gestures", "discrete_gesture"),
    ("gni_handwriting", "handwriting"),
    ("gni_wrist", "wrist_angle"),
]
MANIFEST_COLUMNS = [
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side", "pair_id",
    "official_split", "sample_count", "duration_s", "sample_rate_hz", "source_file", "source_sha256",
]


def _canonical_root(tmp_path) -> Path:
    root = tmp_path / "canonical"
    rows = []
    for dataset_id, task_id, official in NON_NATURE:
        relative = f"{dataset_id}/{task_id}/u/{dataset_id}_{task_id}.h5"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
        rows.append({
            "path": relative, "dataset_id": dataset_id, "task_id": task_id, "subject_id": "u",
            "session_id": "s", "side": "right", "pair_id": "", "official_split": official,
            "sample_count": "8000", "duration_s": "4.0", "sample_rate_hz": "2000",
            "source_file": relative, "source_sha256": "0" * 64,
        })
    for dataset_id, task_id in NATURE:
        relative = f"{dataset_id}/{task_id}/u/{dataset_id}.h5"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
        rows.append({
            "path": relative, "dataset_id": dataset_id, "task_id": task_id, "subject_id": "u",
            "session_id": "s", "side": "right", "pair_id": "", "official_split": "unknown",
            "sample_count": "8000", "duration_s": "4.0", "sample_rate_hz": "2000",
            "source_file": relative, "source_sha256": "1" * 64,
        })
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_build_splits_assigns_non_nature_and_leaves_nature_to_the_corpus(prepare_data, tmp_path):
    root = _canonical_root(tmp_path)
    splits = prepare_data.build_splits(root, seed=0)
    rows = list(csv.DictReader(splits.open(encoding="utf-8", newline="")))

    assert len(rows) == 5
    by_dataset = {row["dataset_id"]: row for row in rows}
    # Official release splits survive untouched.
    assert by_dataset["emg2pose"]["split"] == "train"
    assert by_dataset["emg2qwerty"]["split"] == "val"
    # Nature rows carry no whole-file split: the corpus intervals decide.
    for dataset_id, _ in NATURE:
        assert by_dataset[dataset_id]["split"] == ""


def test_prepare_data_requires_nature_corpus_and_never_invents_it(prepare_data, tmp_path, monkeypatch):
    root = _canonical_root(tmp_path)
    monkeypatch.setenv("E3_DATA_ROOT", str(root))
    monkeypatch.delenv("E3_NATURE_CORPUS", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        prepare_data.main(["--canonical-root", str(root), "--skip-conversion"])
    message = str(excinfo.value)
    assert "nature_corpus.csv" in message
    assert "will not create it" in message
    assert not (root / "nature_corpus.csv").exists()
    assert not (root / "splits.csv").exists()


def test_git_provenance_survives_a_working_directory_outside_the_repository(tmp_path, monkeypatch):
    """Provenance must not depend on the caller's cwd.

    `hydra.job.chdir` moves the process into the run directory, and a server run
    directory lives outside the checkout by design. Resolving `git rev-parse`
    from there used to return None, so `run_metadata.json` lost the commit SHA
    exactly on the runs that matter most.
    """
    from semg_fm import reproducibility

    monkeypatch.chdir(tmp_path)
    commit = reproducibility._git_commit()
    assert commit is not None, "git provenance degraded to null outside the repository"
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
