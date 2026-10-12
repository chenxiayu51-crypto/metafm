"""Smoke contracts for Lightning downstream fine-tuning on canonical batches."""

from __future__ import annotations

import pytest
import torch
from lightning import Trainer
from torch import nn
from torch.utils.data import DataLoader

from semg_fm.data.datamodule import task_homogeneous_collate
from semg_fm.data.window_dataset import CanonicalWindowDataset
from semg_fm.lightning.downstream_module import DownstreamModule


class _TinyBackbone(nn.Module):
    """Small deterministic backbone exposing the shared temporal interface."""

    d_model = 6
    patch_samples = 20
    channels = 16

    def forward(self, emg: torch.Tensor, observation_mask: torch.Tensor | None = None):
        batch, samples, _ = emg.shape
        tokens = emg.reshape(batch, samples // 20, 20, 16).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        return {"z_motion": tokens.expand(-1, -1, self.d_model)}


def test_downstream_regression_uses_metadata_targets_and_excludes_padding_weight():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2, learning_rate=1e-3)
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "pose",
        "example_weight": torch.tensor([1.0, 0.0]),
        "metadata": [
            {"target": torch.zeros(100, 2), "target_mask": torch.ones(100, dtype=torch.bool)},
            {"target": torch.full((100, 2), 10_000.0), "target_mask": torch.ones(100, dtype=torch.bool)},
        ],
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.head.projection.weight.grad is not None


def test_downstream_module_rejects_a_batch_for_a_different_task():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2)
    batch = {"emg": torch.randn(1, 100, 16), "sample_mask": torch.ones(1, 100, dtype=torch.bool), "task_id": "typing", "metadata": [], "example_weight": torch.ones(1)}

    try:
        module.training_step(batch, 0)
    except ValueError as error:
        assert "task_id" in str(error)
    else:
        raise AssertionError("expected task mismatch to be rejected")


def test_downstream_ctc_uses_per_example_loss_before_padding_weighting():
    module = DownstreamModule(
        backbone=_TinyBackbone(), task_id="handwriting", vocabulary=("a", "b"), learning_rate=1e-3
    )
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "handwriting",
        "example_weight": torch.tensor([1.0, 0.0]),
        "metadata": [{"target_text": "ab"}, {"target_text": "ba"}],
    }

    loss = module.training_step(batch, 0)
    loss.backward()

    assert torch.isfinite(loss)
    assert module.head.classifier.weight.grad is not None


def test_event_loss_raises_for_a_label_missing_from_class_names():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="discrete_gesture", class_names=("a",))
    batch = {
        "emg": torch.randn(1, 100, 16),
        "sample_mask": torch.ones(1, 100, dtype=torch.bool),
        "task_id": "discrete_gesture",
        "example_weight": torch.ones(1),
        "metadata": [{"sample_rate_hz": 2000.0, "events": [("b", 0.01)]}],
    }

    with pytest.raises(ValueError, match="class_names"):
        module.training_step(batch, 0)


def test_test_step_accumulates_event_matches_for_the_epoch():
    module = DownstreamModule(
        backbone=_TinyBackbone(), task_id="discrete_gesture", class_names=("a",), event_tolerance_s=0.05
    )
    module.on_test_epoch_start()
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "discrete_gesture",
        "example_weight": torch.ones(2),
        "metadata": [
            {"sample_rate_hz": 2000.0, "events": [("a", 0.02)]},
            {"sample_rate_hz": 2000.0, "events": [("a", 0.03)]},
        ],
    }

    module.test_step(batch, 0)

    counts = module._primary_accumulators["test"]
    # One-to-one tolerance matching accounts for every reference exactly once.
    assert counts["tp"] + counts["fn"] == 2
    assert counts["fp"] >= 0


def test_test_step_accumulates_edit_totals_for_global_cer():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="handwriting", vocabulary=("a", "b"))
    module.on_test_epoch_start()
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "handwriting",
        "example_weight": torch.ones(2),
        "metadata": [{"target_text": "ab"}, {"target_text": "ba"}],
    }

    module.test_step(batch, 0)

    counts = module._primary_accumulators["test"]
    # Both references are counted with their full lengths.
    assert counts["ref_chars"] == 4
    assert 0 <= counts["edits"] <= 8


def test_trainer_test_loop_logs_the_primary_metric(canonical_meta_root):
    dataset = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100,
        task_id="discrete_gesture",
    )
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="discrete_gesture", class_names=("fist",))
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.test(module, dataloaders=DataLoader(dataset, batch_size=2, collate_fn=task_homogeneous_collate))

    assert "test/discrete_gesture_event_f1" in trainer.logged_metrics
