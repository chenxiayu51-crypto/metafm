"""Fine-tuning modes, label budgets, and robustness transforms for downstream tasks."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from semg_fm.data.augment import circular_shift
from semg_fm.lightning.downstream_module import DownstreamModule
from semg_fm.robustness import apply_robustness


class _TinyBackbone(nn.Module):
    """Small deterministic backbone exposing the shared temporal interface."""

    d_model = 6
    patch_samples = 20
    channels = 16

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(3)])

    def forward(self, emg: torch.Tensor, observation_mask: torch.Tensor | None = None):
        batch, samples, _ = emg.shape
        tokens = emg.reshape(batch, samples // 20, 20, 16).mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        features = tokens.expand(-1, -1, self.d_model)
        for block in self.blocks:
            features = block(features)
        return {"z_motion": features}


def _trainable(module):
    return {name for name, parameter in module.named_parameters() if parameter.requires_grad}


def test_probe_mode_freezes_the_backbone_but_not_the_head():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2, finetune_mode="probe")

    assert not _trainable(module.backbone)
    assert _trainable(module.head) == {"projection.weight", "projection.bias"}


def test_partial_mode_unfreezes_only_the_last_blocks():
    module = DownstreamModule(
        backbone=_TinyBackbone(), task_id="pose", output_dim=2,
        finetune_mode="partial", unfrozen_blocks=1,
    )

    assert _trainable(module.backbone) == {"blocks.2.weight", "blocks.2.bias"}
    assert _trainable(module.head)


def test_full_mode_trains_everything():
    module = DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2, finetune_mode="full")

    assert len(_trainable(module.backbone)) == 6
    assert _trainable(module.head)


def test_invalid_finetune_mode_is_rejected():
    with pytest.raises(ValueError, match="finetune_mode"):
        DownstreamModule(backbone=_TinyBackbone(), task_id="pose", output_dim=2, finetune_mode="warm-start")


def test_partial_mode_validates_unfrozen_blocks():
    with pytest.raises(ValueError, match="unfrozen_blocks"):
        DownstreamModule(
            backbone=_TinyBackbone(), task_id="pose", output_dim=2,
            finetune_mode="partial", unfrozen_blocks=9,
        )


def test_label_budget_keeps_deterministic_subject_groups(canonical_meta_root):
    from semg_fm.data.window_dataset import CanonicalWindowDataset

    first = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    first.apply_label_budget(0.5, 7)
    paths = {path for path, _, _ in first._index}
    kept_subjects = {first._records[path]["subject_id"] for path in paths}
    # All windows of a kept record stay together and the budget shrinks data.
    full = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    assert 0 < len(first) < len(full)
    assert len(kept_subjects) >= 1
    # Every window of a kept subject record is present.
    for record_path, record in first._records.items():
        if record["subject_id"] not in kept_subjects:
            assert all(path != record_path for path, _, _ in first._index)
    second = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    second.apply_label_budget(0.5, 7)
    assert {path for path, _, _ in second._index} == paths  # deterministic


def test_label_budget_rejects_out_of_range_values(canonical_meta_root):
    from semg_fm.data.window_dataset import CanonicalWindowDataset

    dataset = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    with pytest.raises(ValueError, match="budget"):
        dataset.apply_label_budget(0.0, 0)
    with pytest.raises(ValueError, match="budget"):
        dataset.apply_label_budget(1.5, 0)


def test_apply_robustness_channel_rotation_rolls_the_ring():
    emg = torch.arange(32, dtype=torch.float32).reshape(2, 1, 16)
    mask = torch.ones(2, 1, dtype=torch.bool)

    rotated, out_mask, observation = apply_robustness(emg, mask, {"name": "channel_rotation", "shift": 3})

    torch.testing.assert_close(rotated, circular_shift(emg, 3))
    assert observation is None
    assert out_mask.equal(mask)


def test_apply_robustness_channel_dropout_zeroes_and_masks():
    emg = torch.ones(2, 40, 16)
    mask = torch.ones(2, 40, dtype=torch.bool)

    dropped, out_mask, observation = apply_robustness(
        emg, mask, {"name": "channel_dropout", "start": 2, "count": 4}
    )

    assert torch.all(dropped[..., 2:6] == 0)
    assert observation is not None and not observation[..., 2:6].any()
    assert observation[..., :2].all() and observation[..., 6:].all()
    assert out_mask.equal(mask)


def test_apply_robustness_temporal_loss_marks_samples_missing():
    emg = torch.ones(2, 40, 16)
    mask = torch.ones(2, 40, dtype=torch.bool)

    dropped, out_mask, observation = apply_robustness(emg, mask, {"name": "temporal_loss", "start": 5, "count": 10})

    assert torch.all(dropped[:, 5:15] == 0)
    assert not out_mask[:, 5:15].any()
    assert out_mask[:, :5].all() and out_mask[:, 15:].all()
    assert observation is None


def test_apply_robustness_gain_and_noise_stay_fully_observed():
    emg = torch.randn(2, 40, 16)
    mask = torch.ones(2, 40, dtype=torch.bool)

    scaled, out_mask, observation = apply_robustness(
        emg, mask, {"name": "gain_drift", "start_gain": 0.8, "end_gain": 1.2}
    )
    assert observation is None and out_mask.equal(mask)
    assert torch.isfinite(scaled).all()

    noisy, out_mask, observation = apply_robustness(emg, mask, {"name": "additive_noise", "std": 0.05, "seed": 0})
    assert observation is None and out_mask.equal(mask)
    assert torch.isfinite(noisy).all()


def test_apply_robustness_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="robustness"):
        apply_robustness(torch.randn(2, 40, 16), torch.ones(2, 40, dtype=torch.bool), {"name": "magic"})


def test_test_step_applies_configured_robustness_transform():
    module = DownstreamModule(
        backbone=_TinyBackbone(), task_id="pose", output_dim=2,
        robustness={"name": "channel_dropout", "start": 0, "count": 1},
    )
    module.on_test_epoch_start()
    batch = {
        "emg": torch.randn(2, 100, 16),
        "sample_mask": torch.ones(2, 100, dtype=torch.bool),
        "task_id": "pose",
        "example_weight": torch.ones(2),
        "metadata": [
            {"target": torch.zeros(100, 2), "target_mask": torch.ones(100, dtype=torch.bool)},
            {"target": torch.zeros(100, 2), "target_mask": torch.ones(100, dtype=torch.bool)},
        ],
    }

    loss = module.test_step(batch, 0)

    assert torch.isfinite(loss)
    # Metadata without subject_id falls back to the "unknown" subject row.
    rows = module.test_subject_metrics
    assert len(rows) == 1
    assert rows[0]["subject_id"] == "unknown"
    assert rows[0]["windows"] == 2
