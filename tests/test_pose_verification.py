import torch

from semg_fm.models import RingCrissCrossEncoder
from semg_fm.lightning.downstream_module import DownstreamModule


def test_pose_head_uses_twenty_dof_and_masked_loss():
    module = DownstreamModule(
        RingCrissCrossEncoder(d_model=8, depth=1, heads=2, patch_samples=40),
        task_id="pose",
        output_dim=20,
        finetune_mode="probe",
    )
    emg = torch.randn(2, 400, 16)
    target = torch.randn(2, 400, 20)
    mask = torch.ones(2, 400, dtype=torch.bool)
    mask[:, 50:75] = False
    batch = {
        "emg": emg,
        "sample_mask": torch.ones(2, 400, dtype=torch.bool),
        "task_id": "pose",
        "example_weight": torch.ones(2),
        "metadata": [
            {"target": target[0], "target_mask": mask[0]},
            {"target": target[1], "target_mask": mask[1]},
        ],
    }
    loss, metrics, encoded = module._loss_and_metrics(batch)
    assert encoded["features"].shape == (2, 10, 8)
    assert module.head(encoded["features"]).shape == (2, 10, 20)
    assert torch.isfinite(loss)
    assert metrics["count"] == mask.sum() * 20
    target[:, 50:75] += 1000
    changed_loss, _, _ = module._loss_and_metrics(batch)
    torch.testing.assert_close(loss, changed_loss, rtol=0, atol=0)
    assert all(not p.requires_grad for p in module.backbone.parameters())
    assert any(p.requires_grad for p in module.head.parameters())


def test_pose_finetune_modes_select_expected_parameters():
    for mode in ("probe", "partial", "full"):
        module = DownstreamModule(
            RingCrissCrossEncoder(d_model=8, depth=2, heads=2, patch_samples=40),
            task_id="pose",
            output_dim=20,
            finetune_mode=mode,
            unfrozen_blocks=1,
        )
        backbone_trainable = [p for p in module.backbone.parameters() if p.requires_grad]
        if mode == "probe":
            assert not backbone_trainable
        elif mode == "partial":
            assert backbone_trainable
            assert all(p.requires_grad for p in module.backbone.blocks[-1].parameters())
            assert all(not p.requires_grad for p in module.backbone.blocks[0].parameters())
        else:
            assert backbone_trainable


def test_synthetic_pose_has_disjoint_official_splits_and_native_targets(tmp_path):
    from semg_fm.pose_verification import make_synthetic_pose
    from semg_fm.data.window_dataset import CanonicalWindowDataset
    manifest = make_synthetic_pose(tmp_path)
    subjects = []
    for split in ("train", "val", "test"):
        ds = CanonicalWindowDataset.from_manifest(manifest, split, 200, 100, root=tmp_path, task_id="pose")
        assert len(ds) > 1
        assert ds[0]["target"].shape == (400, 20)
        subjects.append(ds[0]["subject_id"])
    assert len(set(subjects)) == 3
