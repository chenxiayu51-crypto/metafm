"""Behavioral contracts for Lightning pretraining and EMA persistence."""

from __future__ import annotations

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from semg_fm.data.datamodule import task_homogeneous_collate
import semg_fm.lightning.pretrain_module as pretrain_module
from semg_fm.models import RingCrissCrossEncoder
from semg_fm.lightning.pretrain_module import PretrainModule


class _TinyPretrainDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        generator = torch.Generator().manual_seed(index)
        return {
            "emg": torch.randn(80, 16, generator=generator),
            "sample_mask": torch.ones(80, dtype=torch.bool),
            "task_id": "pose",
        }


def _collate(examples):
    return {
        "emg": torch.stack([example["emg"] for example in examples]),
        "sample_mask": torch.stack([example["sample_mask"] for example in examples]),
        "task_id": examples[0]["task_id"],
    }


def _module() -> PretrainModule:
    return PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        ema_decay=0.5,
        learning_rate=1e-2,
        warmup_steps=0,
        max_steps=1,
    )


def _trainer() -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def _validation_trainer() -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def test_teacher_starts_as_frozen_copy_of_student():
    module = _module()
    for student, teacher in zip(module.student.parameters(), module.teacher.parameters()):
        assert torch.equal(student, teacher)
        assert not teacher.requires_grad


def test_ring_equivariance_is_a_detached_diagnostic_not_an_optimized_loss(monkeypatch):
    module = _module()
    module.latent_weight = 2.0
    module.ring_weight = 3.0
    module.kinematic_weight = 5.0
    monkeypatch.setattr(
        pretrain_module,
        "ring_equivariance_loss",
        lambda field, motion, shifted_field, shifted_motion, shift: field.new_tensor(7.0),
    )
    batch = _collate([_TinyPretrainDataset()[0], _TinyPretrainDataset()[1]])

    total, losses = module._losses(batch)

    assert "ring_equivariance_error" in losses
    assert not losses["ring_equivariance_error"].requires_grad
    assert torch.equal(total, module.latent_weight * losses["latent"] + module.kinematic_weight * losses["kinematic"])


def test_pretraining_uses_tensor_pose_targets_from_standard_collator():
    torch.manual_seed(7)
    raw_items = [
        {
            "emg": torch.randn(80, 16),
            "sample_mask": torch.ones(80, dtype=torch.bool),
            "task_id": "pose",
            "target": torch.full((80, 2), float(index + 1)),
            "target_mask": torch.ones(80, dtype=torch.bool),
        }
        for index in range(2)
    ]
    batch = task_homogeneous_collate(raw_items)
    module = PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        kinematic_dim=2,
        latent_weight=1.5,
        kinematic_weight=2.0,
        warmup_steps=0,
        max_steps=1,
    )

    total, losses = module._losses(batch)

    assert torch.isfinite(losses["kinematic"])
    assert losses["kinematic"].item() > 0
    torch.testing.assert_close(total, module.latent_weight * losses["latent"] + module.kinematic_weight * losses["kinematic"])


def test_pretraining_excludes_zero_weight_duplicate_from_losses():
    torch.manual_seed(19)
    valid = {
        "emg": torch.randn(80, 16),
        "sample_mask": torch.ones(80, dtype=torch.bool),
        "task_id": "pose",
        "target": torch.zeros(80, 2),
        "target_mask": torch.ones(80, dtype=torch.bool),
    }
    duplicate = {
        "emg": torch.full((80, 16), 100.0),
        "sample_mask": torch.ones(80, dtype=torch.bool),
        "task_id": "pose",
        "target": torch.full((80, 2), 100.0),
        "target_mask": torch.ones(80, dtype=torch.bool),
        "is_padding": True,
    }
    reference = task_homogeneous_collate([valid])
    padded = task_homogeneous_collate([valid, duplicate])
    module = PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        kinematic_dim=2,
        latent_weight=1.5,
        kinematic_weight=2.0,
        warmup_steps=0,
        max_steps=1,
    )

    torch.manual_seed(31)
    reference_total, reference_losses = module._losses(reference)
    torch.manual_seed(31)
    padded_total, padded_losses = module._losses(padded)

    torch.testing.assert_close(padded_total, reference_total)
    for name in reference_losses:
        torch.testing.assert_close(padded_losses[name], reference_losses[name])


def test_pretraining_filters_zero_weight_examples_before_model_forwards(monkeypatch):
    valid = _TinyPretrainDataset()[0]
    excluded = dict(_TinyPretrainDataset()[1])
    excluded["is_padding"] = True
    batch = task_homogeneous_collate([valid, excluded])
    module = _module()
    student_forward, teacher_forward = module.student.forward, module.teacher.forward
    student_batch_sizes: list[int] = []
    teacher_batch_sizes: list[int] = []

    def record_student(emg, *args, **kwargs):
        student_batch_sizes.append(int(emg.shape[0]))
        return student_forward(emg, *args, **kwargs)

    def record_teacher(emg, *args, **kwargs):
        teacher_batch_sizes.append(int(emg.shape[0]))
        return teacher_forward(emg, *args, **kwargs)

    monkeypatch.setattr(module.student, "forward", record_student)
    monkeypatch.setattr(module.teacher, "forward", record_teacher)

    module._losses(batch)

    assert student_batch_sizes == [1, 1]
    assert teacher_batch_sizes == [1]


def test_pretraining_all_padding_returns_differentiable_zero_without_masking():
    item = {
        "emg": torch.randn(80, 16),
        "sample_mask": torch.ones(80, dtype=torch.bool),
        "task_id": "pose",
        "is_padding": True,
    }
    batch = task_homogeneous_collate([item, item])
    module = _module()

    total, losses = module._losses(batch)

    assert module._effective_batch_size(batch) == 0
    assert total.requires_grad
    assert torch.isfinite(total)
    assert total.item() == 0
    assert all(torch.isfinite(value) for value in losses.values())
    assert not losses["ring_equivariance_error"].requires_grad
    total.backward()


def test_validation_all_padding_logs_finite_zero_metrics():
    item = {
        "emg": torch.randn(80, 16),
        "sample_mask": torch.ones(80, dtype=torch.bool),
        "task_id": "pose",
        "is_padding": True,
    }
    loader = DataLoader([item, item], batch_size=2, collate_fn=task_homogeneous_collate)

    metrics = _validation_trainer().validate(_module(), dataloaders=loader, verbose=False)[0]

    assert metrics
    assert all(torch.isfinite(torch.as_tensor(value)) and abs(float(value)) < 1e-6 for value in metrics.values())


def test_validation_epoch_loss_ignores_all_padding_batch_weight():
    valid = _TinyPretrainDataset()[0]
    padding = dict(_TinyPretrainDataset()[1])
    padding["is_padding"] = True
    reference_loader = DataLoader([valid], batch_size=1, collate_fn=task_homogeneous_collate)
    mixed_loader = DataLoader([valid, padding], batch_size=1, collate_fn=task_homogeneous_collate)
    module = _module()

    torch.manual_seed(41)
    reference_metrics = _validation_trainer().validate(module, dataloaders=reference_loader, verbose=False)[0]
    torch.manual_seed(41)
    mixed_metrics = _validation_trainer().validate(module, dataloaders=mixed_loader, verbose=False)[0]

    torch.testing.assert_close(mixed_metrics["val/loss"], reference_metrics["val/loss"])


def test_pretraining_steps_log_effective_example_count(monkeypatch):
    module = _module()
    scalar = torch.zeros((), requires_grad=True)
    losses = {"latent": scalar, "ring_equivariance_error": scalar.detach(), "kinematic": scalar}
    logged_batch_sizes: list[tuple[str, int]] = []
    batch = {
        "emg": torch.zeros(2, 80, 16),
        "example_weight": torch.tensor([1.0, 0.0]),
        "task_id": "pose",
    }
    monkeypatch.setattr(module, "_losses", lambda _: (scalar, losses))
    monkeypatch.setattr(module, "_log", lambda stage, total, values, batch_size: logged_batch_sizes.append((stage, batch_size)))
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)

    module.training_step(batch, 0)
    module.validation_step(batch, 0)

    assert logged_batch_sizes == [("train", 1), ("val", 1)]


def test_log_records_ring_equivariance_as_a_metric_without_loss_suffix(monkeypatch):
    module = _module()
    names: list[str] = []
    monkeypatch.setattr(module, "log", lambda name, value, **kwargs: names.append(name))

    module._log(
        "train",
        torch.tensor(1.0),
        {"latent": torch.tensor(2.0), "ring_equivariance_error": torch.tensor(3.0), "kinematic": torch.tensor(4.0)},
        batch_size=2,
    )

    assert "train/ring_equivariance_error" in names
    assert "train/ring_equivariance_error_loss" not in names


def test_ema_updates_only_after_optimizer_step_and_teacher_has_no_grads():
    module = _module()
    before = {key: value.clone() for key, value in module.teacher.state_dict().items()}
    loader = DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate)

    _trainer().fit(module, train_dataloaders=loader)

    assert module.ema_updates == 1
    for key, value in module.teacher.state_dict().items():
        expected = before[key] * 0.5 + module.student.state_dict()[key] * 0.5
        assert torch.equal(value, expected)
    assert all(parameter.grad is None for parameter in module.teacher.parameters())


def test_ema_teacher_is_saved_and_restored_exactly(tmp_path):
    module = _module()
    trainer = _trainer()
    trainer.fit(module, train_dataloaders=DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate))
    before = {key: value.clone() for key, value in module.teacher.state_dict().items()}
    path = tmp_path / "state.ckpt"
    trainer.save_checkpoint(path)

    restored = PretrainModule.load_from_checkpoint(path)

    assert restored.ema_updates == 1
    for key, value in before.items():
        assert torch.equal(value, restored.teacher.state_dict()[key])


def test_direct_student_round_trip_serializes_complete_architecture(tmp_path):
    student = RingCrissCrossEncoder(d_model=8, depth=1, heads=2, patch_samples=40)
    module = PretrainModule(student=student, model_config={"d_model": 8})
    trainer = _trainer()
    trainer.fit(module, train_dataloaders=DataLoader(_TinyPretrainDataset(), batch_size=2, collate_fn=_collate))
    path = tmp_path / "direct-student.ckpt"
    trainer.save_checkpoint(path)

    restored = PretrainModule.load_from_checkpoint(path)

    assert restored.model_config == {
        "d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40,
        "channels": 16, "dropout": 0.0,
    }
    assert restored.student.patch_samples == 40


# The masked-latent objective once degenerated into a parameter-independent
# constant because masked tokens were also marked unobserved, which zeroed the
# student output the loss reads.  These tests fail if that wiring ever returns.


def _gradient_module() -> PretrainModule:
    """A module whose only parameter movement can come from the objective."""
    return PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        ema_decay=0.5,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_steps=0,
        max_steps=1,
    )


def _gradient_batch() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {
        "emg": torch.randn(4, 80, 16, generator=generator),
        "sample_mask": torch.ones(4, 80, dtype=torch.bool),
        "task_id": "pose",
    }


def test_masked_latent_objective_is_not_a_parameter_independent_constant():
    module = _gradient_module()
    _, losses = module._losses(_gradient_batch())

    latent = losses["latent"]
    degenerate = 1.0 + 1.0 / (2.0 * module.student.d_model)

    assert torch.isfinite(latent)
    assert abs(float(latent) - degenerate) > 1e-6


def test_masked_latent_objective_produces_nonzero_student_gradients():
    module = _gradient_module()
    total, _ = module._losses(_gradient_batch())

    module.zero_grad(set_to_none=True)
    total.backward()

    gradients = [p.grad for p in module.student.parameters() if p.grad is not None]
    assert gradients, "objective must reach the student parameters"
    assert sum(float(g.abs().sum()) for g in gradients) > 0.0


def test_mask_token_is_learned_and_receives_gradient():
    module = _gradient_module()
    total, _ = module._losses(_gradient_batch())

    module.zero_grad(set_to_none=True)
    total.backward()

    assert module.student.mask_token.grad is not None
    assert float(module.student.mask_token.grad.abs().sum()) > 0.0


def test_masked_tokens_keep_a_nonzero_output_inside_the_encoder():
    """A masked token must stay observed, so the loss can read a live activation."""
    module = _gradient_module()
    module.eval()
    hidden = torch.zeros(4, 16, 2, dtype=torch.bool)
    hidden[:, 0, 0] = True

    with torch.no_grad():
        outputs = module.student(
            _gradient_batch()["emg"],
            observation_mask=torch.ones(4, 80, 16, dtype=torch.bool),
            patch_mask=hidden,
        )

    assert float(outputs["z_eq"][hidden].abs().sum()) > 0.0


def test_one_optimizer_step_moves_student_parameters_without_weight_decay():
    module = _gradient_module()
    module.train()
    batch = _gradient_batch()
    optimizer = torch.optim.AdamW(module.student.parameters(), lr=1e-2, weight_decay=0.0)
    before = {name: value.detach().clone() for name, value in module.student.state_dict().items()}

    optimizer.zero_grad(set_to_none=True)
    total, _ = module._losses(batch)
    total.backward()
    optimizer.step()

    changed = [
        name
        for name, value in module.student.state_dict().items()
        if not torch.equal(value.detach(), before[name])
    ]
    assert changed, "a single optimizer step must change student parameters"
