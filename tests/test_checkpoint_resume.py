"""Checkpoint state contracts for the EMA teacher and training schedule."""

from __future__ import annotations

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from semg_fm.lightning.pretrain_module import PretrainModule


class _SyntheticDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "emg": torch.full((80, 16), float(index + 1)),
            "sample_mask": torch.ones(80, dtype=torch.bool),
            "task_id": "synthetic",
        }


def _collate(examples):
    return {
        "emg": torch.stack([example["emg"] for example in examples]),
        "sample_mask": torch.stack([example["sample_mask"] for example in examples]),
        "task_id": examples[0]["task_id"],
    }


def _trainer(**overrides):
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        callbacks=[],
        # 测试手动 save_checkpoint/ckpt_path；默认回调会往仓库根写 checkpoints/。
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        **overrides,
    )


def _module():
    return PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        warmup_steps=0,
        max_steps=2,
    )


def test_checkpoint_contains_teacher_and_scheduler_state(tmp_path):
    module = _module()
    trainer = _trainer(max_steps=1)
    trainer.fit(module, train_dataloaders=DataLoader(_SyntheticDataset(), batch_size=2, collate_fn=_collate))
    path = tmp_path / "one_step.ckpt"
    trainer.save_checkpoint(path)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]

    assert any(name.startswith("teacher.") for name in state)
    assert checkpoint.get("lr_schedulers"), "scheduler state must be stored in the checkpoint"
    assert checkpoint.get("optimizer_states"), "optimizer state must be stored in the checkpoint"

    restored = _module()
    restored.load_state_dict(state)
    teacher_names = [name for name in state if name.startswith("teacher.")]
    for name in teacher_names:
        torch.testing.assert_close(restored.state_dict()[name], state[name])
    assert restored.ema_updates == 1


def test_fit_resume_keeps_global_step_and_ema_counter(tmp_path):
    first = _module()
    first_trainer = _trainer(max_steps=1)
    first_trainer.fit(first, train_dataloaders=DataLoader(_SyntheticDataset(), batch_size=2, collate_fn=_collate))
    path = tmp_path / "one_step.ckpt"
    first_trainer.save_checkpoint(path)

    resumed = _module()
    resumed_trainer = _trainer(max_steps=2)
    resumed_trainer.fit(
        resumed,
        train_dataloaders=DataLoader(_SyntheticDataset(), batch_size=2, collate_fn=_collate),
        ckpt_path=str(path),
    )

    # A fresh run would stop at step 1 of 2; a correct resume continues from
    # the stored global step and never resets the EMA update counter.
    assert resumed_trainer.global_step == 2
    assert resumed.ema_updates == 2
