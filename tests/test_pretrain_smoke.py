"""CPU smoke test for the public Lightning pretraining module."""

from __future__ import annotations

import torch
from lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from semg_fm.lightning.pretrain_module import PretrainModule


class _SyntheticDataset(Dataset):
    def __len__(self) -> int:
        return 2

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


def test_circe_pretraining_fast_dev_run_advances_one_global_step():
    module = PretrainModule(
        model_config={"d_model": 8, "depth": 1, "heads": 2, "patch_samples": 40},
        warmup_steps=0,
        max_steps=1,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(module, train_dataloaders=DataLoader(_SyntheticDataset(), batch_size=2, collate_fn=_collate))

    assert trainer.global_step == 1
