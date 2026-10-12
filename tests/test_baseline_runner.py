"""Contracts for the public baseline pretraining/finetuning dispatcher."""

from __future__ import annotations

import pytest
import torch
from hydra import compose, initialize_config_dir
from pathlib import Path

from run_baseline import build_baseline_module
from semg_fm.lightning.baseline_module import BaselinePretrainModule


def test_bendr_baseline_pretraining_module_produces_a_finite_objective():
    module = BaselinePretrainModule("bendr", d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0)
    loss = module.training_step(
        {"emg": torch.randn(2, 100, 16), "sample_mask": torch.ones(2, 100, dtype=torch.bool)}, 0
    )

    assert torch.isfinite(loss)


def test_supervised_tcn_is_rejected_for_baseline_pretraining():
    with pytest.raises(ValueError, match="supervised_tcn"):
        BaselinePretrainModule("supervised_tcn")


def test_baseline_config_dispatches_finetuning_without_claiming_pretraining_support_for_tcn():
    config_dir = str((Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="baseline", overrides=["model=supervised_tcn", "stage=finetune"])

    module = build_baseline_module(cfg)

    assert module.task_id == "pose"
