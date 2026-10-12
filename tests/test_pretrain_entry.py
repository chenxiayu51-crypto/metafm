"""Configuration validation contracts for the public pretraining entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from pretrain import build_datamodule, build_pretrain_module


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_pretrain_requires_explicit_canonical_data_root():
    with pytest.raises(ValueError, match="data.root must be provided"):
        build_datamodule(OmegaConf.create({"root": None}))


def test_pretrain_accepts_documented_data_root_override(tmp_path):
    module = build_datamodule(OmegaConf.create({"root": str(tmp_path)}))
    assert module.root == tmp_path
    assert module.manifest_path == tmp_path / "manifest.csv"


def test_pretrain_module_uses_resolved_training_mappings():
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "ring_criss_cross",
                "d_model": 8,
                "depth": 1,
                "heads": 2,
                "patch_samples": 40,
                "channels": 16,
                "dropout": 0.0,
            },
            "objective": {
                "latent_weight": 2.0,
                "kinematic_weight": 0.5,
                "kinematic_dim": None,
                "ema_decay": 0.9,
            },
            "optimizer": {
                "learning_rate": 0.003,
                "weight_decay": 0.01,
                "warmup_steps": 4,
                "max_steps": 12,
            },
            "mask": {
                "temporal_span": 1,
                "ring_arc": 2,
                "full_channel_probability": 0.0,
                "block_shape": [1, 1],
                "random_patch_probability": 0.0,
            },
        }
    )

    module = build_pretrain_module(cfg)

    assert module.latent_weight == 2.0
    assert module.kinematic_weight == 0.5
    assert module.kinematic_adapter is None
    assert module.ema_decay == 0.9
    assert module.learning_rate == 0.003
    assert module.weight_decay == 0.01
    assert module.warmup_steps == 4
    assert module.max_steps == 12
    assert module.mask_generator.temporal_span == 1
    assert module.mask_generator.ring_arc == 2
    assert module.mask_generator.full_channel_probability == 0.0
    assert module.mask_generator.block_shape == (1, 1)
    assert module.mask_generator.random_patch_probability == 0.0


def test_pretrain_module_propagates_composed_default_training_mappings():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain")

    module = build_pretrain_module(cfg)

    assert module.latent_weight == 1.0
    assert module.kinematic_weight == 1.0
    assert module.kinematic_adapter is None
    assert module.ema_decay == 0.996
    assert module.learning_rate == 0.001
    assert module.weight_decay == 0.05
    assert module.warmup_steps == 1000
    assert module.max_steps == 100000
    assert module.mask_generator.temporal_span == 2
    assert module.mask_generator.ring_arc == 3
    assert module.mask_generator.full_channel_probability == 0.1
    assert module.mask_generator.block_shape == (2, 2)
    assert module.mask_generator.random_patch_probability == 0.15
