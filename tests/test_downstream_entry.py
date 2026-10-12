"""Hydra composition and reconstruction contracts for public downstream entry points."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from evaluate import _module_from_config
from finetune import _module_from_config as finetune_module_from_config


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_finetune_and_evaluate_compose_with_a_task_protocol():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        finetune = compose(config_name="finetune", overrides=["task=handwriting", "model=supervised_tcn"])
        evaluate = compose(config_name="evaluate", overrides=["task=discrete_gesture", "model=supervised_tcn"])

    assert finetune.task.name == "handwriting"
    assert evaluate.task.name == "discrete_gesture"


def test_evaluate_reconstructs_a_downstream_module_from_hydra_config():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="evaluate",
            overrides=["model=supervised_tcn", "task=pose", "task.output_dim=3"],
        )

    module = _module_from_config(cfg)

    assert module.task_id == "pose"
    assert module.head.output_dim == 3


def test_finetune_accepts_event_protocol_settings_without_treating_them_as_head_kwargs():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="finetune", overrides=["model=supervised_tcn", "task=discrete_gesture"])

    module = finetune_module_from_config(cfg)

    assert module.task_id == "discrete_gesture"
