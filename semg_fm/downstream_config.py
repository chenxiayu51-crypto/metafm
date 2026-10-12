"""Shared downstream module construction for finetune/evaluate/baseline entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from semg_fm.lightning.downstream_module import DownstreamModule
from semg_fm.registry import model_from_config


def downstream_module_from_config(cfg: Mapping[str, Any]) -> DownstreamModule:
    """Construct a task head while keeping evaluation-only settings explicit.

    Accepts the composed config of ``finetune``, ``evaluate``, or ``baseline``
    (an OmegaConf ``DictConfig``): the task group supplies head parameters,
    ``robustness``/``save_predictions`` are optional evaluation-only groups,
    and unknown non-null task keys are rejected.
    """
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    task_config = OmegaConf.to_container(cfg.task, resolve=True)
    robustness = OmegaConf.to_container(cfg.robustness, resolve=True) if "robustness" in cfg else None
    assert isinstance(model_config, dict) and isinstance(task_config, dict)
    assert robustness is None or isinstance(robustness, dict)
    module = DownstreamModule(
        backbone=model_from_config(model_config),
        task_id=str(task_config.get("name")),
        output_dim=task_config.get("output_dim"),
        target_name=task_config.get("target_name"),
        class_names=task_config.get("class_names", ()),
        vocabulary=task_config.get("vocabulary", ()),
        event_tolerance_s=float(task_config.get("event_tolerance_s", 0.05)),
        finetune_mode=str(cfg.get("finetune_mode", "full")),
        unfrozen_blocks=int(cfg.get("unfrozen_blocks", 1)),
        robustness=robustness,
        collect_predictions=bool(cfg.get("save_predictions", False)),
        learning_rate=float(cfg.optimizer.learning_rate) if "optimizer" in cfg else 1e-3,
        weight_decay=float(cfg.optimizer.weight_decay) if "optimizer" in cfg else 0.05,
    )
    leftovers = dict(task_config)
    for key in ("name", "output_dim", "target_name", "class_names", "vocabulary", "event_tolerance_s"):
        leftovers.pop(key, None)
    unsupported = sorted(key for key, value in leftovers.items() if value is not None)
    if unsupported:
        raise ValueError(f"unsupported downstream task settings: {unsupported}")
    return module


__all__ = ["downstream_module_from_config"]
