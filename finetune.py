"""Hydra entry point for task-specific fine-tuning of canonical sEMG models."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from semg_fm.config import build_trainer_options, validate_devices, validate_seed
from semg_fm.data.datamodule import build_datamodule
from semg_fm.downstream_config import downstream_module_from_config
from semg_fm.reproducibility import manifest_digest, write_run_metadata

# Backward-compatible alias used by entry tests.
_module_from_config = downstream_module_from_config


@hydra.main(version_base="1.3", config_path="configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    """Build a backbone/head pair, optionally initialize it, then fine-tune."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(cfg.data, task_id=str(cfg.task.name), label_budget=float(cfg.get("label_budget", 1.0)))
    module = _module_from_config(cfg)
    if cfg.pretrained_checkpoint:
        module.load_backbone_checkpoint(str(cfg.pretrained_checkpoint))
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)},
            indent=2,
        ),
        encoding="utf-8",
    )
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_config, dict)
    write_run_metadata(output, manifest_path=datamodule.manifest_path, data_config=data_config, trainer_config=trainer_options, command="finetune")
    Trainer(**build_trainer_options(trainer_options)).fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)


if __name__ == "__main__":
    main()
