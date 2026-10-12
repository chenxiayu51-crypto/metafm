"""Hydra entry point for CIRCE-Myo pretraining."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from semg_fm.config import build_trainer_options, validate_devices, validate_seed
from semg_fm.data.datamodule import build_datamodule
from semg_fm.lightning import PretrainModule
from semg_fm.registry import model_from_config
from semg_fm.reproducibility import manifest_digest, write_run_metadata


def build_pretrain_module(cfg: DictConfig) -> PretrainModule:
    """Build pretraining with every objective, optimizer, and mask setting explicit."""
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    mask_config = OmegaConf.to_container(cfg.mask, resolve=True)
    objective_config = OmegaConf.to_container(cfg.objective, resolve=True)
    optimizer_config = OmegaConf.to_container(cfg.optimizer, resolve=True)
    assert isinstance(model_config, dict)
    assert isinstance(mask_config, dict)
    assert isinstance(objective_config, dict)
    assert isinstance(optimizer_config, dict)
    mask_config["block_shape"] = tuple(mask_config["block_shape"])
    return PretrainModule(
        student=model_from_config(model_config),
        model_name=str(model_config["name"]),
        model_config=model_config,
        mask_config=mask_config,
        latent_weight=objective_config["latent_weight"],
        kinematic_weight=objective_config["kinematic_weight"],
        kinematic_dim=objective_config["kinematic_dim"],
        ema_decay=objective_config["ema_decay"],
        learning_rate=optimizer_config["learning_rate"],
        weight_decay=optimizer_config["weight_decay"],
        warmup_steps=optimizer_config["warmup_steps"],
        max_steps=optimizer_config["max_steps"],
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="pretrain")
def main(cfg: DictConfig) -> None:
    validate_devices(cfg.trainer.devices)
    seed = cfg.get("seed", 0)
    validate_seed(seed)
    seed_everything(seed, workers=True)
    datamodule = build_datamodule(cfg.data)
    module = build_pretrain_module(cfg)
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(json.dumps({"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)}, indent=2), encoding="utf-8")
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_config, dict)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_config, dict)
    write_run_metadata(output, manifest_path=datamodule.manifest_path, data_config=data_config, trainer_config=trainer_config, command="pretrain")
    Trainer(**build_trainer_options(trainer_config)).fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))


if __name__ == "__main__":
    main()
