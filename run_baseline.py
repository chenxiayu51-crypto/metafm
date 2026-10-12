"""Run declared comparison baselines through the same data and Trainer stack."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from semg_fm.config import build_trainer_options, validate_devices, validate_seed
from semg_fm.data.datamodule import build_datamodule
from semg_fm.downstream_config import downstream_module_from_config
from semg_fm.lightning.baseline_module import BaselinePretrainModule
from semg_fm.reproducibility import manifest_digest, write_run_metadata


def build_baseline_module(cfg: DictConfig):
    """Dispatch only the baseline stage/model combinations defined by protocol."""
    if cfg.stage == "finetune":
        return downstream_module_from_config(cfg)
    if cfg.stage == "pretrain":
        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        assert isinstance(model_config, dict)
        name = str(model_config.pop("name"))
        return BaselinePretrainModule(
            name,
            learning_rate=float(cfg.optimizer.learning_rate),
            weight_decay=float(cfg.optimizer.weight_decay),
            **model_config,
        )
    raise ValueError("stage must be 'pretrain' or 'finetune'")


@hydra.main(version_base="1.3", config_path="configs", config_name="baseline")
def main(cfg: DictConfig) -> None:
    """Train one baseline stage and snapshot its data/config provenance."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(
        cfg.data,
        task_id=str(cfg.task.name) if cfg.stage == "finetune" else None,
        label_budget=float(cfg.get("label_budget", 1.0)) if cfg.stage == "finetune" else 1.0,
    )
    module = build_baseline_module(cfg)
    if cfg.stage == "finetune" and cfg.pretrained_checkpoint:
        module.load_backbone_checkpoint(str(cfg.pretrained_checkpoint))
    output = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    (output / "manifest.json").write_text(
        json.dumps({"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)}, indent=2),
        encoding="utf-8",
    )
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_config, dict)
    write_run_metadata(output, manifest_path=datamodule.manifest_path, data_config=data_config, trainer_config=trainer_options, command=f"baseline:{cfg.stage}")
    Trainer(**build_trainer_options(trainer_options)).fit(module, datamodule=datamodule, ckpt_path=cfg.resume_from)


if __name__ == "__main__":
    main()
