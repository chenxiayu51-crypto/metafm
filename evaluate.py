"""Hydra entry point for held-out canonical sEMG task evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from semg_fm.config import build_trainer_options, validate_devices, validate_seed
from semg_fm.data.datamodule import build_datamodule
from semg_fm.downstream_config import downstream_module_from_config
from semg_fm.evaluation import write_evaluation_outputs
from semg_fm.reproducibility import environment_info, manifest_digest, write_run_metadata

# Backward-compatible alias used by entry tests.
_module_from_config = downstream_module_from_config


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    """Run test split evaluation and save configuration/metric provenance."""
    validate_devices(cfg.trainer.devices)
    validate_seed(cfg.seed)
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be provided")
    seed_everything(cfg.seed, workers=True)
    datamodule = build_datamodule(cfg.data, task_id=str(cfg.task.name))
    module = _module_from_config(cfg)
    # weights_only=False: Lightning checkpoints embed non-tensor state (hparams,
    # loop state) beside state_dict. Trust boundary: only load checkpoints
    # produced by this repository's own training entry points.
    checkpoint = torch.load(str(cfg.checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a state_dict mapping")
    module.load_state_dict(state, strict=True)
    output = Path(cfg.output_dir) if cfg.output_dir else Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_options, dict)
    data_config = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data_config, dict)
    write_run_metadata(output, manifest_path=datamodule.manifest_path, data_config=data_config, trainer_config=trainer_options, command="evaluate")
    trainer = Trainer(**build_trainer_options(trainer_options))
    results = trainer.test(module, datamodule=datamodule)
    metrics = results[0] if results else {}
    if not metrics:
        raise RuntimeError(
            "the test split produced no metrics, so metrics.json would be written empty. "
            "Check that the manifest has test records for this task."
        )
    write_evaluation_outputs(
        output,
        metrics=metrics,
        per_subject=module.test_subject_metrics,
        split_manifest_path=datamodule.manifest_path,
        environment=environment_info(),
        predictions=module.test_predictions if cfg.get("save_predictions", False) else None,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {"path": str(datamodule.manifest_path.resolve()), "sha256": manifest_digest(datamodule.manifest_path)},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
