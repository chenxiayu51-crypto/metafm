"""Lightning wrapper for core-mechanism baseline pretraining objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from lightning import LightningModule

from semg_fm.models.aemg import AEMGPretrainingModel
from semg_fm.models.bendr import BendrPretrainingModel
from semg_fm.models.cbramod import CBraModPretrainingModel


class BaselinePretrainModule(LightningModule):
    """Train one declared baseline objective without adding ring-model losses.

    This module purposefully supports only self-supervised comparison baselines.
    ``supervised_tcn`` has no self-supervised objective in this protocol and is
    therefore valid only for downstream supervised fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        *,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        wrappers = {
            "bendr": BendrPretrainingModel,
            "cbramod": CBraModPretrainingModel,
            "aemg": AEMGPretrainingModel,
        }
        if model_name not in wrappers:
            raise ValueError(
                f"{model_name!r} has no supported baseline pretraining objective; "
                "supervised_tcn is fine-tune-only"
            )
        self.model_name = model_name
        self.model = wrappers[model_name](**model_kwargs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters({"model_name": model_name, "learning_rate": learning_rate, "weight_decay": weight_decay, "model_kwargs": model_kwargs})

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Run the declared pretraining objective on a canonical EMG batch."""
        outputs = self._outputs(batch)
        loss = self._objective(outputs)
        if self._trainer is not None:
            self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=int(batch["emg"].shape[0]))
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Evaluate the same objective without mixing it with downstream losses."""
        loss = self._objective(self._outputs(batch))
        if self._trainer is not None:
            self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=int(batch["emg"].shape[0]))
        return loss

    def configure_optimizers(self):
        """Use AdamW for a fair default shared across comparison baselines."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _outputs(self, batch: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        emg, sample_mask = batch.get("emg"), batch.get("sample_mask")
        if not isinstance(emg, torch.Tensor) or not isinstance(sample_mask, torch.Tensor):
            raise ValueError("baseline batch requires tensor emg and sample_mask")
        return self.model(emg, valid_mask=sample_mask.to(device=emg.device, dtype=torch.bool))

    @staticmethod
    def _objective(outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        for name in ("pretrain_loss", "contrastive_loss", "vq_loss", "reconstruction_loss"):
            value = outputs.get(name)
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value):
                    raise ValueError(f"baseline objective {name} is non-finite")
                return value
        raise ValueError("baseline wrapper did not return a recognized objective")


__all__ = ["BaselinePretrainModule"]
