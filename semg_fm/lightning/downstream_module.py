"""Lightning fine-tuning module for the five canonical downstream tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from semg_fm.metrics.events import event_f1, event_f1_from_counts
from semg_fm.metrics.regression import masked_regression_metrics
from semg_fm.metrics.text import levenshtein_distance
from semg_fm.models.heads import CTCTextHead, GestureEventHead, TemporalRegressionHead
from semg_fm.robustness import apply_robustness
from semg_fm.tasks import task_kind, task_spec


def _backbone_block_sequence(backbone: nn.Module) -> list[nn.Module] | None:
    """Return the backbone's transformer blocks for partial fine-tuning."""
    for attribute in ("blocks",):
        sequence = getattr(backbone, attribute, None)
        if isinstance(sequence, (nn.ModuleList, nn.Sequential)):
            return list(sequence)
    contextualizer = getattr(backbone, "contextualizer", None)
    layers = getattr(contextualizer, "layers", None)
    if isinstance(layers, (nn.ModuleList, nn.Sequential)):
        return list(layers)
    encoder = getattr(backbone, "encoder", None)
    if encoder is not None and encoder is not backbone:
        return _backbone_block_sequence(encoder)
    return None


class DownstreamModule(LightningModule):
    """Fine-tune a backbone using task-specific labels from canonical metadata.

    Canonical loaders keep heterogeneous labels in ``batch['metadata']`` so an
    HDF5 window stays faithful to its source format.  Batches are task
    homogeneous; this module validates that invariant before constructing the
    corresponding supervised target.  DDP evaluation duplicates are ignored
    through ``example_weight`` rather than silently counted twice.

    ``finetune_mode`` selects the design-documented protocol: ``"probe"``
    freezes the backbone entirely, ``"partial"`` unfreezes only the last
    ``unfrozen_blocks`` transformer blocks, and ``"full"`` trains everything.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        task_id: str,
        output_dim: int | None = None,
        target_name: str | None = None,
        class_names: Sequence[str] = (),
        vocabulary: Sequence[str] = (),
        event_tolerance_s: float = 0.05,
        finetune_mode: str = "full",
        unfrozen_blocks: int = 1,
        robustness: Mapping[str, Any] | None = None,
        collect_predictions: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if event_tolerance_s < 0:
            raise ValueError("event_tolerance_s must be non-negative")
        if not hasattr(backbone, "d_model") or not isinstance(backbone.d_model, int):
            raise ValueError("backbone must expose integer d_model")
        self.backbone = backbone
        self.task_id = task_spec(task_id).task_id
        self.target_name = target_name
        self.event_tolerance_s = float(event_tolerance_s)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.robustness = dict(robustness or {})
        self.collect_predictions = bool(collect_predictions)
        if finetune_mode not in {"probe", "partial", "full"}:
            raise ValueError("finetune_mode must be 'probe', 'partial', or 'full'")
        self.finetune_mode = finetune_mode
        if finetune_mode == "probe":
            backbone.requires_grad_(False)
        elif finetune_mode == "partial":
            blocks = _backbone_block_sequence(backbone)
            if blocks is None:
                raise ValueError(
                    f"partial fine-tuning needs identifiable backbone blocks; {type(backbone).__name__} has none"
                )
            if not 1 <= unfrozen_blocks <= len(blocks):
                raise ValueError(f"unfrozen_blocks must be in [1, {len(blocks)}]")
            backbone.requires_grad_(False)
            for block in blocks[-unfrozen_blocks:]:
                block.requires_grad_(True)
        kind = task_kind(task_id)
        if kind == "regression":
            if output_dim is None or output_dim < 1:
                raise ValueError("regression tasks require positive output_dim")
            self.head: nn.Module = TemporalRegressionHead(backbone.d_model, output_dim)
        elif kind == "events":
            self.head = GestureEventHead(backbone.d_model, class_names)
        else:
            self.head = CTCTextHead(backbone.d_model, vocabulary)
        # Per-stage primary-metric accumulators (event TP/FP/FN or edit totals)
        # flushed and reported once per epoch end; see on_*_epoch_end hooks.
        self._primary_accumulators: dict[str, dict[str, int]] = {}
        # Test-stage per-subject and per-window artifact collection for
        # per_subject_metrics.csv and the optional predictions.h5.
        self._subject_accumulators: dict[str, dict[str, float]] = {}
        self._test_predictions: list[dict[str, Any]] = []
        # Avoid serializing the potentially large backbone object in hparams;
        # entry points separately snapshot its config and source checkpoint.
        self.save_hyperparameters(ignore=["backbone", "head"])

    def forward(
        self,
        emg: torch.Tensor,
        sample_mask: torch.Tensor | None = None,
        observation_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return temporal backbone features plus a token-validity mask.

        ``observation_mask`` optionally overrides the per-channel visibility
        derived from ``sample_mask`` (used by channel-level robustness
        transforms at evaluation time).
        """
        if emg.ndim != 3:
            raise ValueError("emg must have shape [B,T,C]")
        if sample_mask is None:
            sample_mask = torch.ones(emg.shape[:2], dtype=torch.bool, device=emg.device)
        if sample_mask.shape != emg.shape[:2] or sample_mask.dtype is not torch.bool:
            raise ValueError("sample_mask must be boolean [B,T]")
        if observation_mask is not None:
            if observation_mask.shape != emg.shape or observation_mask.dtype is not torch.bool:
                raise ValueError("observation_mask must be boolean [B,T,C]")
            observed = observation_mask.to(emg.device)
        else:
            observed = sample_mask.to(emg.device).unsqueeze(-1).expand_as(emg)
        outputs = self.backbone(emg, observation_mask=observed)
        if not isinstance(outputs, Mapping) or "z_motion" not in outputs:
            raise ValueError("backbone must return mapping with z_motion [B,N,D]")
        features = outputs["z_motion"]
        if features.ndim != 3:
            raise ValueError("backbone z_motion must have shape [B,N,D]")
        patch_samples = getattr(self.backbone, "patch_samples", None)
        if not isinstance(patch_samples, int) or patch_samples < 1 or emg.shape[1] % patch_samples:
            raise ValueError("backbone must expose patch_samples dividing input length")
        token_mask = observed.reshape(emg.shape[0], -1, patch_samples, emg.shape[-1]).any(dim=(2, 3))
        if features.shape[:2] != token_mask.shape:
            raise ValueError("backbone z_motion shape does not align with patch_samples")
        return {"features": features, "token_mask": token_mask, "backbone_outputs": outputs}

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute and log one task-homogeneous supervised loss."""
        loss, metrics, _ = self._loss_and_metrics(batch)
        self._log("train", loss, metrics, int(batch["emg"].shape[0]))
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute validation loss and accumulate primary task metrics."""
        loss, metrics, encoded = self._loss_and_metrics(batch)
        self._log("val", loss, metrics, int(batch["emg"].shape[0]))
        self._accumulate_primary("val", batch, encoded)
        return loss

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute held-out loss under the configured robustness transform."""
        loss, metrics, encoded = self._loss_and_metrics(batch, transform=self.robustness or None)
        self._log("test", loss, metrics, int(batch["emg"].shape[0]))
        self._accumulate_test(batch, encoded, loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._primary_accumulators["val"] = {"tp": 0, "fp": 0, "fn": 0, "edits": 0, "ref_chars": 0}

    def on_test_epoch_start(self) -> None:
        self._primary_accumulators["test"] = {"tp": 0, "fp": 0, "fn": 0, "edits": 0, "ref_chars": 0}
        self._subject_accumulators.clear()
        self._test_predictions.clear()

    def on_validation_epoch_end(self) -> None:
        self._report_primary("val")

    def on_test_epoch_end(self) -> None:
        self._report_primary("test")

    def configure_optimizers(self):
        """Optimize the selected backbone and its downstream head together."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretraining ``student.*`` state dict into this module's backbone."""
        # weights_only=False mirrors Lightning's own checkpoint format; only
        # load checkpoints produced by this repository's training entry points.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint must contain a state_dict mapping")
        student_state = {
            name.removeprefix("student."): value for name, value in state.items() if name.startswith("student.")
        }
        if not student_state:
            student_state = {name.removeprefix("backbone."): value for name, value in state.items() if name.startswith("backbone.")}
        if not student_state:
            student_state = {name.removeprefix("model.encoder."): value for name, value in state.items() if name.startswith("model.encoder.")}
        if not student_state:
            raise ValueError("checkpoint has neither student.*, backbone.*, nor model.encoder.* weights")
        self.backbone.load_state_dict(student_state, strict=True)

    def _loss_and_metrics(
        self, batch: Mapping[str, Any], transform: Mapping[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if str(batch.get("task_id")) != self.task_id:
            raise ValueError(f"batch task_id {batch.get('task_id')!r} does not match module task_id {self.task_id!r}")
        emg, sample_mask = batch.get("emg"), batch.get("sample_mask")
        if not isinstance(emg, torch.Tensor) or not isinstance(sample_mask, torch.Tensor):
            raise ValueError("batch requires tensor emg and sample_mask")
        metadata = batch.get("metadata")
        if not isinstance(metadata, list) or len(metadata) != emg.shape[0]:
            raise ValueError("batch metadata must be one mapping per example")
        weights = batch.get("example_weight")
        if weights is None:
            weights = torch.ones(emg.shape[0], device=emg.device, dtype=emg.dtype)
        if not isinstance(weights, torch.Tensor) or weights.shape != (emg.shape[0],):
            raise ValueError("example_weight must be [B]")
        observation_mask = None
        if transform is not None:
            emg, sample_mask, observation_mask = apply_robustness(emg, sample_mask, transform)
        encoded = self(
            emg, sample_mask.to(device=emg.device, dtype=torch.bool), observation_mask=observation_mask
        )
        kind = task_kind(self.task_id)
        if kind == "regression":
            loss, metrics = self._regression_loss(encoded["features"], metadata, weights)
        elif kind == "events":
            loss, metrics = self._event_loss(encoded["features"], encoded["token_mask"], metadata, weights, emg.shape[1])
        else:
            loss, metrics = self._ctc_loss(encoded["features"], encoded["token_mask"], metadata, weights)
        return loss, metrics, encoded

    def _accumulate_primary(self, stage: str, batch: Mapping[str, Any], encoded: Mapping[str, torch.Tensor]) -> None:
        """Fold one batch into the stage's primary-metric totals."""
        counts = self._primary_accumulators.get(stage)
        if counts is None:
            return
        metadata = batch.get("metadata")
        if not isinstance(metadata, list):
            return
        weights = batch.get("example_weight")
        if weights is None:
            weights = torch.ones(int(batch["emg"].shape[0]), device=batch["emg"].device, dtype=batch["emg"].dtype)
        included = weights > 0
        kind = task_kind(self.task_id)
        if kind == "events":
            tp, fp, fn, _per_example, _predictions = self._event_matches(encoded, metadata, included)
            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn
        elif kind == "ctc":
            edits, ref_chars, _per_example, _hypotheses = self._edit_totals(encoded, metadata, included)
            counts["edits"] += edits
            counts["ref_chars"] += ref_chars

    def _accumulate_test(
        self, batch: Mapping[str, Any], encoded: Mapping[str, torch.Tensor], loss: torch.Tensor
    ) -> None:
        """Fold one test batch into primary, per-subject, and prediction totals."""
        metadata = batch.get("metadata")
        if not isinstance(metadata, list):
            return
        weights = batch.get("example_weight")
        if weights is None:
            weights = torch.ones(len(metadata), device=batch["emg"].device, dtype=batch["emg"].dtype)
        included = weights > 0
        kind = task_kind(self.task_id)
        if kind == "events":
            tp, fp, fn, per_example, predictions = self._event_matches(encoded, metadata, included)
        elif kind == "ctc":
            edits, ref_chars, per_example, hypotheses = self._edit_totals(encoded, metadata, included)
        else:
            per_example = [(0, 0)] * len(metadata)
            predictions = hypotheses = None
        counts = self._primary_accumulators.get("test")
        if counts is not None:
            if kind == "events":
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
            elif kind == "ctc":
                counts["edits"] += edits
                counts["ref_chars"] += ref_chars
        for index, item in enumerate(metadata):
            if not bool(included[index]):
                continue
            subject = str(item.get("subject_id", "unknown"))
            row = self._subject_accumulators.setdefault(
                subject, {"windows": 0.0, "loss": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0, "edits": 0.0, "ref_chars": 0.0}
            )
            row["windows"] += 1.0
            row["loss"] += float(loss.detach())  # batch-mean loss approximation per window
            if kind == "events":
                entry_tp, entry_fp, entry_fn = per_example[index]
                row["tp"] += entry_tp
                row["fp"] += entry_fp
                row["fn"] += entry_fn
            elif kind == "ctc":
                entry_edits, entry_ref_chars = per_example[index]
                row["edits"] += entry_edits
                row["ref_chars"] += entry_ref_chars
            if not self.collect_predictions:
                continue
            entry: dict[str, Any] = {
                "record_id": str(item.get("record_id", "")),
                "subject_id": subject,
            }
            if kind == "events":
                entry["predicted"] = [[label, float(time_s)] for label, time_s in predictions[index]]
            elif kind == "ctc":
                entry["reference"] = str(item.get("target_text", ""))
                entry["hypothesis"] = hypotheses[index]
            else:
                entry["loss"] = float(loss.detach())
            self._test_predictions.append(entry)

    @property
    def test_subject_metrics(self) -> list[dict[str, Any]]:
        """Per-subject test rows with the task's primary metric derived from totals."""
        kind = task_kind(self.task_id)
        rows = []
        for subject, values in sorted(self._subject_accumulators.items()):
            row: dict[str, Any] = {"subject_id": subject}
            row["windows"] = int(values["windows"])
            row["loss"] = values["loss"] / max(1, int(values["windows"]))
            if kind == "events":
                row["event_f1"] = event_f1_from_counts(
                    values["tp"], values["fp"], values["fn"]
                )
            elif kind == "ctc":
                row["character_error_rate"] = (
                    values["edits"] / values["ref_chars"] if values["ref_chars"] > 0 else 0.0
                )
            rows.append(row)
        return rows

    @property
    def test_predictions(self) -> list[dict[str, Any]]:
        """Per-window predictions collected during the last test stage."""
        return list(self._test_predictions)

    def _event_matches(
        self, encoded: Mapping[str, torch.Tensor], metadata: list[Mapping[str, Any]], included: torch.Tensor
    ) -> tuple[int, int, int, list[tuple[int, int, int]], list[list[tuple[str, float]]]]:
        """Return per-batch true/false positives/negatives via tolerance matching.

        Also returns per-example contributions and decoded predictions so the
        test stage can report per-subject metrics and optional predictions.
        """
        assert isinstance(self.head, GestureEventHead)
        features, token_mask = encoded["features"], encoded["token_mask"]
        logits = self.head(features)
        patch_samples = getattr(self.backbone, "patch_samples")
        batch_size, tokens = logits.shape[:2]
        token_seconds = torch.zeros((batch_size, tokens), dtype=torch.float32, device=logits.device)
        for index, item in enumerate(metadata):
            rate = float(item.get("sample_rate_hz", 0.0))
            if rate <= 0:
                raise ValueError("event metadata requires positive sample_rate_hz")
            # Each token covers patch_samples samples; report its center time.
            token_seconds[index] = (torch.arange(tokens, device=logits.device) * patch_samples + patch_samples / 2) / rate
        decoded = self.head.decode(logits, token_seconds, threshold=0.5)
        tp = fp = fn = 0
        per_example: list[tuple[int, int, int]] = []
        predictions: list[list[tuple[str, float]]] = []
        for index, item in enumerate(metadata):
            if not bool(included[index]):
                per_example.append((0, 0, 0))
                predictions.append([])
                continue
            reference = [(label, float(time_s)) for label, time_s in item.get("events", [])]
            # Token positions outside the observed window never fire: decode only
            # events anchored on observed tokens.
            prediction = [
                (label, time_s) for position, (label, time_s) in enumerate(decoded[index])
                if bool(token_mask[index, position])
            ]
            counts = event_f1(prediction, reference, tolerance_s=self.event_tolerance_s)
            per_example.append(
                (int(counts["true_positive"]), int(counts["false_positive"]), int(counts["false_negative"]))
            )
            predictions.append(prediction)
            tp += int(counts["true_positive"])
            fp += int(counts["false_positive"])
            fn += int(counts["false_negative"])
        return tp, fp, fn, per_example, predictions

    def _edit_totals(
        self, encoded: Mapping[str, torch.Tensor], metadata: list[Mapping[str, Any]], included: torch.Tensor
    ) -> tuple[int, int, list[tuple[int, int]], list[str]]:
        """Return per-batch Levenshtein edit and reference lengths for global CER.

        Also returns per-example contributions and decoded hypotheses for the
        per-subject report and optional predictions.
        """
        assert isinstance(self.head, CTCTextHead)
        hypotheses = self.head.greedy_decode(self.head(encoded["features"]))
        edits = ref_chars = 0
        per_example: list[tuple[int, int]] = []
        for index, item in enumerate(metadata):
            if not bool(included[index]):
                per_example.append((0, 0))
                continue
            reference = str(item.get("target_text", ""))
            edits += levenshtein_distance(reference, hypotheses[index])
            ref_chars += len(reference)
            per_example.append((levenshtein_distance(reference, hypotheses[index]), len(reference)))
        return edits, ref_chars, per_example, hypotheses

    def _report_primary(self, stage: str) -> None:
        """All-gather stage totals and log the declared primary task metric."""
        counts = self._primary_accumulators.pop(stage, None)
        if counts is None or self._trainer is None:
            return
        tensors = {key: torch.tensor(value, dtype=torch.float64, device=self.device) for key, value in counts.items()}
        if int(self.trainer.world_size) > 1:
            gathered = self.all_gather(tensors)
            totals = {key: torch.cat(values).sum() for key, values in gathered.items()}
        else:
            totals = tensors
        kind = task_kind(self.task_id)
        if kind == "events":
            f1 = event_f1_from_counts(totals["tp"], totals["fp"], totals["fn"])
            self.log(f"{stage}/{self.task_id}_event_f1", f1, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        elif kind == "ctc":
            cer = totals["edits"] / totals["ref_chars"] if totals["ref_chars"] > 0 else 0.0
            self.log(f"{stage}/{self.task_id}_character_error_rate", cer, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)

    def _regression_loss(
        self, features: torch.Tensor, metadata: list[Mapping[str, Any]], example_weight: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, TemporalRegressionHead)
        targets, masks = self._regression_targets(metadata, features.device)
        prediction = self.head(features)
        prediction = F.interpolate(prediction.transpose(1, 2), size=targets.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        metrics = masked_regression_metrics(prediction, targets, masks, example_weight=example_weight.to(features))
        loss = metrics["sum_squared_error"] / metrics["count"].clamp_min(1.0)
        return loss, {"mae": metrics["mae"], "rmse": metrics["rmse"], "count": metrics["count"]}

    def _regression_targets(
        self, metadata: list[Mapping[str, Any]], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, masks = [], []
        for item in metadata:
            target, target_mask = item.get("target"), item.get("target_mask")
            if self.task_id == "wrist_angle":
                if not self.target_name or not isinstance(target, Mapping) or not isinstance(target_mask, Mapping):
                    raise ValueError("wrist_angle requires target_name present in target and target_mask mappings")
                if self.target_name not in target or self.target_name not in target_mask:
                    raise ValueError(
                        f"target_name {self.target_name!r} is missing from converted wrist targets; "
                        f"available names: {sorted(target)}"
                    )
                target, target_mask = target[self.target_name], target_mask[self.target_name]
            if not isinstance(target, torch.Tensor) or not isinstance(target_mask, torch.Tensor):
                raise ValueError("regression metadata requires tensor target and target_mask")
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            if target.ndim != 2 or target_mask.shape[0] != target.shape[0]:
                raise ValueError("regression targets must be [T,D] with aligned mask")
            values.append(target.to(device=device, dtype=torch.float32))
            masks.append(target_mask.to(device=device, dtype=torch.bool))
        targets = torch.stack(values)
        if targets.shape[-1] != getattr(self.head, "output_dim"):
            raise ValueError("configured output_dim does not match canonical target dimension")
        return targets, torch.stack(masks)

    def _event_loss(
        self, features: torch.Tensor, token_mask: torch.Tensor, metadata: list[Mapping[str, Any]], example_weight: torch.Tensor,
        input_samples: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, GestureEventHead)
        logits = self.head(features)
        targets = torch.zeros(token_mask.shape, dtype=torch.long, device=features.device)
        for batch, item in enumerate(metadata):
            rate = float(item.get("sample_rate_hz", 0.0))
            if rate <= 0:
                raise ValueError("event metadata requires positive sample_rate_hz")
            for label, relative_time in item.get("events", []):
                if label not in self.head.class_names:
                    raise ValueError(
                        f"event label {label!r} is not in configured class_names {list(self.head.class_names)}; "
                        "update the task config to the converter-emitted vocabulary"
                    )
                index = min(token_mask.shape[1] - 1, max(0, int(float(relative_time) * rate / input_samples * token_mask.shape[1])))
                if token_mask[batch, index]:
                    targets[batch, index] = self.head.class_names.index(label) + 1
        per_token = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        weights = token_mask.to(per_token.dtype) * example_weight.to(per_token).unsqueeze(-1)
        loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, {"event_loss": loss.detach(), "event_tokens": weights.sum()}

    def _ctc_loss(
        self, features: torch.Tensor, token_mask: torch.Tensor, metadata: list[Mapping[str, Any]], example_weight: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert isinstance(self.head, CTCTextHead)
        encoded = []
        for item in metadata:
            text = item.get("target_text", "")
            if not isinstance(text, str):
                raise ValueError("CTC metadata target_text must be a string")
            try:
                encoded.append([self.head.vocabulary.index(character) + 1 for character in text])
            except ValueError as error:
                missing = sorted({character for character in text if character not in self.head.vocabulary})
                raise ValueError(
                    f"target_text contains characters absent from the configured vocabulary: {missing}; "
                    "update the task config vocabulary to cover the converted prompts"
                ) from error
        max_length = max((len(row) for row in encoded), default=0)
        targets = torch.zeros((len(encoded), max_length), dtype=torch.long, device=features.device)
        for index, row in enumerate(encoded):
            if row:
                targets[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=features.device)
        target_lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long, device=features.device)
        per_example = self.head.loss(
            self.head(features), targets=targets, input_lengths=token_mask.sum(dim=-1), target_lengths=target_lengths, reduction="none"
        )
        loss = (per_example * example_weight.to(per_example)).sum() / example_weight.sum().clamp_min(1.0)
        return loss, {"ctc_loss": loss.detach(), "characters": target_lengths.sum().to(features.dtype)}

    def _log(self, stage: str, loss: torch.Tensor, metrics: Mapping[str, torch.Tensor], batch_size: int) -> None:
        # Unit tests and library callers may invoke a step before Lightning has
        # attached a Trainer; the loss remains usable but there is nowhere safe
        # to emit framework metrics yet.
        if self._trainer is None:
            return
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in metrics.items():
            self.log(f"{stage}/{self.task_id}_{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)


__all__ = ["DownstreamModule"]
