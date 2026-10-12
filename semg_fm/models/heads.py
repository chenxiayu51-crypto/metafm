"""Small downstream heads shared by proposed and baseline backbones."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _validate_features(features: torch.Tensor, input_dim: int) -> None:
    """Validate the common temporal backbone output contract."""
    if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[-1] != input_dim:
        raise ValueError(f"features must have shape [B,N,{input_dim}]")


class TemporalRegressionHead(nn.Module):
    """Map temporal backbone features to continuous pose or wrist targets."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return time-aligned ``[B,N,output_dim]`` continuous predictions."""
        _validate_features(features, self.input_dim)
        return self.projection(features)


class GestureEventHead(nn.Module):
    """Per-token gesture-event classifier with an explicit no-event class."""

    def __init__(self, input_dim: int, class_names: Sequence[str]) -> None:
        super().__init__()
        if input_dim < 1 or not class_names or len(set(class_names)) != len(class_names):
            raise ValueError("input_dim must be positive and class_names must be unique and non-empty")
        if any(not isinstance(name, str) or not name for name in class_names):
            raise ValueError("class_names must contain non-empty strings")
        self.input_dim = int(input_dim)
        self.class_names = tuple(class_names)
        self.classifier = nn.Linear(self.input_dim, len(self.class_names) + 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return logits whose index zero is no-event and others match class_names."""
        _validate_features(features, self.input_dim)
        return self.classifier(features)

    def decode(
        self, logits: torch.Tensor, time_s: torch.Tensor, *, threshold: float = 0.5
    ) -> list[list[tuple[str, float]]]:
        """Decode token predictions into events without inventing dense labels."""
        if logits.ndim != 3 or logits.shape[-1] != len(self.class_names) + 1:
            raise ValueError("logits must be [B,N,1+num_classes]")
        if time_s.shape != logits.shape[:2] or not torch.isfinite(time_s).all():
            raise ValueError("time_s must be finite [B,N]")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        probabilities = logits.softmax(dim=-1)
        confidence, classes = probabilities.max(dim=-1)
        decoded: list[list[tuple[str, float]]] = []
        for batch in range(logits.shape[0]):
            events = []
            for token in range(logits.shape[1]):
                class_index = int(classes[batch, token])
                if class_index and float(confidence[batch, token]) >= threshold:
                    events.append((self.class_names[class_index - 1], float(time_s[batch, token])))
            decoded.append(events)
        return decoded


class CTCTextHead(nn.Module):
    """CTC projection and decoding contract for handwriting and typing windows."""

    def __init__(self, input_dim: int, vocabulary: Sequence[str]) -> None:
        super().__init__()
        if input_dim < 1 or not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("input_dim must be positive and vocabulary must be unique and non-empty")
        if any(not isinstance(token, str) or not token for token in vocabulary):
            raise ValueError("vocabulary must contain non-empty strings")
        self.input_dim = int(input_dim)
        self.vocabulary = tuple(vocabulary)
        self.blank_index = 0
        self.classifier = nn.Linear(self.input_dim, len(self.vocabulary) + 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized CTC logits ``[B,N,1+vocabulary_size]``."""
        _validate_features(features, self.input_dim)
        return self.classifier(features)

    def loss(
        self,
        logits: torch.Tensor,
        *,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute finite CTC loss, optionally retaining one loss per example."""
        if logits.ndim != 3 or logits.shape[-1] != len(self.vocabulary) + 1:
            raise ValueError("logits must be [B,N,1+vocabulary_size]")
        if targets.dtype is not torch.long or targets.ndim != 2 or targets.shape[0] != logits.shape[0]:
            raise ValueError("targets must be padded long [B,L]")
        if input_lengths.shape != target_lengths.shape or input_lengths.shape != (logits.shape[0],):
            raise ValueError("input_lengths and target_lengths must be [B]")
        if (input_lengths < 0).any() or (input_lengths > logits.shape[1]).any() or (target_lengths < 0).any():
            raise ValueError("CTC lengths are out of range")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        flattened = torch.cat(
            [targets[index, : int(target_lengths[index])].to(logits.device) for index in range(logits.shape[0])]
        )
        return F.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            flattened,
            input_lengths.to(device=logits.device, dtype=torch.long),
            target_lengths.to(device=logits.device, dtype=torch.long),
            blank=self.blank_index,
            reduction=reduction,
            zero_infinity=True,
        )

    def greedy_decode(self, logits: torch.Tensor) -> list[str]:
        """Collapse CTC repeats/blanks into strings, without event timing claims."""
        if logits.ndim != 3 or logits.shape[-1] != len(self.vocabulary) + 1:
            raise ValueError("logits must be [B,N,1+vocabulary_size]")
        decoded = []
        for indices in logits.argmax(dim=-1).tolist():
            previous = self.blank_index
            tokens = []
            for index in indices:
                if index != self.blank_index and index != previous:
                    tokens.append(self.vocabulary[index - 1])
                previous = index
            decoded.append("".join(tokens))
        return decoded


__all__ = ["CTCTextHead", "GestureEventHead", "TemporalRegressionHead"]
