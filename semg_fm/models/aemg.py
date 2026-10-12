"""AEMG-like discrete-code baseline for canonical 16-channel sEMG.

This is a core-mechanism reproduction: patch waveforms are encoded, vector
quantized, and contextualized.  It intentionally has no circular electrode
bias, so improvements from the proposed ring model remain attributable to its
geometry rather than a shared spatial prior.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..objectives.vector_quantization import VectorQuantizer
from .supervised_tcn import _resolve_channels, observed_samples, token_observation_mask


class AEMGEncoder(nn.Module):
    """Encode sEMG patches as VQ codes followed by a masked Transformer.

    The encoder accepts all canonical observation-mask variants.  Invalid raw
    samples are zeroed before patch projection, while their corresponding
    temporal tokens are excluded from VQ assignment, Transformer attention,
    pooling, and codebook statistics.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 6,
        heads: int = 8,
        patch_samples: int = 40,
        channels: int = 16,
        codebook_size: int = 512,
        commitment_cost: float = 0.25,
        *,
        in_channels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = _resolve_channels(channels, in_channels)
        if d_model < 1 or depth < 1 or heads < 1 or patch_samples < 1 or channels < 1:
            raise ValueError("d_model, depth, heads, patch_samples, and channels must be positive")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.heads = int(heads)
        self.patch_samples = int(patch_samples)
        self.channels = int(channels)
        self.codebook_size = int(codebook_size)
        self.patch_projection = nn.Linear(self.patch_samples * self.channels, self.d_model, bias=False)
        self.quantizer = VectorQuantizer(
            dim=self.d_model, codebook_size=self.codebook_size, commitment_cost=commitment_cost
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.heads,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.contextualizer = nn.TransformerEncoder(layer, num_layers=self.depth, enable_nested_tensor=False)

    def forward(
        self,
        emg: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return contextual tokens, VQ codes, and finite codebook diagnostics."""
        self._validate_emg(emg)
        observed = observed_samples(emg, (observation_mask, valid_mask, sample_mask))
        token_mask = token_observation_mask(observed, self.patch_samples)
        batch, samples, _ = emg.shape
        tokens = self.patch_projection(
            (emg * observed.to(emg.dtype)).reshape(
                batch, samples // self.patch_samples, self.patch_samples * self.channels
            )
        )
        weights = token_mask.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens * weights
        quantized = self.quantizer(tokens, valid_mask=token_mask)

        key_padding_mask = ~token_mask
        # Transformer attention rejects fully padded rows.  Those rows have
        # zero VQ input and are reset to zero after contextualization.
        fully_invalid = key_padding_mask.all(dim=1)
        if fully_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_invalid] = False
        context = self.contextualizer(quantized["quantized"], src_key_padding_mask=key_padding_mask) * weights
        pooled = (context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "z_temporal": context,
            "z_motion": context,
            "pooled": pooled,
            "token_mask": token_mask,
            "quantized": quantized["quantized"],
            "code_indices": quantized["indices"],
            "vq_loss": quantized["loss"],
            "codebook_usage": quantized["usage"],
            "codebook_perplexity": quantized["perplexity"],
            "codebook_active_codes": quantized["active_codes"],
        }

    def _validate_emg(self, emg: Any) -> None:
        if not isinstance(emg, torch.Tensor) or emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        if emg.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {emg.shape[-1]}")
        if emg.shape[1] < self.patch_samples or emg.shape[1] % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")


class AEMGPretrainingModel(nn.Module):
    """Convenience wrapper exposing an AEMG-like VQ objective as ``pretrain_loss``."""

    def __init__(self, encoder: AEMGEncoder | None = None, **encoder_kwargs: Any) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else AEMGEncoder(**encoder_kwargs)

    def forward(self, emg: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Encode an EMG batch and alias VQ loss for generic baseline runners."""
        outputs = self.encoder(emg, **kwargs)
        outputs["pretrain_loss"] = outputs["vq_loss"]
        return outputs


__all__ = ["AEMGEncoder", "AEMGPretrainingModel"]
