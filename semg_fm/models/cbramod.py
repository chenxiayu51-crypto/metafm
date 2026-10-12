"""CBraMod-like time/frequency and ordinary-channel-order baseline.

The model intentionally conditions on the acquisition order ``0..15`` rather
than circular electrode geometry.  It provides a fair contrast to the proposed
ring model while retaining two generic mechanisms used in sEMG foundation work:
time/frequency patch features and masked waveform reconstruction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .bendr import _random_uniform
from .supervised_tcn import _resolve_channels, observed_samples, token_observation_mask


def _channel_token_mask(observed: torch.Tensor, patch_samples: int) -> torch.Tensor:
    """Return one observed flag per temporal patch and ordinary channel index."""
    if observed.ndim != 3 or observed.shape[1] % patch_samples:
        raise ValueError("observed must be [B,T,C] with T divisible by patch_samples")
    return observed.reshape(
        observed.shape[0], observed.shape[1] // patch_samples, patch_samples, observed.shape[2]
    ).any(dim=2)


def sample_channel_reconstruction_mask(
    token_channel_mask: torch.Tensor,
    probability: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample reconstruction targets only at observed patch-channel tokens."""
    if token_channel_mask.ndim != 3 or token_channel_mask.dtype is not torch.bool:
        raise ValueError("token_channel_mask must be boolean [B,N,C]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    draws = _random_uniform(
        tuple(token_channel_mask.shape), device=token_channel_mask.device, generator=generator
    )
    return (draws < probability) & token_channel_mask


class CBraModEncoder(nn.Module):
    """Encode ordered channel patches with time/FFT features and reconstruction.

    ``reconstruction_mask`` has shape ``[B,N,C]`` and replaces those observed
    patch-channel features with a learned mask token before contextualization.
    Loss is evaluated only on observed samples selected by that mask, preventing
    either padded values or unmasked input waveforms from leaking a target.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 6,
        heads: int = 8,
        patch_samples: int = 40,
        channels: int = 16,
        *,
        in_channels: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = _resolve_channels(channels, in_channels)
        if d_model < 1 or depth < 1 or heads < 1 or patch_samples < 2 or channels < 1:
            raise ValueError("d_model, depth, heads, channels must be positive and patch_samples at least 2")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.heads = int(heads)
        self.patch_samples = int(patch_samples)
        self.channels = int(channels)
        spectral_bins = self.patch_samples // 2 + 1
        self.feature_projection = nn.Linear(self.patch_samples + spectral_bins, self.d_model, bias=False)
        # This learned table intentionally treats channels as an ordinary line.
        # It is therefore not rotation equivariant and has no ring-distance bias.
        self.ordered_channel_embedding = nn.Embedding(self.channels, self.d_model)
        self.mask_token = nn.Parameter(torch.zeros(self.d_model))
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
        self.waveform_decoder = nn.Linear(self.d_model, self.patch_samples)

    def forward(
        self,
        emg: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        reconstruction_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return temporal features and a masked-waveform reconstruction loss."""
        self._validate_emg(emg)
        observed = observed_samples(emg, (observation_mask, valid_mask, sample_mask))
        batch, samples, channels = emg.shape
        patches = (emg * observed.to(emg.dtype)).reshape(
            batch, samples // self.patch_samples, self.patch_samples, channels
        ).permute(0, 1, 3, 2)
        observed_patches = observed.reshape(
            batch, samples // self.patch_samples, self.patch_samples, channels
        ).permute(0, 1, 3, 2)
        channel_mask = _channel_token_mask(observed, self.patch_samples)
        token_mask = token_observation_mask(observed, self.patch_samples)
        if reconstruction_mask is None:
            reconstruction_mask = torch.zeros_like(channel_mask)
        if reconstruction_mask.shape != channel_mask.shape or reconstruction_mask.dtype is not torch.bool:
            raise ValueError("reconstruction_mask must be boolean [B,N,C]")
        reconstruction_mask = reconstruction_mask.to(emg.device) & channel_mask

        # FFT magnitudes complement raw time samples without adding a temporal
        # coordinate embedding. ``log1p`` controls high-amplitude artifacts.
        spectrum = torch.log1p(torch.fft.rfft(patches, dim=-1).abs())
        features = self.feature_projection(torch.cat((patches, spectrum), dim=-1))
        positions = torch.arange(self.channels, device=emg.device)
        features = features + self.ordered_channel_embedding(positions).view(1, 1, channels, -1)
        features = torch.where(
            reconstruction_mask.unsqueeze(-1), self.mask_token.view(1, 1, 1, -1), features
        )
        weights = channel_mask.unsqueeze(-1).to(features.dtype)
        features = features * weights
        flattened = features.reshape(batch, -1, self.d_model)
        flat_mask = channel_mask.reshape(batch, -1)
        key_padding_mask = ~flat_mask
        fully_invalid = key_padding_mask.all(dim=1)
        if fully_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_invalid] = False
        channel_tokens = self.contextualizer(flattened, src_key_padding_mask=key_padding_mask)
        channel_tokens = channel_tokens.reshape(batch, -1, channels, self.d_model) * weights
        reconstruction = self.waveform_decoder(channel_tokens)
        reconstruction_loss = self._reconstruction_loss(
            reconstruction, patches, observed_patches, reconstruction_mask
        )
        token_weights = channel_mask.unsqueeze(-1).to(channel_tokens.dtype)
        temporal = channel_tokens.sum(dim=2) / token_weights.sum(dim=2).clamp_min(1.0)
        temporal = temporal * token_mask.unsqueeze(-1).to(temporal.dtype)
        pooled = temporal.sum(dim=1) / token_mask.sum(dim=1, keepdim=True).to(temporal.dtype).clamp_min(1.0)
        return {
            "z_temporal": temporal,
            "z_motion": temporal,
            "pooled": pooled,
            "token_mask": token_mask,
            "channel_token_mask": channel_mask,
            "channel_tokens": channel_tokens,
            "reconstruction": reconstruction,
            "reconstruction_mask": reconstruction_mask,
            "reconstruction_loss": reconstruction_loss,
        }

    @staticmethod
    def _reconstruction_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        observed: torch.Tensor,
        reconstruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a differentiable zero when a batch has no masked targets."""
        selected = reconstruction_mask.unsqueeze(-1) & observed
        if not selected.any():
            return prediction.sum() * 0.0
        return F.mse_loss(prediction[selected], target[selected])

    def _validate_emg(self, emg: Any) -> None:
        if not isinstance(emg, torch.Tensor) or emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        if emg.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {emg.shape[-1]}")
        if emg.shape[1] < self.patch_samples or emg.shape[1] % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")


class CBraModPretrainingModel(nn.Module):
    """Attach random patch-channel masking to the CBraMod-like encoder."""

    def __init__(
        self, encoder: CBraModEncoder | None = None, *, mask_probability: float = 0.15, **encoder_kwargs: Any
    ) -> None:
        super().__init__()
        if not 0.0 <= mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0, 1]")
        self.encoder = encoder if encoder is not None else CBraModEncoder(**encoder_kwargs)
        self.mask_probability = float(mask_probability)

    def forward(self, emg: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Mask observed patch-channel tokens when no explicit mask is supplied."""
        generator = kwargs.pop("generator", None)
        if kwargs.get("reconstruction_mask") is None:
            observed = observed_samples(
                emg, (kwargs.get("observation_mask"), kwargs.get("valid_mask"), kwargs.get("sample_mask"))
            )
            kwargs["reconstruction_mask"] = sample_channel_reconstruction_mask(
                _channel_token_mask(observed, self.encoder.patch_samples),
                self.mask_probability,
                generator=generator,
            )
        outputs = self.encoder(emg, **kwargs)
        outputs["pretrain_loss"] = outputs["reconstruction_loss"]
        return outputs


__all__ = ["CBraModEncoder", "CBraModPretrainingModel", "sample_channel_reconstruction_mask"]
