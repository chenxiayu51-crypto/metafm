"""Time- and frequency-domain patch embeddings for electrode-ring sEMG."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MultiScalePatchEncoder(nn.Module):
    """Encode time-major sEMG into one embedding grid per electrode.

    The temporal convolution is grouped by electrode, so it does not mix an
    arbitrary channel ordering before circular attention can model the ring.
    """

    def __init__(
        self,
        channels: int = 16,
        d_model: int = 128,
        patch_samples: int = 40,
        temporal_features: int = 8,
        spectral_bins: int = 8,
    ) -> None:
        super().__init__()
        if channels < 1 or d_model < 1 or patch_samples < 2:
            raise ValueError("channels and d_model must be positive and patch_samples must be at least 2")
        if temporal_features < 1 or spectral_bins < 1:
            raise ValueError("temporal_features and spectral_bins must be positive")

        self.channels = channels
        self.patch_samples = patch_samples
        self.spectral_bins = spectral_bins
        self.temporal_features = temporal_features
        self.temporal = nn.Conv1d(
            channels,
            channels * temporal_features,
            kernel_size=patch_samples,
            stride=patch_samples,
            groups=channels,
        )
        self.projection = nn.Linear(temporal_features + spectral_bins, d_model)

    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """Return channel-preserving tokens of shape ``[B, C, N, D]``."""
        if emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        batch, samples, channels = emg.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if samples < self.patch_samples or samples % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")

        temporal = self.temporal(emg.transpose(1, 2))
        patches = samples // self.patch_samples
        temporal = temporal.reshape(batch, channels, self.temporal_features, patches)
        temporal = temporal.permute(0, 1, 3, 2)

        waveform_patches = emg.transpose(1, 2).unfold(-1, self.patch_samples, self.patch_samples)
        spectrum = torch.log1p(torch.fft.rfft(waveform_patches, dim=-1).abs())
        spectrum = F.adaptive_avg_pool1d(
            spectrum.reshape(-1, spectrum.shape[-1]), self.spectral_bins
        ).reshape(batch, channels, patches, self.spectral_bins)

        return self.projection(torch.cat((temporal, spectrum), dim=-1))
