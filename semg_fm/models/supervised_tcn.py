"""Causal temporal-convolution baseline for canonical 16-channel sEMG.

This is a compact supervised-TCN *core-mechanism reproduction*, not an
official implementation of any particular published model.  It deliberately
uses ordinary channel mixing and contains neither circular-electrode bias nor
channel-position embeddings; it is the non-ring comparison baseline.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _resolve_channels(channels: int, in_channels: int | None) -> int:
    """Resolve the legacy ``in_channels`` spelling without ambiguity."""
    if in_channels is None:
        return channels
    if channels != 16 and channels != in_channels:
        raise ValueError("channels and in_channels disagree")
    return in_channels


def observed_samples(
    emg: torch.Tensor,
    masks: Iterable[torch.Tensor | None],
) -> torch.Tensor:
    """Combine sample/electrode masks into a boolean ``[B,T,C]`` observation grid.

    A time-only validity mask is common after collating variable-length H5
    windows, whereas channel masks occur after electrode dropout.  Combining
    every supplied mask avoids treating padding or a missing electrode as an
    observed zero-valued signal.
    """
    if emg.ndim != 3:
        raise ValueError("emg must have shape [batch, time, channels]")
    batch, samples, channels = emg.shape
    combined = torch.ones((batch, samples, channels), dtype=torch.bool, device=emg.device)
    for mask in masks:
        if mask is None:
            continue
        if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
            raise ValueError("observation masks must be boolean tensors")
        if mask.shape == (batch, samples):
            expanded = mask.to(emg.device).unsqueeze(-1).expand_as(combined)
        elif mask.shape == (batch, samples, channels):
            expanded = mask.to(emg.device)
        elif mask.shape == (batch, channels):
            expanded = mask.to(emg.device).unsqueeze(1).expand_as(combined)
        else:
            raise ValueError("observation masks must be [B,T], [B,T,C], or [B,C]")
        combined &= expanded
    return combined


def token_observation_mask(observed: torch.Tensor, patch_samples: int) -> torch.Tensor:
    """Reduce sample-level observations to one valid flag per temporal token."""
    if observed.ndim != 3 or observed.shape[1] % patch_samples:
        raise ValueError("observed samples must be [B,T,C] with T divisible by patch_samples")
    return observed.reshape(
        observed.shape[0], observed.shape[1] // patch_samples, patch_samples, observed.shape[2]
    ).any(dim=(2, 3))


class _CausalResidualBlock(nn.Module):
    """One zero-preserving causal residual block operating on ``[B,N,D]`` tokens."""

    def __init__(self, d_model: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.norm = nn.LayerNorm(d_model)
        # Bias-free convolutions ensure an invalid zero token cannot inject a
        # learned constant into later valid tokens through the causal state.
        self.convolution = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, dilation=dilation, bias=False
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        mask = token_mask.unsqueeze(-1).to(tokens.dtype)
        normalized = self.norm(tokens) * mask
        convolution = self.convolution(F.pad(normalized.transpose(1, 2), (self.left_padding, 0)))
        convolution = F.gelu(convolution.transpose(1, 2)) * mask
        tokens = (tokens + self.dropout(convolution)) * mask
        return (tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))) * mask


class SupervisedTCN(nn.Module):
    """Encode canonical ``[B,T,16]`` sEMG with a masked causal TCN.

    ``forward`` returns the shared backbone mapping (``z_temporal`` /
    ``z_motion`` / ``pooled`` / ``token_mask``) required by the downstream
    module; use :meth:`extract_features` when only the temporal features are
    needed.  Masked input values are multiplied out before the feature stem
    and invalid tokens are re-zeroed after every residual block, so changing
    a masked raw value cannot affect any output.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 6,
        patch_samples: int = 40,
        channels: int = 16,
        *,
        in_channels: int | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = _resolve_channels(channels, in_channels)
        if d_model < 1 or depth < 1 or patch_samples < 1 or channels < 1:
            raise ValueError("d_model, depth, patch_samples, and channels must be positive")
        if kernel_size < 2 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer of at least 3")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.depth = depth
        self.patch_samples = patch_samples
        self.channels = channels
        self.kernel_size = kernel_size
        self.stem = nn.Conv1d(channels, d_model, kernel_size=patch_samples, stride=patch_samples, bias=False)
        self.blocks = nn.ModuleList(
            _CausalResidualBlock(d_model, dilation=2**index, kernel_size=kernel_size, dropout=dropout)
            for index in range(depth)
        )

    def forward(
        self,
        emg: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the shared backbone mapping with ``z_motion`` for downstream heads."""
        return self.extract_features(
            emg, observation_mask=observation_mask, valid_mask=valid_mask, sample_mask=sample_mask
        )

    def extract_features(
        self,
        emg: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return temporal features, their validity mask, and masked-mean pooling."""
        self._validate_emg(emg)
        observed = observed_samples(emg, (observation_mask, valid_mask, sample_mask))
        token_mask = token_observation_mask(observed, self.patch_samples)
        masked_emg = emg * observed.to(emg.dtype)
        tokens = self.stem(masked_emg.transpose(1, 2)).transpose(1, 2)
        token_values = token_mask.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens * token_values
        for block in self.blocks:
            tokens = block(tokens, token_mask)
        pooled = self.pool(tokens, token_mask)
        return {
            "z_temporal": tokens,
            # Alias used by shared downstream code: unlike CIRCE-Myo, the TCN
            # has no separate electrode field to pool before temporal decoding.
            "z_motion": tokens,
            "pooled": pooled,
            "token_mask": token_mask,
        }

    @staticmethod
    def pool(tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Compute a finite masked mean over a sequence of temporal features."""
        if tokens.ndim != 3 or token_mask.shape != tokens.shape[:2] or token_mask.dtype is not torch.bool:
            raise ValueError("tokens must be [B,N,D] and token_mask must be boolean [B,N]")
        weights = token_mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _validate_emg(self, emg: torch.Tensor) -> None:
        if not isinstance(emg, torch.Tensor) or emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        if emg.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {emg.shape[-1]}")
        if emg.shape[1] < self.patch_samples or emg.shape[1] % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")
