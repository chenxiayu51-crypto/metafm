"""Circular relative-position attention across an electrode ring."""

from __future__ import annotations

import math

import torch
from torch import nn


def signed_ring_offsets(channels: int) -> torch.Tensor:
    """Return shortest signed offsets from every query electrode to every key."""
    if channels < 2:
        raise ValueError("a ring needs at least two channels")
    positions = torch.arange(channels)
    offsets = positions[None, :] - positions[:, None]
    half = channels // 2
    return torch.remainder(offsets + half, channels) - half


class RingMultiheadAttention(nn.Module):
    """Attention across electrodes independently for every temporal patch."""

    def __init__(self, d_model: int, num_heads: int, channels: int = 16) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError("d_model must be divisible by a positive num_heads")
        if channels < 2:
            raise ValueError("channels must be at least 2")
        self.d_model = d_model
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.relative_bias = nn.Parameter(torch.zeros(num_heads, 2 * channels - 1))
        self.register_buffer("offset_indices", signed_ring_offsets(channels) + channels - 1)

    def forward(
        self,
        tokens: torch.Tensor,
        electrode_observed: torch.Tensor | None = None,
        *,
        observation_mask: torch.Tensor | None = None,
        electrode_observation_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Apply ring attention to ``tokens`` shaped ``[B, C, N, D]``.

        ``electrode_observed`` can be ``[B, C]`` or ``[B, C, N]``. Missing
        electrodes are excluded as keys and emit zero-valued output tokens.
        """
        masks = [mask for mask in (electrode_observed, observation_mask, electrode_observation_mask) if mask is not None]
        if len(masks) > 1:
            raise ValueError("pass only one electrode observation mask")
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [batch, channels, patches, d_model]")
        batch, channels, patches, d_model = tokens.shape
        if channels != self.channels or d_model != self.d_model:
            raise ValueError(f"expected tokens with {self.channels} channels and d_model={self.d_model}")

        observed = self._observed_mask(masks[0] if masks else None, batch, channels, patches, tokens.device)
        qkv = self.qkv(tokens).reshape(batch, channels, patches, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=3)
        query = query.permute(0, 2, 3, 1, 4)
        key = key.permute(0, 2, 3, 1, 4)
        value = value.permute(0, 2, 3, 1, 4)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        bias = self.relative_bias[:, self.offset_indices].unsqueeze(0).unsqueeze(0)
        scores = scores + bias
        key_observed = observed.permute(0, 2, 1)[:, :, None, None, :]
        query_observed = observed.permute(0, 2, 1)[:, :, None, :, None]
        weights = torch.softmax(scores.masked_fill(~key_observed, -torch.finfo(scores.dtype).max), dim=-1)
        weights = weights * key_observed.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        weights = weights * query_observed.to(weights.dtype)

        attended = torch.matmul(weights, value)
        attended = attended.permute(0, 3, 1, 2, 4).reshape(batch, channels, patches, d_model)
        output = self.out_proj(attended) * observed.unsqueeze(-1).to(attended.dtype)
        if return_attention:
            return output, weights
        return output

    @staticmethod
    def _observed_mask(
        mask: torch.Tensor | None,
        batch: int,
        channels: int,
        patches: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, channels, patches, dtype=torch.bool, device=device)
        if mask.shape == (batch, channels):
            return mask.to(device=device, dtype=torch.bool).unsqueeze(-1).expand(-1, -1, patches)
        if mask.shape == (batch, channels, patches):
            return mask.to(device=device, dtype=torch.bool)
        raise ValueError("electrode observation mask must have shape [B, C] or [B, C, N]")
