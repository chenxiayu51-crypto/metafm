"""Criss-cross temporal and circular-electrode encoding for sEMG."""

from __future__ import annotations

import torch
from torch import nn

from .patch_encoder import MultiScalePatchEncoder
from .ring_attention import RingMultiheadAttention


class RingCrissCrossBlock(nn.Module):
    """Fuse temporal and ring attention computed from one normalized token grid."""

    def __init__(self, d_model: int, heads: int, channels: int = 16) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(d_model, heads, dropout=0.0, batch_first=True)
        self.ring_attention = RingMultiheadAttention(d_model, heads, channels=channels)
        self.gate = nn.Linear(d_model, 2)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def fusion_weights(self, normalized_tokens: torch.Tensor) -> torch.Tensor:
        """Return the temporal/ring convex-combination weights for each token."""
        return torch.softmax(self.gate(normalized_tokens), dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        electrode_observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply a pre-norm attention and pointwise-FFN residual pair."""
        batch, channels, patches, width = tokens.shape
        if electrode_observed is None:
            electrode_observed = torch.ones(
                batch, channels, patches, dtype=torch.bool, device=tokens.device
            )
        elif electrode_observed.shape == (batch, channels):
            electrode_observed = electrode_observed.to(device=tokens.device, dtype=torch.bool)
            electrode_observed = electrode_observed.unsqueeze(-1).expand(-1, -1, patches)
        elif electrode_observed.shape == (batch, channels, patches):
            electrode_observed = electrode_observed.to(device=tokens.device, dtype=torch.bool)
        else:
            raise ValueError("electrode observation mask must have shape [B, C] or [B, C, N]")
        normalized = self.attention_norm(tokens)
        sequence_observed = electrode_observed.reshape(batch * channels, patches)
        key_padding_mask = ~sequence_observed
        # MultiheadAttention produces NaNs when every key is padded. Let such
        # sequences attend only for the computation, then zero their queries.
        fully_padded = key_padding_mask.all(dim=1)
        if fully_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_padded] = False
        temporal_input = normalized.reshape(batch * channels, patches, width)
        temporal_output, _ = self.temporal_attention(
            temporal_input,
            temporal_input,
            temporal_input,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        temporal_output = temporal_output.reshape(batch, channels, patches, width)
        token_observed = electrode_observed.unsqueeze(-1).to(temporal_output.dtype)
        temporal_output = temporal_output * token_observed
        ring_output = self.ring_attention(normalized, electrode_observed=electrode_observed)
        weights = self.fusion_weights(normalized)
        fused = weights[..., :1] * temporal_output + weights[..., 1:] * ring_output
        tokens = (tokens + fused) * token_observed
        return (tokens + self.ffn(self.ffn_norm(tokens))) * token_observed


class RingCrissCrossEncoder(nn.Module):
    """Encode a circular electrode ring into equivariant and pooled motion tokens."""

    def __init__(
        self,
        d_model: int,
        depth: int,
        heads: int,
        patch_samples: int,
        channels: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0 to preserve train-mode circular equivariance")
        if channels < 2:
            raise ValueError("channels must be at least 2")

        self.d_model = d_model
        self.channels = channels
        self.patch_samples = patch_samples
        # Encode every electrode through the same single-channel encoder.  This
        # avoids any channel-index parameters before circular attention.
        self.patch_encoder = MultiScalePatchEncoder(
            channels=1, d_model=d_model, patch_samples=patch_samples
        )
        self.blocks = nn.ModuleList(
            RingCrissCrossBlock(d_model, heads, channels=channels)
            for _ in range(depth)
        )
        # Learnable placeholder for tokens the model must predict rather than
        # read.  It is shared across electrodes, so it carries no channel index
        # and the encoder stays equivariant under circular shifts.  Keeping it
        # distinct from the observation mask is what makes masked-latent
        # distillation possible: a masked token stays *observed* (it attends and
        # keeps a non-zero output) while its content is replaced.
        self.mask_token = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        emg: torch.Tensor,
        electrode_observed: torch.Tensor | None = None,
        *,
        observation_mask: torch.Tensor | None = None,
        electrode_observation_mask: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``z_eq`` [B, C, N, D] and masked-mean ``z_motion`` [B, N, D].

        Observation masks may describe electrodes as ``[B, C]``, patches as
        ``[B, C, N]``, or individual input samples as ``[B, T, C]``.  Electrode
        and patch masks are broadcast to input samples.  A sample mask makes a
        patch valid when at least one sample in that patch is observed; masked
        samples are zeroed before patch encoding in every case.

        ``patch_mask`` [B, C, N] marks tokens the model should *predict*.  Those
        tokens stay observed -- they attend and are attended to, and their output
        stays non-zero -- but their patch embedding is replaced by the learnable
        ``mask_token``.  Pass ``patch_mask`` only together with an observation
        mask that marks the same positions observed, otherwise the observation
        mask wins and the output is zeroed.
        """
        masks = [
            mask
            for mask in (electrode_observed, observation_mask, electrode_observation_mask)
            if mask is not None
        ]
        if len(masks) > 1:
            raise ValueError("pass only one electrode observation mask")
        if emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        batch, samples, channels = emg.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if samples < self.patch_samples or samples % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")

        patches = samples // self.patch_samples
        sample_observed, observed = self._observation_masks(
            masks[0] if masks else None, batch, samples, patches, emg.device
        )
        masked_emg = emg * sample_observed.to(emg.dtype)
        channel_major = masked_emg.transpose(1, 2).reshape(batch * channels, samples, 1)
        tokens = self.patch_encoder(channel_major).reshape(batch, channels, patches, self.d_model)
        if patch_mask is not None:
            if patch_mask.dtype is not torch.bool or patch_mask.shape != (batch, channels, patches):
                raise ValueError("patch_mask must be boolean with shape [batch, channels, patches]")
            tokens = torch.where(
                patch_mask.to(device=tokens.device).unsqueeze(-1),
                self.mask_token.to(tokens.dtype).view(1, 1, 1, self.d_model),
                tokens,
            )
        for block in self.blocks:
            tokens = block(tokens, electrode_observed=observed)

        pooled_weights = observed.unsqueeze(-1).to(tokens.dtype)
        z_motion = (tokens * pooled_weights).sum(dim=1) / pooled_weights.sum(dim=1).clamp_min(1.0)
        return {"z_eq": tokens, "z_motion": z_motion}

    def _observation_masks(
        self,
        mask: torch.Tensor | None,
        batch: int,
        samples: int,
        patches: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            sample_observed = torch.ones(batch, samples, self.channels, dtype=torch.bool, device=device)
        elif mask.shape == (batch, self.channels):
            sample_observed = mask.to(device=device, dtype=torch.bool).unsqueeze(1).expand(-1, samples, -1)
        elif mask.shape == (batch, self.channels, patches):
            patch_observed = mask.to(device=device, dtype=torch.bool)
            sample_observed = patch_observed.permute(0, 2, 1).repeat_interleave(self.patch_samples, dim=1)
        elif mask.shape == (batch, samples, self.channels):
            sample_observed = mask.to(device=device, dtype=torch.bool)
        else:
            raise ValueError("electrode observation mask must have shape [B, C], [B, C, N], or [B, T, C]")
        patch_observed = sample_observed.reshape(
            batch, patches, self.patch_samples, self.channels
        ).any(dim=2).permute(0, 2, 1)
        return sample_observed, patch_observed
