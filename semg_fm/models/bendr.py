"""BENDR-like continuous/contextual baseline components.

The implementation preserves the two mechanisms needed for a fair comparison:
a continuous convolutional target encoder and a masked contextual transformer.
It is a compact core-mechanism reproduction rather than an official BENDR
implementation.  In particular, it intentionally uses no ordinary temporal
position embedding and no circular-electrode bias.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .supervised_tcn import _resolve_channels, observed_samples, token_observation_mask


def _random_uniform(
    shape: tuple[int, ...], *, device: torch.device, generator: torch.Generator | None
) -> torch.Tensor:
    """Sample uniformly even when a caller supplies a CPU generator for CUDA work."""
    if generator is None:
        return torch.rand(shape, device=device)
    generator_device = torch.device(generator.device)
    values = torch.rand(shape, device=generator_device, generator=generator)
    return values.to(device=device)


def sample_bendr_mask(
    token_mask: torch.Tensor,
    probability: float,
    span: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample forward contiguous corruption spans only at valid temporal tokens.

    The start probability is explicit rather than tied to a hidden global
    schedule, which lets comparison runs reproduce the masking regime from
    their Hydra configuration.  Only observed tokens may start a span, so an
    internal padding/artifact gap cannot corrupt a valid successor.  Spans are
    clipped at sequence end rather than wrapped: time is a line, unlike the
    16-electrode ring.
    """
    if token_mask.ndim != 2 or token_mask.dtype is not torch.bool:
        raise ValueError("token_mask must be boolean [B,N]")
    if not 0.0 <= probability <= 1.0 or span < 1:
        raise ValueError("probability must be in [0, 1] and span must be positive")
    starts = (
        _random_uniform(tuple(token_mask.shape), device=token_mask.device, generator=generator)
        < probability
    ) & token_mask
    corrupted = starts.clone()
    active = starts
    for _ in range(1, min(span, token_mask.shape[1])):
        # Propagate only active spans.  Re-applying ``token_mask`` at each step
        # makes an invalid token a hard boundary rather than a transparent hole.
        next_active = torch.zeros_like(token_mask)
        next_active[:, 1:] = active[:, :-1]
        active = next_active & token_mask
        corrupted |= active
    return corrupted


def same_sequence_negative_indices(
    token_mask: torch.Tensor,
    num_negatives: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample valid negative positions from the same record, never the positive.

    The result is ``(indices, usable)`` with shape ``[B,N,K]``.  ``usable`` is
    false for padding slots or when a sequence has fewer than two valid
    positions.  Consumers must use that flag rather than accidentally treating
    the harmless zero-filled index padding as a real negative.
    """
    if token_mask.ndim != 2 or token_mask.dtype is not torch.bool:
        raise ValueError("token_mask must be boolean [B,N]")
    if num_negatives < 1:
        raise ValueError("num_negatives must be positive")
    batch, positions = token_mask.shape
    indices = torch.zeros((batch, positions, num_negatives), dtype=torch.long, device=token_mask.device)
    usable = torch.zeros_like(indices, dtype=torch.bool)
    if positions == 0:
        return indices, usable

    candidate_positions = torch.arange(positions, device=token_mask.device)
    # Candidate rows are indexed by (record, positive position, candidate
    # position); the first dimension is never flattened, preventing cross-record
    # negatives even when a batch contains windows from different participants.
    candidates = token_mask[:, None, :] & (candidate_positions[None, None, :] != candidate_positions[None, :, None])
    take = min(num_negatives, positions)
    scores = _random_uniform((batch, positions, positions), device=token_mask.device, generator=generator)
    scores = scores.masked_fill(~candidates, float("-inf"))
    selected = scores.topk(take, dim=-1).indices
    selected_usable = candidates.gather(dim=-1, index=selected)
    indices[..., :take] = selected
    usable[..., :take] = selected_usable
    return indices, usable


class BendrLikeContrastiveLoss(nn.Module):
    """InfoNCE over masked positions with strictly within-record negatives."""

    def __init__(self, num_negatives: int = 20, temperature: float = 0.1) -> None:
        super().__init__()
        if num_negatives < 1 or temperature <= 0:
            raise ValueError("num_negatives and temperature must be positive")
        self.num_negatives = int(num_negatives)
        self.temperature = float(temperature)
        # Diagnostic-only state for tests/logging; it is not learned state and
        # therefore intentionally excluded from checkpoints.
        self.last_usable_queries = 0

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        prediction_mask: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return finite contrastive loss or a differentiable zero with no negatives."""
        if context.ndim != 3 or target.shape != context.shape:
            raise ValueError("context and target must share shape [B,N,D]")
        if prediction_mask.shape != context.shape[:2] or prediction_mask.dtype is not torch.bool:
            raise ValueError("prediction_mask must be boolean [B,N]")
        if token_mask is None:
            token_mask = torch.ones_like(prediction_mask)
        if token_mask.shape != context.shape[:2] or token_mask.dtype is not torch.bool:
            raise ValueError("token_mask must be boolean [B,N]")
        if not torch.isfinite(context).all() or not torch.isfinite(target).all():
            raise ValueError("context and target must be finite")

        indices, negative_usable = same_sequence_negative_indices(
            token_mask, self.num_negatives, generator=generator
        )
        # A query is only eligible if it was corrupted, valid, and has at least
        # one true same-record negative.  One-token records are skipped rather
        # than using their positive target as a fake negative.
        usable_queries = prediction_mask & token_mask & negative_usable.any(dim=-1)
        self.last_usable_queries = int(usable_queries.sum().item())
        if not usable_queries.any():
            return context.sum() * 0.0

        computation_dtype = (
            torch.float32 if context.dtype in (torch.float16, torch.bfloat16) else context.dtype
        )
        queries = F.normalize(context.to(computation_dtype), dim=-1, eps=torch.finfo(computation_dtype).eps)
        targets = F.normalize(target.detach().to(computation_dtype), dim=-1, eps=torch.finfo(computation_dtype).eps)
        batch_index = torch.arange(context.shape[0], device=context.device)[:, None, None]
        negative_targets = targets[batch_index, indices]
        positive_logits = (queries * targets).sum(dim=-1, keepdim=True)
        negative_logits = (queries.unsqueeze(-2) * negative_targets).sum(dim=-1)
        # Invalid padded slots receive the minimum finite logit.  Cross entropy
        # then remains finite and ignores them relative to the true positives.
        negative_logits = negative_logits.masked_fill(
            ~negative_usable, torch.finfo(computation_dtype).min
        )
        logits = torch.cat((positive_logits, negative_logits), dim=-1) / self.temperature
        labels = torch.zeros(int(usable_queries.sum().item()), dtype=torch.long, device=context.device)
        return F.cross_entropy(logits[usable_queries], labels)


class BendrEncoder(nn.Module):
    """Continuous convolutional target encoder plus masked temporal Transformer.

    The returned ``target`` comes from the uncorrupted continuous encoder;
    ``context`` comes from a Transformer that sees a learnable mask token at
    controllably corrupted positions.  No positional embeddings are allocated,
    so this baseline does not accidentally acquire the proposed model's ring
    geometry or an extra learned temporal coordinate system.
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
        mask_probability: float = 0.0,
        mask_span: int = 1,
    ) -> None:
        super().__init__()
        channels = _resolve_channels(channels, in_channels)
        if d_model < 1 or depth < 1 or heads < 1 or patch_samples < 1 or channels < 1:
            raise ValueError("d_model, depth, heads, patch_samples, and channels must be positive")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= mask_probability <= 1.0 or mask_span < 1:
            raise ValueError("mask_probability must be in [0, 1] and mask_span must be positive")
        self.d_model = d_model
        self.depth = depth
        self.heads = heads
        self.patch_samples = patch_samples
        self.channels = channels
        self.mask_probability = float(mask_probability)
        self.mask_span = int(mask_span)
        self.continuous_encoder = nn.Conv1d(
            channels, d_model, kernel_size=patch_samples, stride=patch_samples, bias=False
        )
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # ``enable_nested_tensor=False`` keeps mask behavior explicit and
        # deterministic across PyTorch versions for padded H5 windows.
        self.contextualizer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)

    def forward(
        self,
        emg: torch.Tensor,
        observation_mask: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        corruption_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return continuous targets and contextual features for one EMG batch."""
        self._validate_emg(emg)
        observed = observed_samples(emg, (observation_mask, valid_mask, sample_mask))
        token_mask = token_observation_mask(observed, self.patch_samples)
        target = self.continuous_encoder((emg * observed.to(emg.dtype)).transpose(1, 2)).transpose(1, 2)
        weights = token_mask.unsqueeze(-1).to(target.dtype)
        target = target * weights

        if corruption_mask is None:
            corruption_mask = sample_bendr_mask(
                token_mask, self.mask_probability, self.mask_span, generator=generator
            )
        elif corruption_mask.shape != token_mask.shape or corruption_mask.dtype is not torch.bool:
            raise ValueError("corruption_mask must be boolean [B,N]")
        else:
            corruption_mask = corruption_mask.to(device=emg.device) & token_mask
        contextual_input = torch.where(
            corruption_mask.unsqueeze(-1), self.mask_token.view(1, 1, -1), target
        ) * weights

        key_padding_mask = ~token_mask
        # PyTorch attention returns NaNs if all keys in a row are padded.  Let
        # those rows compute with zero input, then mask every output back to
        # zero, which preserves the semantic all-invalid representation.
        fully_invalid = key_padding_mask.all(dim=1)
        if fully_invalid.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_invalid] = False
        context = self.contextualizer(contextual_input, src_key_padding_mask=key_padding_mask) * weights
        pooled = (context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "context": context,
            "target": target,
            "z_temporal": context,
            "z_motion": context,
            "pooled": pooled,
            "token_mask": token_mask,
            "corruption_mask": corruption_mask,
        }

    def _validate_emg(self, emg: Any) -> None:
        if not isinstance(emg, torch.Tensor) or emg.ndim != 3:
            raise ValueError("emg must have shape [batch, time, channels]")
        if emg.shape[-1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {emg.shape[-1]}")
        if emg.shape[1] < self.patch_samples or emg.shape[1] % self.patch_samples:
            raise ValueError("time length must be a positive multiple of patch_samples")


class BendrPretrainingModel(nn.Module):
    """Convenience wrapper that attaches the BENDR-like contrastive objective."""

    def __init__(self, encoder: BendrEncoder | None = None, *, num_negatives: int = 20, temperature: float = 0.1, **encoder_kwargs: Any) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else BendrEncoder(**encoder_kwargs)
        self.contrastive = BendrLikeContrastiveLoss(num_negatives=num_negatives, temperature=temperature)

    def forward(self, emg: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Encode ``emg`` and append a stable within-record contrastive loss."""
        outputs = self.encoder(emg, **kwargs)
        outputs["contrastive_loss"] = self.contrastive(
            outputs["context"],
            outputs["target"],
            outputs["corruption_mask"],
            outputs["token_mask"],
            generator=kwargs.get("generator"),
        )
        return outputs


# Both spellings remain public because manuscripts typically use uppercase
# BENDR, while the pre-existing plan used the ``BendrEncoder`` Python style.
BENDRLikeEncoder = BendrEncoder
BendrContrastiveLoss = BendrLikeContrastiveLoss

__all__ = [
    "BENDRLikeEncoder",
    "BendrContrastiveLoss",
    "BendrEncoder",
    "BendrLikeContrastiveLoss",
    "BendrPretrainingModel",
    "same_sequence_negative_indices",
    "sample_bendr_mask",
]
