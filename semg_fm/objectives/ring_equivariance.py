"""Loss for circularly equivariant fields and invariant motion tokens."""

from __future__ import annotations

import numbers

import torch
from torch.nn import functional as F


def ring_equivariance_loss(field: torch.Tensor, motion: torch.Tensor, shifted_field: torch.Tensor, shifted_motion: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    """Inverse-align a rotated field and match its pooled motion representation."""
    if field.ndim < 3 or shifted_field.shape != field.shape:
        raise ValueError("field tensors must share shape [batch, channels, ...]")
    if motion.shape != field.shape[:1] + field.shape[2:] or shifted_motion.shape != motion.shape:
        raise ValueError("motion tensors must match field after pooling its channel axis")
    if not all(torch.isfinite(value).all() for value in (field, motion, shifted_field, shifted_motion)):
        raise ValueError("field and motion tensors must be finite")
    return F.mse_loss(_inverse_align(shifted_field, shift), field) + F.mse_loss(shifted_motion, motion)


def _inverse_align(shifted_field: torch.Tensor, shift: int | torch.Tensor) -> torch.Tensor:
    if isinstance(shift, numbers.Integral):
        return torch.roll(shifted_field, shifts=-int(shift), dims=1)
    if not isinstance(shift, torch.Tensor) or shift.ndim > 1:
        raise ValueError("shift must be an integer or one integer shift per batch element")
    if shift.ndim == 0:
        return torch.roll(shifted_field, shifts=-int(shift.item()), dims=1)
    if shift.shape[0] != shifted_field.shape[0]:
        raise ValueError("per-example shifts must match the batch size")
    if shift.is_floating_point() and not torch.equal(shift, shift.round()):
        raise ValueError("shifts must be integral")
    return torch.stack([torch.roll(sample, shifts=-int(value.item()), dims=0) for sample, value in zip(shifted_field, shift)])
