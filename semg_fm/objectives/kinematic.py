"""Masked supervision for optional kinematic labels."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def kinematic_loss(prediction: torch.Tensor, target: torch.Tensor | None, target_mask: torch.Tensor | None) -> torch.Tensor:
    """Return SmoothL1 on valid labels, or a differentiable zero when absent."""
    if not isinstance(prediction, torch.Tensor) or not prediction.is_floating_point():
        raise ValueError("prediction must be a floating-point tensor")
    if target is None or target_mask is None:
        return prediction.reshape(-1)[:0].sum()
    if not isinstance(target, torch.Tensor) or target.shape != prediction.shape:
        raise ValueError("target must share prediction's shape")
    if not target.is_floating_point() or target_mask.dtype is not torch.bool:
        raise ValueError("target must be floating point and target_mask must be boolean")
    valid = _expand_target_mask(target_mask, prediction.shape)
    if not valid.any():
        return prediction.reshape(-1)[:0].sum()
    valid_prediction, valid_target = prediction[valid], target[valid]
    if not torch.isfinite(valid_prediction).all() or not torch.isfinite(valid_target).all():
        raise ValueError("valid prediction and target values must be finite")
    return F.smooth_l1_loss(valid_prediction, valid_target)


def _expand_target_mask(mask: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if mask.shape == target_shape:
        return mask
    if mask.shape == target_shape[:-1]:
        return mask.unsqueeze(-1).expand(target_shape)
    raise ValueError("target_mask must match target or target without its final feature axis")
