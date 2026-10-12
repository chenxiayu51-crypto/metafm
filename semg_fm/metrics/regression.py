"""Masked continuous-target metrics for pose and wrist-angle evaluation."""

from __future__ import annotations

import torch


def masked_regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    example_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return finite MAE/RMSE totals over valid, non-padding targets.

    ``target_mask`` may be ``[B,T]`` or match the full prediction shape.  A
    zero ``example_weight`` is how DDP evaluation marks sampler-added duplicate
    records, so those examples must not influence either numerator or count.
    """
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same rank >= 2 shape")
    if target_mask.dtype is not torch.bool:
        raise ValueError("target_mask must be boolean")
    if target_mask.shape == prediction.shape[:-1]:
        target_mask = target_mask.unsqueeze(-1).expand_as(prediction)
    elif target_mask.shape != prediction.shape:
        raise ValueError("target_mask must be [B,...] without features or match prediction")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if example_weight is None:
        example_weight = torch.ones(prediction.shape[0], dtype=prediction.dtype, device=prediction.device)
    if example_weight.shape != (prediction.shape[0],):
        raise ValueError("example_weight must be [B]")
    if not torch.isfinite(example_weight).all() or (example_weight < 0).any():
        raise ValueError("example_weight must be finite and non-negative")
    weights = target_mask.to(prediction.dtype) * example_weight.to(prediction.dtype).view(
        prediction.shape[0], *([1] * (prediction.ndim - 1))
    )
    error = prediction - target
    count = weights.sum()
    denominator = count.clamp_min(1.0)
    return {
        "count": count,
        "mae": (error.abs() * weights).sum() / denominator,
        "rmse": ((error.square() * weights).sum() / denominator).sqrt(),
        "sum_absolute_error": (error.abs() * weights).sum(),
        "sum_squared_error": (error.square() * weights).sum(),
    }


__all__ = ["masked_regression_metrics"]
