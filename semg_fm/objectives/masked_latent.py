"""Masked latent distillation objective."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_latent_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Match normalized student and detached-teacher features at masked tokens."""
    if student.ndim != 4 or teacher.shape != student.shape:
        raise ValueError("student and teacher must share shape [batch, 16, patches, features]")
    if mask.dtype is not torch.bool or mask.shape != student.shape[:-1]:
        raise ValueError("mask must be boolean with shape [batch, 16, patches]")
    if not mask.any():
        raise ValueError("mask must contain at least one masked position")
    masked_student = student[mask]
    masked_teacher = teacher.detach()[mask]
    if not torch.isfinite(masked_student).all() or not torch.isfinite(masked_teacher).all():
        raise ValueError("masked student and teacher features must be finite")
    computation_dtype = (
        torch.float32
        if masked_student.dtype in (torch.float16, torch.bfloat16)
        else masked_student.dtype
    )
    normalized_student = F.normalize(
        masked_student.to(computation_dtype), dim=-1, eps=torch.finfo(computation_dtype).eps
    )
    normalized_teacher = F.normalize(
        masked_teacher.to(computation_dtype), dim=-1, eps=torch.finfo(computation_dtype).eps
    )
    smooth_l1 = F.smooth_l1_loss(normalized_student, normalized_teacher)
    cosine = (1.0 - F.cosine_similarity(normalized_student, normalized_teacher, dim=-1)).mean()
    return smooth_l1 + cosine
