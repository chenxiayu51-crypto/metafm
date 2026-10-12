"""Numerically safe vector quantization used by the AEMG-like baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with transparent codebook diagnostics.

    ``valid_mask`` excludes padded or unobserved temporal tokens from both the
    nearest-code search and codebook statistics.  Invalid tokens are returned
    as exact zeros and use index ``-1`` so downstream metrics cannot mistake
    padding for a learned discrete code.
    """

    def __init__(self, dim: int, codebook_size: int, commitment_cost: float = 0.25) -> None:
        super().__init__()
        if dim < 1 or codebook_size < 2:
            raise ValueError("dim must be positive and codebook_size must be at least 2")
        if commitment_cost < 0.0:
            raise ValueError("commitment_cost must be non-negative")
        self.dim = int(dim)
        self.codebook_size = int(codebook_size)
        self.commitment_cost = float(commitment_cost)
        self.codebook = nn.Embedding(self.codebook_size, self.dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(
        self, values: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Quantize ``[B,N,D]`` values and return loss plus usage statistics."""
        if values.ndim != 3 or values.shape[-1] != self.dim:
            raise ValueError(f"values must have shape [B,N,{self.dim}]")
        if not torch.isfinite(values).all():
            raise ValueError("values must be finite")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        if valid_mask.shape != values.shape[:2] or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be boolean [B,N]")
        valid_mask = valid_mask.to(values.device)

        indices = torch.full(values.shape[:2], -1, dtype=torch.long, device=values.device)
        zero = values.sum() * 0.0
        usage = torch.zeros(self.codebook_size, dtype=values.dtype, device=values.device)
        if not valid_mask.any():
            return {
                "quantized": torch.zeros_like(values),
                "loss": zero,
                "indices": indices,
                "usage": usage,
                "perplexity": zero.detach(),
                "active_codes": torch.zeros((), dtype=torch.long, device=values.device),
            }

        flat_values = values[valid_mask]
        # Squared Euclidean distance is evaluated without materializing a
        # [B,N,K,D] tensor, which keeps a larger codebook practical on A100s.
        codebook = self.codebook.weight
        distances = (
            flat_values.square().sum(dim=-1, keepdim=True)
            + codebook.square().sum(dim=-1).unsqueeze(0)
            - 2.0 * flat_values @ codebook.transpose(0, 1)
        )
        selected = distances.argmin(dim=-1)
        chosen = self.codebook(selected)
        indices[valid_mask] = selected

        commitment = F.mse_loss(flat_values, chosen.detach())
        codebook_loss = F.mse_loss(chosen, flat_values.detach())
        loss = codebook_loss + self.commitment_cost * commitment
        # The straight-through path updates the encoder from the reconstruction
        # objective while the explicit codebook term above updates embeddings.
        selected_st = flat_values + (chosen - flat_values).detach()
        quantized = torch.zeros_like(values)
        quantized[valid_mask] = selected_st

        counts = torch.bincount(selected, minlength=self.codebook_size).to(values.dtype)
        usage = counts / counts.sum().clamp_min(1.0)
        nonzero = usage > 0
        entropy = -(usage[nonzero] * usage[nonzero].log()).sum()
        return {
            "quantized": quantized,
            "loss": loss,
            "indices": indices,
            "usage": usage,
            "perplexity": entropy.exp(),
            "active_codes": nonzero.sum().to(torch.long),
        }


__all__ = ["VectorQuantizer"]
