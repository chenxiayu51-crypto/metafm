"""Structured masks for the 16-electrode circular token grid."""

from __future__ import annotations

import math

import torch


class MaskGenerator:
    """Compose temporal, circular-channel, block, and patch masks."""

    channels = 16

    def __init__(
        self,
        *,
        temporal_span: int = 2,
        ring_arc: int = 3,
        full_channel_probability: float = 0.1,
        block_shape: tuple[int, int] = (2, 2),
        random_patch_probability: float = 0.15,
    ) -> None:
        if temporal_span < 0 or ring_arc < 0:
            raise ValueError("temporal_span and ring_arc must be non-negative")
        if len(block_shape) != 2 or any(size < 0 for size in block_shape):
            raise ValueError("block_shape must contain two non-negative sizes")
        probabilities = (full_channel_probability, random_patch_probability)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("mask probabilities must be finite values in [0, 1]")
        self.temporal_span = temporal_span
        self.ring_arc = ring_arc
        self.full_channel_probability = full_channel_probability
        self.block_shape = block_shape
        self.random_patch_probability = random_patch_probability

    def __call__(
        self,
        batch_size: int,
        num_patches: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return a deterministic-with-generator boolean mask ``[B, 16, N]``."""
        if batch_size < 1 or num_patches < 1:
            raise ValueError("batch_size and num_patches must be positive")
        output_device = torch.device("cpu") if device is None else torch.device(device)
        sampling_device = self._sampling_device(generator)
        mask = torch.zeros((batch_size, self.channels, num_patches), dtype=torch.bool, device=sampling_device)
        for batch_index in range(batch_size):
            if self.temporal_span:
                start = self._randint(num_patches, generator, sampling_device)
                indices = (torch.arange(min(self.temporal_span, num_patches), device=sampling_device) + start) % num_patches
                mask[batch_index, :, indices] = True
            if self.ring_arc:
                start = self._randint(self.channels, generator, sampling_device)
                indices = (torch.arange(min(self.ring_arc, self.channels), device=sampling_device) + start) % self.channels
                mask[batch_index, indices, :] = True
            if self.full_channel_probability:
                full_channels = torch.rand(self.channels, generator=generator, device=sampling_device) < self.full_channel_probability
                mask[batch_index, full_channels, :] = True
            block_channels, block_patches = self.block_shape
            if block_channels and block_patches:
                channel_start = self._randint(self.channels, generator, sampling_device)
                patch_start = self._randint(num_patches, generator, sampling_device)
                channels = (torch.arange(min(block_channels, self.channels), device=sampling_device) + channel_start) % self.channels
                patches = (torch.arange(min(block_patches, num_patches), device=sampling_device) + patch_start) % num_patches
                mask[batch_index, channels[:, None], patches[None, :]] = True
            if self.random_patch_probability:
                mask[batch_index] |= torch.rand((self.channels, num_patches), generator=generator, device=sampling_device) < self.random_patch_probability
            self._ensure_both_states(mask[batch_index], generator, sampling_device)
        return mask.to(output_device)

    @staticmethod
    def _randint(upper: int, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
        return torch.randint(upper, (), generator=generator, device=device)

    @staticmethod
    def _sampling_device(generator: torch.Generator | None) -> torch.device:
        """Choose a generator-compatible device without coupling it to output."""
        if generator is None:
            return torch.device("cpu")
        try:
            return torch.device(generator.device)
        except (AttributeError, TypeError):
            return torch.device("cpu")

    @staticmethod
    def _ensure_both_states(mask: torch.Tensor, generator: torch.Generator | None, device: torch.device) -> None:
        if mask.numel() <= 1:
            return
        if not mask.any():
            mask.flatten()[MaskGenerator._randint(mask.numel(), generator, device)] = True
        elif mask.all():
            mask.flatten()[MaskGenerator._randint(mask.numel(), generator, device)] = False
