"""Physical sEMG augmentations that keep channel geometry intact."""

from __future__ import annotations

import torch


def _validate_emg(emg: torch.Tensor) -> None:
    if not isinstance(emg, torch.Tensor) or emg.ndim < 2:
        raise ValueError("emg must be a tensor with time and channel dimensions")
    if emg.shape[-1] != 16:
        raise ValueError("emg must have 16 channels on its final axis")


def _observed_like(emg: torch.Tensor) -> torch.Tensor:
    return torch.ones(emg.shape, dtype=torch.bool, device=emg.device)


def circular_shift(emg: torch.Tensor, shift: int) -> torch.Tensor:
    """Rotate the physical electrode ring, always along the final channel axis."""
    _validate_emg(emg)
    return torch.roll(emg, shifts=int(shift), dims=-1)


def contiguous_channel_dropout_indices(
    emg: torch.Tensor, start: int, count: int
) -> tuple[torch.Tensor, list[int]]:
    """Zero a contiguous electrode run and return only its missing channel IDs."""
    _validate_emg(emg)
    if not 0 <= start < emg.shape[-1]:
        raise ValueError("start must name an existing channel")
    if not 0 <= count <= emg.shape[-1]:
        raise ValueError("count must be between zero and the number of channels")
    missing_channels = sorted((start + offset) % emg.shape[-1] for offset in range(count))
    augmented = emg.clone()
    if missing_channels:
        augmented[..., missing_channels] = 0
    return augmented, missing_channels


def contiguous_channel_dropout(
    emg: torch.Tensor, start: int, count: int
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Apply channel dropout and return its explicit boolean observation mask."""
    augmented, missing_channels = contiguous_channel_dropout_indices(emg, start, count)
    observation_mask = _observed_like(emg)
    if missing_channels:
        observation_mask[..., missing_channels] = False
    return augmented, missing_channels, observation_mask


def contiguous_channel_dropout_with_mask(
    emg: torch.Tensor, start: int, count: int
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """Compatibility alias for the canonical mask-returning channel dropout."""
    return contiguous_channel_dropout(emg, start, count)


def gain_drift(
    emg: torch.Tensor, start_gain: float = 0.9, end_gain: float = 1.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a linear gain drift over time and report all values as observed."""
    _validate_emg(emg)
    if not emg.is_floating_point():
        raise ValueError("gain drift requires floating-point emg")
    gains = torch.linspace(
        start_gain, end_gain, emg.shape[-2], dtype=emg.dtype, device=emg.device
    )
    shape = (1,) * (emg.ndim - 2) + (emg.shape[-2], 1)
    return emg * gains.reshape(shape), _observed_like(emg)


def additive_noise(
    emg: torch.Tensor, std: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add zero-mean sensor noise without hiding any observations."""
    _validate_emg(emg)
    if not emg.is_floating_point():
        raise ValueError("additive noise requires floating-point emg")
    if std < 0:
        raise ValueError("std must be non-negative")
    noise = torch.randn(emg.shape, dtype=emg.dtype, device=emg.device, generator=generator)
    return emg + noise * std, _observed_like(emg)


def time_block_dropout(
    emg: torch.Tensor, start: int, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero a contiguous time block and mark each removed sample-channel value missing."""
    _validate_emg(emg)
    time_steps = emg.shape[-2]
    if not 0 <= start <= time_steps or not 0 <= count <= time_steps - start:
        raise ValueError("time block must be contained in the emg time axis")
    augmented = emg.clone()
    observation_mask = _observed_like(emg)
    augmented[..., start : start + count, :] = 0
    observation_mask[..., start : start + count, :] = False
    return augmented, observation_mask
