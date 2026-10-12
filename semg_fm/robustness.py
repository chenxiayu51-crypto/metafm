"""In-memory evaluation-time robustness transforms for sEMG windows.

Transforms operate on already-loaded batches and never modify HDF5 files.
Each returns ``(emg, sample_mask, observation_mask)`` where ``sample_mask``
is ``[B, T]`` and ``observation_mask`` is ``[B, T, C]`` or ``None``; the
downstream module folds both into the backbone call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from semg_fm.data.augment import (
    additive_noise,
    circular_shift,
    contiguous_channel_dropout,
    gain_drift,
    time_block_dropout,
)


def _seed_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def apply_robustness(
    emg: torch.Tensor,
    sample_mask: torch.Tensor,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply one configured perturbation to a batch of ``[B, T, 16]`` windows."""
    name = str((config or {}).get("name", "none"))
    if config is None or name in {"", "none"}:
        return emg, sample_mask, None
    seed = int(config.get("seed", 0))
    if name == "channel_rotation":
        shift = int(config.get("shift", 1))
        if not 1 <= shift < emg.shape[-1]:
            raise ValueError("channel_rotation shift must be in [1, 15]")
        return circular_shift(emg, shift), sample_mask, None
    if name == "channel_dropout":
        count = int(config.get("count", 1))
        raw_start = config.get("start")
        if raw_start is None:
            start = torch.randint(emg.shape[-1], (1,), generator=_seed_generator(seed)).item()
        else:
            start = int(raw_start)
        augmented, _missing, observation = contiguous_channel_dropout(emg, start, count)
        return augmented, sample_mask, observation
    if name == "gain_drift":
        start_gain = float(config.get("start_gain", 0.9))
        end_gain = float(config.get("end_gain", 1.1))
        augmented, _observed = gain_drift(emg, start_gain=start_gain, end_gain=end_gain)
        return augmented, sample_mask, None
    if name == "additive_noise":
        std = float(config.get("std", 0.1))
        augmented, _observed = additive_noise(emg, std, generator=_seed_generator(seed))
        return augmented, sample_mask, None
    if name == "temporal_loss":
        count = int(config.get("count", 10))
        start = int(config.get("start", 0))
        augmented, observation = time_block_dropout(emg, start, count)
        sample_observed = sample_mask & observation.all(dim=-1)
        return augmented, sample_observed, None
    raise ValueError(f"unknown robustness transform {name!r}")


__all__ = ["apply_robustness"]
