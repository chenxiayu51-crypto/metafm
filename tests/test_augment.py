import torch

from semg_fm.data.augment import (
    additive_noise,
    circular_shift,
    contiguous_channel_dropout,
    contiguous_channel_dropout_indices,
    contiguous_channel_dropout_with_mask,
    gain_drift,
    time_block_dropout,
)


def test_circular_shift_wraps_channel_fifteen_to_zero():
    emg = torch.arange(16, dtype=torch.float32).reshape(1, 16)

    shifted = circular_shift(emg, 1)

    assert shifted[0, 0].item() == 15
    assert shifted[0, 1].item() == 0


def test_channel_dropout_wraps_across_channel_boundary_and_returns_sorted_missing_channels():
    emg = torch.ones(2, 16, dtype=torch.float32)

    dropped, missing_channels, observation_mask = contiguous_channel_dropout(emg, start=15, count=2)

    assert missing_channels == [0, 15]
    assert torch.equal(dropped[:, [15, 0]], torch.zeros(2, 2))
    assert observation_mask.dtype == torch.bool
    assert observation_mask.shape == emg.shape
    assert not observation_mask[:, [15, 0]].any()
    assert observation_mask[:, 1:15].all()


def test_mask_capable_channel_dropout_marks_exactly_the_dropped_channels():
    emg = torch.ones(3, 16, dtype=torch.float32)

    dropped, missing_channels, observation_mask = contiguous_channel_dropout_with_mask(
        emg, start=15, count=2
    )

    assert missing_channels == [0, 15]
    assert torch.equal(dropped[:, missing_channels], torch.zeros(3, 2))
    assert observation_mask.dtype == torch.bool
    assert observation_mask.shape == emg.shape
    assert not observation_mask[:, missing_channels].any()
    assert observation_mask[:, 1:15].all()


def test_indices_only_channel_dropout_preserves_the_two_value_compatibility_api():
    dropped, missing_channels = contiguous_channel_dropout_indices(
        torch.ones(1, 16, dtype=torch.float32), start=15, count=2
    )

    assert missing_channels == [0, 15]
    assert torch.equal(dropped[:, missing_channels], torch.zeros(1, 2))


def test_physical_augmentations_preserve_shape_dtype_and_return_observation_masks():
    emg = torch.ones(6, 16, dtype=torch.float32)

    for augmented, observation_mask in (
        gain_drift(emg, start_gain=0.9, end_gain=1.1),
        additive_noise(emg, std=0.1, generator=torch.Generator().manual_seed(7)),
        time_block_dropout(emg, start=2, count=3),
    ):
        assert augmented.shape == emg.shape
        assert augmented.dtype == emg.dtype
        assert observation_mask.shape == emg.shape
        assert observation_mask.dtype == torch.bool

    _, time_mask = time_block_dropout(emg, start=2, count=3)
    assert not time_mask[2:5].any()
    assert time_mask[:2].all()
    assert time_mask[5:].all()
