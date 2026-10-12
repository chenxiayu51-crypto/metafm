import torch
import pytest

from semg_fm.models.ring_criss_cross import RingCrissCrossBlock, RingCrissCrossEncoder


def _encoder() -> RingCrissCrossEncoder:
    return RingCrissCrossEncoder(
        d_model=16,
        depth=2,
        heads=4,
        patch_samples=40,
        channels=16,
        dropout=0.0,
    )


def test_ring_criss_cross_returns_channel_equivariant_and_motion_tokens():
    encoder = _encoder()

    outputs = encoder(torch.randn(2, 400, 16))

    assert outputs["z_eq"].shape == (2, 16, 10, 16)
    assert outputs["z_motion"].shape == (2, 10, 16)


def test_ring_criss_cross_is_circularly_equivariant_in_eval_mode():
    torch.manual_seed(7)
    encoder = _encoder().eval()
    emg = torch.randn(2, 400, 16)

    reference = encoder(emg)
    shifted = encoder(torch.roll(emg, shifts=5, dims=-1))

    torch.testing.assert_close(
        shifted["z_eq"], torch.roll(reference["z_eq"], shifts=5, dims=1), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(shifted["z_motion"], reference["z_motion"], atol=1e-5, rtol=1e-4)


def test_ring_criss_cross_all_missing_mask_has_finite_pooled_outputs():
    encoder = _encoder()
    observed = torch.zeros(2, 16, dtype=torch.bool)

    outputs = encoder(torch.randn(2, 400, 16), electrode_observed=observed)

    assert torch.isfinite(outputs["z_eq"]).all()
    assert torch.isfinite(outputs["z_motion"]).all()
    assert torch.equal(outputs["z_motion"], torch.zeros_like(outputs["z_motion"]))


def test_ring_criss_cross_fusion_gate_is_a_two_way_distribution():
    encoder = _encoder()
    tokens = torch.randn(2, 16, 10, 16)

    gate = encoder.blocks[0].fusion_weights(tokens)

    assert gate.shape == (2, 16, 10, 2)
    torch.testing.assert_close(gate.sum(dim=-1), torch.ones(2, 16, 10))


def test_ring_criss_cross_backpropagates_through_encoder_parameters():
    encoder = _encoder()
    emg = torch.randn(2, 400, 16, requires_grad=True)

    outputs = encoder(emg)
    (outputs["z_eq"].square().mean() + outputs["z_motion"].square().mean()).backward()

    assert emg.grad is not None
    assert torch.isfinite(emg.grad).all()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


@pytest.mark.parametrize("mask_kind", ["electrode", "patch", "sample"])
def test_ring_criss_cross_ignores_values_at_masked_samples_for_all_mask_shapes(mask_kind: str):
    encoder = _encoder().eval()
    torch.manual_seed(11)
    emg = torch.randn(2, 400, 16)
    if mask_kind == "electrode":
        observed = torch.ones(2, 16, dtype=torch.bool)
        observed[:, 3] = False
        sample_observed = observed[:, None, :].expand(-1, 400, -1)
    elif mask_kind == "patch":
        observed = torch.ones(2, 16, 10, dtype=torch.bool)
        observed[:, 3, 4] = False
        sample_observed = observed.permute(0, 2, 1).repeat_interleave(40, dim=1)
    else:
        observed = torch.ones(2, 400, 16, dtype=torch.bool)
        observed[:, 160:180, 3] = False
        sample_observed = observed

    modified = emg.masked_fill(~sample_observed, 10_000.0)
    reference = encoder(emg, electrode_observed=observed)
    changed = encoder(modified, electrode_observed=observed)

    torch.testing.assert_close(changed["z_eq"], reference["z_eq"], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(changed["z_motion"], reference["z_motion"], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mask_kind", ["electrode", "patch", "sample"])
def test_ring_criss_cross_returns_zero_equivariant_tokens_for_invalid_patches(mask_kind: str):
    encoder = _encoder()
    if mask_kind == "electrode":
        observed = torch.ones(1, 16, dtype=torch.bool)
        observed[:, 4] = False
        expected_invalid = observed[:, :, None].expand(-1, -1, 10)
    elif mask_kind == "patch":
        observed = torch.ones(1, 16, 10, dtype=torch.bool)
        observed[:, 4, 6] = False
        expected_invalid = observed
    else:
        observed = torch.ones(1, 400, 16, dtype=torch.bool)
        observed[:, 240:280, 4] = False
        expected_invalid = observed.reshape(1, 10, 40, 16).any(dim=2).permute(0, 2, 1)

    outputs = encoder(torch.randn(1, 400, 16), electrode_observed=observed)

    assert torch.equal(outputs["z_eq"][~expected_invalid], torch.zeros_like(outputs["z_eq"][~expected_invalid]))


def test_ring_criss_cross_is_circularly_equivariant_in_train_mode_without_dropout():
    torch.manual_seed(13)
    encoder = _encoder().train()
    emg = torch.randn(2, 400, 16)

    reference = encoder(emg)
    shifted = encoder(torch.roll(emg, shifts=7, dims=-1))

    torch.testing.assert_close(
        shifted["z_eq"], torch.roll(reference["z_eq"], shifts=7, dims=1), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(shifted["z_motion"], reference["z_motion"], atol=1e-5, rtol=1e-4)


def test_ring_criss_cross_rejects_nonzero_ordinary_dropout():
    with pytest.raises(ValueError, match="dropout"):
        RingCrissCrossEncoder(d_model=16, depth=1, heads=4, patch_samples=40, dropout=0.1)


def test_ring_criss_cross_block_keeps_optional_observation_mask_optional():
    block = RingCrissCrossBlock(d_model=16, heads=4)

    outputs = block(torch.randn(2, 16, 10, 16))

    assert outputs.shape == (2, 16, 10, 16)


def test_ring_criss_cross_block_accepts_electrode_level_observation_masks():
    block = RingCrissCrossBlock(d_model=16, heads=4)
    observed = torch.ones(2, 16, dtype=torch.bool)
    observed[:, 5] = False

    outputs = block(torch.randn(2, 16, 10, 16), electrode_observed=observed)

    assert torch.equal(outputs[:, 5], torch.zeros_like(outputs[:, 5]))
