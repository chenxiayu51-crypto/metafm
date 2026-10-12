import torch

from semg_fm.models.patch_encoder import MultiScalePatchEncoder


def test_patch_encoder_preserves_sixteen_channel_axis():
    encoder = MultiScalePatchEncoder(channels=16, d_model=32, patch_samples=40)

    out = encoder(torch.randn(2, 400, 16))

    assert out.shape == (2, 16, 10, 32)


def test_patch_encoder_uses_finite_spectral_features():
    encoder = MultiScalePatchEncoder(channels=16, d_model=16, patch_samples=40)

    out = encoder(torch.zeros(1, 80, 16))

    assert torch.isfinite(out).all()
