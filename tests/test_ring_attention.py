import torch

from semg_fm.models.ring_attention import RingMultiheadAttention, signed_ring_offsets


def test_ring_offsets_make_endpoints_neighbors():
    offsets = signed_ring_offsets(16)

    assert abs(int(offsets[0, 15])) == 1
    assert abs(int(offsets[15, 0])) == 1
    assert int(offsets[0, 8]) in {-8, 8}


def test_ring_attention_masks_missing_electrode_without_nonfinite_values():
    attention = RingMultiheadAttention(d_model=8, num_heads=2, channels=16)
    tokens = torch.randn(2, 16, 3, 8)
    observed = torch.ones(2, 16, dtype=torch.bool)
    observed[:, 5] = False

    out, weights = attention(tokens, electrode_observed=observed, return_attention=True)

    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    assert torch.equal(out[:, 5], torch.zeros_like(out[:, 5]))
    assert torch.equal(weights[..., 5], torch.zeros_like(weights[..., 5]))
