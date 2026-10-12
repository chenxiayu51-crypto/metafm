"""Behavioral contracts for the CBraMod-like and AEMG-like baselines.

These compact models reproduce comparison mechanisms, not the authors' exact
training stacks.  The tests therefore make their ordinary channel ordering,
mask handling, reconstruction, and vector-quantizer behavior explicit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from semg_fm.models.aemg import AEMGEncoder
from semg_fm.models.cbramod import CBraModEncoder, CBraModPretrainingModel
from semg_fm.objectives.vector_quantization import VectorQuantizer
from semg_fm.registry import build_model, model_from_config


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_cbramod_exposes_ordinary_channel_tokens_and_masked_waveform_reconstruction():
    model = CBraModEncoder(d_model=12, depth=1, heads=3, patch_samples=20, dropout=0.0)
    reconstruction_mask = torch.zeros(2, 5, 16, dtype=torch.bool)
    reconstruction_mask[:, 1, :4] = True

    outputs = model(torch.randn(2, 100, 16), reconstruction_mask=reconstruction_mask)

    assert outputs["z_temporal"].shape == (2, 5, 12)
    assert outputs["channel_tokens"].shape == (2, 5, 16, 12)
    assert outputs["reconstruction"].shape == (2, 5, 16, 20)
    assert torch.isfinite(outputs["reconstruction_loss"])
    assert outputs["reconstruction_loss"].requires_grad


def test_cbramod_ignores_masked_raw_values_and_all_invalid_windows_are_zero():
    torch.manual_seed(13)
    model = CBraModEncoder(d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0).eval()
    emg = torch.randn(1, 100, 16)
    observed = torch.ones(1, 100, 16, dtype=torch.bool)
    observed[:, 20:40, 5] = False
    changed = emg.masked_fill(~observed, 99_999.0)

    reference = model(emg, observation_mask=observed)
    result = model(changed, observation_mask=observed)
    empty = model(emg, valid_mask=torch.zeros(1, 100, dtype=torch.bool))

    torch.testing.assert_close(result["z_temporal"], reference["z_temporal"])
    assert torch.equal(empty["z_temporal"], torch.zeros_like(empty["z_temporal"]))
    assert torch.equal(empty["pooled"], torch.zeros_like(empty["pooled"]))


def test_cbramod_wrapper_accepts_generator_with_an_explicit_reconstruction_mask():
    encoder = CBraModEncoder(d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0)
    model = CBraModPretrainingModel(encoder=encoder)
    explicit_mask = torch.zeros(1, 5, 16, dtype=torch.bool)
    explicit_mask[:, 0, 0] = True

    outputs = model(
        torch.randn(1, 100, 16),
        reconstruction_mask=explicit_mask,
        generator=torch.Generator().manual_seed(7),
    )

    assert torch.equal(outputs["reconstruction_mask"], explicit_mask)
    assert torch.isfinite(outputs["pretrain_loss"])


def test_vector_quantizer_reports_finite_usage_and_handles_all_invalid_tokens():
    quantizer = VectorQuantizer(dim=6, codebook_size=8, commitment_cost=0.25)
    values = torch.randn(2, 4, 6, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])

    outputs = quantizer(values, valid_mask=valid)
    outputs["loss"].backward()
    empty = quantizer(torch.randn(1, 3, 6), valid_mask=torch.zeros(1, 3, dtype=torch.bool))

    assert outputs["quantized"].shape == values.shape
    assert torch.isfinite(outputs["loss"])
    assert torch.isfinite(outputs["perplexity"])
    assert 0.0 < outputs["perplexity"].item() <= 8.0
    assert outputs["active_codes"].item() >= 1
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.equal(empty["quantized"], torch.zeros_like(empty["quantized"]))
    assert empty["loss"].item() == 0.0
    assert empty["active_codes"].item() == 0
    assert torch.equal(empty["indices"], torch.full_like(empty["indices"], -1))


def test_aemg_quantizes_masked_temporal_tokens_and_exposes_codebook_metrics():
    model = AEMGEncoder(
        d_model=8, depth=1, heads=2, patch_samples=20, codebook_size=8, dropout=0.0
    )
    outputs = model(torch.randn(2, 100, 16))

    assert outputs["z_temporal"].shape == (2, 5, 8)
    assert outputs["quantized"].shape == (2, 5, 8)
    assert outputs["code_indices"].shape == (2, 5)
    assert torch.isfinite(outputs["vq_loss"])
    assert torch.isfinite(outputs["codebook_perplexity"])
    assert outputs["codebook_active_codes"].item() >= 1


@pytest.mark.parametrize("model_name", ["cbramod", "aemg"])
def test_cbramod_and_aemg_compose_through_hydra_and_registry(model_name: str):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="baseline", overrides=[f"model={model_name}"])

    model = model_from_config(dict(cfg.model))
    direct = build_model(str(cfg.model.name), **{key: value for key, value in cfg.model.items() if key != "name"})

    assert type(model) is type(direct)
    assert cfg.model.channels == 16
