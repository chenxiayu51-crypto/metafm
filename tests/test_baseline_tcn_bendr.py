"""Behavioral contracts for the supervised-TCN and BENDR-like baselines.

These tests exercise the public baseline interfaces rather than an official
implementation of either cited method.  They make masking and negative-sample
semantics explicit so later training modules can use the models safely.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from semg_fm.models.bendr import (
    BendrEncoder,
    BendrLikeContrastiveLoss,
    BendrPretrainingModel,
    same_sequence_negative_indices,
    sample_bendr_mask,
)
from semg_fm.models.supervised_tcn import SupervisedTCN
from semg_fm.registry import build_model, model_from_config


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_supervised_tcn_returns_masked_temporal_features_and_pooled_summary():
    model = SupervisedTCN(d_model=12, depth=2, patch_samples=20, channels=16, dropout=0.0)
    outputs = model.extract_features(torch.randn(2, 100, 16))

    assert outputs["z_temporal"].shape == (2, 5, 12)
    assert outputs["z_motion"].shape == (2, 5, 12)
    assert outputs["pooled"].shape == (2, 12)
    assert outputs["token_mask"].dtype is torch.bool


def test_supervised_tcn_forward_returns_the_shared_backbone_mapping():
    model = SupervisedTCN(d_model=12, depth=2, patch_samples=20, channels=16, dropout=0.0)
    outputs = model(torch.randn(2, 100, 16))

    assert isinstance(outputs, dict)
    assert outputs["z_motion"].shape == (2, 5, 12)


def test_supervised_tcn_masked_input_values_cannot_change_outputs():
    torch.manual_seed(5)
    model = SupervisedTCN(d_model=8, depth=2, patch_samples=20, dropout=0.0).eval()
    emg = torch.randn(1, 100, 16)
    observed = torch.ones(1, 100, 16, dtype=torch.bool)
    observed[:, 20:40, 3] = False
    changed = emg.masked_fill(~observed, 50_000.0)

    reference = model.extract_features(emg, observation_mask=observed)
    result = model.extract_features(changed, observation_mask=observed)

    torch.testing.assert_close(result["z_temporal"], reference["z_temporal"])
    torch.testing.assert_close(result["pooled"], reference["pooled"])


def test_supervised_tcn_all_invalid_tokens_are_finite_zeroes():
    model = SupervisedTCN(d_model=8, depth=1, patch_samples=20)
    outputs = model.extract_features(
        torch.randn(2, 100, 16), valid_mask=torch.zeros(2, 100, dtype=torch.bool)
    )

    assert torch.isfinite(outputs["z_temporal"]).all()
    assert torch.equal(outputs["z_temporal"], torch.zeros_like(outputs["z_temporal"]))
    assert torch.equal(outputs["pooled"], torch.zeros_like(outputs["pooled"]))


def test_bendr_exposes_uncontextualized_targets_and_masked_context():
    model = BendrEncoder(d_model=12, depth=1, heads=3, patch_samples=20, dropout=0.0)
    corruption = torch.tensor([[False, True, False, False, False], [True, False, False, False, False]])

    outputs = model(torch.randn(2, 100, 16), corruption_mask=corruption)

    assert outputs["context"].shape == outputs["target"].shape == (2, 5, 12)
    assert outputs["z_temporal"].shape == (2, 5, 12)
    assert torch.equal(outputs["corruption_mask"], corruption)
    assert torch.isfinite(outputs["context"]).all()


def test_bendr_masked_signal_values_cannot_change_target_or_context():
    torch.manual_seed(8)
    model = BendrEncoder(d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0).eval()
    emg = torch.randn(1, 100, 16)
    observed = torch.ones(1, 100, dtype=torch.bool)
    observed[:, 40:60] = False
    changed = emg.clone()
    changed[:, 40:60] = -90_000.0

    reference = model(emg, observation_mask=observed)
    result = model(changed, observation_mask=observed)

    torch.testing.assert_close(result["target"], reference["target"])
    torch.testing.assert_close(result["context"], reference["context"])


@pytest.mark.parametrize(
    ("token_mask", "seed"),
    [
        (torch.tensor([[False, True, True]]), 11),
        (torch.tensor([[True, False, True]]), 5),
    ],
)
def test_bendr_mask_spans_never_expand_from_invalid_starts(token_mask: torch.Tensor, seed: int):
    # At p=0.2 each chosen seed samples only the invalid position.  A buggy
    # expansion masks the valid token immediately to its right; valid starts
    # must instead be selected before their forward span is expanded.
    mask = sample_bendr_mask(
        token_mask, probability=0.2, span=2, generator=torch.Generator().manual_seed(seed)
    )

    assert not mask.any()
    assert not mask[~token_mask].any()


def test_bendr_mask_spans_stop_at_an_internal_invalid_gap():
    """A valid start cannot bridge a missing-token gap to a later token."""
    token_mask = torch.tensor([[True, False, True]])

    # seed=3 selects only token 0 at p=0.2.  A three-token span must stop at
    # token 1, rather than corrupting valid token 2 across the gap.
    mask = sample_bendr_mask(
        token_mask, probability=0.2, span=3, generator=torch.Generator().manual_seed(3)
    )

    assert torch.equal(mask, torch.tensor([[True, False, False]]))


def test_same_sequence_negative_sampler_excludes_positive_and_other_records():
    valid = torch.tensor([[True, True, True, False], [False, True, True, True]])
    indices, usable = same_sequence_negative_indices(valid, num_negatives=5, generator=torch.Generator().manual_seed(3))

    assert indices.shape == usable.shape == (2, 4, 5)
    for batch in range(valid.shape[0]):
        for position in range(valid.shape[1]):
            selected = indices[batch, position][usable[batch, position]]
            assert torch.all(selected != position)
            assert torch.all(valid[batch, selected])


def test_bendr_contrastive_loss_is_finite_when_a_record_has_no_safe_negative():
    objective = BendrLikeContrastiveLoss(num_negatives=4, temperature=0.2)
    context = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.randn(2, 3, 8)
    predicted = torch.tensor([[True, False, False], [True, False, False]])
    valid = torch.tensor([[True, False, False], [True, True, False]])

    loss = objective(context, targets, predicted, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(context.grad).all()
    # The first record has one valid position only and is excluded rather than
    # borrowing a negative from the second record.
    assert objective.last_usable_queries == 1


def test_bendr_all_invalid_tokens_are_zero_and_early_zero_loss_backpropagates_finitely():
    encoder = BendrEncoder(d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0).eval()
    objective = BendrLikeContrastiveLoss(num_negatives=2, temperature=0.2)
    emg = torch.randn(2, 100, 16, requires_grad=True)

    outputs = encoder(emg, valid_mask=torch.zeros(2, 100, dtype=torch.bool))
    loss = objective(
        outputs["context"], outputs["target"], outputs["corruption_mask"], outputs["token_mask"]
    )
    loss.backward()

    for key in ("context", "target", "pooled"):
        assert torch.isfinite(outputs[key]).all()
        assert torch.equal(outputs[key], torch.zeros_like(outputs[key]))
    assert torch.equal(outputs["token_mask"], torch.zeros_like(outputs["token_mask"]))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert emg.grad is not None and torch.isfinite(emg.grad).all()


def test_bendr_pretraining_wrapper_uses_caller_generator_for_negative_sampling():
    encoder = BendrEncoder(d_model=8, depth=1, heads=2, patch_samples=20, dropout=0.0).eval()
    model = BendrPretrainingModel(encoder=encoder, num_negatives=2, temperature=0.2).eval()
    emg = torch.randn(1, 100, 16)
    corruption = torch.tensor([[True, True, True, False, False]])

    torch.manual_seed(1)
    _ = torch.randn(17)
    expected_indices, expected_usable = same_sequence_negative_indices(
        torch.ones(1, 5, dtype=torch.bool), num_negatives=2, generator=torch.Generator().manual_seed(31)
    )
    torch.manual_seed(9_999)
    _ = torch.randn(1_000)
    actual_indices, actual_usable = same_sequence_negative_indices(
        torch.ones(1, 5, dtype=torch.bool), num_negatives=2, generator=torch.Generator().manual_seed(31)
    )
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_usable, expected_usable)

    # These deliberately different global states prove the wrapper does not
    # silently let contrastive negative sampling fall back to global RNG.
    # Mutation proof: removing the wrapper's ``generator=...`` forwarding
    # makes the assertion below fail with these exact two states.
    torch.manual_seed(1)
    _ = torch.randn(17)
    first = model(emg, corruption_mask=corruption, generator=torch.Generator().manual_seed(31))
    torch.manual_seed(9_999)
    _ = torch.randn(1_000)
    second = model(emg, corruption_mask=corruption, generator=torch.Generator().manual_seed(31))

    torch.testing.assert_close(first["contrastive_loss"], second["contrastive_loss"])


@pytest.mark.parametrize("model_name", ["supervised_tcn", "bendr"])
def test_baseline_models_compose_through_hydra_and_registry(model_name: str):
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="baseline", overrides=[f"model={model_name}"])

    model = model_from_config(dict(cfg.model))
    direct = build_model(str(cfg.model.name), **{key: value for key, value in cfg.model.items() if key != "name"})

    assert type(model) is type(direct)
    assert cfg.model.channels == 16
