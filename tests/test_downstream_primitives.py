"""Contracts for downstream heads, task routing, and task-appropriate metrics."""

from __future__ import annotations

import torch

from semg_fm.metrics.events import event_f1, event_f1_from_counts
from semg_fm.metrics.regression import masked_regression_metrics
from semg_fm.metrics.text import character_error_rate, levenshtein_distance
from semg_fm.models.heads import CTCTextHead, GestureEventHead, TemporalRegressionHead
from semg_fm.tasks import task_kind, task_spec


def test_masked_regression_metrics_ignore_invalid_targets_and_padded_examples():
    prediction = torch.tensor([[[2.0], [99.0]], [[5.0], [6.0]]])
    target = torch.tensor([[[1.0], [0.0]], [[3.0], [8.0]]])
    mask = torch.tensor([[True, False], [True, True]])
    example_weight = torch.tensor([1.0, 0.0])

    metrics = masked_regression_metrics(prediction, target, mask, example_weight=example_weight)

    assert metrics["count"].item() == 1
    assert metrics["mae"].item() == 1.0
    assert metrics["rmse"].item() == 1.0


def test_event_f1_matches_each_label_once_within_tolerance():
    reference = [("pinch", 0.10), ("pinch", 0.40), ("fist", 0.20)]
    predicted = [("pinch", 0.11), ("pinch", 0.13), ("fist", 0.30)]

    metrics = event_f1(predicted, reference, tolerance_s=0.05)

    assert metrics == {"true_positive": 1, "false_positive": 2, "false_negative": 2, "precision": 1 / 3, "recall": 1 / 3, "f1": 1 / 3}


def test_text_metrics_use_levenshtein_without_character_timing_assumptions():
    assert levenshtein_distance("abc", "axc") == 1
    assert character_error_rate("abc", "axc") == 1 / 3
    assert character_error_rate("", "") == 0.0


def test_task_heads_return_expected_shapes_and_ctc_loss_is_finite():
    features = torch.randn(2, 5, 8, requires_grad=True)
    regression = TemporalRegressionHead(8, output_dim=3)
    gesture = GestureEventHead(8, class_names=("pinch", "fist"))
    ctc = CTCTextHead(8, vocabulary=("a", "b", "c"))

    reg_prediction = regression(features)
    gesture_logits = gesture(features)
    text_logits = ctc(features)
    loss = ctc.loss(
        text_logits,
        targets=torch.tensor([[1, 2], [2, 0]], dtype=torch.long),
        input_lengths=torch.tensor([5, 5]),
        target_lengths=torch.tensor([2, 1]),
    )
    loss.backward()

    assert reg_prediction.shape == (2, 5, 3)
    assert gesture_logits.shape == (2, 5, 3)  # includes no-event class
    assert text_logits.shape == (2, 5, 4)  # includes CTC blank
    assert torch.isfinite(loss)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_task_registry_routes_all_five_canonical_tasks_to_supported_protocols():
    assert task_kind("pose") == "regression"
    assert task_kind("wrist_angle") == "regression"
    assert task_kind("discrete_gesture") == "events"
    assert task_kind("handwriting") == "ctc"
    assert task_kind("typing") == "ctc"
    assert task_spec("typing").requires_pair_group is True


def test_event_f1_from_counts_is_the_precision_recall_harmonic_mean():
    # The superficially similar 2*TP/(TP+FP+FN) returns 2.0 for a perfect
    # match, which is why the accumulated-count path shares this helper.
    assert event_f1_from_counts(1, 0, 0) == 1.0
    assert event_f1_from_counts(3, 0, 0) == 1.0
    assert event_f1_from_counts(0, 3, 2) == 0.0
    assert event_f1_from_counts(0, 0, 0) == 0.0
    # precision 3/5, recall 3/8 -> 0.45 / 0.975
    assert abs(event_f1_from_counts(3, 2, 5) - 0.4615384615384615) < 1e-12


def test_event_f1_from_counts_never_exceeds_one():
    for counts in [(1, 0, 0), (3, 2, 5), (0, 0, 4), (7, 1, 2), (5, 5, 5), (100, 0, 0)]:
        value = event_f1_from_counts(*counts)
        assert 0.0 <= value <= 1.0, (counts, value)
