"""Event-level gesture metrics that respect event time rather than window IDs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence


Event = tuple[str, float]


def event_f1_from_counts(true_positive: float, false_positive: float, false_negative: float) -> float:
    """F1 from accumulated one-to-one match counts.

    This is ``2*TP / (2*TP + FP + FN)``, the harmonic mean of precision and
    recall.  The superficially similar ``2*TP / (TP + FP + FN)`` is not F1 and
    can exceed 1.0, so callers must share this helper rather than open-code it.
    """
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator > 0 else 0.0


def event_f1(predicted: Sequence[Event], reference: Sequence[Event], *, tolerance_s: float) -> dict[str, float | int]:
    """Compute label-aware greedy event F1 with one-to-one tolerance matching.

    Events are sorted within each label and a prediction can match at most one
    reference event.  This avoids over-crediting several detections around a
    single gesture onset, while requiring no synthetic per-frame labels.
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be non-negative")
    groups: dict[str, list[float]] = defaultdict(list)
    for label, time_s in reference:
        if not isinstance(label, str) or not isinstance(time_s, (int, float)):
            raise ValueError("reference events must be (str, finite float)")
        if time_s != time_s or abs(float(time_s)) == float("inf"):
            raise ValueError("reference event times must be finite")
        groups[label].append(float(time_s))
    predicted_groups: dict[str, list[float]] = defaultdict(list)
    for label, time_s in predicted:
        if not isinstance(label, str) or not isinstance(time_s, (int, float)):
            raise ValueError("predicted events must be (str, finite float)")
        if time_s != time_s or abs(float(time_s)) == float("inf"):
            raise ValueError("predicted event times must be finite")
        predicted_groups[label].append(float(time_s))

    true_positive = 0
    false_positive = 0
    for label, predictions in predicted_groups.items():
        references = sorted(groups.get(label, []))
        used = [False] * len(references)
        for time_s in sorted(predictions):
            candidates = [
                (abs(time_s - ref_time), index)
                for index, ref_time in enumerate(references)
                if not used[index] and abs(time_s - ref_time) <= tolerance_s
            ]
            if not candidates:
                false_positive += 1
                continue
            _, matched = min(candidates)
            used[matched] = True
            true_positive += 1
    false_negative = len(reference) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


__all__ = ["Event", "event_f1"]
