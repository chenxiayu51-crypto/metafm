"""Explicit downstream task protocol registry for the five canonical datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Evaluation constraints for one canonical task, independent of a model."""

    task_id: str
    kind: str
    primary_metric: str
    requires_pair_group: bool = False


TASK_SPECS = {
    "pose": TaskSpec("pose", "regression", "masked_rmse"),
    "wrist_angle": TaskSpec("wrist_angle", "regression", "masked_rmse"),
    "discrete_gesture": TaskSpec("discrete_gesture", "events", "event_f1"),
    "handwriting": TaskSpec("handwriting", "ctc", "character_error_rate"),
    # Left/right typing recordings share a pair_id and must stay within one
    # split. That protects evaluation from paired-record leakage, not a claim
    # that the two wrists are positive contrastive views.
    "typing": TaskSpec("typing", "ctc", "character_error_rate", requires_pair_group=True),
}


def task_spec(task_id: str) -> TaskSpec:
    """Return the supported protocol for a canonical task ID."""
    try:
        return TASK_SPECS[task_id]
    except KeyError as error:
        raise ValueError(f"unsupported task_id {task_id!r}; choose from {sorted(TASK_SPECS)}") from error


def task_kind(task_id: str) -> str:
    """Return the downstream head/loss family for ``task_id``."""
    return task_spec(task_id).kind


__all__ = ["TASK_SPECS", "TaskSpec", "task_kind", "task_spec"]
