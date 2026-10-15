"""Server-side training diagnostics and fail-closed divergence guards for E3.

This callback is purely additive.  It does not touch the encoder, the masked
latent objective, the EMA update rule or the data module; it only observes the
run and guarantees the E3 production requirements that native Lightning does
not provide on its own:

* per-corpus (per-task) training and validation batch counts,
* student/teacher update status and student feature statistics,
* a NaN/Inf guard that stops the run and persists the offending step.

``TrainingDiagnostics`` in ``semg_fm.local_callbacks`` covers the same ground
for the local smoke profile, but it is smoke-specific (it hardcodes a
``stage='smoke'`` report, asserts a 32-bit AMP status and retains every step in
memory).  A 100k-step production run would accumulate ~100k rows in RAM and
write a metrics file of the same order, so this module keeps only bounded
aggregates in memory and streams per-step rows to a JSONL file instead.
"""
from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

KNOWN_TASKS = ("discrete_gesture", "handwriting", "pose", "typing", "wrist_angle")


def _finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors if torch.is_tensor(t))


def _fingerprint(state: dict[str, torch.Tensor]) -> dict[str, float]:
    """Cheap change detector: exact value hashing stays out of the hot path."""
    total = 0.0
    for value in state.values():
        if torch.is_tensor(value) and value.is_floating_point():
            total += float(value.detach().float().sum())
    return {"sum": total}


class E3RunDiagnostics(Callback):
    """Bounded-memory diagnostics plus a hard NaN/Inf stop for production runs.

    Parameters
    ----------
    output_dir:
        Directory for ``train_metrics.jsonl``, ``diagnostics.json`` and, on
        divergence, ``e3_error.json``.  Defaults to the Hydra run directory
        (``hydra.job.chdir`` makes it the working directory).
    param_check_interval:
        Scan the model parameters for non-finite values every N optimizer
        steps.  Losses and gradients are checked on every step.
    """

    def __init__(self, output_dir: str = ".", param_check_interval: int = 10) -> None:
        if param_check_interval < 1:
            raise ValueError("param_check_interval must be at least 1")
        self.output_dir = Path(output_dir)
        self.param_check_interval = int(param_check_interval)
        self._rows_path: Path | None = None
        self._started = 0.0
        self._feature_std = 0.0
        self._feature_hook = None
        self._batch_started = 0.0
        self._initial_student: dict[str, float] = {}
        self._initial_teacher: dict[str, float] = {}
        self._train_batches: dict[str, int] = {}
        self._val_batches: dict[str, int] = {}
        self._feature_std_min = math.inf
        self._feature_std_max = -math.inf
        self._feature_std_sum = 0.0
        self._feature_std_count = 0
        self._steps_recorded = 0
        self._error_path: Path | None = None

    # -- lifecycle ---------------------------------------------------------

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rows_path = self.output_dir / "train_metrics.jsonl"
        self._error_path = self.output_dir / "e3_error.json"
        self._started = time.perf_counter()
        self._initial_student = _fingerprint(pl_module.student.state_dict())
        self._initial_teacher = _fingerprint(pl_module.teacher.state_dict())
        # The encoder returns a dict; z_motion is the kinematic/motion latent.
        self._feature_hook = pl_module.student.register_forward_hook(self._capture_features)

    def _capture_features(self, _module, _inputs, output) -> None:
        if isinstance(output, dict) and torch.is_tensor(output.get("z_motion")):
            self._feature_std = float(output["z_motion"].detach().float().std())

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._batch_started = time.perf_counter()

    # -- per-step observation ---------------------------------------------

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        task = self._task_of(batch)
        self._train_batches[task] = self._train_batches.get(task, 0) + 1
        step = int(trainer.global_step)
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        loss_value = float(loss.detach()) if torch.is_tensor(loss) else float("nan")
        grads = [p.grad.detach().float().norm() for p in pl_module.student.parameters() if p.grad is not None]
        grad_norm = float(torch.stack(grads).norm()) if grads else 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        row = {
            "step": step,
            "task_id": task,
            "loss": loss_value,
            "latent_loss": self._metric(trainer, "train/latent_loss"),
            "student_gradient_norm": grad_norm,
            "feature_std": self._feature_std,
            "learning_rate": trainer.optimizers[0].param_groups[0]["lr"],
            "gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "step_time_s": time.perf_counter() - self._batch_started,
            "ema_updates": int(pl_module.ema_updates),
        }
        self._observe(row)
        self._guard(row, pl_module, step)

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
        task = self._task_of(batch)
        self._val_batches[task] = self._val_batches.get(task, 0) + 1

    # -- reporting ---------------------------------------------------------

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module) -> None:
        if self._feature_hook is not None:
            self._feature_hook.remove()
            self._feature_hook = None
        student_after = _fingerprint(pl_module.student.state_dict())
        teacher_after = _fingerprint(pl_module.teacher.state_dict())
        report = {
            "stage": "server",
            "host": platform.node(),
            "global_step": int(trainer.global_step),
            "elapsed_s": time.perf_counter() - self._started,
            "precision": str(trainer.precision),
            "world_size": int(trainer.world_size),
            "expected_tasks": list(KNOWN_TASKS),
            "train_batches_per_task": dict(sorted(self._train_batches.items())),
            "val_batches_per_task": dict(sorted(self._val_batches.items())),
            "train_batch_total": sum(self._train_batches.values()),
            "missing_train_tasks": sorted(set(KNOWN_TASKS) - set(self._train_batches)),
            "student_updated": student_after != self._initial_student,
            "teacher_updated": teacher_after != self._initial_teacher,
            "teacher_has_no_grads": all(p.grad is None for p in pl_module.teacher.parameters()),
            "ema_updates": int(pl_module.ema_updates),
            "feature_std": {
                "last": self._feature_std,
                "mean": (self._feature_std_sum / self._feature_std_count) if self._feature_std_count else 0.0,
                "min": self._feature_std_min if self._feature_std_count else 0.0,
                "max": self._feature_std_max if self._feature_std_count else 0.0,
            },
            "steps_recorded": self._steps_recorded,
        }
        self._write("diagnostics.json", report)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _metric(trainer, key: str) -> float:
        value = trainer.callback_metrics.get(key)
        return float(value) if torch.is_tensor(value) else 0.0

    @staticmethod
    def _task_of(batch) -> str:
        return str(batch.get("task_id", "unknown")) if isinstance(batch, dict) else "unknown"

    def _observe(self, row: dict) -> None:
        value = row["feature_std"]
        if math.isfinite(value):
            self._feature_std_min = min(self._feature_std_min, value)
            self._feature_std_max = max(self._feature_std_max, value)
            self._feature_std_sum += value
            self._feature_std_count += 1
        # Per-step rows stream to disk so memory stays bounded across 100k steps;
        # the end-of-fit report is built from the aggregates kept above.
        with self._rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self._steps_recorded += 1

    def _guard(self, row: dict, pl_module, step: int) -> None:
        """Stop on divergence and persist the exact step that produced it."""
        suspects = {
            "loss": row["loss"],
            "feature_std": row["feature_std"],
            "student_gradient_norm": row["student_gradient_norm"],
        }
        bad = sorted(name for name, value in suspects.items() if not math.isfinite(value))
        if not bad and step % self.param_check_interval == 0:
            params = [p.detach() for p in pl_module.student.parameters()]
            if not _finite(*params):
                bad = ["student_parameters"]
        if not bad:
            return
        payload = {
            "error": "non_finite_training_state",
            "non_finite": bad,
            "global_step": step,
            "task_id": row["task_id"],
            "observed": row,
            "action": "training stopped before the checkpoint could be poisoned",
        }
        self._write("e3_error.json", payload)
        raise RuntimeError(
            f"non-finite training state at step {step} ({', '.join(bad)}); "
            f"details written to {self._error_path}"
        )

    def _write(self, name: str, payload: dict) -> None:
        (self.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
