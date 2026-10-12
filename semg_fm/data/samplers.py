"""Deterministic task-balanced samplers for heterogeneous sEMG training data."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


def _task_ids(dataset) -> list[str]:
    """Read task IDs without opening HDF5 windows when using the canonical dataset."""
    if hasattr(dataset, "task_ids"):
        task_ids = list(dataset.task_ids)
        if len(task_ids) == len(dataset):
            return [str(task_id) for task_id in task_ids]
    if hasattr(dataset, "_index") and hasattr(dataset, "_records"):
        return [str(dataset._records[path]["task_id"]) for path, _, _ in dataset._index]
    return [str(dataset[index]["task_id"]) for index in range(len(dataset))]


def _task_groups(dataset) -> dict[str, np.ndarray]:
    """Group window ordinals by task, as arrays rather than Python lists.

    A corpus-scale split holds tens of millions of windows in one task; a
    ``list[int]`` of those ordinals costs about 1 GB and a ``list[str]`` of
    task IDs several times more.
    """
    if hasattr(dataset, "window_indices_by_task"):
        groups = dataset.window_indices_by_task()
        if not groups:
            raise ValueError("dataset must contain at least one task")
        return groups
    collected: dict[str, list[int]] = defaultdict(list)
    for index, task_id in enumerate(_task_ids(dataset)):
        collected[task_id].append(index)
    if not collected:
        raise ValueError("dataset must contain at least one task")
    return {
        task_id: np.asarray(indices, dtype=np.int64)
        for task_id, indices in sorted(collected.items())
    }


class EvaluationIndex(int):
    """An integer dataset index that identifies a deterministic padded duplicate."""

    def __new__(cls, value: int, is_padding: bool = False):
        instance = int.__new__(cls, value)
        instance.is_padding = bool(is_padding)
        return instance


class TaskBalancedBatchSampler(Sampler[list[int]]):
    """Yield homogeneous batches, interleaving each task equally within an epoch."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        epoch_getter: Callable[[], int] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self._epoch_getter = epoch_getter
        self.epoch = 0
        self._groups = _task_groups(dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        """Persist the shuffle epoch so DataLoader resume replays deterministically."""
        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get("epoch", 0))

    def _rank_batch_count(self, indices: np.ndarray) -> int:
        """Full batches this rank contributes from one task's windows."""
        return len(indices) // (self.batch_size * self.world_size)

    def _task_batches(self, indices: np.ndarray, generator: torch.Generator) -> Iterator[list[int]]:
        """Shuffle one task's windows and yield this rank's batches in order.

        The permutation is drawn on the first ``next`` call, so drawing from a
        generator shared by every task reproduces the previous per-task draw
        order without materializing every batch of the epoch at once.
        """
        order = torch.randperm(len(indices), generator=generator).numpy()
        samples_per_rank = self._rank_batch_count(indices) * self.batch_size
        rank_start = self.rank * samples_per_rank
        rank_indices = indices[order][rank_start : rank_start + samples_per_rank]
        for start in range(0, samples_per_rank, self.batch_size):
            yield rank_indices[start : start + self.batch_size].tolist()

    def __iter__(self) -> Iterator[list[int]]:
        if self._epoch_getter is not None:
            self.set_epoch(self._epoch_getter())
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pending = {
            task_id: self._task_batches(indices, generator)
            for task_id, indices in self._groups.items()
        }
        # Interleave one batch per task per round until every task is exhausted.
        # A min() cutoff would starve the whole epoch whenever one task has
        # fewer windows than a batch; skip exhausted tasks instead.
        while pending:
            for task_id in list(pending):
                try:
                    yield next(pending[task_id])
                except StopIteration:
                    del pending[task_id]

    def __len__(self) -> int:
        return sum(
            self._rank_batch_count(indices) for indices in self._groups.values()
        )


class TaskEvaluationBatchSampler(Sampler[list[int]]):
    """Yield every evaluation index exactly once, while keeping each batch task-homogeneous.

    With ``batch_budget`` set, the total yielded count is clamped to the
    budget (divided evenly across tasks, at least one batch per task) and
    tasks are interleaved round-robin instead of being exhausted in turn.
    This keeps a Lightning ``limit_val_batches`` prefix spanning every task:
    exhausting the alphabetically first task first would otherwise make the
    monitored ``val/loss`` a single-task metric.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        batch_budget: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if batch_budget is not None and batch_budget <= 0:
            raise ValueError("batch_budget must be positive")
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_budget = batch_budget
        self._groups = _task_groups(dataset)

    @property
    def per_task_cap(self) -> int | None:
        """Per-task batch cap implied by the total budget, or None."""
        if self.batch_budget is None:
            return None
        return max(1, self.batch_budget // max(1, len(self._groups)))

    def _task_batch(self, indices: np.ndarray, offset: int) -> np.ndarray:
        return indices[offset * self.batch_size : (offset + 1) * self.batch_size]

    def _batch_counts(self, indices: np.ndarray) -> tuple[int, int]:
        """Return (actual, assigned) batch counts for one task on this rank."""
        actual = (len(indices) + self.batch_size - 1) // self.batch_size
        cap = self.per_task_cap
        if cap is not None:
            actual = min(actual, cap)
        return actual, actual + (-actual) % self.world_size

    def _task_batches(self) -> list[list[list[int]]]:
        batches = []
        for indices in self._groups.values():
            actual, assigned = self._batch_counts(indices)
            task_batches = []
            for batch_index in range(self.rank, assigned, self.world_size):
                is_padding = batch_index >= actual
                # A padded duplicate repeats an existing batch, so every rank
                # sees the same number of batches without inventing windows.
                source = batch_index % actual if is_padding else batch_index
                task_batches.append([
                    EvaluationIndex(int(index), is_padding)
                    for index in self._task_batch(indices, source)
                ])
            batches.append(task_batches)
        return batches

    def _batches(self) -> Iterator[list[int]]:
        # Batches are produced as they are consumed: the test splits of the
        # converted corpus still hold millions of windows, and a list of every
        # batch would hold them all at once.
        per_task = self._task_batches()
        if self.batch_budget is None:
            for task_batches in per_task:
                yield from task_batches
            return
        # Round-robin so an outer batch limit (Lightning's `limit_val_batches`)
        # truncates evenly across tasks instead of cutting before later tasks.
        for column in range(max((len(b) for b in per_task), default=0)):
            for task_batches in per_task:
                if column < len(task_batches):
                    yield task_batches[column]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return sum(len(task_batches) for task_batches in self._task_batches())
