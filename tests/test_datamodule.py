import h5py
import pytest
import pandas as pd
import torch
from torch.utils.data import Dataset

from semg_fm.data.datamodule import CanonicalDataModule, task_homogeneous_collate
from semg_fm.data.samplers import TaskBalancedBatchSampler, TaskEvaluationBatchSampler


def _item(task_id, length, value=1.0):
    return {
        "emg": torch.full((length, 16), value, dtype=torch.float32),
        "sample_mask": torch.ones(length, dtype=torch.bool),
        "task_id": task_id,
        "record_id": f"{task_id}-{length}",
    }


def test_collate_rejects_mixed_task_ids():
    with pytest.raises(ValueError, match="one task_id"):
        task_homogeneous_collate([_item("pose", 2), _item("typing", 3)])


def test_collate_pads_time_and_returns_sample_mask_and_metadata():
    batch = task_homogeneous_collate([_item("pose", 2, 1), _item("pose", 4, 2)])

    assert batch["emg"].shape == (2, 4, 16)
    assert batch["sample_mask"].dtype == torch.bool
    assert batch["sample_mask"].tolist() == [[True, True, False, False], [True, True, True, True]]
    assert batch["task_id"] == "pose"
    assert [metadata["record_id"] for metadata in batch["metadata"]] == ["pose-2", "pose-4"]


def test_collate_promotes_tensor_pose_targets_and_preserves_metadata():
    short = _item("pose", 2)
    long = _item("pose", 4)
    short["target"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    long["target"] = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    short["target_mask"] = torch.tensor([True, False])
    long["target_mask"] = torch.tensor([True, True, False, True])

    batch = task_homogeneous_collate([short, long])

    assert batch["target"].shape == (2, 4, 2)
    assert batch["target_mask"].dtype == torch.bool
    assert batch["target_mask"].tolist() == [[True, False, False, False], [True, True, False, True]]
    torch.testing.assert_close(batch["target"][0, :2], short["target"])
    torch.testing.assert_close(batch["target"][1], long["target"])
    torch.testing.assert_close(batch["metadata"][0]["target"], short["target"])
    assert torch.equal(batch["metadata"][1]["target_mask"], long["target_mask"])


def test_collate_keeps_non_tensor_targets_in_metadata_only():
    first = _item("wrist_angle", 2)
    second = _item("wrist_angle", 3)
    first["target"] = {"flexion": torch.tensor([1.0, 2.0])}
    second["target"] = {"flexion": torch.tensor([3.0, 4.0, 5.0])}
    first["target_mask"] = {"flexion": torch.tensor([True, True])}
    second["target_mask"] = {"flexion": torch.tensor([True, True, True])}

    batch = task_homogeneous_collate([first, second])

    assert "target" not in batch
    assert "target_mask" not in batch
    assert batch["metadata"][0]["target"] is first["target"]
    assert batch["metadata"][1]["target_mask"] is second["target_mask"]


def test_collate_rejects_tensor_target_mask_with_incompatible_trailing_shape():
    first = _item("pose", 2)
    second = _item("pose", 3)
    first["target"] = torch.ones(2, 2)
    second["target"] = torch.ones(3, 2)
    first["target_mask"] = torch.ones(2, 1, dtype=torch.bool)
    second["target_mask"] = torch.ones(3, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="target_mask"):
        task_homogeneous_collate([first, second])


def test_collate_returns_zero_weight_for_padded_evaluation_examples():
    real_item = _item("pose", 2)
    padded_item = _item("pose", 2)
    real_item["is_padding"] = False
    padded_item["is_padding"] = True

    batch = task_homogeneous_collate([real_item, padded_item])

    assert batch["is_padding"].dtype == torch.bool
    assert batch["is_padding"].tolist() == [False, True]
    assert batch["example_weight"].tolist() == [1.0, 0.0]


class _TaskDataset(Dataset):
    def __init__(self):
        self.items = [_item(task, 2) for task in ("a", "a", "a", "a", "b", "b", "b", "b")]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_task_balanced_sampler_set_epoch_is_deterministic_and_changes_order():
    dataset = _TaskDataset()
    sampler = TaskBalancedBatchSampler(dataset, batch_size=2, seed=17)

    sampler.set_epoch(0)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    sampler.set_epoch(0)
    repeated_epoch_zero = list(sampler)

    assert epoch_zero == repeated_epoch_zero
    assert epoch_zero != epoch_one
    assert [dataset[batch[0]]["task_id"] for batch in epoch_zero] == ["a", "b", "a", "b"]


def test_task_balanced_sampler_rank_partitions_do_not_overlap():
    dataset = _TaskDataset()
    first_rank = TaskBalancedBatchSampler(dataset, batch_size=1, seed=3, rank=0, world_size=2)
    second_rank = TaskBalancedBatchSampler(dataset, batch_size=1, seed=3, rank=1, world_size=2)
    first_rank.set_epoch(2)
    second_rank.set_epoch(2)

    first_indices = {index for batch in first_rank for index in batch}
    second_indices = {index for batch in second_rank for index in batch}

    assert not first_indices.intersection(second_indices)
    assert len(first_indices) == sum(len(batch) for batch in first_rank)
    assert len(second_indices) == sum(len(batch) for batch in second_rank)


def test_task_balanced_sampler_keeps_distributed_rank_batch_counts_equal():
    class UnevenTaskDataset(Dataset):
        def __init__(self):
            self.items = [_item(task, 2) for task in ("a",) * 7 + ("b",) * 7]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = UnevenTaskDataset()
    first_rank = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3, rank=0, world_size=2)
    second_rank = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3, rank=1, world_size=2)

    assert len(list(first_rank)) == len(list(second_rank)) == 2


def test_task_balanced_sampler_reads_epoch_when_its_loader_starts():
    epoch = [0]
    sampler = TaskBalancedBatchSampler(
        _TaskDataset(), batch_size=2, seed=3, epoch_getter=lambda: epoch[0]
    )

    list(sampler)
    epoch[0] = 5
    list(sampler)

    assert sampler.epoch == 5


def test_task_balanced_sampler_yields_batches_when_one_task_is_too_small():
    class UnevenTaskDataset(Dataset):
        def __init__(self):
            self.items = [_item(task, 2) for task in ("a",) * 8 + ("b",) * 3]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = UnevenTaskDataset()
    sampler = TaskBalancedBatchSampler(dataset, batch_size=4, seed=3)

    # Task b has 3 windows (< one batch); task a still trains instead of the
    # epoch collapsing to zero batches.
    batches = list(sampler)
    assert len(batches) == 2
    assert {dataset[batch[0]]["task_id"] for batch in batches} == {"a"}


def test_task_balanced_sampler_length_matches_iteration():
    dataset = _TaskDataset()
    sampler = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3)
    assert len(sampler) == len(list(sampler))


def test_task_balanced_sampler_state_dict_round_trip_preserves_epoch():
    dataset = _TaskDataset()
    sampler = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3)
    sampler.set_epoch(7)

    restored = TaskBalancedBatchSampler(dataset, batch_size=2, seed=3)
    restored.load_state_dict(sampler.state_dict())

    assert restored.epoch == 7
    assert list(restored) == list(sampler)


def test_datamodule_yields_task_homogeneous_train_batch(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root,
        window_ms=100,
        stride_ms=100,
        batch_size=2,
        num_workers=0,
        sampler_seed=5,
    )
    module.setup("fit")

    batch = next(iter(module.train_dataloader()))

    assert batch["emg"].shape[0] == 2
    assert batch["emg"].shape[2] == 16
    assert batch["sample_mask"].dtype == torch.bool
    assert all(metadata["task_id"] == batch["task_id"] for metadata in batch["metadata"])


def test_datamodule_uses_attached_trainer_rank_when_building_loader(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root,
        window_ms=100,
        stride_ms=100,
        batch_size=1,
        num_workers=0,
        sampler_seed=5,
    )
    module.setup("fit")

    class FakeTrainer:
        global_rank = 1
        world_size = 2
        current_epoch = 0

    module.trainer = FakeTrainer()
    rank_one_sampler = module.train_dataloader().batch_sampler
    rank_zero_sampler = TaskBalancedBatchSampler(
        module.train_dataset, batch_size=1, seed=5, rank=0, world_size=2
    )

    assert rank_one_sampler.rank == 1
    assert rank_one_sampler.world_size == 2
    assert list(rank_one_sampler) != list(rank_zero_sampler)


def _manifest_for_split(canonical_meta_root, split):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    manifest["official_split"] = split
    path = canonical_meta_root / f"{split}_only.csv"
    manifest.to_csv(path, index=False)
    return path


def test_setup_initializes_only_the_requested_split_lifecycle(canonical_meta_root):
    val_module = CanonicalDataModule(
        split_paths={"val": _manifest_for_split(canonical_meta_root, "val")},
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    val_module.setup("validate")
    assert val_module.val_dataset is not None
    assert val_module.train_dataset is None
    assert val_module.test_dataset is None

    test_module = CanonicalDataModule(
        split_paths={"test": _manifest_for_split(canonical_meta_root, "test")},
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    test_module.setup("test")
    assert test_module.test_dataset is not None
    assert test_module.train_dataset is None
    assert test_module.val_dataset is None

    fit_module = CanonicalDataModule(
        split_paths={
            "train": _manifest_for_split(canonical_meta_root, "train"),
            "val": _manifest_for_split(canonical_meta_root, "val"),
        },
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    fit_module.setup("fit")
    assert fit_module.train_dataset is not None
    assert fit_module.val_dataset is not None
    assert fit_module.test_dataset is None


class _ImbalancedTaskDataset(Dataset):
    def __init__(self):
        self.items = [_item("major", 2) for _ in range(20)] + [_item("minor", 2) for _ in range(2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def test_eval_loaders_cover_each_imbalanced_task_index_once_without_epoch_variation(canonical_meta_root):
    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root, window_ms=100, stride_ms=100, batch_size=2, num_workers=0,
    )
    dataset = _ImbalancedTaskDataset()
    module.val_dataset = dataset
    module.test_dataset = dataset

    for loader in (module.val_dataloader(), module.test_dataloader()):
        sampler = loader.batch_sampler
        first_epoch = [index for batch in sampler for index in batch]
        module.set_epoch(9)
        second_epoch = [index for batch in sampler for index in batch]
        assert sorted(first_epoch) == list(range(22))
        assert len(first_epoch) == 22
        assert second_epoch == first_epoch


def test_evaluation_sampler_budget_interleaves_tasks_round_robin():
    """A total budget must truncate across tasks, not exhaust the first one.

    Lightning's `limit_val_batches` takes the first N batches, so exhausting
    task 'a' (alphabetically first) before 'b' would make the monitored
    `val/loss` a single-task metric.
    """

    class TwoTaskDataset(Dataset):
        def __init__(self):
            self.items = [_item("a", 2) for _ in range(12)] + [_item("b", 2) for _ in range(4)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = TwoTaskDataset()

    unbudgeted = TaskEvaluationBatchSampler(dataset, batch_size=2)
    assert [dataset[batch[0]]["task_id"] for batch in unbudgeted] == ["a"] * 6 + ["b"] * 2

    sampler = TaskEvaluationBatchSampler(dataset, batch_size=2, batch_budget=4)
    tasks = [dataset[batch[0]]["task_id"] for batch in sampler]
    # Budget 4 over 2 tasks: cap 2 per task, interleaved a,b,a,b.
    assert tasks == ["a", "b", "a", "b"]
    assert len(sampler) == len(tasks) == 4


def test_distributed_evaluation_padding_keeps_rank_lengths_equal_and_marks_duplicates():
    class UnevenEvaluationDataset(Dataset):
        def __init__(self):
            self.items = [_item("major", 2) for _ in range(5)] + [_item("minor", 2) for _ in range(2)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    dataset = UnevenEvaluationDataset()
    first_rank = TaskEvaluationBatchSampler(dataset, batch_size=2, rank=0, world_size=2)
    second_rank = TaskEvaluationBatchSampler(dataset, batch_size=2, rank=1, world_size=2)

    first_batches = list(first_rank)
    second_batches = list(second_rank)
    entries = [entry for batch in first_batches + second_batches for entry in batch]
    nonpadding = [int(entry) for entry in entries if not entry.is_padding]
    padding = [entry for entry in entries if entry.is_padding]

    assert len(first_rank) == len(second_rank)
    assert sorted(nonpadding) == list(range(7))
    assert len(nonpadding) == 7
    assert padding
    assert all(entry.is_padding for entry in padding)


def test_datamodule_rejects_an_empty_training_split(canonical_meta_root):
    """Training on zero windows used to run zero steps and report nothing."""
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    manifest["official_split"] = ""
    manifest["split"] = "test"
    path = canonical_meta_root / "no_train.csv"
    manifest.to_csv(path, index=False)
    module = CanonicalDataModule(
        manifest_path=path, root=canonical_meta_root, window_ms=100, stride_ms=100,
        batch_size=2, num_workers=0,
    )

    with pytest.raises(ValueError, match="training split is empty"):
        module.setup("fit")


def test_evaluation_sampler_reads_task_ids_without_opening_h5_files(canonical_meta_root, monkeypatch):
    """Grouping windows by task must read the manifest index, not the signals.

    The sampler is built from the unwrapped canonical dataset, so it can read
    ``_index``/``_records`` directly.  Handing it the padded loader wrapper
    instead would fall back to per-window indexing and open one HDF5 file per
    window on every evaluation epoch.
    """
    real_file = h5py.File
    calls = []

    def observed_file(*args, **kwargs):
        calls.append(args[0])
        return real_file(*args, **kwargs)

    module = CanonicalDataModule(
        manifest_path=canonical_meta_root / "manifest.csv",
        root=canonical_meta_root,
        window_ms=100,
        stride_ms=100,
        batch_size=2,
        num_workers=0,
    )
    module.setup("test")
    monkeypatch.setattr("semg_fm.data.window_dataset.h5py.File", observed_file)

    module.test_dataloader()

    assert calls == [], f"evaluation sampler opened {len(calls)} HDF5 file(s)"
