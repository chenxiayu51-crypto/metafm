import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from semg_fm.data.window_dataset import CanonicalWindowDataset


def _task_manifest(canonical_meta_root, task_id):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    path = canonical_meta_root / f"{task_id}_manifest.csv"
    manifest.loc[manifest.task_id == task_id].to_csv(path, index=False)
    return path


@pytest.fixture
def pose_dataset(canonical_meta_root):
    return CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )


@pytest.fixture
def handwriting_dataset(canonical_meta_root):
    return CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "handwriting"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )


def test_window_uses_milliseconds_and_returns_time_major_emg(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    item = ds[0]
    assert item["emg"].ndim == 2
    assert item["emg"].shape[1] == 16
    assert item["emg"].shape[0] == 400
    assert item["emg"].dtype == torch.float32
    assert item["sample_mask"].dtype == torch.bool
    assert item["sample_mask"].all()


def test_dataset_construction_does_not_open_h5_files(canonical_meta_root, monkeypatch):
    real_file = h5py.File
    calls = []

    def observed_file(*args, **kwargs):
        calls.append(args[0])
        return real_file(*args, **kwargs)

    monkeypatch.setattr("semg_fm.data.window_dataset.h5py.File", observed_file)
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    assert calls == []
    ds[0]
    assert len(calls) == 1


def _write_split_manifest(canonical_meta_root, official, split_column):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    manifest["official_split"] = official
    manifest["split"] = split_column
    path = canonical_meta_root / "split_manifest.csv"
    manifest.to_csv(path, index=False)
    # Keep H5 root attributes consistent with the rewritten manifest so item
    # access passes attribute validation.
    for row in manifest.itertuples():
        with h5py.File(canonical_meta_root / row.path, "r+") as h5:
            h5.attrs["official_split"] = str(row.official_split)
    return manifest


def test_from_manifest_falls_back_to_split_column_when_official_split_unset(canonical_meta_root):
    manifest = _write_split_manifest(
        canonical_meta_root,
        official=["", "", "", "", "", ""],
        split_column=["train", "train", "train", "val", "test", "test"],
    )
    train = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "split_manifest.csv", split="train", window_ms=200, stride_ms=100,
        root=canonical_meta_root,
    )
    train_paths = {path.name for path, _, _ in train._index}
    expected = {Path(row).name for row in manifest.loc[manifest.split == "train", "path"]}
    assert train_paths == expected
    assert len(train) > 0
    assert train[0]["emg"].shape[0] == 400  # item access still validates attributes


def test_from_manifest_prefers_official_split_over_split_column(canonical_meta_root):
    manifest = _write_split_manifest(
        canonical_meta_root,
        official=["val", "", "", "", "", ""],
        split_column=["train", "train", "train", "val", "test", "test"],
    )
    val = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "split_manifest.csv", split="val", window_ms=200, stride_ms=100,
        root=canonical_meta_root,
    )
    val_paths = {path.name for path, _, _ in val._index}
    assert Path(manifest.loc[0, "path"]).name in val_paths  # official_split wins over split column
    train = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "split_manifest.csv", split="train", window_ms=200, stride_ms=100,
        root=canonical_meta_root,
    )
    train_paths = {path.name for path, _, _ in train._index}
    assert Path(manifest.loc[0, "path"]).name not in train_paths


def test_from_manifest_filters_records_by_task_id(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100,
        task_id="pose",
    )
    tasks = {ds._records[path]["task_id"] for path, _, _ in ds._index}
    assert tasks == {"pose"}


def test_from_manifest_rejects_a_task_id_absent_from_the_manifest(canonical_meta_root):
    with pytest.raises(ValueError, match="task_id"):
        CanonicalWindowDataset.from_manifest(
            canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100,
            task_id="nonexistent",
        )


def test_pose_invalid_ik_frames_become_target_mask(pose_dataset):
    item = pose_dataset[0]
    assert item["target"].shape[0] == item["emg"].shape[0]
    assert item["target_mask"].dtype == torch.bool
    assert not item["target_mask"][3]


def test_handwriting_returns_prompt_text_without_character_times(handwriting_dataset):
    item = handwriting_dataset[0]
    assert item["target_text"] == "hello"
    assert "character_times" not in item


def test_gesture_events_are_relative_to_window_start(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    assert ds[1]["events"] == [("fist", pytest.approx(0.05))]


def test_label_times_follow_the_record_clock_at_a_window_boundary(canonical_meta_root):
    path = canonical_meta_root / "discrete_gesture" / "discrete_gesture_right.h5"
    with h5py.File(path, "r+") as h5:
        time_s = h5["signal/time_s"][:]
        time_s[200:] += 0.01
        h5["signal/time_s"][:] = time_s
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    item = ds[1]
    assert item["start_s"] == pytest.approx(0.11)
    assert item["events"] == [("fist", pytest.approx(0.04))]


@pytest.mark.parametrize(
    ("dataset_name", "dtype", "expected"),
    [("emg", np.float64, "float32"), ("time_s", np.float32, "float64")],
)
def test_dataset_rejects_noncanonical_signal_dtypes(canonical_meta_root, dataset_name, dtype, expected):
    path = canonical_meta_root / "pose" / "pose_right.h5"
    with h5py.File(path, "r+") as h5:
        values = h5[f"signal/{dataset_name}"][:].astype(dtype)
        del h5[f"signal/{dataset_name}"]
        h5["signal"].create_dataset(dataset_name, data=values)
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    with pytest.raises(ValueError, match=expected):
        ds[0]


def test_dataset_rejects_missing_or_mismatched_canonical_root_attributes(canonical_meta_root):
    path = canonical_meta_root / "pose" / "pose_right.h5"
    with h5py.File(path, "r+") as h5:
        del h5.attrs["dataset_id"]
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "pose"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    with pytest.raises(ValueError, match="dataset_id"):
        ds[0]
    with h5py.File(path, "r+") as h5:
        h5.attrs["dataset_id"] = "not-canonical"
    with pytest.raises(ValueError, match="does not match manifest"):
        ds[0]


def test_window_index_stores_only_per_record_geometry(canonical_meta_root):
    """A corpus split holds tens of millions of windows and one tuple each costs
    about 10 GB, so the index may only keep per-record geometry.
    """
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    assert ds._index.offsets.shape == (len(ds._index.paths) + 1,)
    assert len(ds._index) == len(ds) > 0
    # Indexing, slicing, and iteration must agree with each other.
    assert ds._index[0] == next(iter(ds._index))
    assert ds._index[-1] == list(ds._index)[-1]
    assert ds._index[:2] == [ds._index[0], ds._index[1]]
    with pytest.raises(IndexError):
        ds._index[len(ds)]


def test_window_indices_by_task_covers_every_window_exactly_once(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    groups = ds.window_indices_by_task()
    assert set(groups) == {ds._records[path]["task_id"] for path in ds._index.paths}
    assert all(isinstance(indices, np.ndarray) for indices in groups.values())
    combined = np.sort(np.concatenate(list(groups.values())))
    assert combined.tolist() == list(range(len(ds)))


def test_window_size_follows_the_declared_rate_not_the_gapped_clock_span(canonical_meta_root):
    """Dropped samples make the clock span exceed (T-1)/rate, so the span cannot
    estimate the rate.

    This record holds 600 samples but its clock spans 0.6 s, which would infer
    1000 Hz.  The window must follow the declared 2000 Hz and return all 400
    samples rather than being silently halved.
    """
    path = canonical_meta_root / "pose" / "pose_right.h5"
    with h5py.File(path, "r+") as h5:
        time_s = h5["signal/time_s"][:]
        time_s[2:] += 0.3
        h5["signal/time_s"][:] = time_s
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    # Push the span far enough from the declared rate that the assertion cannot
    # pass by coincidence.
    manifest.loc[manifest.path.str.contains("pose_right"), "duration_s"] = 0.5995
    manifest.to_csv(canonical_meta_root / "manifest.csv", index=False)
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train", window_ms=200, stride_ms=100
    )
    assert ds[0]["emg"].shape == (400, 16)


def test_gapped_record_clock_controls_the_open_window_end(canonical_meta_root):
    path = canonical_meta_root / "discrete_gesture" / "discrete_gesture_right.h5"
    with h5py.File(path, "r+") as h5:
        time_s = h5["signal/time_s"][:]
        time_s[201:] += np.arange(time_s.size - 201) * 0.0005
        h5["signal/time_s"][:] = time_s
        events = h5["labels/events"][:]
        events["time_s"] = 0.22
        h5["labels/events"][:] = events
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    assert ds[1]["events"] == [("fist", pytest.approx(0.12))]


def test_adjacent_clock_windows_do_not_drop_events_between_midpoint_and_next_sample(canonical_meta_root):
    path = canonical_meta_root / "discrete_gesture" / "discrete_gesture_right.h5"
    with h5py.File(path, "r+") as h5:
        events = h5["labels/events"][:]
        events["time_s"] = 0.0999
        h5["labels/events"][:] = events
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "discrete_gesture"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    adjacent_events = ds[0]["events"] + ds[1]["events"]
    assert [name for name, _ in adjacent_events] == ["fist"]


def test_typing_preserves_pair_id_for_each_wrist(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "typing"), split="train", window_ms=100,
        stride_ms=100, root=canonical_meta_root,
    )
    items = [ds[0], ds[3]]
    assert {item["side"] for item in items} == {"left", "right"}
    assert {item["pair_id"] for item in items} == {"pair-1"}
    assert all("pair_positive" not in item for item in items)
    window = ds[1]
    assert window["keystrokes"] == [("a", pytest.approx(0.05), pytest.approx(0.051))]
    assert window["prompt_start_s"] == pytest.approx(-0.1)
    assert window["prompt_end_s"] == pytest.approx(0.1995)


def test_wrist_targets_are_name_keyed_and_aligned(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        _task_manifest(canonical_meta_root, "wrist_angle"), split="train", window_ms=200,
        stride_ms=100, root=canonical_meta_root,
    )
    item = ds[0]
    assert set(item["target"]) == {"wrist_angles"}
    assert all(value.shape == (400, 2) for value in item["target"].values())
    assert all(value.dtype == torch.bool for value in item["target_mask"].values())


# Previously a manifest no split could be selected from, or a label budget that
# kept no group, silently produced an empty dataset and a run that took zero
# steps.  These tests pin the loud failures.


def test_from_manifest_rejects_a_manifest_that_cannot_select_any_split(canonical_meta_root):
    manifest = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    manifest["official_split"] = ""
    path = canonical_meta_root / "unsplittable.csv"
    manifest.to_csv(path, index=False)

    with pytest.raises(ValueError, match="cannot select a split"):
        CanonicalWindowDataset.from_manifest(
            path, split="train", window_ms=200, stride_ms=100, root=canonical_meta_root
        )


def test_dataset_rejects_records_shorter_than_one_window(tmp_path):
    record = {
        "path": tmp_path / "short.h5", "dataset_id": "d", "task_id": "pose",
        "record_id": "r", "subject_id": "s", "session_id": "1", "side": "left",
        "pair_id": "", "official_split": "train", "sample_count": 100,
        "sample_rate_hz": 2000.0,
    }

    with pytest.raises(ValueError, match="no windows fit"):
        CanonicalWindowDataset([record], window_ms=200, stride_ms=100)


def test_label_budget_rejects_a_budget_that_keeps_no_group(pose_dataset):
    seed = 7
    points = []
    for record in pose_dataset._records.values():
        key = f"{seed}:{record['dataset_id']}:{record['subject_id']}:{record['pair_id']}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
        points.append(int.from_bytes(digest, "big") / 2**64)
    # Strictly below the smallest group point, so no group can be selected.
    budget = min(points) / 2

    with pytest.raises(ValueError, match="kept none"):
        pose_dataset.apply_label_budget(budget, seed)
