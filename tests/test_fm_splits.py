import pandas as pd
import pytest

from semg_fm.data.splits import assign_group_splits, assert_no_leakage


@pytest.fixture
def manifest_frame():
    return pd.DataFrame(
        {
            "path": ["a.h5", "b.h5", "c.h5", "d.h5"],
            "dataset_id": ["d1", "d1", "d2", "d3"],
            "subject_id": ["s1", "s1", "s2", "s3"],
            "pair_id": ["pair-1", "pair-1", "", ""],
            "official_split": ["", "", "", ""],
        }
    )


def test_qwerty_pair_never_crosses_splits(manifest_frame):
    assigned = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    pair_rows = assigned[assigned.pair_id == "pair-1"]
    assert pair_rows.split.nunique() == 1
    assert_no_leakage(assigned)


def test_assignment_is_deterministic(manifest_frame):
    a = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    b = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    assert a[["path", "split"]].equals(b[["path", "split"]])


def test_invalid_ratios_fail(manifest_frame):
    with pytest.raises(ValueError, match="ratios"):
        assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.2, 0.2))


def test_nonfinite_ratios_fail(manifest_frame):
    with pytest.raises(ValueError, match="ratios"):
        assign_group_splits(manifest_frame, seed=17, ratios=(float("nan"), 0.2, 0.8))


def test_forced_split_leak_raises(manifest_frame):
    leaked = manifest_frame.assign(split=["train", "test", "train", "val"])
    with pytest.raises(ValueError, match="leak"):
        assert_no_leakage(leaked)


def test_subject_cannot_leak_across_pair_ids_or_unpaired_rows():
    frame = pd.DataFrame(
        {
            "dataset_id": ["d", "d", "d"],
            "subject_id": ["s", "s", "s"],
            "pair_id": ["p1", "p2", ""],
            "split": ["train", "val", "test"],
        }
    )
    with pytest.raises(ValueError, match="leak"):
        assert_no_leakage(frame)


def test_connected_subject_pair_component_gets_one_split():
    frame = pd.DataFrame(
        {
            "path": ["a", "b", "c"],
            "dataset_id": ["d", "d", "d"],
            "subject_id": ["s", "s", "s"],
            "pair_id": ["p1", "p2", ""],
        }
    )
    assigned = assign_group_splits(frame, seed=17, ratios=(0.7, 0.15, 0.15))
    assert assigned.split.nunique() == 1


def test_same_pair_id_in_different_datasets_is_not_merged():
    frame = pd.DataFrame(
        {
            "path": ["a", "b"],
            "dataset_id": ["d1", "d2"],
            "subject_id": ["s1", "s2"],
            "pair_id": ["p", "p"],
            "official_split": ["", ""],
        }
    )
    # Namespaced pair keys make the two rows independent components, so their
    # hashed assignments diverge for at least one seed.
    outcomes = [
        assign_group_splits(frame, seed=seed).loc[0, "split"]
        == assign_group_splits(frame, seed=seed).loc[1, "split"]
        for seed in range(24)
    ]
    assert not all(outcomes)
