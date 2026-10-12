import pandas as pd
import pytest

from semg_fm.data.manifest import load_manifest


def test_manifest_resolves_records_and_preserves_task_ids(canonical_meta_root):
    frame = load_manifest(canonical_meta_root / "manifest.csv", canonical_meta_root)
    assert set(frame.task_id) == {"pose", "typing", "discrete_gesture", "handwriting", "wrist_angle"}
    assert frame.resolved_path.map(lambda p: p.exists()).all()


def test_manifest_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"path": ["x.h5"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path, tmp_path)


def test_manifest_rejects_a_missing_sample_rate_column(canonical_meta_root, tmp_path):
    """The rate must come from the manifest; a gapped clock cannot supply it."""
    frame = pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)
    path = tmp_path / "manifest.csv"
    frame.drop(columns=["sample_rate_hz"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        load_manifest(path, canonical_meta_root)


def test_manifest_rejects_directory_record_path(tmp_path):
    path = tmp_path / "manifest.csv"
    columns = {
        "path": ["record.h5"],
        "dataset_id": ["canonical"], "task_id": ["pose"], "subject_id": ["s"],
        "session_id": ["session"], "side": ["right"], "pair_id": [""],
        "official_split": ["train"], "sample_count": [1], "duration_s": [0.0],
        "sample_rate_hz": [2000.0], "source_file": ["record.h5"], "source_sha256": ["0" * 64],
    }
    (tmp_path / "record.h5").mkdir()
    pd.DataFrame(columns).to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="missing record paths"):
        load_manifest(path, tmp_path)


@pytest.mark.parametrize("record_path", ["C:/outside.h5", "../outside.h5"])
def test_manifest_rejects_paths_outside_root(tmp_path, record_path):
    path = tmp_path / "manifest.csv"
    columns = {
        "path": [record_path],
        "dataset_id": ["canonical"], "task_id": ["pose"], "subject_id": ["s"],
        "session_id": ["session"], "side": ["right"], "pair_id": [""],
        "official_split": ["train"], "sample_count": [1], "duration_s": [0.0],
        "sample_rate_hz": [2000.0], "source_file": ["outside.h5"], "source_sha256": ["0" * 64],
    }
    pd.DataFrame(columns).to_csv(path, index=False)
    with pytest.raises(ValueError, match="outside root"):
        load_manifest(path, tmp_path)


def test_canonical_fixture_has_root_schema_attributes(canonical_meta_root):
    import h5py

    with h5py.File(canonical_meta_root / "pose" / "pose_right.h5", "r") as h5:
        assert h5.attrs["schema_version"]
        assert h5.attrs["source_file"]
        assert h5.attrs["preprocessing_json"]
        assert h5["labels/source_json"].asstr()[()]
        assert h5["meta/source_json"].asstr()[()]
        assert h5["meta/quality/sample_count"][()] == 600
        assert h5["meta/quality/duration_s"][()] == pytest.approx(0.2995)
        assert h5["meta/quality/nan_count"][()] == 0
        assert h5["meta/quality/channel_rms"].shape == (16,)
