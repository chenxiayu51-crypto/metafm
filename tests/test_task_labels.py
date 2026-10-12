import h5py
import numpy as np
import pytest
import torch

from semg_fm.data.task_labels import parse_target


def test_pose_target_and_mask_are_sliced(tmp_path):
    path = tmp_path / "pose.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("signal").create_dataset("emg", data=np.zeros((4, 16), dtype=np.float32))
        labels = h5.create_group("labels")
        labels.create_dataset("joint_angles", data=np.arange(12, dtype=np.float32).reshape(4, 3))
        labels.create_dataset("no_ik_failure", data=[True, False, True, True])
    with h5py.File(path) as h5:
        result = parse_target("pose", h5, 1.0, 1.002, 1, 3)
    assert result["target"].shape == (2, 3)
    assert result["target_mask"].tolist() == [False, True]


@pytest.mark.parametrize(
    ("joint_angles", "message"),
    [(np.arange(4, dtype=np.float32), "rank-2"), (np.zeros((3, 2), dtype=np.float32), "does not align")],
)
def test_pose_rejects_non_matrix_or_misaligned_joint_angles(tmp_path, joint_angles, message):
    path = tmp_path / "bad_pose.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("signal").create_dataset("emg", data=np.zeros((4, 16), dtype=np.float32))
        h5.create_group("labels").create_dataset("joint_angles", data=joint_angles)
    with h5py.File(path) as h5:
        with pytest.raises(ValueError, match=message):
            parse_target("pose", h5, 0.0, 0.001, 0, 2)


def test_pose_rejects_nonfinite_joint_angles(tmp_path):
    path = tmp_path / "nonfinite_pose.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("signal").create_dataset("emg", data=np.zeros((4, 16), dtype=np.float32))
        joint_angles = np.zeros((4, 2), dtype=np.float32)
        joint_angles[1, 0] = np.nan
        h5.create_group("labels").create_dataset("joint_angles", data=joint_angles)
    with h5py.File(path) as h5:
        with pytest.raises(ValueError, match="finite"):
            parse_target("pose", h5, 0.0, 0.001, 0, 2)


def test_wrist_rejects_scalar_or_non_numeric_targets(tmp_path):
    path = tmp_path / "bad_wrist.h5"
    with h5py.File(path, "w") as h5:
        h5.create_group("signal").create_dataset("emg", data=np.zeros((1, 16), dtype=np.float32))
        targets = h5.create_group("labels").create_group("targets")
        targets.create_dataset("scalar", data=1.0)
    with h5py.File(path) as h5:
        with pytest.raises(ValueError, match="scalar"):
            parse_target("wrist_angle", h5, 0.0, 0.1, 0, 1)


def test_handwriting_uses_containing_prompt_and_decodes_bytes(tmp_path):
    path = tmp_path / "handwriting.h5"
    text = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_group("labels").create_dataset(
            "prompts", data=np.array([("hello", 0.0, 1.0)], dtype=[("text", text), ("start_s", float), ("end_s", float)])
        )
    with h5py.File(path) as h5:
        result = parse_target("handwriting", h5, 0.2, 0.4, 2, 4)
    assert result == {"target_text": "hello", "prompt_start_s": 0.0, "prompt_end_s": 1.0}


def test_unknown_task_is_rejected(tmp_path):
    with h5py.File(tmp_path / "empty.h5", "w") as h5:
        with pytest.raises(ValueError, match="Unsupported task_id"):
            parse_target("unknown", h5, 0.0, 0.1, 0, 1)
