import h5py
import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def canonical_meta_root(tmp_path):
    """Small canonical FM dataset used by manifest and converter tests."""
    import h5py

    root = tmp_path / "canonical"
    root.mkdir()
    records = []
    tasks = ["pose", "typing", "discrete_gesture", "handwriting", "wrist_angle"]
    for task in tasks:
        sides = ["left", "right"] if task == "typing" else ["right"]
        for side in sides:
            task_dir = root / task
            task_dir.mkdir(exist_ok=True)
            path = task_dir / f"{task}_{side}.h5"
            t = np.arange(600, dtype=np.float64) / 2000.0
            signal = np.full((600, 16), 0.1, dtype=np.float32)
            with h5py.File(path, "w") as h5:
                signal_group = h5.create_group("signal")
                signal_group.create_dataset("emg", data=signal)
                signal_group.create_dataset("time_s", data=t)
                attrs = {
                    "schema_version": "1.0",
                    "dataset_id": "canonical",
                    "task_id": task,
                    "subject_id": "subject-1",
                    "session_id": f"session-{task}",
                    "side": side,
                    "pair_id": "pair-1" if task == "typing" else "",
                    "official_split": "train",
                    "sample_rate_hz": 2000.0,
                    "source_file": path.name,
                    "preprocessing_json": "{}",
                }
                for key, value in attrs.items():
                    h5.attrs[key] = value
                if task == "pose":
                    labels = h5.create_group("labels")
                    labels.create_dataset("joint_angles", data=np.zeros((600, 3), dtype=np.float32))
                    no_ik_failure = np.ones(600, dtype=np.bool_)
                    no_ik_failure[3] = False
                    labels.create_dataset("no_ik_failure", data=no_ik_failure)
                elif task == "typing":
                    labels = h5.create_group("labels")
                    text = h5py.string_dtype("utf-8")
                    labels.create_dataset("keystrokes", data=np.array([("a", 0.15, 0.151)], dtype=[("key", text), ("press_s", np.float64), ("release_s", np.float64)]))
                    labels.create_dataset("prompts", data=np.array([("hello", 0.0, 0.2995)], dtype=[("text", text), ("start_s", np.float64), ("end_s", np.float64)]))
                elif task == "discrete_gesture":
                    labels = h5.create_group("labels")
                    text = h5py.string_dtype("utf-8")
                    labels.create_dataset("events", data=np.array([("fist", 0.15)], dtype=[("name", text), ("time_s", np.float64)]))
                elif task == "handwriting":
                    labels = h5.create_group("labels")
                    text = h5py.string_dtype("utf-8")
                    labels.create_dataset("prompts", data=np.array([("hello", 0.0, 0.2995)], dtype=[("text", text), ("start_s", np.float64), ("end_s", np.float64)]))
                else:
                    # 真实 gni_wrist H5 的 /labels/targets/wrist_angles 是
                    # (T, 2)（flexion/extension 与 radial/ulnar），夹具保持同形。
                    labels = h5.create_group("labels")
                    targets = labels.create_group("targets")
                    targets.create_dataset(
                        "wrist_angles",
                        data=np.stack(
                            [np.arange(600, dtype=np.float32), np.arange(600, dtype=np.float32) + 1],
                            axis=1,
                        ),
                    )
                labels.create_dataset("source_json", data=json.dumps({"task_id": task}), dtype=h5py.string_dtype("utf-8"))
                meta = h5.create_group("meta")
                meta.create_dataset("source_json", data=json.dumps({"source": "canonical"}), dtype=h5py.string_dtype("utf-8"))
                quality = meta.create_group("quality")
                quality.create_dataset("sample_count", data=600)
                quality.create_dataset("duration_s", data=float(t[-1]))
                quality.create_dataset("nan_count", data=0)
                quality.create_dataset("channel_rms", data=np.sqrt(np.mean(signal**2, axis=0)))
                records.append({
                    "path": str(path.relative_to(root)), "dataset_id": "canonical", "task_id": task,
                    "subject_id": "subject-1", "session_id": f"session-{task}", "side": side,
                    "pair_id": "pair-1" if task == "typing" else "", "official_split": "train",
                    "sample_count": 600, "duration_s": 0.2995, "sample_rate_hz": 2000.0,
                    "source_file": path.name, "source_sha256": "0" * 64,
                })
    pd.DataFrame(records).to_csv(root / "manifest.csv", index=False)
    return root


@pytest.fixture
def canonical_manifest_frame(canonical_meta_root):
    return pd.read_csv(canonical_meta_root / "manifest.csv", keep_default_na=False)


@pytest.fixture
def emg2pose_h5(tmp_path):
    source_path = tmp_path / "user_1" / "session_1.h5"
    source_path.parent.mkdir()
    dtype = np.dtype(
        [
            ("emg", np.float32, (16,)),
            ("joint_angles", np.float32, (3,)),
            ("time", np.float64),
            ("no_ik_failure", np.bool_),
        ]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["time"] = np.arange(4) / 2000 + 10.0
    rows["joint_angles"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    rows["no_ik_failure"] = [True, True, False, True]
    with h5py.File(source_path, "w") as h5:
        group = h5.create_group("emg2pose")
        group.attrs["subject_id"] = "user_1"
        group.attrs["session_id"] = "session_1"
        group.attrs["side"] = "left"
        group.attrs["stage"] = "capture"
        group.create_dataset("timeseries", data=rows)
    return source_path


@pytest.fixture
def qwerty_h5(tmp_path):
    source_path = tmp_path / "user_2" / "typing_1.h5"
    source_path.parent.mkdir()
    dtype = np.dtype(
        [("emg_left", np.float32, (16,)), ("emg_right", np.float32, (16,)), ("time", np.float64)]
    )
    rows = np.zeros(4, dtype=dtype)
    rows["emg_left"] = 1
    rows["emg_right"] = 2
    rows["time"] = np.arange(4) / 2000 + 20.0
    with h5py.File(source_path, "w") as h5:
        group = h5.create_group("emg2qwerty")
        group.attrs["subject_id"] = "user_2"
        group.attrs["session_id"] = "typing_1"
        group.attrs["keystrokes_json"] = json.dumps(
            [{"key": "a", "press_s": 20.0005, "release_s": 20.001}]
        )
        group.attrs["prompts_json"] = json.dumps(
            [{"text": "a", "start_s": 20.0, "end_s": 20.0015}]
        )
        group.create_dataset("timeseries", data=rows)
    return source_path


@pytest.fixture
def gni_h5_by_task(tmp_path):
    sources = {}
    task_specs = {
        "discrete_gesture": ("discrete_gestures", {"name": ["fist"], "time": [30.0005]}),
        "handwriting": (
            "handwriting",
            {"prompt": ["a"], "start": [30.0], "end": [30.0015]},
        ),
        "wrist_angle": ("wrist", None),
    }
    for canonical_task, (source_task, prompts) in task_specs.items():
        source_path = tmp_path / f"{source_task}.h5"
        fields = [("emg", np.float32, (16,)), ("time", np.float64)]
        if source_task == "wrist":
            # 与真实 gni_wrist 发布一致：wrist_angles 为 (T, 2)。
            fields.append(("wrist_angles", np.float32, (2,)))
        rows = np.zeros(4, dtype=np.dtype(fields))
        rows["emg"] = 3
        rows["time"] = np.arange(4) / 2000 + 30.0
        if source_task == "wrist":
            rows["wrist_angles"] = np.arange(8, dtype=np.float32).reshape(4, 2)
        with h5py.File(source_path, "w") as h5:
            data = h5.create_dataset("data", data=rows)
            data.attrs["task"] = source_task
            data.attrs["subject_id"] = "gni_user"
            data.attrs["session_id"] = source_task
            data.attrs["side"] = "right"
        if prompts is not None:
            pd.DataFrame(prompts).to_hdf(source_path, key="prompts", mode="a")
        sources[canonical_task] = source_path
    return sources
