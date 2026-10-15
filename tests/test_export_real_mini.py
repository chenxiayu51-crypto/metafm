"""Export of a portable real-data mini must preserve the canonical contract.

The fixture here mimics the REAL server corpus rather than the synthetic one:
Nature recordings carry a non-zero `source_time_origin_s`, emg2pose ships an
official `test` split that must never be exported, and emg2qwerty records come
in left/right pairs.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("export_real_mini", REPO / "scripts" / "export_real_mini.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return _load()


FS = 2000
SAMPLES = 16000           # 8 s, enough for 2 s windows INSIDE each corpus interval
NATURE_ORIGIN = 100.0     # non-zero on purpose: the clock must survive the export
# Every interval must be longer than one 2000 ms window, or the loader would
# legitimately produce zero windows for it. This is a real constraint on the
# official corpus, not a fixture artefact -- server_preflight.py checks it.
CORPUS = {
    "gni_discrete_gestures": [(100.5, 104.0, "train"), (104.5, 107.5, "val")],
    "gni_handwriting": [(100.5, 104.0, "train"), (104.5, 107.5, "val")],
    "gni_wrist": [(100.5, 104.0, "train"), (104.5, 107.5, "val")],
}
TEXT = h5py.string_dtype("utf-8")


def _time_s():
    return np.arange(SAMPLES, dtype=np.float64) / FS


def _write_canonical(path: Path, *, dataset_id, task_id, subject, session, side, pair_id,
                     official_split, origin, emg=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = _time_s()
    emg = np.zeros((SAMPLES, 16), dtype=np.float32) if emg is None else emg
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "schema_version": "1.0", "dataset_id": dataset_id, "task_id": task_id,
            "subject_id": subject, "session_id": session, "side": side, "pair_id": pair_id,
            "official_split": official_split, "sample_rate_hz": float(FS),
            "source_file": path.name, "preprocessing_json": "{}",
        })
        signal = h5.create_group("signal")
        signal.create_dataset("emg", data=emg)
        signal.create_dataset("time_s", data=t)
        labels = h5.create_group("labels")
        labels.create_dataset("source_json", data="{}", dtype=TEXT)
        if task_id == "pose":
            labels.create_dataset("joint_angles", data=np.zeros((SAMPLES, 20), dtype=np.float32))
            labels.create_dataset("no_ik_failure", data=np.ones(SAMPLES, dtype=np.bool_))
        elif task_id == "typing":
            labels.create_dataset("keystrokes", data=np.array(
                [(b"a", 1.0, 1.05), (b"b", 3.5, 3.55), (b"c", 5.5, 5.55)],
                dtype=[("key", TEXT), ("press_s", "f8"), ("release_s", "f8")]))
            labels.create_dataset("prompts", data=np.array(
                [(b"ab", 0.5, 2.0), (b"c", 5.0, 5.9)],
                dtype=[("text", TEXT), ("start_s", "f8"), ("end_s", "f8")]))
        elif task_id == "discrete_gesture":
            labels.create_dataset("events", data=np.array(
                [(b"fist", 1.2), (b"pinch", 3.7), (b"wave", 5.4)],
                dtype=[("name", TEXT), ("time_s", "f8")]))
        elif task_id == "handwriting":
            labels.create_dataset("prompts", data=np.array(
                [(b"ab", 0.5, 2.0), (b"c", 5.0, 5.9)],
                dtype=[("text", TEXT), ("start_s", "f8"), ("end_s", "f8")]))
        else:
            targets = labels.create_group("targets")
            targets.create_dataset("angle", data=np.stack(
                [np.arange(SAMPLES, dtype=np.float32), np.arange(SAMPLES, dtype=np.float32) + 1], axis=1))
        meta = h5.create_group("meta")
        meta.create_dataset("source_json", data=json.dumps({"source_time_origin_s": origin}), dtype=TEXT)
        quality = meta.create_group("quality")
        quality.create_dataset("sample_count", data=SAMPLES)
        quality.create_dataset("duration_s", data=float(t[-1]))
        quality.create_dataset("nan_count", data=0)
        quality.create_dataset("channel_rms", data=np.ones(16))


@pytest.fixture
def realistic_root(tmp_path):
    """A data root shaped like the real server corpus."""
    root = tmp_path / "server_canonical"
    root.mkdir()
    rows, corpus_rows = [], []

    def row(rel, dataset_id, task_id, subject, session, side, pair_id, official_split, split):
        # The converters write the raw source BASENAME into both the H5 attribute
        # and the manifest row, and window_dataset compares them for equality.
        rows.append({
            "path": rel, "dataset_id": dataset_id, "task_id": task_id, "subject_id": subject,
            "session_id": session, "side": side, "pair_id": pair_id, "official_split": official_split,
            "sample_count": str(SAMPLES), "duration_s": "6.0", "sample_rate_hz": str(FS),
            "source_file": Path(rel).name,
            "source_sha256": hashlib.sha256(rel.encode()).hexdigest(), "split": split,
        })

    # emg2pose: official train / val / test. The test file must be left behind.
    for split in ("train", "val", "test"):
        rel = f"emg2pose/pose/u_{split}/s_{split}_right.h5"
        _write_canonical(root / rel, dataset_id="emg2pose", task_id="pose", subject=f"u_{split}",
                         session=f"s_{split}", side="right", pair_id="", official_split=split, origin=0.0)
        row(rel, "emg2pose", "pose", f"u_{split}", f"s_{split}", "right", "", split, split)

    # emg2qwerty: two sessions, each a left/right pair sharing a pair_id.
    for session, split in (("s1", "train"), ("s2", "val")):
        for side in ("left", "right"):
            rel = f"emg2qwerty/typing/u_{session}/{session}_{side}.h5"
            _write_canonical(root / rel, dataset_id="emg2qwerty", task_id="typing", subject=f"u_{session}",
                             session=session, side=side, pair_id=f"emg2qwerty:{session}",
                             official_split=split, origin=0.0)
            row(rel, "emg2qwerty", "typing", f"u_{session}", session, side,
                f"emg2qwerty:{session}", split, split)

    # GNI: two sources per dataset, each carrying BOTH a train and a val interval,
    # with a non-zero source clock.
    for dataset_id, task_id in (("gni_discrete_gestures", "discrete_gesture"),
                                ("gni_handwriting", "handwriting"),
                                ("gni_wrist", "wrist_angle")):
        for index in (1, 2):
            # Basenames must be unique across ALL Nature sources: the loaders key
            # the corpus on the basename and reject duplicates.
            basename = f"{dataset_id}_rec_{index}_right.h5"
            rel = f"{dataset_id}/{task_id}/u_{index}/{basename}"
            _write_canonical(root / rel, dataset_id=dataset_id, task_id=task_id, subject=f"u_{index}",
                             session=f"rec_{index}", side="right", pair_id="",
                             official_split="unknown", origin=NATURE_ORIGIN)
            row(rel, dataset_id, task_id, f"u_{index}", f"rec_{index}", "right", "", "unknown", "")
            for start, end, split in CORPUS[dataset_id]:
                corpus_rows.append({"dataset": basename, "split": split,
                                    "start": str(start), "end": str(end)})

    columns = list(rows[0])
    with (root / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "nature_corpus.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "split", "start", "end"])
        writer.writeheader()
        writer.writerows(corpus_rows)
    return root


def _source_clock(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        times = h5["signal/time_s"][:]
        origin = float(json.loads(h5["meta/source_json"].asstr()[()])["source_time_origin_s"])
    return times + origin


# ---------------------------------------------------------------------------

def test_export_skips_official_test_and_keeps_five_tasks(exporter, realistic_root, tmp_path):
    out = tmp_path / "mini"
    metadata = exporter.export(realistic_root, out, manifest_name="splits.csv",
                               corpus_name="nature_corpus.csv", per_split=2, segment_seconds=30.0,
                               margin_seconds=1.0, seed=0, overwrite=False)

    assert metadata["schema"] == "e3-real-mini-v1"
    assert set(metadata["records_per_task"]) == set(exporter.TASKS)
    for counts in metadata["records_per_task"].values():
        assert counts["train"] > 0 and counts["val"] > 0

    exported = list(csv.DictReader((out / "splits.csv").open(encoding="utf-8")))
    assert {r["dataset_id"] for r in exported} == set(exporter.TASKS)
    # The official test recording never becomes a mini record.
    assert not any("test" in r["path"] for r in exported)
    assert not any(r["official_split"] == "test" for r in exported)
    # Official splits are carried through untouched.
    assert {r["official_split"] for r in exported if r["dataset_id"] == "emg2pose"} == {"train", "val"}


def test_export_preserves_the_source_clock(exporter, realistic_root, tmp_path):
    """`time_s + origin` must be identical to the original at every kept sample.

    This is what keeps the official corpus intervals meaningful: they are
    expressed in source time and are never rewritten.
    """
    out = tmp_path / "mini"
    exporter.export(realistic_root, out, manifest_name="splits.csv", corpus_name="nature_corpus.csv",
                    per_split=2, segment_seconds=30.0, margin_seconds=1.0, seed=0, overwrite=False)

    checked = 0
    for row in csv.DictReader((out / "splits.csv").open(encoding="utf-8")):
        original = _source_clock(realistic_root / row["path"])
        mini = _source_clock(out / row["path"])
        start = int(np.searchsorted(original, mini[0] - 1e-9))
        assert start + len(mini) <= len(original), row["path"]
        # Tolerance is far below one sample period (1/2000 s = 5e-4).
        np.testing.assert_allclose(mini, original[start:start + len(mini)], atol=1e-9, rtol=0)
        checked += 1
    assert checked > 0 and checked == len(list(csv.DictReader((out / "splits.csv").open(encoding="utf-8"))))


def test_export_keeps_nature_intervals_inside_the_window(exporter, realistic_root, tmp_path):
    out = tmp_path / "mini"
    exporter.export(realistic_root, out, manifest_name="splits.csv", corpus_name="nature_corpus.csv",
                    per_split=2, segment_seconds=30.0, margin_seconds=1.0, seed=0, overwrite=False)

    corpus = list(csv.DictReader((out / "nature_corpus.csv").open(encoding="utf-8")))
    assert corpus, "the mini must ship its official corpus rows"
    # A retained interval must be covered by the window of the record it names.
    mini_by_basename = {
        Path(r["source_file"]).name: r for r in csv.DictReader((out / "splits.csv").open(encoding="utf-8"))
        if r["dataset_id"] in exporter.NATURE_DATASETS
    }
    for entry in corpus:
        assert entry["split"] in ("train", "val")
        row = mini_by_basename[Path(entry["dataset"]).name]
        clock = _source_clock(out / row["path"])
        assert float(entry["start"]) >= clock[0] - 1e-9
        assert float(entry["end"]) <= clock[-1] + 1e-9


def test_mini_validates_against_the_real_dataset(exporter, realistic_root, tmp_path):
    out = tmp_path / "mini"
    exporter.export(realistic_root, out, manifest_name="splits.csv", corpus_name="nature_corpus.csv",
                    per_split=2, segment_seconds=30.0, margin_seconds=1.0, seed=0, overwrite=False)
    report = exporter.validate(out)
    for split in ("train", "val"):
        assert set(report[split]) == set(exporter.TASKS.values())
        assert all(count > 0 for count in report[split].values())


def test_mini_ships_checksums_and_metadata(exporter, realistic_root, tmp_path):
    out = tmp_path / "mini"
    exporter.export(realistic_root, out, manifest_name="splits.csv", corpus_name="nature_corpus.csv",
                    per_split=1, segment_seconds=30.0, margin_seconds=1.0, seed=0, overwrite=False)

    recorded = {}
    for line in (out / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    assert recorded, "checksums.sha256 must not be empty"
    for name, digest in recorded.items():
        actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert actual == digest, name

    metadata = json.loads((out / "export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["sha256"]
    assert metadata["source_corpus"]["sha256"]
    assert metadata["parameters"]["seed"] == 0

    # The mini must be relocatable: no absolute path may leak into the manifest.
    for row in csv.DictReader((out / "splits.csv").open(encoding="utf-8")):
        assert not Path(row["path"]).is_absolute()
        assert not row["path"].startswith("/")


def test_export_refuses_when_corpus_is_missing(exporter, realistic_root, tmp_path):
    (realistic_root / "nature_corpus.csv").unlink()
    with pytest.raises(exporter.ExportError, match="official Nature corpus"):
        exporter.export(realistic_root, tmp_path / "mini", manifest_name="splits.csv",
                        corpus_name="nature_corpus.csv", per_split=1, segment_seconds=30.0,
                        margin_seconds=1.0, seed=0, overwrite=False)


def test_export_refuses_nonempty_output_without_overwrite(exporter, realistic_root, tmp_path):
    out = tmp_path / "mini"
    out.mkdir()
    (out / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(exporter.ExportError, match="not empty"):
        exporter.export(realistic_root, out, manifest_name="splits.csv", corpus_name="nature_corpus.csv",
                        per_split=1, segment_seconds=30.0, margin_seconds=1.0, seed=0, overwrite=False)


def test_export_rejects_output_inside_the_data_root(exporter, realistic_root):
    with pytest.raises(exporter.ExportError, match="must not live inside"):
        exporter.main(["--data-root", str(realistic_root), "--output", str(realistic_root / "mini"),
                       "--records-per-task-per-split", "1"])


def test_tight_segment_fails_rather_than_dropping_an_official_interval(exporter, realistic_root, tmp_path):
    """A window too short to contain both intervals of a shared source must raise."""
    with pytest.raises(exporter.ExportError, match="window"):
        exporter.export(realistic_root, tmp_path / "mini", manifest_name="splits.csv",
                        corpus_name="nature_corpus.csv", per_split=2, segment_seconds=0.2,
                        margin_seconds=0.0, seed=0, overwrite=False)
