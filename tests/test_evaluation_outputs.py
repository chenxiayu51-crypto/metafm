"""Machine-readable evaluation output contracts."""

from __future__ import annotations

import json

import h5py
import pandas as pd


def test_evaluation_writes_machine_readable_outputs(tmp_path):
    from semg_fm.evaluation import write_evaluation_outputs

    manifest = tmp_path / "splits.csv"
    manifest.write_text("path,official_split\n", encoding="utf-8")

    write_evaluation_outputs(
        tmp_path,
        metrics={"test/cer": 0.25},
        per_subject=[{"subject_id": "s1", "character_error_rate": 0.25}],
        split_manifest_path=manifest,
        environment={"python": "3.13", "torch": "2.14"},
        predictions=[{"record_id": "r1", "subject_id": "s1", "reference": "ab", "hypothesis": "ac"}],
    )

    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["test/cer"] == 0.25
    frame = pd.read_csv(tmp_path / "per_subject_metrics.csv")
    assert frame.loc[0, "subject_id"] == "s1"
    assert (tmp_path / "split_manifest.csv").read_text(encoding="utf-8") == "path,official_split\n"
    assert "torch=2.14" in (tmp_path / "environment.txt").read_text(encoding="utf-8")
    with h5py.File(tmp_path / "predictions.h5", "r") as h5:
        assert h5["reference"][0].decode() == "ab"
        assert h5["hypothesis"][0].decode() == "ac"


def test_evaluation_writes_events_predictions_as_json_lists(tmp_path):
    from semg_fm.evaluation import write_evaluation_outputs

    write_evaluation_outputs(
        tmp_path,
        metrics={},
        predictions=[{"record_id": "r1", "subject_id": "s1", "predicted": [["thumb_tap", 0.12]]}],
    )

    with h5py.File(tmp_path / "predictions.h5", "r") as h5:
        assert "predicted" in h5
        assert h5["predicted"][0].decode() == '[["thumb_tap", 0.12]]'


def test_evaluation_skips_optional_outputs_when_absent(tmp_path):
    from semg_fm.evaluation import write_evaluation_outputs

    write_evaluation_outputs(tmp_path, metrics={"loss": 1.0})

    assert (tmp_path / "metrics.json").is_file()
    assert not (tmp_path / "per_subject_metrics.csv").exists()
    assert not (tmp_path / "split_manifest.csv").exists()
    assert not (tmp_path / "environment.txt").exists()
    assert not (tmp_path / "predictions.h5").exists()
