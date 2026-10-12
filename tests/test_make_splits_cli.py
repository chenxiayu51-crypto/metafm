import subprocess
import sys

import pandas as pd
import pytest

from semg_fm.data.make_splits import main


def test_root_make_splits_assigns_unknown_and_preserves_official(tmp_path):
    record = tmp_path / "record.h5"
    record.write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["record.h5"], "dataset_id": ["d"], "task_id": ["t"],
            "subject_id": ["s"], "session_id": ["x"], "side": ["right"],
            "pair_id": [""], "official_split": ["unknown"], "sample_count": [1],
            "duration_s": [0.0], "sample_rate_hz": [2000.0],
            "source_file": ["record.h5"], "source_sha256": ["0" * 64],
        }
    ).to_csv(manifest, index=False)
    output = tmp_path / "split.csv"
    subprocess.run(
        [sys.executable, "make_splits.py", "--manifest", str(manifest), "--root", str(tmp_path),
         "--output", str(output), "--seed", "17"], check=True,
    )
    written = pd.read_csv(output)
    assert written.split.iloc[0] in {"train", "val", "test"}
    assert "resolved_path" not in written.columns


def test_output_is_independent_of_manifest_root(tmp_path):
    roots = [tmp_path / "one", tmp_path / "two"]
    outputs = []
    for root in roots:
        root.mkdir()
        (root / "record.h5").write_bytes(b"x")
        manifest = root / "manifest.csv"
        pd.DataFrame(
            {
                "path": ["record.h5"], "dataset_id": ["d"], "task_id": ["t"], "subject_id": ["s"],
                "session_id": ["x"], "side": ["right"], "pair_id": [""], "official_split": [""],
                "sample_count": [1], "duration_s": [0.0], "sample_rate_hz": [2000.0],
                "source_file": ["record.h5"], "source_sha256": ["0" * 64],
            }
        ).to_csv(manifest, index=False)
        output = root / "split.csv"
        main(["--manifest", str(manifest), "--root", str(root), "--output", str(output), "--seed", "17"])
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_invalid_official_split_is_rejected(tmp_path):
    (tmp_path / "record.h5").write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["record.h5"], "dataset_id": ["d"], "task_id": ["t"], "subject_id": ["s"],
            "session_id": ["x"], "side": ["right"], "pair_id": [""], "official_split": ["holdout"],
            "sample_count": [1], "duration_s": [0.0], "sample_rate_hz": [2000.0],
                "source_file": ["record.h5"], "source_sha256": ["0" * 64],
        }
    ).to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="invalid official_split"):
        main(["--manifest", str(manifest), "--root", str(tmp_path), "--output", str(tmp_path / "out.csv"), "--seed", "1"])


def test_official_splits_spanning_one_subject_are_preserved(tmp_path, capsys):
    """A release may publish session-level splits that put one subject's
    recordings in several splits.

    emg2pose does exactly that to define its stage-generalization benchmark, so
    the leakage assertion must apply only to the splits we assign ourselves.
    """
    for name in ("a.h5", "b.h5"):
        (tmp_path / name).write_bytes(b"x")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "path": ["a.h5", "b.h5"], "dataset_id": ["d", "d"], "task_id": ["t", "t"],
            "subject_id": ["s", "s"], "session_id": ["x", "y"], "side": ["right", "right"],
            "pair_id": ["", ""], "official_split": ["train", "test"],
            "sample_count": [1, 1], "duration_s": [0.0, 0.0], "sample_rate_hz": [2000.0, 2000.0],
            "source_file": ["a.h5", "b.h5"], "source_sha256": ["0" * 64, "1" * 64],
        }
    ).to_csv(manifest, index=False)
    output = tmp_path / "split.csv"

    main(["--manifest", str(manifest), "--root", str(tmp_path),
          "--output", str(output), "--seed", "1"])

    written = pd.read_csv(output)
    assert set(written.split) == {"train", "test"}
    assert "the release's own protocol" in capsys.readouterr().out
