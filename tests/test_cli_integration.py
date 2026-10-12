import subprocess
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_converter_cli_writes_a_valid_tree_and_manifest(tmp_path, qwerty_h5):
    output_root = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str((REPOSITORY_ROOT / "semg_h5" / "convert_emg2qwerty.py").resolve()),
            "--input-root",
            str(qwerty_h5.parent),
            "--output-root",
            str(output_root),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(output_root / "manifest.csv")) == 2
