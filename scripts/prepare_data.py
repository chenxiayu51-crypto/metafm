#!/usr/bin/env python3
"""Build a canonical five-corpus data root from raw HDF5 for E3 pretraining.

    python scripts/prepare_data.py --raw-root /srv/datasets/raw \
                                   --canonical-root /srv/datasets/meta_canonical

This wraps the repository's five existing converters and its `make_splits`
tool.  It writes no new data format and invents no splits.

Two hard rules it enforces, both of which the E3 pipeline depends on:

1. ``nature_corpus.csv`` is NEVER generated.  The three GNI corpora are sliced
   by the *official* Meta corpus intervals, so that file is an input the
   operator must supply.  This script stops with an explanation instead of
   guessing.

2. ``make_splits`` refuses any manifest containing a Nature row (they must be
   split by corpus interval, never by whole-file random assignment).  The
   non-Nature rows are therefore passed through it on their own and the Nature
   rows are re-attached untouched.

Raw layout -- each corpus in its own directory, overridable by environment
variable when the download layout differs:

    <raw-root>/emg2pose/                 E3_RAW_EMG2POSE
    <raw-root>/emg2qwerty/               E3_RAW_EMG2QWERTY
    <raw-root>/gni_discrete_gestures/    E3_RAW_GNI_DISCRETE_GESTURES
    <raw-root>/gni_handwriting/          E3_RAW_GNI_HANDWRITING
    <raw-root>/gni_wrist/                E3_RAW_GNI_WRIST
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# dataset_id -> converter module. Mirrors the canonical ids used everywhere else.
CONVERTERS = {
    "emg2pose": "convert_emg2pose.py",
    "emg2qwerty": "convert_emg2qwerty.py",
    "gni_discrete_gestures": "convert_gni_discrete_gestures.py",
    "gni_handwriting": "convert_gni_handwriting.py",
    "gni_wrist": "convert_gni_wrist.py",
}
NATURE_DATASETS = frozenset({"gni_discrete_gestures", "gni_handwriting", "gni_wrist"})
ENVIRONMENT_VARIABLES = {name: f"E3_RAW_{name.upper()}" for name in CONVERTERS}

CORPUS_HELP = """\
[E3] nature_corpus.csv was not found in {root}

    This file is REQUIRED and this script will not create it. The three GNI
    corpora (gni_wrist, gni_handwriting, gni_discrete_gestures) are pretrained
    on the OFFICIAL Meta corpus time intervals, not on whole recordings, and
    those intervals are part of the released protocol. Generating or guessing
    them would silently train on validation or test data.

    Supply the official corpus as a CSV with exactly these columns:

        dataset,split,start,end

      dataset  source recording filename (the basename is what is matched)
      split    one of train | val | test
      start    interval start in SOURCE-recording seconds (half-open)
      end      interval end in SOURCE-recording seconds (must be > start)

    then either place it at {root}/nature_corpus.csv or point at it with
    E3_NATURE_CORPUS=/path/to/nature_corpus.csv.

    See docs/DATA_LAYOUT.md for the full contract.
"""


def raw_root_for(dataset_id: str, raw_root: Path) -> Path:
    override = os.environ.get(ENVIRONMENT_VARIABLES[dataset_id])
    return Path(override).expanduser() if override else raw_root / dataset_id


def run_converter(dataset_id: str, input_root: Path, output_root: Path, overwrite: bool) -> None:
    script = REPO / "semg_h5" / CONVERTERS[dataset_id]
    if not script.is_file():
        raise SystemExit(f"[E3] converter missing from the repository: {script}")
    argv = [sys.executable, str(script), "--input-root", str(input_root), "--output-root", str(output_root)]
    if overwrite:
        argv.append("--overwrite")
    print(f"[E3] converting {dataset_id}: {input_root} -> {output_root}", flush=True)
    completed = subprocess.run(argv, cwd=REPO)
    if completed.returncode:
        raise SystemExit(
            f"[E3] {dataset_id} conversion FAILED (exit {completed.returncode}).\n"
            f"     Re-run it alone to see the full error:\n"
            f"       {' '.join(argv)}"
        )


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_splits(canonical: Path, seed: int) -> Path:
    """Attach a `split` column to manifest.csv, preserving official splits."""
    sys.path.insert(0, str(REPO))
    from semg_fm.data.make_splits import main as make_splits_main

    manifest = canonical / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"[E3] no canonical manifest produced at {manifest}")
    rows, columns = read_rows(manifest)
    if not rows:
        raise SystemExit(f"[E3] canonical manifest is empty: {manifest}")

    present = {row["dataset_id"] for row in rows}
    absent = sorted(set(CONVERTERS) - present)
    if absent:
        raise SystemExit(
            f"[E3] canonical manifest is missing corpora: {absent}\n"
            "     All five must be converted before E3 pretraining."
        )

    nature_rows = [row for row in rows if row["dataset_id"] in NATURE_DATASETS]
    other_rows = [row for row in rows if row["dataset_id"] not in NATURE_DATASETS]

    # make_splits validates group/pair consistency and refuses Nature rows, so
    # it only ever sees the non-Nature subset. Its output is the authoritative
    # source of the `split` column for emg2pose / emg2qwerty.
    work = canonical / ".e3_non_nature_manifest.csv"
    assigned_path = canonical / ".e3_non_nature_splits.csv"
    try:
        write_rows(work, other_rows, columns)
        make_splits_main([
            "--manifest", str(work),
            "--root", str(canonical),
            "--output", str(assigned_path),
            "--seed", str(seed),
        ])
        assigned_rows, assigned_columns = read_rows(assigned_path)
    finally:
        work.unlink(missing_ok=True)
        assigned_path.unlink(missing_ok=True)

    # Nature rows keep their converter-written official_split (usually
    # 'unknown') and an empty `split`: their split comes from the corpus
    # intervals at load time, never from this file.
    for row in nature_rows:
        row.setdefault("split", "")
        row["split"] = ""

    out_columns = assigned_columns if "split" in assigned_columns else [*columns, "split"]
    combined = [{column: row.get(column, "") for column in out_columns} for row in [*assigned_rows, *nature_rows]]

    splits = canonical / "splits.csv"
    write_rows(splits, combined, out_columns)
    print(f"[E3] wrote {splits} ({len(combined)} records: {len(assigned_rows)} split-assigned, {len(nature_rows)} corpus-driven)")
    return splits


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default=os.environ.get("E3_RAW_ROOT"),
                        help="directory holding one subdirectory per corpus")
    parser.add_argument("--canonical-root", default=os.environ.get("E3_DATA_ROOT"),
                        help="destination canonical data root")
    parser.add_argument("--seed", type=int, default=0, help="seed for non-official group splits")
    parser.add_argument("--overwrite", action="store_true", help="replace existing canonical records")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="canonical HDF5 already exists; only (re)build splits.csv")
    args = parser.parse_args(argv)

    if not args.canonical_root:
        raise SystemExit("[E3] --canonical-root (or E3_DATA_ROOT) is required")
    canonical = Path(args.canonical_root).expanduser().resolve()
    canonical.mkdir(parents=True, exist_ok=True)

    if not args.skip_conversion:
        if not args.raw_root:
            raise SystemExit("[E3] --raw-root (or E3_RAW_ROOT) is required unless --skip-conversion is used")
        raw_root = Path(args.raw_root).expanduser().resolve()
        if not raw_root.is_dir():
            raise SystemExit(f"[E3] raw root does not exist: {raw_root}")
        if canonical == raw_root or raw_root in canonical.parents:
            raise SystemExit(
                f"[E3] canonical root {canonical} must not sit inside the raw root {raw_root};\n"
                "     the converters refuse that layout too."
            )
        resolved = {name: raw_root_for(name, raw_root) for name in CONVERTERS}
        missing = {name: path for name, path in resolved.items() if not path.is_dir()}
        if missing:
            lines = [f"[E3] these raw corpus directories were not found:"]
            lines += [f"       {name:<24} {path}   (set {ENVIRONMENT_VARIABLES[name]})" for name, path in missing.items()]
            lines.append("     Every corpus must be present: E3 pretrains on all five jointly.")
            raise SystemExit("\n".join(lines))
        for name, path in resolved.items():
            run_converter(name, path, canonical, args.overwrite)

    corpus = Path(os.environ.get("E3_NATURE_CORPUS", canonical / "nature_corpus.csv")).expanduser()
    if not corpus.is_file():
        raise SystemExit(CORPUS_HELP.format(root=canonical))

    splits = build_splits(canonical, args.seed)

    print("\n[E3] data root is ready. Verify it before training:\n")
    print(f"      export E3_DATA_ROOT={canonical}")
    print(f"      export E3_RUN_ROOT=/srv/experiments/e3")
    print(f"      bash scripts/verify_server.sh {canonical}")
    print(f"\n    splits.csv  : {splits}")
    print(f"    corpus      : {corpus}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
