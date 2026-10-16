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
import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import h5py

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


def resolve_raw_roots(args) -> dict[str, Path]:
    """Locate each corpus: explicit flag, then env var, then <raw-root>/<corpus>.

    The five public releases are often unpacked in unrelated directories (some
    on a different mount entirely), so a single parent root is a convenience,
    not a requirement.
    """
    explicit = {
        "emg2pose": args.raw_emg2pose,
        "emg2qwerty": args.raw_emg2qwerty,
        "gni_discrete_gestures": args.raw_gni_discrete_gestures,
        "gni_handwriting": args.raw_gni_handwriting,
        "gni_wrist": args.raw_gni_wrist,
    }
    base = Path(args.raw_root).expanduser().resolve() if args.raw_root else None
    resolved: dict[str, Path] = {}
    unresolved: list[str] = []
    for name in CONVERTERS:
        value = explicit[name] or os.environ.get(ENVIRONMENT_VARIABLES[name])
        if value:
            resolved[name] = Path(value).expanduser().resolve()
        elif base is not None:
            resolved[name] = base / name
        else:
            unresolved.append(name)
    if unresolved:
        raise SystemExit(
            "[E3] no location given for: " + ", ".join(unresolved) + "\n"
            "     Pass --raw-<corpus> for each one (or set the matching environment\n"
            "     variable, or give --raw-root containing <corpus>/ subdirectories)."
        )
    return resolved


def _converter_module(dataset_id: str):
    spec = importlib.util.spec_from_file_location(
        f"convert_{dataset_id}", REPO / "semg_h5" / CONVERTERS[dataset_id])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_files(root: Path, limit: int = 3) -> list[Path]:
    found = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in (".h5", ".hdf5") and p.is_file()]
    return found[:limit]


def probe_corpus(dataset_id: str, root: Path, samples: int = 2) -> dict:
    """Try the converter's OWN reader on real files. Read-only; nothing is written.

    Running the actual reader is the only honest pre-check: a structural guess
    can pass while the converter still rejects the file, and a wrong guess that
    survives costs hours of conversion before it fails.
    """
    report: dict = {"root": str(root), "status": "UNKNOWN"}
    if not root.is_dir():
        report.update(status="FAIL", error=f"not a directory: {root}")
        return report
    files = [p for p in sorted(root.rglob("*"))
             if p.suffix.lower() in (".h5", ".hdf5") and p.is_file()]
    report["hdf5_files"] = len(files)
    if not files:
        report.update(status="FAIL", error="no .h5/.hdf5 files found below this directory")
        return report

    module = _converter_module(dataset_id)
    sample = files[0]
    report["sample"] = str(sample)
    try:
        with h5py.File(sample, "r") as handle:
            report["top_level"] = sorted(handle.keys())
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        report.update(status="FAIL", error=f"cannot open as HDF5: {exc}")
        return report

    reader = getattr(module, "_read_emg2pose", None) or getattr(module, "read_emg2qwerty", None) \
        or getattr(module, "_read_gni", None)
    if reader is None:
        report.update(status="FAIL", error="converter exposes no reader to test with")
        return report
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = reader(sample, "unknown")
            except TypeError:
                parsed = reader(sample)
    except Exception as exc:  # noqa: BLE001 - the message is the whole point
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        return report

    records = parsed if isinstance(parsed, list) else [parsed]

    def field(record, name):
        """The converters return both dataclasses and plain dicts."""
        value = record.get(name) if isinstance(record, dict) else getattr(record, name, None)
        return value

    report["records_per_file"] = len(records)
    report["status"] = "PASS"
    report["parsed"] = []
    for record in records[:4]:
        emg = field(record, "emg")
        report["parsed"].append({
            "task_id": field(record, "task_id"), "subject_id": field(record, "subject_id"),
            "side": field(record, "side"), "pair_id": field(record, "pair_id"),
            "samples": int(len(emg)) if emg is not None else 0,
        })
    return report


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
    parser.add_argument("--raw-emg2pose", default=None, help="raw emg2pose release (overrides --raw-root)")
    parser.add_argument("--raw-emg2qwerty", default=None, help="raw emg2qwerty release")
    parser.add_argument("--raw-gni-discrete-gestures", default=None, help="raw GNI discrete gestures release")
    parser.add_argument("--raw-gni-handwriting", default=None, help="raw GNI handwriting release")
    parser.add_argument("--raw-gni-wrist", default=None, help="raw GNI wrist release")
    parser.add_argument("--canonical-root", default=os.environ.get("E3_DATA_ROOT"),
                        help="destination canonical data root (or set E3_DATA_ROOT)")
    parser.add_argument("--seed", type=int, default=0, help="seed for non-official group splits")
    parser.add_argument("--overwrite", action="store_true", help="replace existing canonical records")
    parser.add_argument("--skip-conversion", action="store_true",
                        help="canonical HDF5 already exists; only (re)build splits.csv")
    parser.add_argument("--inspect", action="store_true",
                        help="read-only: test-read a sample from each raw corpus with the "
                             "converter's own reader, then exit without converting anything")
    args = parser.parse_args(argv)

    # --inspect never writes, so it does not need a destination.
    if args.inspect:
        resolved = resolve_raw_roots(args)
        print("[E3] inspecting raw corpora (read-only; nothing is written)\n")
        failures = 0
        for dataset_id in CONVERTERS:
            report = probe_corpus(dataset_id, resolved[dataset_id])
            mark = "PASS" if report["status"] == "PASS" else "FAIL"
            failures += report["status"] != "PASS"
            print(f"  [{mark}] {dataset_id}")
            print(f"         path    : {report['root']}")
            print(f"         hdf5    : {report.get('hdf5_files', 0)} file(s) found")
            if report.get("top_level"):
                print(f"         groups  : {', '.join(report['top_level'])}")
            if report["status"] == "PASS":
                print(f"         sample  : {report['sample']}")
                for record in report.get("parsed", [])[:2]:
                    print(f"         parsed  : task={record['task_id']} subject={record['subject_id']} "
                          f"side={record['side']} samples={record['samples']}")
            else:
                print(f"         error   : {report.get('error')}")
            print()
        if failures:
            print(f"[E3] {failures} corpus/corpora could not be read by the converter.")
            print("     Fix the layout (see docs/DATA_LAYOUT.md section 4) before converting.")
            return 1
        print("[E3] all five corpora read cleanly. Convert with the same flags, minus --inspect.")
        return 0

    if not args.canonical_root:
        raise SystemExit("[E3] --canonical-root (or E3_DATA_ROOT) is required")
    canonical = Path(args.canonical_root).expanduser().resolve()
    canonical.mkdir(parents=True, exist_ok=True)

    if not args.skip_conversion:
        resolved = resolve_raw_roots(args)
        missing = {name: path for name, path in resolved.items() if not path.is_dir()}
        if missing:
            lines = ["[E3] these raw corpus directories were not found:"]
            lines += [f"       {name:<24} {path}" for name, path in missing.items()]
            lines.append("     Every corpus must be present: E3 pretrains on all five jointly.")
            lines.append("     Run with --inspect first to check what is actually there.")
            raise SystemExit("\n".join(lines))
        for name, path in resolved.items():
            if canonical == path or path in canonical.parents:
                raise SystemExit(
                    f"[E3] canonical root {canonical} sits inside the raw corpus {path};\n"
                    "     the converters refuse that layout too."
                )
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
