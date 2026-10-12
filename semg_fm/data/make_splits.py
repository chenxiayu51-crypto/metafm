"""CLI for adding deterministic leakage-safe splits to a manifest."""

import argparse

from .manifest import load_manifest
from .intervals import anchored_group_splits


def _cross_split_groups(frame) -> int:
    """Count (dataset, subject) groups whose official rows span several splits."""
    if frame.empty:
        return 0
    grouped = frame.groupby(["dataset_id", "subject_id"], dropna=False)["split"].nunique(dropna=False)
    return int((grouped > 1).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)

    frame = load_manifest(args.manifest, args.root)
    assigned=anchored_group_splits(frame,args.seed)
    assigned=assigned.drop(columns=['resolved_path'],errors='ignore')
    official_overlap=_cross_split_groups(assigned.loc[assigned['official_split'].isin(['train','val','test'])])
    if official_overlap:
        print(f'note: {official_overlap} subject group(s) span splits in the official assignment; that is the release\'s own protocol and is preserved as-is.')
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    assigned.to_csv(args.output,index=False,lineterminator='\n')


if __name__ == "__main__":
    main()
