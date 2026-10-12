"""CLI for adding deterministic leakage-safe splits to a manifest."""

import argparse

from .manifest import load_manifest
from .splits import assert_no_leakage, assign_group_splits


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
    official = frame["official_split"].fillna("").astype(str).str.strip()
    invalid = sorted(set(official) - {"", "unknown", "train", "val", "test"})
    if invalid:
        raise ValueError(f"invalid official_split values: {', '.join(invalid)}")
    assigned = frame.copy()
    assigned["split"] = official
    empty = official.isin(["", "unknown"])
    if empty.any():
        assigned.loc[empty, "split"] = assign_group_splits(
            frame.loc[empty], seed=args.seed, ratios=(0.8, 0.1, 0.1)
        )["split"].to_numpy()
        # Only the splits we just assigned have to be leakage-free.  A release
        # may deliberately publish session-level splits that put one subject's
        # recordings in several splits -- emg2pose does exactly that to define
        # its stage-generalization benchmark -- and those must be preserved
        # rather than rejected.
        assert_no_leakage(assigned.loc[empty])
    official_overlap = _cross_split_groups(assigned.loc[~empty])
    if official_overlap:
        print(
            f"note: {official_overlap} subject group(s) span splits in the official "
            "assignment; that is the release's own protocol and is preserved as-is."
        )
    assigned = assigned.drop(columns=["resolved_path"], errors="ignore")
    assigned.to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
