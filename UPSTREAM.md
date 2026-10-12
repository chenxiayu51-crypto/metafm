# Upstream snapshot

Source: https://github.com/Amazing-hw/metafmodel
Branch: main
Retrieved: 2026-10-12
Archive: https://codeload.github.com/Amazing-hw/metafmodel/zip/refs/heads/main
Archive SHA256: 851c3630b625711f453689f9b71eb47c389b296d8e0588a57c2b2726defb08e8

This directory is a source snapshot, not a Git submodule. Commit identity was not available because the Git endpoint timed out; do not represent this snapshot as a pinned Git commit.
Original source/tests/docs are retained. No repository-level LICENSE file was present in the retrieved archive. The user identified this as a colleague's project and authorized reuse in this workspace; upstream attribution is retained here.

Local modifications:

- `semg_h5/convert_emg2pose.py`: preserve source HDF5 official split; reject conflicting overrides; use source split for CLI filtering; support official metadata `filename` and extensionless lookup.
- `semg_fm/data/manifest.py`: load CSV columns as strings to preserve leading-zero identities and hashes.

All other upstream source files are retained unchanged. The original 280 tests pass locally. Integration code and additional regression tests live outside this snapshot, in `meta_emg/canonical.py` and `tests/`.

See ../../docs/metafmodel_review.md for analysis, data restrictions and integration commands. Prefer the root project's canonical CLI for official anchors and Nature corpus intervals; the original whole-record training/split CLI does not incorporate those additions.
