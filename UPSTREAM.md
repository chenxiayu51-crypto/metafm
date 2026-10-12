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

On 2026-10-12 this small source snapshot was extracted from
`../MetaEMGFM/third_party/metafmodel` into the sole formal research repository.
The extraction baseline is local commit `bf322d1`; it is not an upstream commit.
The unchanged imported baseline suite was freshly run: 280 passed; evidence is
`reports/upstream_baseline_tests.log`.

Subsequent integration changes extend native window sampling with official
Nature corpus intervals, raw HDF5 adapters, metadata identity checks and safe
split anchoring; add local Hydra profiles, Lightning diagnostic callbacks and
RNG/resume-configuration checks; extend strict downstream loading and explicit
val evaluation. `semg_fm/models/` and the objectives remain the original source.
The detailed change/validation table is in `reports/final_delivery.md`.

Only format/time helper functions in `semg_fm/data/raw_signal.py` were migrated
from `../MetaEMGFM/meta_emg/data.py`. The earlier CNN and its trainer were not
migrated. Additional raw task label metadata uses upstream decoder helpers.
Historical comparison: `../MetaEMGFM/docs/metafmodel_review.md`.
