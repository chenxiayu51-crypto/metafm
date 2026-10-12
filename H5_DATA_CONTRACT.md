# Canonical H5 Data Contract

Exactly what the loaders require from each converted record. This mirrors
the checks in `semg_fm/data/window_dataset.py` and
`semg_fm/data/manifest.py`.

## File layout

```text
CANONICAL_META/
  manifest.csv
  <dataset_id>/<task_id>/<subject>/<session>_<side>.h5
```

`manifest.csv` columns (fixed): `path`, `dataset_id`, `task_id`,
`subject_id`, `session_id`, `side`, `pair_id`, `official_split`,
`sample_count`, `duration_s`, `sample_rate_hz`, `source_file`,
`source_sha256`. Paths are relative to the manifest's root; the loader rejects
paths escaping the root.

`sample_rate_hz` is the rate the windows are sized with, so it must be the
declared device rate rather than a value derived from `sample_count` and
`duration_s`: `/signal/time_s` keeps the release's own clock, and dropped
samples make that span longer than `(T-1)/rate`, which would understate the
rate and shrink every window.

## Root attributes (validated on every window read)

`dataset_id`, `task_id`, `subject_id`, `session_id`, `side`, `pair_id`,
`official_split`, `sample_rate_hz` — the first seven must equal the manifest
row; `sample_rate_hz` must be positive and equal the manifest's value.

## Datasets

| path | dtype/shape | notes |
|---|---|---|
| `/signal/emg` | float32 `[T, 16]` | raw signal, no filtering/rescaling by the converters |
| `/signal/time_s` | float64 `[T]` | record-relative clock, monotonic |

Finite values are required for every selected window. Windows are half-open
clock intervals: a non-terminal window ends exactly at its successor's first
timestamp.

## Task labels (per window, made window-relative)

| task_id | required |
|---|---|
| `pose` | `/labels/joint_angles` float `[T, D]`; optional `/labels/no_ik_failure` bool `[T]` |
| `wrist_angle` | one or more float `[T]` or `[T, D]` arrays under `/labels/targets` (the GNI release ships a single `wrist_angles` target of shape `[T, 2]`) |
| `discrete_gesture` | `/labels/events` structured `(name: str, time_s: f64)` |
| `handwriting` | `/labels/prompts` structured `(text: str, start_s: f64, end_s: f64)` |
| `typing` | `/labels/keystrokes` `(key, press_s, release_s)` and `/labels/prompts` `(text, start_s, end_s)` |

## Validation commands

The loaders validate on first access; for a full preflight check use the test
fixtures' equivalent checks or:

```bash
python make_splits.py --manifest CANONICAL_META/manifest.csv --root CANONICAL_META \
  --output CANONICAL_META/splits.csv --seed 0     # manifest schema + paths + leakage assertions
python -m pytest tests/test_fm_manifest.py tests/test_window_dataset.py -q
```

The five standalone converters in `semg_h5/` produce this contract; they
share no code and each validates its source layout before writing.
