# E3 Data Layout and Canonical HDF5 Contract

Everything E3 needs from a data root, what each file must contain, and where it
comes from. If a file is missing, the run stops — nothing here is guessed.

```
<E3_DATA_ROOT>/
  splits.csv            # manifest + a `split` column (see §2)
  manifest.csv          # produced by the converters; splits.csv is derived from it
  nature_corpus.csv     # OFFICIAL Meta intervals for the three GNI corpora (see §3)
  <dataset_id>/<task_id>/<subject>/<session>_<side>.h5
```

---

## 1. Canonical HDF5 contract

One file per recording. `schema_version` must be `"1.0"`.

### Root attributes

Must be present and must equal the matching `splits.csv` row, except
`sample_rate_hz` which must equal the manifest value numerically:

| attribute | notes |
|---|---|
| `dataset_id` | one of the five ids in the table in [SERVER_DEPLOY.md](SERVER_DEPLOY.md) |
| `task_id` | must match the `dataset_id` pairing exactly |
| `subject_id` | subject identity; drives leakage-safe grouping |
| `session_id` | session identity |
| `side` | `left` / `right` |
| `pair_id` | links the two wrists of one emg2qwerty session; empty when unpaired |
| `official_split` | `train` / `val` / `test`, or `unknown` when the release does not ship one |
| `sample_rate_hz` | **2000**; the declared device rate, never inferred from the clock |
| `source_file` | provenance path of the raw recording |
| `preprocessing_json` | JSON string describing preprocessing |

### Datasets

| path | dtype | shape | notes |
|---|---|---|---|
| `/signal/emg` | `float32` | `[T, 16]` | raw, unfiltered, unrescaled |
| `/signal/time_s` | `float64` | `[T]` | record-relative, monotonic |
| `/meta/source_json` | `string` | — | must contain `source_time_origin_s` |

`source_time_origin_s` is what maps the record-relative clock onto the source
recording clock, and therefore what maps the official corpus intervals onto
sample indices. It is required for the three GNI corpora.

Task labels (only read by the task that needs them):

| task | label datasets |
|---|---|
| `pose` | `/labels/joint_angles` `[T, D]`, optional `/labels/no_ik_failure` `[T]` bool |
| `typing` | `/labels/keystrokes` `(key, press_s, release_s)`, `/labels/prompts` `(text, start_s, end_s)` |
| `discrete_gesture` | `/labels/events` `(name, time_s)` |
| `handwriting` | `/labels/prompts` `(text, start_s, end_s)` |
| `wrist_angle` | one or more `[T]` / `[T, D]` arrays under `/labels/targets` |

Converters also write `/meta/quality/{sample_count,duration_s,nan_count,channel_rms}`
and optionally `/labels/stage`.

**emg2qwerty is special.** Each source session produces **two** records — one
per wrist — sharing a `pair_id` of `emg2qwerty:<session>`. Unlabelled
pretraining treats them as independent single-wrist samples, but the pairing
and provenance are preserved so a downstream task can rejoin them. The two
sides must never be split across train and val.

---

## 2. `manifest.csv` and `splits.csv`

Both use the same 13 columns, in order:

```
path, dataset_id, task_id, subject_id, session_id, side, pair_id,
official_split, sample_count, duration_s, sample_rate_hz, source_file, source_sha256
```

- `path` is relative to the data root and must not escape it.
- `sample_count` must equal `T`; `sample_rate_hz` must be `2000`.

**`manifest.csv`** is produced by the five converters in `semg_h5/`, which
append to the same file in the output root.

**`splits.csv`** is `manifest.csv` plus a trailing `split` column, and is what
training reads.

- Non-Nature rows: `official_split` wins when it is `train`/`val`/`test`.
  Otherwise `make_splits` assigns whole subject groups deterministically.
  Official val/test recordings are **never** relabelled as train.
- Nature rows: `split` is intentionally **empty**. Their split comes from
  `nature_corpus.csv` at load time. `make_splits` refuses a manifest containing
  Nature rows outright, which is why `prepare_data.py` passes the non-Nature
  rows through it separately and re-attaches the Nature rows untouched.

Use `bash scripts/prepare_data.sh` rather than hand-writing this file.

---

## 3. `nature_corpus.csv` — required, never generated

The three GNI corpora are pretrained on the **official Meta corpus intervals**,
which are part of the released protocol. This file is an input you must supply.

```
dataset,split,start,end
```

| column | meaning |
|---|---|
| `dataset` | source recording filename; matched on **basename** |
| `split` | `train`, `val` or `test` |
| `start` | interval start, in **source-recording** seconds, half-open |
| `end` | interval end in source-recording seconds; must be `> start` |

Rules enforced at load time (the run fails closed on any violation):

- Basenames of Nature source files must be unique — an ambiguous match is an error.
- Intervals for one source must not overlap.
- `start`/`end` must be finite with `start < end`.
- If a record declares an `official_split`, it must agree with the corpus row;
  a conflict is an error.
- Every one of the three GNI corpora must yield at least one train **and** one
  val interval.

Intervals are mapped onto samples with
`searchsorted(time_s, bound - source_time_origin_s)`, so they are half-open
`[start, end)` in source time.

`prepare_data.py` will not create this file. If it is missing the run stops and
explains what is required and why guessing is unsafe.

---

## 4. Raw layouts the converters expect

Each converter takes `--input-root` and `--output-root` and expects one layout
per corpus. Override the discovery directory with the matching environment
variable when your download differs.

| corpus | raw layout | override |
|---|---|---|
| emg2pose | HDF5 group `emg2pose/timeseries`, compound fields `emg`, `joint_angles`, `time`, optional `no_ik_failure`; group attrs `split`, `subject_id`, `session_id`, `side` | `E3_RAW_EMG2POSE` |
| emg2qwerty | group `emg2qwerty/timeseries`, fields `emg_left`, `emg_right`, `time`; keystroke/prompt JSON attrs | `E3_RAW_EMG2QWERTY` |
| GNI discrete gestures | `data` compound with `data.attrs['task'] == "discrete_gestures"` and a 16-channel EMG field, plus `prompts` | `E3_RAW_GNI_DISCRETE_GESTURES` |
| GNI handwriting | `data` compound with `data.attrs['task'] == "handwriting"`, plus `prompts` | `E3_RAW_GNI_HANDWRITING` |
| GNI wrist | `data` compound with `data.attrs['task'] == "wrist"`; every other numeric field aligned with EMG becomes a target | `E3_RAW_GNI_WRIST` |

Each converter validates its source layout before writing and refuses an output
root nested inside its input root.

---

## 5. What the preflight verifies

`bash scripts/verify_server.sh <data-root>` checks, against the real loaders:

- `splits.csv` parses, has every required column, and contains all five corpora.
- `nature_corpus.csv` parses and its intervals are valid and non-overlapping.
- Root attributes match the manifest row for a sample of records.
- `/signal/emg` is `float32 [T,16]`, `/signal/time_s` is `float64 [T]`,
  `sample_rate_hz` is 2000, `sample_count` matches, the clock is monotonic and
  the first window is finite.
- Nature records carry `/meta/source_json` with a finite `source_time_origin_s`
  and overlap at least one corpus interval.
- Non-Nature records are not reused across splits, and no `pair_id` or
  `source_sha256` spans two splits.
- Every one of the five tasks has train and val windows, and the real
  DataLoader yields a five-task batch stream.
