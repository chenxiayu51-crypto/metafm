# E3 Server Deployment

Deploy and run E3 five-corpus foundation-model pretraining on a Linux GPU host
(NVIDIA A800 / A100 / Ampere or newer).

E3 is the single supported pretraining target. It trains one Ring Criss-Cross
encoder on five corpora jointly:

| corpus | `dataset_id` | `task_id` |
|---|---|---|
| emg2pose | `emg2pose` | `pose` |
| emg2qwerty | `emg2qwerty` | `typing` |
| GNI Discrete Gestures | `gni_discrete_gestures` | `discrete_gesture` |
| GNI Handwriting | `gni_handwriting` | `handwriting` |
| GNI Wrist | `gni_wrist` | `wrist_angle` |

Input is always 16-channel, 2000 Hz sEMG. The pipeline is:

```
canonical HDF5 -> official train/val -> task-balanced sampler
  -> Ring Criss-Cross encoder -> masked student + EMA teacher
  -> masked latent loss -> Lightning -> student checkpoint
```

---

## 1. Directory layout

Source and data never mix. The repository is disposable; the data and the runs
are not.

```
/srv/projects/metafmodel/          cloned source (this repository)
/srv/datasets/meta_canonical/      canonical data root  -> E3_DATA_ROOT
/srv/experiments/e3/               run outputs          -> E3_RUN_ROOT
/srv/datasets/raw/                 optional: raw corpora, converted in place
```

All three paths are configurable; nothing is hardcoded. See
[DATA_LAYOUT.md](DATA_LAYOUT.md) for the exact content of the data root.

> Outputs must live **outside** the checkout. `e3_oneclick.py` refuses to start
> a server run whose output root is inside the repository, so checkpoints and
> logs can never be committed or overwritten by a `git clean`.

---

## 2. Install

```bash
git clone <your-fork-url> /srv/projects/metafmodel
cd /srv/projects/metafmodel
```

Create a dedicated environment. Do not upgrade a shared CUDA environment that
other projects depend on.

```bash
python3 -m venv .venv
export E3_PYTHON=/srv/projects/metafmodel/.venv/bin/python
```

Install PyTorch from the CUDA wheel index that matches the host driver, then the
pinned dependencies. `scripts/setup_server.sh` prints the exact command for the
detected driver instead of installing anything itself:

```bash
bash scripts/setup_server.sh        # inspect and report; installs nothing
```

Follow the commands it prints, then re-run it until it reports
`Environment is ready.` Typically that is:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server.txt
```

The verified version set is [requirements-server.txt](../requirements-server.txt).
It is a tested-together set, not a guess: changing a version can alter training
numerics or break the Lightning/Hydra wiring.

---

## 3. Prepare the data

If you already have canonical HDF5 with `splits.csv` and `nature_corpus.csv`,
skip to section 4.

If you only have **raw** HDF5 downloads:

```bash
bash scripts/prepare_data.sh /srv/datasets/raw /srv/datasets/meta_canonical
```

This runs the five existing converters, then `make_splits`, and stops.

**It cannot do everything, and it will not pretend to.** Two things are
required and are never generated for you:

1. **`nature_corpus.csv` — the official Meta corpus intervals.** The three GNI
   corpora are pretrained on the released time intervals, not on whole
   recordings. Guessing them would silently train on validation or test data.
   The script stops with an explanation if this file is missing.
2. **Official splits for emg2pose / emg2qwerty.** `make_splits` preserves the
   release's own `official_split` and only assigns groups the release left
   unassigned. Validation and test recordings are never relabelled as train,
   and the official test split is never used for pretraining.

See [DATA_LAYOUT.md](DATA_LAYOUT.md) for the file formats.

---

## 4. Verify before training

```bash
export E3_DATA_ROOT=/srv/datasets/meta_canonical
export E3_RUN_ROOT=/srv/experiments/e3
bash scripts/verify_server.sh "$E3_DATA_ROOT"
```

Read-only. It checks the GPU and bf16 support, every dependency, the repository
entry points, the data root, `splits.csv`, `nature_corpus.csv`, that all five
corpora have valid train **and** val windows, that the canonical HDF5 agrees
with the manifest metadata, and that the real DataLoader yields five-task
batches. It reuses the repository's own validators, so the preflight exercises
the same code path a launch uses.

Exit status is 0 only when nothing failed. Do not launch on a FAIL.

---

## 5. Launch pretraining

```bash
bash scripts/run_e3_server.sh /srv/datasets/meta_canonical
```

That is the whole command. It delegates to `scripts/e3_oneclick.py server`,
which runs the repository's own `pretrain.py` through Hydra — there is no
second training entry point to keep in sync.

Environment overrides:

| variable | default | meaning |
|---|---|---|
| `E3_DATA_ROOT` | — | canonical data root (positional argument wins) |
| `E3_RUN_ROOT` | `/srv/experiments/e3` | where runs are written |
| `E3_NATURE_CORPUS` | `<data-root>/nature_corpus.csv` | official corpus CSV |
| `E3_PYTHON` | `python` | interpreter to use |

Extra arguments are forwarded, e.g. `--max-steps 2000` for a short pilot.

### Default training configuration

| setting | value |
|---|---|
| input channels | 16 |
| sampling rate | 2000 Hz |
| window / stride | 2000 ms / 1000 ms |
| patch samples | 40 |
| model | Ring Criss-Cross, `d_model=128`, depth 6, heads 8 |
| objective | masked latent + EMA teacher (`ema_decay=0.996`) |
| kinematic auxiliary loss | **off** (`kinematic_dim=null`, weight `0.0`) |
| precision | bf16-mixed (server mode refuses to run without bf16) |
| batch size | 8 |
| optimizer | AdamW, lr 3e-4, weight decay 0.05, 1000-step warmup, cosine decay |
| total steps | 100000 |
| validation | every 2000 steps, 250 batches |

These are the intended starting configuration, **not tuned optima**.

---

## 6. What the run writes

Everything lands under `$E3_RUN_ROOT/e3_server_<UTC-timestamp>/`:

```
e3_status.json          DATAFLOW_OK -> COMPLETE; preflight and checkpoint summary
command.json            the exact Hydra argv used
e3_train.log            full stdout/stderr of pretrain.py
resolved_config.yaml    the fully resolved Hydra configuration
run_metadata.json       git commit SHA, manifest sha256, environment, trainer config
manifest.json           data manifest path + sha256
train_metrics.jsonl     one JSON row per optimizer step
diagnostics.json        per-corpus batch counts, student/teacher status, feature stats
e3_error.json           ONLY on divergence: the non-finite step
checkpoints/last.ckpt               most recent, written every 1000 steps
checkpoints/epoch=N-step=M.ckpt     best val/loss
```

Reliability guarantees:

- **Checkpointing.** `last.ckpt` every 1000 steps; best `val/loss` retained
  (`save_top_k=1`). A crash costs at most 1000 steps.
- **Contents.** Student, teacher, optimizer, LR scheduler, global step and
  RNG state (python/numpy/torch/CUDA). The launcher refuses to report success
  unless all of these are present.
- **Per-corpus accounting.** `diagnostics.json` records how many batches each
  corpus actually contributed, in train and in validation.
- **Student/teacher reporting.** `diagnostics.json` records whether each was
  updated, that the teacher received no gradients, the EMA update count, and
  feature-std statistics.
- **NaN/Inf stop.** Non-finite loss, feature std or gradient norm — or
  non-finite parameters — stops the run and writes `e3_error.json` naming the
  offending step and task. A bad update is never written into a checkpoint.
- **Provenance.** Git commit SHA, random seed, full resolved config, manifest
  SHA-256 and environment are saved per run.

---

## 7. Resume after an interruption

```bash
bash scripts/run_e3_server.sh /srv/datasets/meta_canonical \
  --resume /srv/experiments/e3/e3_server_<timestamp>/checkpoints/last.ckpt
```

Resume is strict: global step, optimizer, LR schedule, EMA counter and RNG
streams are all restored.

The configuration is re-validated on load. If any training-relevant
hyperparameter differs from the checkpoint (model geometry, mask settings,
weights, learning rate, `max_steps`, `ema_decay`), the run **refuses to start**
with `resume configuration mismatch for <key>` rather than silently training
with different semantics. Start a new experiment instead.

Resuming a run that already reached its step limit performs no further work and
completes immediately; `e3_status.json` then reports
`"checkpoint_source": "resume_input"`.

---

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `SERVER mode requires CUDA` | No visible GPU. Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`. Server mode never falls back to CPU. |
| `SERVER default requires verified bf16 support` | The device or torch build lacks bf16. Reinstall a matching CUDA wheel. |
| `Nature corpus not found` | Supply the official `nature_corpus.csv`. It is never inferred. |
| `Synthetic HDF5 fixture is forbidden` | The data root is the local smoke fixture. Point at real canonical data. |
| `Train/Val interval or file missing` | A corpus lacks a train or val window. All five must have both. |
| `resume configuration mismatch for X` | Changing `X` on resume. Restore the original value or start a new run. |
| `Server output root must live outside the Git repository` | Export `E3_RUN_ROOT` to a path outside the checkout. |
| `non-finite training state at step N` | Divergence. Inspect `e3_error.json`, then resume from the previous `last.ckpt`, possibly lowering the learning rate. |

---

## 9. Licensing

This repository contains **no LICENSE file**, so no open-source grant is
implied. The source snapshot in `semg_h5/` and `semg_fm/models/` was imported
from an upstream project (see [UPSTREAM.md](../UPSTREAM.md)); confirm the
upstream licence and your colleague's authorization before publishing or
redistributing. Third-party dependencies carry their own licences — notably
PyTorch (BSD-3-Clause), Lightning (Apache-2.0), Hydra/OmegaConf (MIT),
h5py (BSD-3-Clause), NumPy (BSD-3-Clause), pandas (BSD-3-Clause),
PyTables (BSD-3-Clause). Review these before any redistribution.
