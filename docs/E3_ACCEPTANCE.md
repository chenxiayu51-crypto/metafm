# E3 Acceptance Report

Evidence for the E3 five-corpus pretraining release. Every line below records
something that was **actually executed**. Anything not executed is listed in
§8 as **not verified** — no pass is claimed for it.

Summary: the whole pipeline passes on a real-data mini at workstation scale.
Nothing here is evidence about the real five-corpus corpus, A800 hardware or
model quality.

---

## 1. Environment used for verification

| item | value |
|---|---|
| interpreter | Python 3.12.14 |
| torch | 2.8.0+cu128 |
| lightning | 2.6.6 |
| hydra-core / omegaconf | 1.3.7 / 2.3.1 |
| h5py / numpy / pandas | 3.16.0 / 2.5.3 / 3.0.6 |
| GPU | NVIDIA GeForce RTX 4060 Ti, 8 GiB, compute capability sm_89 |
| driver | 610.62, max CUDA 13.3 |
| precision | bf16-mixed (bf16 supported) |

Runs are executed on a **CUDA GPU at full model size** (`d_model=128`, depth 6,
heads 8, 1 591 588 parameters). The reduced-geometry CPU fallback was not used.

---

## 2. Test suite

```
python -m pytest tests/ -q
361 passed, 44 warnings in 22.23s
```

- The pre-existing suite is unchanged: no test removed, no assertion relaxed.
- New in this release: `tests/test_export_real_mini.py` (9),
  `tests/test_e3_fault_injection.py` (15),
  `tests/test_local_real_verification.py` (7),
  plus 8 in `tests/test_e3_server_release.py`.

`tests/test_e3_oneclick.py` (10 tests, the loader / Nature-corpus contract) is
unchanged and passing.

---

## 3. Local end-to-end training

### 3a. Synthetic fixture — `python scripts/e3_oneclick.py local`

```
[E3] Real Train/Val Dataset and five-task windows: PASS
[E3] Real Train/Val DataLoader batch coverage: PASS
`Trainer.fit` stopped: `max_steps=5` reached.
[E3] COMPLETE: .../checkpoints/last.ckpt
```

Checkpoint: `step=5, student=125, teacher=125, optimizer_states=1,
lr_schedulers=1, rng_state=true`. This proves the plumbing and nothing about
the science.

### 3b. Real mini — `python scripts/verify_local_real.py`

```
== DATA ==      PASS   pose train=7 val=7 | typing 14/14 | discrete_gesture 4/4
                       handwriting 4/4 | wrist_angle 4/4
== SAMPLING ==  PASS
== MODEL ==     PASS   parameters=1,591,588  amp=torch.bfloat16 (residual torch.float32)
                       feature std: z_motion=3.6334  z_eq=3.6333
== TRAINING ==  PASS   50 steps in 5.5s (0.078s/step)
                       loss 0.5446 -> 0.0011
                       peak GPU memory 407.9 MiB
== CHECKPOINT == PASS
ALL SECTIONS PASSED
```

The MODEL section asserts, per corpus: finite non-zero student gradients,
the student actually moves, the teacher moves **only** via EMA and never
accumulates gradients, bf16 reaches the matmul path, and features have not
collapsed.

---

## 4. The per-corpus sampling ratio, measured

`TaskBalancedBatchSampler` interleaves one batch per corpus per **round**. That
fixes the ordering only: each corpus contributes exactly
`floor(windows / batch_size)` batches, so its share of optimiser steps tracks
its **window count**, not equality.

Measured on the real mini (batch size 4), and independently reproduced by a
20-step server run (batch size 4) in `diagnostics.json`:

| corpus | local mini share | server run batches |
|---|---|---|
| pose | 14.3 % | 3 |
| typing | 42.9 % | 8 |
| discrete_gesture | 14.3 % | 3 |
| handwriting | 14.3 % | 3 |
| wrist_angle | 14.3 % | 3 |

An equal split would be 20 % each. On the real corpus the spread will be far
wider, because the GNI corpora dwarf emg2pose. **The corpus mix is therefore a
design decision that must be made deliberately and logged — it is not something
the sampler guarantees.** Changing it changes the training data distribution.

---

## 5. Resume and its guard

| scenario | observed |
|---|---|
| resume with unchanged configuration | reaches the target step; `ema_updates` continues from the checkpoint |
| resume with `max_steps` changed | **refused to start**, exit 2, `resume configuration mismatch for max_steps` |
| reload a checkpoint into a fresh module | student and teacher state dicts compare exactly equal |

**Reproducibility caveat, stated precisely:** resume restores weights, optimizer
state, LR scheduler, EMA counter, global step and RNG streams. It does **not**
replay the interrupted sample order — `CanonicalDataModule` implements no
`state_dict`/`load_state_dict`, and the sampler persists only its epoch. Weights
are unaffected, but a resumed run is not bit-identical sample-for-sample.

---

## 6. Mini export — `scripts/export_real_mini.py`

Verified by `tests/test_export_real_mini.py`, including:

- **Official test is never exported**; `official_split` is copied verbatim.
- **The source clock is preserved exactly.** For every exported record,
  `time_s + source_time_origin_s` matches the original recording at every kept
  sample to within 1e-9 s (four orders of magnitude below one sample period),
  so the official corpus intervals still map to the same source instants.
- Nature intervals survive truncation, and an interval shorter than one window
  is an error rather than a silent drop.
- The mini re-validates through the project's own `CanonicalDataModule`.
- `checksums.sha256` verifies, and no absolute path leaks into `splits.csv`.

---

## 7. Deployment surface

| script | result |
|---|---|
| `scripts/setup_server.sh` | `Environment is ready.` — resolved sm_89 onto the wheel's sm_86 cubin (binary-compatible), parsed the driver and max CUDA, and accepted torch's `+cu128` local tag against the `==2.8.0` pin |
| `scripts/verify_server.sh` | `PASS (0 failures)` — full deep path |
| `scripts/server_preflight.py` | all four sections PASS (ENVIRONMENT / DATA / DATALOADER / GPU_SMOKE) |
| `scripts/run_e3_server.sh` | preflight PASS → training `COMPLETE` at step 20; `run_metadata.json` records the git commit, manifest SHA-256 and environment |
| `scripts/prepare_data.sh` | fail-closed verified: lists missing raw corpora; refuses to invent `nature_corpus.csv` |
| `scripts/run_e3_server.sh` output guard | refuses an output root **inside** the repository before any work starts |

`tests/test_e3_fault_injection.py` drives 15 failure modes and requires each to
fail loudly: missing data root, missing manifest, missing corpus, overlapping
official intervals, missing HDF5, empty train, empty val, NaN in EMG, wrong
channel count, checkpoint/config mismatch, no CUDA, unwritable output, synthetic
fixture used for a server run, official test entering training, and an
unreadable record in the DataLoader.

---

## 8. Defects found by running the pipeline, and fixed

1. **Provenance was silently `null` on exactly the runs that matter.**
   `hydra.job.chdir` moves the process into the run directory, and a server run
   directory lives outside the checkout by design — so `git rev-parse HEAD`
   found no repository and `run_metadata.json` recorded `git_commit: null`.
   Git calls are now anchored to the package's own checkout. Regression test
   included.
2. **A batch size larger than a corpus's window count silently removed that
   corpus from training.** The preflight now detects it and names the batch size
   as the cause rather than blaming the data.
3. **The bf16 check asserted the wrong tensor.** Under autocast, `LayerNorm` and
   `softmax` are on PyTorch's fp32 list, so the residual stream is promoted back
   to fp32 — that is what "mixed" precision means, and it is the numerically
   safer arrangement. The check now asserts on the matmul path, and the run
   reports both dtypes.

Two further defects in the verification tooling itself were found the same way
(an inverted half-open overlap comparison, and a forward hook whose return value
replaced the patch encoder's output). Both are now covered by tests.

---

## 9. Not verified

1. **Real five-corpus data.** The official corpora were not present, so the
   official interval path is exercised against a faithful fixture, not the
   released corpus. No real `nature_corpus.csv` was available.
2. **A800 / A100 hardware.** No such GPU here. `devices: 1`; multi-GPU DDP was
   not exercised.
3. **Full 100000-step training.** The longest run was 50 steps, by design.
4. **Convergence or downstream accuracy.** Nothing here supports a claim about
   representation quality. Synthetic and mini metrics prove nothing about
   generalisation.
5. **Actual conversion of a real raw corpus.** `prepare_data.sh` orchestrates
   the five converters, but no raw release was converted in this session; its
   fail-closed branches were verified instead.

---

## 10. Source hygiene

- 211 tracked files in the release. No data, checkpoints or weights.
- No credentials, tokens, private keys or API keys in any tracked file.
- No machine-specific absolute paths or user names in any tracked file.
- `reports/` (30 files of machine-local evidence, including an inventory of the
  pre-existing project's private dataset holdings) is untracked and gitignored;
  working-tree copies are preserved.
- **Git history still contains those files** in earlier commits. Publishing that
  history would expose them despite `.gitignore`. Publish from the release
  branch, or rewrite history with `git filter-repo` (destructive; changes SHAs).

---

## 11. Licensing

No LICENSE file exists in this repository, so no open-source grant is implied.
`semg_h5/` and `semg_fm/models/` originate from an upstream snapshot recorded in
`UPSTREAM.md`; confirm the upstream licence and authorization before
redistributing. Dependency licences are listed in
[SERVER_DEPLOY.md](SERVER_DEPLOY.md) §9.
