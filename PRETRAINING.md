# CIRCE-Myo Pretraining

Self-supervised pretraining of the ring criss-cross sEMG encoder over the five
converted Meta 16-channel wrist corpora.

## Input representation

- Windows: `[B, T, 16]` float32 raw sEMG, time-major, 16 channels.
- Defaults: `window_ms=200` (400 samples at 2 kHz), `stride_ms=100`,
  `batch_size=8` (`configs/data/canonical_meta.yaml`).
- Windows are lazy HDF5 slices; no dataset-level normalization or filtering is
  applied (by design — see `H5_DATA_CONTRACT.md`).
- A half-open window boundary is the successor window's first timestamp; the
  final window extrapolates half a local interval.

## Architecture (`semg_fm/models/ring_criss_cross.py`, `patch_encoder.py`, `ring_attention.py`)

1. **Multi-scale patch encoder**, shared across electrodes: grouped temporal
   `Conv1d` with `kernel=stride=patch_samples` (default 40 samples = 20 ms)
   plus a `log1p(|rfft|)` spectral branch pooled to 8 bins, projected to
   `d_model` (default 128). Sharing parameters means no channel-index
   parameters exist before circular attention.
2. **Ring Criss-Cross blocks** (default `depth=6`, `heads=8`): pre-norm
   residual with parallel temporal attention (`[B*C, N, D]`) and circular
   attention (`[B*N, C, D]`, learnable signed circular relative-position bias
   indexed by `(c_i - c_j) mod 16`), fused by a learned softmax gate, then an
   FFN. **`dropout` must stay 0.0** to preserve circular equivariance in
   train mode.
3. **Outputs**: `z_eq [B,16,N,D]` (equivariant electrode field) and
   `z_motion [B,N,D]` (masked mean over the channel orbit, invariant).

## Objectives (`semg_fm/objectives/`)

`total = latent_weight * L_masked + kinematic_weight * L_kin`

- **Masked latent distillation** (`masked_latent.py`): normalized Smooth-L1 +
  cosine distance between normalized student and detached EMA-teacher latents,
  applied **only at masked token positions**. `MaskGenerator` composes
  temporal spans, ring arcs, full-channel, spatial-temporal block, and random
  patch masks and guarantees both mask states appear.
- **Ring circular equivariance** (`ring_equivariance.py`): the zero-dropout
  architecture guarantees circular equivariance. It is evaluated and logged as
  the detached `ring_equivariance_error` diagnostic, not optimized. The old
  same-student rotated comparison supplied no independent learning signal
  because exact structural equivariance makes its gradient numerically zero up
  to floating-point error. The shifted input's `z_eq` inverse-roll is compared with the original `z_eq`
  (equivariant branch), while `z_motion` is compared without a roll (invariant
  branch); the shift is fixed to 1 channel.
- **Kinematic anchoring** (`kinematic.py`): optional Smooth-L1 regression of
  `z_motion` to pose targets on valid frames. **Disabled by default**
  (`kinematic_dim=None`). The standard loader supports pose-only pretraining:
  set `+data.task_id=pose objective.kinematic_dim=20`. Tensor pose
  `target`/`target_mask` labels are padded into top-level batch fields; other,
  non-tensor labels remain in `metadata` and are not used by this objective.

## Pretraining configuration (`configs/pretrain.yaml`)

- `mask`: `temporal_span=2`, `ring_arc=3`,
  `full_channel_probability=0.1`, `block_shape=[2, 2]`, and
  `random_patch_probability=0.15`.
- `objective`: `latent_weight=1.0`, `kinematic_weight=1.0`,
  `kinematic_dim=null`, and `ema_decay=0.996`. `kinematic_dim=null` keeps the
  optional kinematic objective disabled by default.
- `optimizer`: `learning_rate=0.001`, `weight_decay=0.05`,
  `warmup_steps=1000`, and `max_steps=100000`.

## Teacher and optimizer

- The teacher is a frozen `deepcopy` of the student, updated after each
  optimizer step with `ema_decay=0.996`; `_ema_updates` counts updates.
- AdamW (`learning_rate=1e-3`, `weight_decay=0.05`), linear warmup
  (1000 steps) then cosine decay to `max_steps=100_000`.

## Data mixing

One manifest lists all five corpora. The task-balanced batch sampler
interleaves one homogeneous batch per task per round until tasks exhaust
(homogeneous batches are required by the collator). Windows beyond
`len(task_windows) // (batch_size * world_size)` full batches per task and
rank are dropped from each epoch — with corpus-scale data this is a fraction
of a percent of windows.

## Command

```bash
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR="/absolute/path/to/outputs/run"
python pretrain.py trainer=a100_ddp \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  data.batch_size=8 trainer.devices=4 trainer.accumulate_grad_batches=2 \
  hydra.run.dir="${RUN_DIR}" seed=0
```

The variables above make all CLI paths absolute (`hydra.job.chdir=true`). The manifest may be the
converter-written `manifest.csv` or a `make_splits.py` output CSV; both are
accepted.

## Resume

`Trainer.fit(…, ckpt_path=cfg.resume_from)` restores the student/teacher
model state, optimizer, scheduler, `global_step`, and EMA counter
(`_ema_updates`). Only epoch-boundary checkpoints guarantee deterministic
continuation. A generic PyTorch dataloader cannot reproduce an arbitrary
in-epoch position, so a resume from an interruption during an epoch must not
be claimed to do so.

```bash
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR="/absolute/path/to/outputs/run"
python pretrain.py trainer=a100_ddp \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  resume_from="${RUN_DIR}/checkpoints/last.ckpt"
```

## Outputs

Per run (`outputs/<date>/<time>/`): `resolved_config.yaml`, `manifest.json`
(manifest path + SHA-256), `run_metadata.json` (git commit, manifest digest,
configs, effective global batch = `batch_size × devices ×
accumulate_grad_batches`, environment), `checkpoints/` (`last.ckpt` plus the
best `val/loss` checkpoint), and TensorBoard logs under `lightning_logs/`.

Validation honours `limit_val_batches` as a per-task-divided budget: the
evaluation sampler clamps each task to its share and interleaves tasks, so
the monitored `val/loss` — and therefore the best-checkpoint selection —
spans every task in the corpus instead of only the first one. Per-task
losses are logged as `val/<task>_loss` alongside the aggregate `val/loss`.
