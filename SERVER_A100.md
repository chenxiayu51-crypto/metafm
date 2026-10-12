# A100 Server Guide

Launch, resume, and troubleshoot multi-GPU runs. All commands assume the
repository root as working directory and absolute data paths
(`hydra.job.chdir=true`).

## Setup

```bash
python -m pip install -r requirements-train.txt
# Sanity: one CPU step with synthetic fixtures
python -m pytest tests -q
```

## Launch by device count

`strategy=auto` selects DDP for multiple GPUs. Keep the **effective global
batch size** fixed when changing device count: with the base
`batch_size=8`:

| devices | accumulate_grad_batches | effective global batch |
|---|---|---|
| 1 | 8 | 64 |
| 2 | 4 | 64 |
| 4 | 2 | 64 |
| 8 | 1 | 64 |

```bash
# Pretraining on 1 / 2 / 4 / 8 devices
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR_1="/absolute/path/to/outputs/run_1"
RUN_DIR_2="/absolute/path/to/outputs/run_2"
RUN_DIR_4="/absolute/path/to/outputs/run_4"
RUN_DIR_8="/absolute/path/to/outputs/run_8"
python pretrain.py trainer=a100_ddp trainer.devices=1 trainer.accumulate_grad_batches=8 \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  hydra.run.dir="${RUN_DIR_1}" seed=0
python pretrain.py trainer=a100_ddp trainer.devices=2 trainer.accumulate_grad_batches=4 \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  hydra.run.dir="${RUN_DIR_2}" seed=0
python pretrain.py trainer=a100_ddp trainer.devices=4 trainer.accumulate_grad_batches=2 \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  hydra.run.dir="${RUN_DIR_4}" seed=0
python pretrain.py trainer=a100_ddp trainer.devices=8 trainer.accumulate_grad_batches=1 \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  hydra.run.dir="${RUN_DIR_8}" seed=0

# Downstream fine-tune and evaluate use the same trainer group
RUN_DIR="/absolute/path/to/outputs/finetune_pose"
python finetune.py trainer=a100_ddp trainer.devices=4 task=pose \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  finetune_mode=full label_budget=1.0 hydra.run.dir="${RUN_DIR}"
CHECKPOINT_RUN_DIR="/absolute/path/to/outputs/pretrain_pose"
RUN_DIR="/absolute/path/to/outputs/evaluate_pose"
python evaluate.py trainer=a100_ddp trainer.devices=4 task=pose \
  checkpoint_path="${CHECKPOINT_RUN_DIR}/checkpoints/last.ckpt" data.root="${CANONICAL_META}" \
  data.manifest_path="${CANONICAL_META}/splits.csv" hydra.run.dir="${RUN_DIR}"
```

No Slurm integration is required; launch from a plain shell on the node.

## Resume

Checkpoints retain optimizer, scheduler, EMA teacher, and global step; resume
with the same effective batch size as the original run (pass the same
`trainer.devices`, `trainer.accumulate_grad_batches`, and `data.batch_size`
as the launch command above). Pretraining runs one long step-based epoch
(`max_epochs: -1`), so every `last.ckpt` is an in-epoch checkpoint:

```bash
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR="/absolute/path/to/outputs/run"
python pretrain.py trainer=a100_ddp trainer.devices=4 \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  data.batch_size=8 trainer.accumulate_grad_batches=2 \
  resume_from="${RUN_DIR}/checkpoints/last.ckpt"
```

Generic PyTorch dataloaders cannot reproduce an arbitrary in-epoch position,
so a mid-run resume continues from the stored global step but does not replay
the interrupted position deterministically. Deterministic continuation would
require an epoch boundary, which a `max_steps` pretraining schedule never
produces by design.

To start a fine-tune from pretrained weights, do not pass the pretraining
checkpoint to `resume_from` (it expects a fine-tuning checkpoint): use
`pretrained_checkpoint`, which loads the backbone:

```bash
python finetune.py trainer=a100_ddp trainer.devices=4 task=pose \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  pretrained_checkpoint="${RUN_DIR}/checkpoints/last.ckpt"
```

## DDP smoke test (one batch)

Before long runs, verify NCCL across N devices:

```bash
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR="/absolute/path/to/outputs/run"
python run_baseline.py stage=pretrain model=bendr trainer=a100_ddp trainer.devices=4 \
  +trainer.fast_dev_run=true data.root="${CANONICAL_META}" \
  data.manifest_path="${CANONICAL_META}/splits.csv" data.num_workers=4 \
  hydra.run.dir="${RUN_DIR}"
```

`+trainer.fast_dev_run=true` needs the `+` because the a100_ddp trainer
config does not declare that key and Hydra runs in struct mode. The default
`use_distributed_sampler: false` in the same config keeps the rank-sharded
batch samplers working under DDP; leave it in place.

The test suite does not claim NCCL coverage; run this smoke on the server.

## Diagnostics

```bash
CANONICAL_META="/absolute/path/to/canonical_meta"
RUN_DIR="/absolute/path/to/outputs/run"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET python pretrain.py trainer=a100_ddp \
  data.root="${CANONICAL_META}" data.manifest_path="${CANONICAL_META}/splits.csv" \
  hydra.run.dir="${RUN_DIR}"
nvidia-smi                          # device health and memory
```

For recovery, confirm that the selected `last.ckpt` was written after a
completed epoch before relying on deterministic continuation.

## Out-of-memory reductions (preserving effective batch)

1. `data.batch_size=4` with `trainer.accumulate_grad_batches=16` (or the
   matching multiple for your device count).
2. `trainer.precision=bf16-mixed` is the default; do not switch to fp32
   before trying the above.
3. Reduce model size: `model.d_model=64 model.depth=4`.
4. `data.num_workers=4` (or 8) with `trainer.devices>1` to keep dataloading
   off the critical path.

Every run writes `run_metadata.json` with the effective global batch size,
manifest digest, git commit, and environment — attach it to any reported
number.
