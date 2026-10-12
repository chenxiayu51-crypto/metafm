# Experiment Protocol

Fixed conventions for every reported configuration so tables stay
comparable.

## Main table

For each of the five tasks:

| column | convention |
|---|---|
| methods | `ring_criss_cross`, `supervised_tcn`, `bendr`, `cbramod`, `aemg` |
| pretraining | one run per method, `seed=0` (self-supervised methods only) |
| fine-tuning | `finetune_mode=full`, `label_budget=1.0`, seeds `0, 1, 2` |
| data | `make_splits.py --seed 0` output CSV used by every run |
| metrics | report the task's primary metric from `metrics.json` (masked RMSE / event F1 / CER) |
| compute | report parameter count, training tokens (windows × samples), steps, effective global batch size, GPU-hours |

## Low-label transfer

For budgets `0.01`, `0.05`, `0.1`, `1.0`, repeat the fine-tuning table with
`label_budget=<fraction>` and the same three seeds. Budget selection is
deterministic per `data.sampler_seed`; keep `sampler_seed` fixed across the
budget sweep so that each budget is a subset of larger budgets' subject
groups.

## Fine-tune mode ablation

`finetune_mode=probe`, `partial` (`unfrozen_blocks=1`), and `full`, each with
the pretrained `ring_criss_cross` backbone at `label_budget=1.0`.

## Robustness table

Run `evaluate.py` with `robustness=none|channel_rotation|channel_dropout|
gain_drift|additive_noise|temporal_loss` over the same held-out checkpoint
and split file; report the primary metric per perturbation and severity
(e.g. `channel_rotation.shift=1,2,4,8`). State that synthetic rotations test
algorithmic channel-phase robustness, not physical re-wearing.

## Ablations and comparisons

- Ring topology contribution: `bendr`/`cbramod` (no ring bias) vs
  `ring_criss_cross`.
- Self-supervised pretraining contribution: `supervised_tcn` vs
  `ring_criss_cross` at the same budget.
- Masked continuous prediction without equivariance: `bendr` (contrastive)
  vs `cbramod` (reconstruction) vs CIRCE-Myo.
- Quantized token prediction: `aemg`.

The kinematic objective (`kinematic_dim`) and the `latent_weight` /
`kinematic_weight` mix are constructor-level options of `PretrainModule`;
report them in the configuration column when changed. The ring-equivariance
term is a detached diagnostic only — it appears in metrics but never enters
the optimized loss, and the `ring_weight` attribute does not scale it (pinned
by `tests/test_pretrain_module.py`).

## Reporting checklist

- [ ] Split file digest (`split_manifest.csv` + `run_metadata.json`) in the appendix
- [ ] Git commit of every run
- [ ] Seeds, effective global batch size, precision, and device count
- [ ] Parameter count per method
- [ ] Event `class_names` and CTC vocabularies verified against converted data
- [ ] Robustness results labeled "synthetic", not re-wearing

## Sampler note

The task-balanced sampler trains
`floor(windows_per_task / (batch_size × world_size))` full batches per task
and rank; the remainder of each task is not trained in that epoch. At corpus
scale this drops well under 1% of windows; do not use tiny toy manifests
(< `batch_size × world_size` windows per task) for reported numbers.
