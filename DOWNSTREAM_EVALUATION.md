# Downstream Tasks and Evaluation

Fine-tuning and held-out evaluation for the five converted corpora.

## Tasks, labels, and metrics

| task_id | kind | labels (per window) | primary metric |
|---|---|---|---|
| `pose` | regression | `/labels/joint_angles [T,20]` + optional `/labels/no_ik_failure` mask | masked RMSE |
| `wrist_angle` | regression | one named array under `/labels/targets` | masked RMSE |
| `discrete_gesture` | events | `/labels/events` `(name, time_s)` made window-relative | event F1 (tolerance 0.05 s) |
| `handwriting` | CTC | `/labels/prompts` containing/intersecting the window | character error rate |
| `typing` | CTC | prompt text for the window (keystrokes kept for pair grouping) | character error rate |

Event F1 uses label-aware one-to-one tolerance matching (`semg_fm/metrics/events.py`);
CER uses per-example Levenshtein distance aggregated globally. Primary metrics
are accumulated over the whole val/test stage, all-gathered across ranks, and
logged once per epoch (`test/<task>_event_f1`, `test/<task>_character_error_rate`).

Task configs live in `configs/task/*.yaml`. Before training, verify:

- `discrete_gesture.yaml` `class_names` matches the converter-emitted event
  vocabulary (9 GNI classes; a mismatch raises at the first batch).
- `pose.yaml` `output_dim: 20` matches the converted `joint_angles` width.
- `wrist_angle.yaml` `target_name` exists under `/labels/targets` (a missing
  name raises listing the available names).
- CTC vocabularies cover the converted prompts (a missing character raises
  naming it).

## Splits

- The loader prefers non-empty `official_split` values; records without one
  use the deterministic `split` column written by `make_splits.py`.
- `make_splits.py` assigns train/val/test at subject-group level (union-find
  over subject and `pair_id` components, SHA-256 hashed with the seed,
  ratios 0.8/0.1/0.1) and asserts no subject, session, or qwerty pair leaks
  across splits. Its output CSV is the manifest passed to every run.

## Fine-tuning protocols

```bash
python finetune.py task=pose finetune_mode=full label_budget=1.0 ...
```

- `finetune_mode=probe` freezes the backbone; `partial` trains only the last
  `unfrozen_blocks` backbone blocks; `full` trains everything.
- `label_budget` is the fraction of **subject groups** used for training:
  `0.01`, `0.05`, `0.1`, or `1.0` (1/5/10/100%). Selection is deterministic
  per seed (`data.sampler_seed`); validation and test splits are never
  budgeted.
- `pretrained_checkpoint=<ckpt>` initializes the backbone from a pretraining
  checkpoint (`student.*`/`backbone.*`/`model.encoder.*` prefixes are
  stripped automatically).

## Evaluation and robustness

```bash
python evaluate.py task=discrete_gesture \
  checkpoint_path="'/absolute/path/to/run/checkpoints/epoch=0-step=4.ckpt'" \
  robustness=channel_rotation robustness.shift=4 data.root=... data.manifest_path=... \
  save_predictions=true
```

`checkpoint_path` must be absolute (Hydra chdirs into the run directory), and
a checkpoint filename contains `=` (`epoch=<n>-step=<n>.ckpt`), so wrap the
path in a second pair of quotes to keep Hydra from parsing it as an override.

`configs/robustness/` provides in-memory, evaluation-only transforms that
never modify HDF5 files:

- `none` — unperturbed reference;
- `channel_rotation` — circular ring shift (`shift` 1/2/4/8 = 22.5/45/90/180°);
- `channel_dropout` — zero a contiguous electrode run (`count=1` is random
  single-channel dropout; fixed `start` simulates adjacent missing
  electrodes);
- `gain_drift` — linear per-window gain between `start_gain` and `end_gain`;
- `additive_noise` — zero-mean Gaussian noise with `std` and a fixed `seed`;
- `temporal_loss` — zero `count` samples and mark them missing.

Each run writes `metrics.json`, `per_subject_metrics.csv`, `split_manifest.csv`
(a copy of the consumed split file), `environment.txt`, `run_metadata.json`,
`resolved_config.yaml`, and optionally `predictions.h5` (decoded CTC text or
event lists). Synthetic rotations establish algorithmic channel-phase
robustness only; they are not evidence of real re-wearing robustness without
physical re-wearing metadata.
