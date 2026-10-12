# Comparison Baselines

All baselines are **core-mechanism reproductions, not official
implementations** of the cited methods. They share the canonical data
pipeline, windowing, splits, task heads, and trainer configuration with
CIRCE-Myo so that differences come from the method, not the harness.

```bash
python run_baseline.py stage=pretrain model=bendr ...        # self-supervised stage
python run_baseline.py stage=finetune model=supervised_tcn task=pose ...  # downstream stage
```

The default `configs/baseline.yaml` model is `bendr` (works for both stages);
the proposed `ring_criss_cross` model is not a baseline and is excluded from
`stage=pretrain` with a clear error.

## supervised_tcn

Causal dilated temporal-convolution encoder (`semg_fm/models/supervised_tcn.py`)
trained directly on each downstream task without self-supervised
pretraining. Establishes whether pretraining beats a competitive supervised
model. Bias-free convolutions, LayerNorm + FFN residuals, masked-value
invariance. Fine-tune-only: `stage=pretrain` is rejected.

## bendr (BENDR-style)

Conv1d latent encoder + masked temporal Transformer
(`semg_fm/models/bendr.py`) with within-record masked contrastive prediction:
negative positions come from other tokens of the same record
(`num_negatives=20`, `temperature=0.1`), no positional embeddings. Tests
whether gains come from ordinary continuous masked pretraining without
explicit ring topology. Logs `contrastive_loss`.

## cbramod (CBraMod-style)

Time-domain + FFT patch encoder with **linear** channel embedding and
parallel temporal + ordinary spatial attention (`semg_fm/models/cbramod.py`),
trained by masked waveform patch reconstruction. Uses the fixed channel
order — no circular relative-position bias, no equivariance loss — to
isolate the contribution of ring topology. Logs `reconstruction_loss`.

## aemg (AEMG-style)

Patch projection + vector-quantized codebook (512 codes, commitment 0.25)
+ masked token classification (`semg_fm/models/aemg.py`). Logs `vq_loss`
plus codebook diagnostics (usage, perplexity, active/dead codes).

## Fair-comparison rules

- Same manifest, split file, window/stride, batch collation, and seeds.
- Same task head, optimizer family, and maximum update budget in
  `stage=finetune`; the same fine-tune modes and label budgets apply.
- Parameter count, training tokens, steps, and effective global batch size
  are reported per configuration (see `EXPERIMENT_PROTOCOL.md`).
- Deviations from each paper are stated in the module docstrings and in this
  file; none of the baselines is claimed to be an official implementation.
