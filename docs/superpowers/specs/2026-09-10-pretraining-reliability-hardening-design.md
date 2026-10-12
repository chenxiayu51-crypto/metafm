# Pretraining reliability hardening

## Objective

Make the public Meta 16-channel sEMG pretraining workflow reproducible and
methodologically honest without changing the canonical HDF5 data contract or
the ring-criss-cross backbone.

## Decisions

1. The ring criss-cross architecture is exactly circularly equivariant when
   dropout is zero.  The present student-versus-rotated-student ring loss is
   therefore identically zero (up to floating-point noise) and supplies no
   gradient.  It will become a detached `ring_equivariance_error` diagnostic,
   rather than be included in the optimized total loss.
2. Masked latent distillation remains the sole default self-supervised loss.
   Optional kinematic supervision remains disabled unless explicitly
   configured.  This avoids presenting a redundant term as an independent
   learning signal.
3. Every pretraining control currently hidden in Python defaults is exposed in
   `configs/pretrain.yaml`: mask geometry/probabilities, objective weights,
   EMA decay, optimizer parameters, and scheduler length.  The entry point
   passes this resolved configuration to `PretrainModule`, so checkpoints and
   run metadata identify the actual method.
4. Checkpoint restoration guarantees model, optimizer, scheduler, EMA, and
   global-step restoration.  The documented deterministic-replay guarantee is
   limited to epoch-boundary checkpoints; a generic PyTorch DataLoader does
   not resume an arbitrary in-epoch iterator position.

## Data flow

`pretrain.yaml` -> Hydra resolved config -> `PretrainModule` -> masked student
and full-input EMA teacher -> optimized latent (and optional kinematic) loss.
The same student and a channel-rotated view are evaluated only to log the
detached circular-equivariance error.  The HDF5 files remain raw and unchanged.

## Validation

- A red test first proves the pretraining entry point propagates configured
  objective and optimizer values, rather than using constructor defaults.
- A red test first proves the ring diagnostic does not contribute a gradient
  to the optimized loss.
- Existing EMA/checkpoint, CPU smoke, Hydra, data-contract, and downstream
  tests run unchanged after the minimal implementation.
- Documentation commands are exercised by the existing command-contract tests.

## Non-goals

- No reordering, filtering, resampling, or normalizing the stored sEMG data.
- No claim of sample-exact mid-epoch recovery.
- No new contrastive or reconstruction objective without a separately reviewed
  research design.
