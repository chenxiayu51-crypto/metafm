# Pretraining Reliability Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Meta 16-channel sEMG pretraining controls explicit, prevent the zero-gradient ring objective from being optimized, and document checkpoint replay accurately.

**Architecture:** `pretrain.py` will build `PretrainModule` from explicit Hydra objective, optimizer, and mask mappings. Masked latent and optional kinematic losses remain optimized; circular equivariance becomes a detached diagnostic. Documentation will distinguish persisted state from unsupported sample-exact mid-epoch iterator recovery.

**Tech Stack:** Python 3.13, PyTorch, Lightning, Hydra/OmegaConf, pytest.

---

### Task 1: Expose the actual pretraining method in Hydra

**Files:**

- Modify: `configs/pretrain.yaml`
- Modify: `pretrain.py`
- Modify: `tests/test_hydra_config.py`
- Modify: `tests/test_pretrain_entry.py`

- [ ] **Step 1: Write the failing configuration tests**

Add to `tests/test_hydra_config.py`:

```python
def test_pretrain_config_declares_all_optimized_objective_controls():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain")
    assert cfg.objective.latent_weight == 1.0
    assert cfg.objective.kinematic_weight == 1.0
    assert cfg.objective.kinematic_dim is None
    assert cfg.objective.ema_decay == 0.996
    assert cfg.optimizer.learning_rate == 1e-3
    assert cfg.optimizer.warmup_steps == 1_000
    assert cfg.optimizer.max_steps == 100_000
    assert cfg.mask.temporal_span == 2
    assert cfg.mask.random_patch_probability == 0.15
```

Add this import and test to `tests/test_pretrain_entry.py`:

```python
from pretrain import build_pretrain_module


def test_pretrain_module_uses_resolved_training_mappings():
    cfg = OmegaConf.create({
        "model": {"name": "ring_criss_cross", "d_model": 8, "depth": 1, "heads": 2,
                  "patch_samples": 40, "channels": 16, "dropout": 0.0},
        "objective": {"latent_weight": 2.0, "kinematic_weight": 0.5,
                      "kinematic_dim": None, "ema_decay": 0.9},
        "optimizer": {"learning_rate": 0.003, "weight_decay": 0.01,
                      "warmup_steps": 4, "max_steps": 12},
        "mask": {"temporal_span": 1, "ring_arc": 2, "full_channel_probability": 0.0,
                 "block_shape": [1, 1], "random_patch_probability": 0.0},
    })
    module = build_pretrain_module(cfg)
    assert module.latent_weight == 2.0
    assert module.kinematic_weight == 0.5
    assert module.ema_decay == 0.9
    assert module.learning_rate == 0.003
    assert module.weight_decay == 0.01
    assert module.warmup_steps == 4
    assert module.max_steps == 12
    assert module.mask_generator.temporal_span == 1
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest -q tests/test_hydra_config.py::test_pretrain_config_declares_all_optimized_objective_controls tests/test_pretrain_entry.py::test_pretrain_module_uses_resolved_training_mappings --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: FAIL because `objective`, `optimizer`, `mask`, and `build_pretrain_module` do not exist.

- [ ] **Step 3: Add explicit mappings and a module builder**

Append to `configs/pretrain.yaml`:

```yaml
mask:
  temporal_span: 2
  ring_arc: 3
  full_channel_probability: 0.1
  block_shape: [2, 2]
  random_patch_probability: 0.15
objective:
  latent_weight: 1.0
  kinematic_weight: 1.0
  kinematic_dim: null
  ema_decay: 0.996
optimizer:
  learning_rate: 0.001
  weight_decay: 0.05
  warmup_steps: 1000
  max_steps: 100000
```

In `pretrain.py`, add `build_pretrain_module(cfg)`. Resolve model, mask,
objective, and optimizer with `OmegaConf.to_container`; construct the model;
then pass the mapping values by their matching `PretrainModule` argument names.
Replace the inline module construction in `main` with this helper.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest -q tests/test_hydra_config.py::test_pretrain_config_declares_all_optimized_objective_controls tests/test_pretrain_entry.py::test_pretrain_module_uses_resolved_training_mappings --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add configs/pretrain.yaml pretrain.py tests/test_hydra_config.py tests/test_pretrain_entry.py`

Run: `git commit -m "feat: configure pretraining objectives explicitly"`

### Task 2: Convert circular equivariance to a detached diagnostic

**Files:**

- Modify: `semg_fm/lightning/pretrain_module.py`
- Modify: `tests/test_pretrain_module.py`

- [ ] **Step 1: Write the failing objective-separation test**

Add to `tests/test_pretrain_module.py`:

```python
def test_ring_equivariance_is_a_detached_diagnostic_not_an_optimized_loss():
    module = _module()
    batch = _collate([_TinyPretrainDataset()[0], _TinyPretrainDataset()[1]])
    total, losses = module._losses(batch)
    assert "ring_equivariance_error" in losses
    assert not losses["ring_equivariance_error"].requires_grad
    torch.testing.assert_close(total, losses["latent"] + losses["kinematic"])
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m pytest -q tests/test_pretrain_module.py::test_ring_equivariance_is_a_detached_diagnostic_not_an_optimized_loss --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: FAIL because the current losses contain `ring` and include it in `total`.

- [ ] **Step 3: Make the smallest loss-path change**

In `PretrainModule._losses`, retain the full-input teacher and masked student
paths. Compute the rotated-student check inside `torch.no_grad()`, pass
detached field and motion tokens to `ring_equivariance_loss`, and call it
`ring_equivariance_error`. Return:

```python
total = self.latent_weight * latent + self.kinematic_weight * kinematic
return total, {
    "latent": latent,
    "ring_equivariance_error": ring_equivariance_error,
    "kinematic": kinematic,
}
```

Do not introduce another rotation-target loss: for this exact-equivariant
model, same-model rotation targets are redundant.

- [ ] **Step 4: Run module and objective tests**

Run: `python -m pytest -q tests/test_pretrain_module.py tests/test_pretrain_objectives.py --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add semg_fm/lightning/pretrain_module.py tests/test_pretrain_module.py`

Run: `git commit -m "fix: log ring equivariance without optimizing it"`

### Task 3: Correct the public operational contract

**Files:**

- Modify: `PRETRAINING.md`
- Modify: `SERVER_A100.md`
- Modify: `tests/test_documented_commands.py`

- [ ] **Step 1: Write the failing documentation contract test**

Add to `tests/test_documented_commands.py`:

```python
def test_pretraining_docs_limit_deterministic_resume_to_epoch_boundaries():
    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in ["PRETRAINING.md", "SERVER_A100.md"]
    )
    assert "epoch-boundary" in text
    assert "mid-epoch resume replays deterministically" not in text
```

- [ ] **Step 2: Run the documentation test and verify it fails**

Run: `python -m pytest -q tests/test_documented_commands.py::test_pretraining_docs_limit_deterministic_resume_to_epoch_boundaries --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: FAIL because current documentation promises deterministic mid-epoch replay.

- [ ] **Step 3: Update docs**

In `PRETRAINING.md`, describe masked latent plus optional kinematic anchoring
as the optimized total; describe ring equivariance as a diagnostic; list the
new Hydra mappings; and limit deterministic replay to epoch-boundary
checkpoints. In `SERVER_A100.md`, use the same wording and identify
`last.ckpt` as an end-of-completed-epoch checkpoint.

- [ ] **Step 4: Run docs tests and full regression**

Run: `python -m pytest -q tests/test_documented_commands.py --basetemp=D:\Fmodel\.pytest_tmp_review`

Expected: PASS.

Run: `python -m pytest -q --basetemp=D:\Fmodel\.pytest_tmp_review -o cache_dir=D:\Fmodel\.pytest_cache_review`

Expected: all tests pass; known Lightning informational warnings do not fail the suite.

- [ ] **Step 5: Commit**

Run: `git add PRETRAINING.md SERVER_A100.md tests/test_documented_commands.py`

Run: `git commit -m "docs: clarify pretraining recovery guarantees"`
