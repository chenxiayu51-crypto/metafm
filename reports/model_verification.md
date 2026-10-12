# model verification

Source: `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121246_587071`

```json
{
  "parameters": 35140,
  "architecture": {
    "d_model": 32,
    "depth": 2,
    "heads": 4,
    "patch_samples": 40
  },
  "measurements": [
    {
      "input": [
        2,
        400,
        16
      ],
      "output": {
        "z_eq": [
          2,
          16,
          10,
          32
        ],
        "z_motion": [
          2,
          10,
          32
        ]
      },
      "forward_s": 1.0661520001012832,
      "backward_s": 0.3092102000955492,
      "peak_memory_bytes": 21364224,
      "finite": true,
      "feature_std": 0.9825783371925354
    },
    {
      "input": [
        2,
        4000,
        16
      ],
      "output": {
        "z_eq": [
          2,
          16,
          100,
          32
        ],
        "z_motion": [
          2,
          100,
          32
        ]
      },
      "forward_s": 0.0160959999775514,
      "backward_s": 0.018010100000537932,
      "peak_memory_bytes": 46891008,
      "finite": true,
      "feature_std": 0.9790263175964355
    }
  ],
  "ring_equivariance_max_abs": 4.76837158203125e-07,
  "motion_invariance_max_abs": 2.384185791015625e-07,
  "loss_audit": {
    "ring_diagnostic_no_grad": true,
    "ring_weight_optimized": false,
    "total": "latent_weight * latent + kinematic_weight * kinematic"
  },
  "real_electrode_rotation_robustness": "NOT_RUN: electrode numbering/hand conventions unverified"
}
```
