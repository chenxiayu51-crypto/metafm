# model verification

Source: `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121912_773000`

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
      "forward_s": 0.8925480999751016,
      "backward_s": 0.37918210006318986,
      "peak_memory_bytes": 21364224,
      "finite": true,
      "feature_std": 0.9486349821090698
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
      "forward_s": 0.010368800023570657,
      "backward_s": 0.00954429991543293,
      "peak_memory_bytes": 46891008,
      "finite": true,
      "feature_std": 0.9481415152549744
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
