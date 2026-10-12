# pose verification

Source: `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121912_773000`

```json
{
  "weight_loading": {
    "checkpoint": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel\\outputs\\pretrain\\smoke_wrist_20261012_121912_773000\\resumed\\checkpoint_last.ckpt",
    "missing_keys": [],
    "unexpected_keys": [],
    "loaded_tensors": 45,
    "strict": true,
    "configuration_checked": true
  },
  "forward": {
    "input": [
      2,
      400,
      16
    ],
    "features": [
      2,
      10,
      32
    ],
    "head": [
      2,
      10,
      20
    ],
    "target": [
      2,
      400,
      20
    ],
    "loss": 8.162158012390137,
    "metrics": {
      "mae": 2.102165937423706,
      "rmse": 2.8569490909576416,
      "count": 16000.0
    },
    "split": "val",
    "training": false,
    "time_alignment": "Same sample indices for EMG/labels/mask; linear interpolation token predictions to target length, align_corners=False",
    "units": "Source angles preserved without scaling; radian convention requires confirmation from official metadata",
    "valid_windows": 21048
  },
  "synthetic_test": {
    "probe": {
      "synthetic_loss": 1.5743143558502197,
      "frozen_parameters": 43,
      "frozen_unchanged": true,
      "head_updated": true,
      "checkpoint": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel\\outputs\\finetune\\synthetic_pose_20261012_121912_773000\\probe\\last.ckpt",
      "global_step": 2,
      "save_load_exact": true
    },
    "partial": {
      "synthetic_loss": 1.415610432624817,
      "frozen_parameters": 24,
      "frozen_unchanged": true,
      "head_updated": true,
      "checkpoint": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel\\outputs\\finetune\\synthetic_pose_20261012_121912_773000\\partial\\last.ckpt",
      "global_step": 2,
      "save_load_exact": true
    },
    "full": {
      "synthetic_loss": 1.6645479202270508,
      "frozen_parameters": 0,
      "frozen_unchanged": true,
      "head_updated": true,
      "checkpoint": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel\\outputs\\finetune\\synthetic_pose_20261012_121912_773000\\full\\last.ckpt",
      "global_step": 2,
      "save_load_exact": true
    }
  },
  "real_training": false,
  "independent_performance": false,
  "evaluation_interface": {
    "test/loss": 7.244800567626953,
    "test/pose_mae": 2.0695252418518066,
    "test/pose_rmse": 2.6916167736053467,
    "test/pose_count": 16000.0
  }
}
```
