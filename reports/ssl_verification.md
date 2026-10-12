# ssl verification

Source: `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121246_587071`

```json
{
  "stage": "SMOKE TEST",
  "batch_size": 2,
  "window_ms": 200,
  "first": {
    "stage": "smoke",
    "steps": [
      {
        "step": 1,
        "student_gradient_norm": 98.96739959716797,
        "feature_std": 0.5731073021888733,
        "loss": 0.8175375461578369,
        "loss_latent": 0.8175375461578369,
        "learning_rate": 0.0009938441702975688,
        "GPU_memory_bytes": 21515776,
        "step_time_s": 0.9609497000928968
      },
      {
        "step": 2,
        "student_gradient_norm": 22.30592155456543,
        "feature_std": 1.3067809343338013,
        "loss": 0.6296691298484802,
        "loss_latent": 0.6296691298484802,
        "learning_rate": 0.0009755282581475768,
        "GPU_memory_bytes": 22477824,
        "step_time_s": 0.05342840007506311
      },
      {
        "step": 3,
        "student_gradient_norm": 8.557161331176758,
        "feature_std": 1.3261476755142212,
        "loss": 0.5255230069160461,
        "loss_latent": 0.5255230069160461,
        "learning_rate": 0.0009455032620941839,
        "GPU_memory_bytes": 22479872,
        "step_time_s": 0.054554900038056076
      },
      {
        "step": 4,
        "student_gradient_norm": 6.048893451690674,
        "feature_std": 1.52810800075531,
        "loss": 0.5518441200256348,
        "loss_latent": 0.5518441200256348,
        "learning_rate": 0.0009045084971874737,
        "GPU_memory_bytes": 22483968,
        "step_time_s": 0.05573100002948195
      },
      {
        "step": 5,
        "student_gradient_norm": 6.560628890991211,
        "feature_std": 0.8844493627548218,
        "loss": 0.3678833246231079,
        "loss_latent": 0.3678833246231079,
        "learning_rate": 0.0008535533905932737,
        "GPU_memory_bytes": 22483968,
        "step_time_s": 0.052184500033035874
      },
      {
        "step": 6,
        "student_gradient_norm": 6.359366416931152,
        "feature_std": 0.8920457363128662,
        "loss": 0.26979488134384155,
        "loss_latent": 0.26979488134384155,
        "learning_rate": 0.0007938926261462366,
        "GPU_memory_bytes": 22479872,
        "step_time_s": 0.05858569999691099
      },
      {
        "step": 7,
        "student_gradient_norm": 4.390637397766113,
        "feature_std": 1.1384025812149048,
        "loss": 0.4112584590911865,
        "loss_latent": 0.4112584590911865,
        "learning_rate": 0.0007269952498697733,
        "GPU_memory_bytes": 22471680,
        "step_time_s": 0.060707799973897636
      },
      {
        "step": 8,
        "student_gradient_norm": 3.183382749557495,
        "feature_std": 1.0032382011413574,
        "loss": 0.2787296175956726,
        "loss_latent": 0.2787296175956726,
        "learning_rate": 0.0006545084971874737,
        "GPU_memory_bytes": 22473728,
        "step_time_s": 0.059377299970947206
      },
      {
        "step": 9,
        "student_gradient_norm": 1.9047317504882812,
        "feature_std": 1.194994330406189,
        "loss": 0.3531495928764343,
        "loss_latent": 0.3531495928764343,
        "learning_rate": 0.0005782172325201155,
        "GPU_memory_bytes": 22473728,
        "step_time_s": 0.05889360001310706
      },
      {
        "step": 10,
        "student_gradient_norm": 2.074868679046631,
        "feature_std": 1.1109340190887451,
        "loss": 0.32353463768959045,
        "loss_latent": 0.32353463768959045,
        "learning_rate": 0.0005,
        "GPU_memory_bytes": 22473728,
        "step_time_s": 0.051877700025215745
      }
    ],
    "student_updated": true,
    "teacher_updated": true,
    "teacher_has_no_grads": true,
    "ema_updates": 10,
    "global_step": 10,
    "elapsed_s": 1.7958912000758573,
    "resume": {},
    "amp_status": "NOT_RUN: precision 32",
    "sample_sequence_reproducible": false
  },
  "resumed": {
    "stage": "smoke",
    "steps": [
      {
        "step": 11,
        "student_gradient_norm": 1.8517632484436035,
        "feature_std": 1.0549657344818115,
        "loss": 0.3193764388561249,
        "loss_latent": 0.3193764388561249,
        "learning_rate": 0.0004217827674798847,
        "GPU_memory_bytes": 21814784,
        "step_time_s": 0.8725983999902382
      },
      {
        "step": 12,
        "student_gradient_norm": 2.4590494632720947,
        "feature_std": 1.6499545574188232,
        "loss": 0.3096262812614441,
        "loss_latent": 0.3096262812614441,
        "learning_rate": 0.00034549150281252633,
        "GPU_memory_bytes": 22471680,
        "step_time_s": 0.05650209996383637
      },
      {
        "step": 13,
        "student_gradient_norm": 2.342060089111328,
        "feature_std": 1.6954914331436157,
        "loss": 0.6372452974319458,
        "loss_latent": 0.6372452974319458,
        "learning_rate": 0.00027300475013022663,
        "GPU_memory_bytes": 22483968,
        "step_time_s": 0.0603902000002563
      },
      {
        "step": 14,
        "student_gradient_norm": 1.7100589275360107,
        "feature_std": 1.0306358337402344,
        "loss": 0.2014930099248886,
        "loss_latent": 0.2014930099248886,
        "learning_rate": 0.00020610737385376348,
        "GPU_memory_bytes": 22465536,
        "step_time_s": 0.06529319996479899
      },
      {
        "step": 15,
        "student_gradient_norm": 4.134342193603516,
        "feature_std": 1.4698530435562134,
        "loss": 0.33722078800201416,
        "loss_latent": 0.33722078800201416,
        "learning_rate": 0.00014644660940672628,
        "GPU_memory_bytes": 22477824,
        "step_time_s": 0.06019960006233305
      },
      {
        "step": 16,
        "student_gradient_norm": 2.0693929195404053,
        "feature_std": 0.9775229096412659,
        "loss": 0.1612827181816101,
        "loss_latent": 0.1612827181816101,
        "learning_rate": 9.549150281252633e-05,
        "GPU_memory_bytes": 22469632,
        "step_time_s": 0.06392550002783537
      },
      {
        "step": 17,
        "student_gradient_norm": 2.5645196437835693,
        "feature_std": 1.8632389307022095,
        "loss": 0.5066304802894592,
        "loss_latent": 0.5066304802894592,
        "learning_rate": 5.449673790581611e-05,
        "GPU_memory_bytes": 22465536,
        "step_time_s": 0.06338249996770173
      },
      {
        "step": 18,
        "student_gradient_norm": 1.9597684144973755,
        "feature_std": 1.0652027130126953,
        "loss": 0.19239754974842072,
        "loss_latent": 0.19239754974842072,
        "learning_rate": 2.4471741852423235e-05,
        "GPU_memory_bytes": 22473728,
        "step_time_s": 0.05868150002788752
      },
      {
        "step": 19,
        "student_gradient_norm": 1.5044420957565308,
        "feature_std": 0.985028088092804,
        "loss": 0.21148961782455444,
        "loss_latent": 0.21148961782455444,
        "learning_rate": 6.15582970243117e-06,
        "GPU_memory_bytes": 22496256,
        "step_time_s": 0.057962599908933043
      },
      {
        "step": 20,
        "student_gradient_norm": 1.4687663316726685,
        "feature_std": 1.1972012519836426,
        "loss": 0.34405308961868286,
        "loss_latent": 0.34405308961868286,
        "learning_rate": 0.0,
        "GPU_memory_bytes": 22463488,
        "step_time_s": 0.04805099999066442
      }
    ],
    "student_updated": true,
    "teacher_updated": true,
    "teacher_has_no_grads": true,
    "ema_updates": 20,
    "global_step": 20,
    "elapsed_s": 1.8313626000890508,
    "resume": {
      "source": "C:/Users/18165/Documents/ChatGPT/metafmodel/outputs/pretrain/smoke_wrist_20261012_121246_587071/first/checkpoint_last.ckpt",
      "saved_step": 10,
      "student_exact": true,
      "teacher_exact": true,
      "rng_saved": true,
      "global_step": 10,
      "optimizer_exact": true,
      "scheduler_exact": true
    },
    "amp_status": "NOT_RUN: precision 32",
    "sample_sequence_reproducible": false
  },
  "checkpoint": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel\\outputs\\pretrain\\smoke_wrist_20261012_121246_587071\\resumed\\checkpoint_last.ckpt",
  "checkpoint_sha256": "b2c01a366bf965d760a4642516a66a7977a2412afd01f8f02ddeefb938380512"
}
```
