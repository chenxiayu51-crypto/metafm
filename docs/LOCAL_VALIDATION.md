# 本地验证（local validation）

在花掉服务器 A800 工时之前，先在工作站上把 E3 五语料训练通路跑通。本地验证分两级：**合成 fixture** 证明管道连通，**real mini** 在真实信号上证明整条通路可用。两者都不代表模型质量。

---

## 1. 两级本地验证

| 级别 | 命令 | 数据 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| 管道（plumbing） | `python scripts/e3_oneclick.py local` | 合成 fixture | 数据加载、五任务、真实 encoder/EMA/Lightning、checkpoint 写出 | 与真实信号无关；**对模型质量零信息** |
| 真实（real） | `python scripts/verify_local_real.py --data-root D:/emg_data/real_mini` | real mini（见 [REAL_MINI_DATASET.md](REAL_MINI_DATASET.md)） | 真实信号、工作站规模下的完整训练通路 | 不代表服务器全量吞吐/质量 |

### 1.1 `e3_oneclick.py local` —— 合成 fixture

`mode == "local"` 时，它在 `<repo>/outputs/e3_synthetic_v3` 下用 `create_synthetic()` 生成一个**最小 canonical fixture**，然后跑**仓库原始** `pretrain.py`（`trainer=e3_server`）：

- fixture 里有 `.E3_SYNTHETIC_DO_NOT_TRAIN` 标记文件（防止误用于生产）。
- 几何：`fs=2000`、`samples=8000`（4 秒，够放 2 s/1 s 窗口）；每语料的 train 与 val 各一条记录（typing 分左右）；EMG 为随机噪声 + 65 Hz 正弦；`labels/*` 按 task 写齐；`splits.csv` 的 `official_split = train/val`；`nature_corpus.csv` 为 `[0,4)` 半开区间。
- 训练 overrides（local）：`steps=5`、`batch_size=1`、`num_workers=0`；CPU 时缩小模型（`d_model=32`、`depth=2`、`heads=4`）、`kinematic_weight=0.0`；precision 用 `bf16-mixed`（CUDA 且支持 bf16）否则 `32-true`。
- 结束时用 `validate_checkpoint()` 检查 `last.ckpt` 含 student/teacher 权重、optimizer 状态、LR scheduler、RNG state。

**它只证明"管道连通"**：数据→五任务→真实前/反向→checkpoint 全链路不炸。它**不说明任何模型质量**，也不验证真实数据的时钟/corpus 语义。

> 注意：`local` 的 fixture 与运行输出会写入仓库内的 `outputs/`。只有 `server` 模式才强制输出根在仓库之外。

### 1.2 `verify_local_real.py` —— 真实信号

```bash
python scripts/verify_local_real.py --data-root D:/emg_data/real_mini
```

它**不重新实现任何东西**：每个检查都走项目自身的 `CanonicalDataModule`、`TaskBalancedBatchSampler`、`RingCrissCrossEncoder`、`PretrainModule`、`E3RunDiagnostics` 与 Lightning `Trainer`，并用服务器同一套 Hydra 配置树（`data=local_real trainer=local_real`）驱动。

参数（默认值已核对源码）：

| flag | 默认 | 含义 |
|---|---|---|
| `--data-root` | 环境变量 `E3_DATA_ROOT` | real mini 数据根（二选一必填） |
| `--manifest` | `splits.csv` | mini 根下的 manifest |
| `--corpus` | `nature_corpus.csv` | mini 根下的官方语料 |
| `--steps` | `50` | 训练 section 的优化步数 |
| `--resume-steps` | `5` | resume 后额外步数 |
| `--batch-size` | `4` | batch size |
| `--num-workers` | `0` | DataLoader workers |
| `--val-every` | `25` | `trainer.val_check_interval` |
| `--val-batches` | `5` | `trainer.limit_val_batches` |
| `--seed` | `0` | 随机种子 |
| `--output` | 环境变量 `E3_LOCAL_RUN_ROOT` 或 `outputs/local_real_runs` | 运行输出根 |
| `--skip-train` | 关 | 只跑 section 1–3（DATA/SAMPLING/MODEL） |

它要求 `<data-root>/splits.csv` 与 `<data-root>/nature_corpus.csv` 同时存在。运行目录为 `<output>/local_real_<UTCstamp>/`，报告写到 `<run>/verification.json`。**退出码 0 仅当每个 section 都通过**，否则列出失败项并返回 1。

### 1.3 PowerShell 包裹

```powershell
.\scripts\run_local_real.ps1 -DataRoot D:/emg_data/real_mini
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-DataRoot` | **必填** | mini 数据根 |
| `-Steps` | `50` | 透传 `--steps` |
| `-ResumeSteps` | `5` | 透传 `--resume-steps` |
| `-BatchSize` | `4` | 透传 `--batch-size` |
| `-NumWorkers` | `0` | 透传 `--num-workers` |
| `-ValEvery` | `25` | 透传 `--val-every` |
| `-ValBatches` | `5` | 透传 `--val-batches` |
| `-SkipTrain` | 关（switch） | 透传 `--skip-train` |

包裹脚本会先检查 `-DataRoot` 存在且含 `splits.csv`/`nature_corpus.csv`，否则 `throw`；Python 解释器按 `E3_PYTHON` → `META_PYTHON` → `python` 依次选取；底层脚本非零退出即 `throw`。

---

## 2. 五个 section 各断言什么

section 名会以大写打印（`DATA`、`SAMPLING`、`MODEL`、`TRAINING`、`CHECKPOINT`），报告里的键为小写。

### DATA

- 五语料：`setup("fit")` 后 train 与 val 的 `window_indices_by_task()` 必须**恰好**暴露 `{pose, typing, discrete_gesture, handwriting, wrist_angle}`。
- train/val 非空：每个 task 的 train、val 窗口数都必须 > 0。
- 每个 split、每个 task 抽一条真实窗口，验证契约：EMG 为 `[T,16]`、`T == 4000`（2000 ms @ 2 kHz）、dtype `float32`、全部有限（无 NaN/Inf）、`sample_mask` 全为 True、`sample_rate_hz == 2000.0`；有 `target`/`target_mask` 时其时间轴必须与 EMG 对齐。
- 泄漏检查（用 `intervals.NATURE_DATASETS`）：
  - 按 `(dataset_id, record_id)` 收集 `[sample_start, sample_end)` 半开区间；
  - 同时出现在 train 与 val 的记录**只允许**是 Nature（同一录音的官方 train/val 区间是发布协议，不是泄漏）；对共享记录校验区间两两不重叠（`a1 < b2 and b1 < a2` 即报错）；
  - 任何 `pair_id` 不得同时出现在两个 split。
- 报告：`records_per_split`、`shared_records`、`shared_records_all_nature`、`shared_pairs=0`、`overlapping_sample_ranges=0`。

### SAMPLING

- 直接枚举**真实** `TaskBalancedBatchSampler`（`rank=0, world_size=1, seed=sampler_seed`），把每个 batch 的首个 ordinal 用 `np.searchsorted(dataset._index.offsets, batch[0], side="right") - 1` 映回记录、得出 task，**实测**每语料的 batch 占比。
- 断言 sampler 至少产出一个 batch，且五个 task 都出现。
- 报告：`batches_per_epoch`、`per_task_batches`、`per_task_share`、`equal_share_would_be`（0.2）、`max_over_min_share_ratio`、`windows_per_task`、`enumeration_seconds`、`conclusion`。

> 见 §3：这里**不假定**比例相等，而是量出来。

### MODEL

- `RingCrissCrossEncoder(**model_cfg)`（去 `name`）建到 device，统计参数量。
- 逐 task 取一条真实 batch：`PretrainModule._losses(batch)` → loss 必须有限；`backward()` 后学生梯度范数有限且 > 0、无 NaN/Inf；否则报错。
- bf16（仅 CUDA）：`autocast(dtype=bfloat16)` 下学生 encoder 激活（`student(emg)["z_eq"]`）的 dtype 必须含 `bfloat16`。（loss 是 fp32 计算的，所以不能拿 loss dtype 证明 bf16。）
- EMA：跑**一次** Lightning `Trainer.fit`（单步）而不是手调 `optimizer_step`，使被测的 hook 顺序就是生产的：
  - 学生权重必须变化；
  - teacher 权重必须变化（经 EMA）且 `ema_updates >= 1`；
  - teacher **不得**有任何梯度（只经 EMA 移动）。
- 坍缩检查（CPU 上探测，与 device 无关）：`z_motion` 的 std 与 `z_eq` 的 std 都必须 `> 1e-6`。
- 报告：`parameters`、`architecture`、`device`、`amp_forward_dtype`、`per_task_loss`、`per_task_grad_norm`、`student_updated`、`teacher_updated_via_ema`、`teacher_has_no_grads`、`ema_updates`、`feature_std_z_motion`、`feature_std_z_eq`、`representation_collapsed=False`。

### TRAINING

- `seed_everything(seed, workers=True)`；用完整配置（mask / objective 权重 / ema_decay / lr / weight_decay / warmup_steps / max_steps）建 `PretrainModule`；`Trainer(**build_trainer_options(...))`。
- checkpoint 与 diagnostics 回调被重定向到本次运行目录（`_redirect_callbacks`：`dirpath`/`output_dir`/`default_root_dir` 指到 `run_dir`）。
- 断言：`<run>/diagnostics.json` 存在；`trainer.global_step >= steps`；`missing_train_tasks` 为空；`student_updated`/`teacher_updated`/`teacher_has_no_grads` 全为真；`<run>/train_metrics.jsonl` 行数 == steps 且每行 loss 有限。
- 报告：`steps_requested`、`global_step`、`wall_seconds`、`seconds_per_step`、`steps_per_second`、`peak_gpu_memory_bytes`、`train_batches_per_task`、`val_batches_per_task`、`train_loss_first/last`、`feature_std`、`ema_updates`、`precision`、`manifest_sha256`、`checkpoint_dir`。

### CHECKPOINT

- 读 `<run>/checkpoints/last.ckpt`，断言含 `state_dict`、`optimizer_states`、`lr_schedulers`、`global_step`、`rng_state`，且 `global_step == steps`。
- 重建 `PretrainModule` 并 `load_state_dict(strict=True)`：重载后的 student 与 teacher 权重必须与 checkpoint **逐张量相等**；EMA 计数 `_ema_updates` 必须等于 global step。
- resume：在独立目录 `<run>_resumed` 用 `ckpt_path=<checkpoint>` 续跑到 `steps + resume_steps`，断言 `global_step >= total_steps` 且 EMA 计数**继续增长**。
- 报告：`checkpoint`、`checkpoint_bytes`、`student_exact`、`teacher_exact`、`optimizer_states`、`lr_schedulers`、`rng_state_present`、`resumed_to_step`、`ema_continued`、`weight_sequence_reproducible`、`sample_sequence_reproducible`、`sample_sequence_note`。

> 训练与 checkpoint 两个 section 仅在未加 `--skip-train` 时运行。checkpoint 里的 `optimizer.max_steps` 是**烘焙进 checkpoint hparams 的 LR-schedule 视界**，因此 driver 把它固定成 `steps + resume_steps`，让 resume 腿能重载 hparams 匹配的 checkpoint（`PretrainModule.on_load_checkpoint` 拒绝变更的 `max_steps`），而第一腿用 `trainer.max_steps=steps` 提前停。

---

## 3. 关键发现：`TaskBalancedBatchSampler` 只平衡"顺序"，不平衡"比例"

这是必须诚实写下的结论。

`TaskBalancedBatchSampler.__iter__` 每 **ROUND** 从每个 task 取一个 batch 轮流产出，直到某 task 耗尽。这只固定了 batch 的**顺序**，**不固定数量**。

每个语料在一个 epoch 里贡献的 batch 数恰好是：

```
floor(windows / (batch_size * world_size))
```

（`_rank_batch_count`）。所以在单卡（`world_size=1`）下，**每语料的优化步占比 = 该语料的窗口数占比**，而非均分。`SAMPLING` section 正是从真实 sampler 实测这个比例并打印出来。

本仓库 mini 实测（batch_size=4，单轮 16 个 batch）：

| task | batches | 占比 |
|---|---|---|
| pose | 3 | 18.8% |
| typing | 7 | 43.8% |
| discrete_gesture | 2 | 12.5% |
| handwriting | 2 | 12.5% |
| wrist_angle | 2 | 12.5% |
| （均分应为） | — | 每项 20.0% |

可见 typing（emg2qwerty，成对记录多）占了近半，pose 只有 18.8%，三个 GNI 各 12.5%。

**后果：** 在真实全量语料上，emg2pose 的窗口数相对其他语料更少，其优化步占比会**远低于 20%**——除非改动 sampler。而改 sampler 会直接改变训练数据分布，因此这必须是一个**有意的、记录在案的**决定，而不是顺手调平。

---

## 4. 可复现性说明（resume）

resume 恢复的是：**global step、优化器状态、LR scheduler、EMA 计数、RNG 流**。这些都已验证（`CHECKPOINT` section）。

但**采样顺序不重放**：

- `CanonicalDataModule` **没有** `state_dict` / `load_state_dict`；
- `TaskBalancedBatchSampler` 只持久化它的 `epoch`（`samplers.py` 的 `state_dict` 仅返回 `{"epoch": ...}`）。

因此**中途中止的 epoch 在 resume 后不会重放被打断的样本顺序**。`CHECKPOINT` section 会把 `sample_sequence_reproducible` 记为 `False` 并附上说明。**权重与优化器状态不受影响**——不要声称样本序列可复现。

（`verify_local_real.py` 用 `hasattr(CanonicalDataModule, "state_dict") and hasattr(CanonicalDataModule, "load_state_dict")` 判定，为假。）

---

## 5. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `[verify ERROR] --data-root or E3_DATA_ROOT is required` | 传 `--data-root` 或设 `E3_DATA_ROOT` |
| `[verify ERROR] data root does not exist: <root>` | 路径写错或 mini 未就位 |
| `[verify ERROR] splits.csv is missing from <root>`（或 `nature_corpus.csv`） | mini 不完整，先用 `scripts/export_real_mini.py` 导出（见 [REAL_MINI_DATASET.md](REAL_MINI_DATASET.md)） |
| `no windows fit in the selected records: each recording is shorter than one <window_ms> ms window` | 某官方区间短于一个窗口，重新导出并用更大的 `--segment-seconds`/`--records-per-task-per-split` |
| `train split exposes tasks [...], expected [...]`（或 val、或 `task '<t>' has no train/val windows`） | mini 未覆盖五语料，重新导出 |
| `expected 4000 samples (2000 ms @ 2 kHz), got N` / `EMG contains NaN/Inf` | 数据契约被破坏，检查源 canonical 文件 |
| `the train sampler never emitted: [...]` | 某 task 窗口少于一个 batch；调小 `--batch-size` 或增大 mini |
| `bf16 autocast produced <dtype> encoder activations` | CUDA 上 bf16 未生效，检查 PyTorch/CUDA |
| `<task>: loss is not finite` / `student gradient norm is <n>` | 真实 batch 反传出问题，检查数据与学习率 |
| `a training step did not change the student` / `... the teacher (EMA is not running)` | 优化器或 EMA 未生效 |
| `the run produced no diagnostics at <path>` | diagnostics 回调未写到 `run_dir` |
| `training stopped at step X, expected Y` | 训练被提前中断 |
| `checkpoint is missing '<key>'` / `resumed run reached step X, expected Y` | checkpoint 不完整或未恢复优化器/global step |
| `[verify] WARNING: no CUDA device; section 3 will run on CPU` | 无 GPU：MODEL section 在 CPU 上跑，**不是 GPU 验收** |

---

## 6. 产物与退出码

运行目录 `<output>/local_real_<UTCstamp>/` 内含：

```
verification.json          # 完整报告（schema: e3-local-real-verification-v1）
diagnostics.json           # 训练诊断（五任务 batch 计数、feature std、EMA 计数、precision）
train_metrics.jsonl        # 每步一行：loss、step_time_s
checkpoints/last.ckpt      # 供 CHECKPOINT section 与后续 resume
```

退出码：**0** 仅当所有 section 通过；否则 **1**，并逐条列出失败项。报告即使失败也会写出。

---

## 7. 相关配置

| 文件 | 作用 |
|---|---|
| `configs/data/local_real.yaml` | `window_ms: 2000`、`stride_ms: 1000`、`batch_size: 4`、`num_workers: 0`、`sampler_seed: 0`；`root`/`manifest_path` 为 `null`，始终由 CLI/环境注入 |
| `configs/trainer/local_real.yaml` | 继承 `e3_server`（bf16、梯度裁剪、task-balanced samplers、E3RunDiagnostics、周期 `last.ckpt`），只改单卡相关：`devices: 1`、`precision: bf16-mixed`；`ModelCheckpoint` 监视 `val/loss`、`save_last: true`、`every_n_train_steps: 25` |
