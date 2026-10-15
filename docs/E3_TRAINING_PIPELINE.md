# E3 端到端训练流水线

本文回答一件事：**E3 训练到底是什么**。它从三个运行模式、数据路径、单个训练步的精确计算、一次 run 落盘的文件，到可靠性保证与 preflight 门禁，给出可核对的参考。

数据契约本身（HDF5 schema、`splits.csv` / `nature_corpus.csv` 的列）见 [DATA_LAYOUT.md](DATA_LAYOUT.md)；部署与运维见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。本文不重复其内容，只在其之上描述「训练」这一环节。

占位符约定：`<repo>` 是源码检出目录，`<data-root>` 是 canonical 数据根，`<run-root>` 是输出根。三者互不嵌套，源码目录是可丢弃的，数据与 run 不是。

---

## 1. 三种运行模式

E3 只有一个训练实现。三种模式只是**同一实现的三种装配**：编码器、EMA teacher-student、masked-latent 目标、数据集与 collate、优化器、checkpoint 语义完全一致，差异仅在数据集规模、batch size、worker 数、步数和输出路径。

| 模式 | 入口 | 数据 | 目的 | 关键默认 |
|---|---|---|---|---|
| `local_synthetic` | `python scripts/e3_oneclick.py local` | 现场生成的合成 fixture（`.E3_SYNTHETIC_DO_NOT_TRAIN` 标记） | 工程冒烟：证明管线能跑通 | 5 步；batch 1；workers 0；无 CUDA 时用缩小模型 `d_model=32, depth=2, heads=4`，有 CUDA 时用完整模型 |
| `local_real` | `python scripts/verify_local_real.py --data-root <data-root>` | 真实 mini（少量真实录制） | 在工作站尺度上用**真实信号**验证同一条管线，再上服务器 | 5 个 section；`data=local_real trainer=local_real`；batch 4；workers 0 |
| `server_full` | `bash scripts/run_e3_server.sh <data-root>` | 完整五语料 canonical 数据根 | 生产预训练 | 100000 步；batch 8；workers 8；bf16-mixed 强制 |

三种模式共同走同一条代码路径：

- `scripts/e3_oneclick.py` 的 `command()`（`scripts/e3_oneclick.py:415`）始终以 `pretrain.py trainer=e3_server` 启动，即 `local_synthetic` 与 `server_full` 都是 **`pretrain.py` → Hydra → Lightning `Trainer`**，没有第二条训练入口。
- `scripts/verify_local_real.py` 不重新实现任何东西：它直接构造项目自身的 `CanonicalDataModule`、`TaskBalancedBatchSampler`、`RingCrissCrossEncoder`、`PretrainModule`、`E3RunDiagnostics` 与 Lightning `Trainer`，并用与服务器相同的 Hydra 配置树（`data=local_real trainer=local_real`，而 `trainer/local_real.yaml` 继承 `e3_server`）。
- 因此三者共享：`semg_fm/lightning/pretrain_module.py` 的 `PretrainModule`、`semg_fm/models/ring_criss_cross.py` 的 `RingCrissCrossEncoder`、`semg_fm/objectives/masking.py` 的 `MaskGenerator`、`semg_fm/objectives/masked_latent.py` 的 `masked_latent_loss`、`semg_fm/data/datamodule.py` 的 `task_homogeneous_collate`、以及 AdamW + LambdaLR 与 checkpoint/EMA 语义。

`server_full` 的运行链是：

```
<data-root> -> scripts/server_preflight.py（独立进程）
            -> 任一 section 失败则 STOP
            -> scripts/e3_oneclick.py server
            -> <repo>/pretrain.py（Hydra，trainer=e3_server）
```

`local_synthetic` 与 `local_real` 是**工程结果**，不是表示质量的证据——见第 7 节。

---

## 2. 数据路径

```
原始 HDF5
  -> 五个转换器 (semg_h5/convert_emg2pose.py, convert_emg2qwerty.py,
     convert_gni_discrete_gestures.py, convert_gni_handwriting.py, convert_gni_wrist.py)
  -> <data-root>/manifest.csv
  -> make_splits（只处理非 Nature 行）-> <data-root>/splits.csv（manifest + split 列）
  +  <data-root>/nature_corpus.csv（官方 Meta 语料区间，人工提供，绝不生成）
```

- 五个转换器由 `scripts/prepare_data.py`（`CONVERTERS`，`scripts/prepare_data.py:43`）逐一调用，追加写入同一个 `manifest.csv`。
- `splits.csv` 是训练唯一读取的清单：`CanonicalWindowDataset.from_manifest` 优先取 `official_split`，否则回退到 `split` 列（`semg_fm/data/window_dataset.py:198-203`）。
- **`make_splits` 拒绝任何含 Nature 行的 manifest。** `anchored_group_splits` 在发现 `dataset_id ∈ NATURE_DATASETS` 时直接 `raise ValueError('Nature requires official corpus intervals, not whole-file random splits')`（`semg_fm/data/intervals.py:40-41`）。因此 `prepare_data.py:133-159` 把 Nature 行拆出去、只把非 Nature 行喂给 `make_splits`，再原样拼回。
- 三个 GNI 语料（`gni_discrete_gestures` / `gni_handwriting` / `gni_wrist`）的 split 来自 corpus 区间，在加载期按半开区间 `[start, end)` 展开成采样范围（`semg_fm/data/intervals.py:108-126` `selected_records`，用 `searchsorted(time_s, bound - source_time_origin_s)` 映射）。

`manifest.csv` 的 13 列、`nature_corpus.csv` 的四列、HDF5 schema 与各转换器的原始布局，全部见 [DATA_LAYOUT.md](DATA_LAYOUT.md)，本文不再复述。

---

## 3. 单个训练步真正计算什么

以 `PretrainModule._losses`（`semg_fm/lightning/pretrain_module.py:170`）为准逐步展开。

### 3.1 张量形状

| 量 | 形状 | 来源 |
|---|---|---|
| `emg` | `[B, 4000, 16]` | 4000 采样 = 2000 ms @ 2 kHz；16 电极 |
| `patches` | 100 | `4000 / patch_samples(40)` |
| token 网格 `[B, 16, 100]` | `16 × 100` | `[batch, channels, patches]` |
| `d_model` / `depth` / `heads` | 128 / 6 / 8 | `configs/model/ring_criss_cross.yaml` |

`emg.ndim` 必须是 3，`channels` 必须等于 `student.channels`，`samples % patch_samples` 必须为 0，否则在 `_losses` 里直接抛错。

### 3.2 单通道通道优先 patch 编码器

`RingCrissCrossEncoder` 用**同一个单通道编码器**编码每个电极：

- `self.patch_encoder = MultiScalePatchEncoder(channels=1, d_model=d_model, patch_samples=patch_samples)`（`semg_fm/models/ring_criss_cross.py:102-104`）。
- 前向先做 `channel_major = masked_emg.transpose(1, 2).reshape(batch * channels, samples, 1)`，编码后 reshape 回 `[B, C, patches, D]`（`ring_criss_cross.py:161-162`）。

这样在进入环形注意力之前**不存在任何通道索引参数**（`ring_criss_cross.py:100-101` 注释），循环等变性因此得以保持。通道之间唯一的交互发生在 `RingMultiheadAttention` 的环形相对位置偏置里（`signed_ring_offsets`，`semg_fm/models/ring_attention.py:11`）。

### 3.3 MaskGenerator 的五个组成部分

`MaskGenerator`（`semg_fm/objectives/masking.py:10`）按 batch 逐个采样，组合五个组件（默认值来自 `configs/pretrain.yaml:11-16`）：

| 组件 | 默认 | 行为 |
|---|---|---|
| `temporal_span` | 2 | 随机起点，模回绕地标记连续的 2 个 patch（`masking.py:52-55`） |
| `ring_arc` | 3 | 随机起点，在环形通道上模回绕地标记 3 个通道的全部 patch（`masking.py:56-59`） |
| `full_channel_probability` | 0.1 | 每个通道以 10% 概率整条被标记（`masking.py:60-62`） |
| `block_shape` | (2, 2) | 随机起点的 2×2 通道×patch 块，双向模回绕（`masking.py:63-69`） |
| `random_patch_probability` | 0.15 | 每个 token 独立 15% 概率被标记（`masking.py:70-71`） |

每个样本采样完后调用 `_ensure_both_states`（`masking.py:72`、`masking.py:90`）：若全 False 就置一个 True，若全 True 就置一个 False，保证 mask 既非空也非满。

在 `_losses` 里，采样出的 mask 还要与「该 token 至少有一个采样点是观测到的」相交：`token_mask = mask_generator(...) & token_observed`；若相交后为空，则退化为 `token_observed`（`pretrain_module.py:186-189`）。这样 mask 里绝不会包含数据本身已缺失的 token。

### 3.4 teacher 前向（完整观测信号，no_grad）

```python
self.teacher.eval()
with torch.no_grad():
    teacher = self.teacher(emg * observed.to(emg.dtype), observation_mask=observed)
```

（`pretrain_module.py:191-193`）teacher 看到的是**完整观测信号**，在 `no_grad` 下产生目标特征 `teacher["z_eq"]`。teacher 是 student 的深拷贝，`requires_grad_(False)` 且恒为 `eval()`（`pretrain_module.py:64-66`）——它只通过 EMA 移动，从不接收梯度。

### 3.5 student 前向（masked token 被替换，但仍保持 observed）

```python
student_observed = observed & _expand_grid(~token_mask, self.student.patch_samples)
student_input = emg * student_observed.to(emg.dtype)
student = self.student(student_input, observation_mask=observed, patch_mask=token_mask)
```

（`pretrain_module.py:190, 194, 200`）这里有三点必须精确理解：

1. **输入被置零**：`student_observed` 把被 mask 的 token 的原始采样清零，所以模型看不到真值。
2. **但观测掩码仍是 `observed`**：被 mask 的 token 在编码器内部**保持 observed** —— 它正常参与注意力、被别人注意，输出非零。
3. **`patch_mask=token_mask` 替换 embedding**：`RingCrissCrossEncoder` 用可学习的 `mask_token` 替换这些 token 的 patch embedding（`ring_criss_cross.py:163-170`），而 `mask_token` 在所有电极间共享，所以不带通道索引、不破坏等变性（`ring_criss_cross.py:109-115`）。

**为什么必须这样**：若把被 mask 的 token 当成「未观测」，编码器会把它们的输出清零（见 `ring_criss_cross.py:174` 的 `pooled_weights`、以及各 block 结尾的 `* token_observed`），student 与 teacher 在这些位置的特征都会变成同一个常量，蒸馏损失随之变成常数、**整条预训练目标失去梯度**。这正是 `pretrain_module.py:195-199` 那段注释记录的原因；`mask_token` 与观测掩码分离，是 masked-latent 蒸馏得以成立的前提。

### 3.6 损失

- `latent = masked_latent_loss(student["z_eq"], teacher["z_eq"], token_mask)`（`pretrain_module.py:201`）。
- `masked_latent_loss`（`semg_fm/objectives/masked_latent.py:9`）**只在被 mask 的位置**取值：`student[mask]` 与 `teacher.detach()[mask]`（`masked_latent.py:17-18`），各自做 L2 归一化后返回
  `smooth_l1(normalized_student, normalized_teacher) + (1 - cosine_similarity).mean()`（`masked_latent.py:32-34`）。
- 当输入是 bf16/fp16 时，计算提升到 **fp32**：`computation_dtype` 在 `masked_latent.py:21-25` 决定。
- 该函数在 mask 为空、或 student/teacher 特征非有限时直接抛错（`masked_latent.py:15-20`），所以「masked 位置无有效梯度」这种错误状态不会静默通过。

### 3.7 total、kinematic 与 equivariance

```python
total = self.latent_weight * latent + self.kinematic_weight * kinematic
```

（`pretrain_module.py:212`）其中 `kinematic` 来自 `_kinematic`（`pretrain_module.py:240`）。

**在 E3 默认配置下**（`kinematic_dim: null`、`kinematic_weight: 0.0`，见 `configs/pretrain.yaml:18-22` 与 `scripts/e3_oneclick.py:450-451`）：

- `kinematic_adapter` 为 `None`，`_kinematic` 返回 `motion.reshape(-1)[:0].sum()` —— 一个**可微的零**（`pretrain_module.py:241-242`）。
- 于是 `total = 1.0 * latent + 0.0 * 0 == latent`。**目标退化为纯粹的 masked latent 蒸馏**，kinematic 辅助损失关闭。

`ring_equivariance_error` 在同一函数内计算，但**整体包在 `torch.no_grad()` 内**：把 `student_input` 沿通道轴 `torch.roll(..., 1, -1)` 再前向一次，用 `ring_equivariance_loss` 比较 `z_eq`/`z_motion`（`pretrain_module.py:202-210`）。它**只被记录，从不进入 `total`**，因此不产生梯度、不影响优化。

### 3.8 EMA 更新

在 `optimizer_step` 中，先执行 `super().optimizer_step(...)`（AdamW 步进），随后在 `no_grad` 下遍历 student 的 `state_dict`：

```python
teacher_value.mul_(self.ema_decay).add_(student_value.detach(), alpha=1 - self.ema_decay)
```

（`pretrain_module.py:157-168`）即 `teacher = 0.996 * teacher + 0.004 * student`。非浮点缓冲（如计数器/整型状态）直接 `copy_` 覆盖。随后 `self._ema_updates.add_(1)`。

- `ema_decay` 是**常量 0.996，没有 warmup ramp**（`configs/pretrain.yaml:22`）。
- EMA 严格发生在优化器更新**之后**，teacher 与 student 参数一一对应（按名字对齐，`teacher_state[name]`）。
- teacher 从不累积梯度（`requires_grad_(False)`），`diagnostics.json` 的 `teacher_has_no_grads` 会核验这一点。

### 3.9 学习率调度

`_lr_scale`（`pretrain_module.py:270-274`）配合 `LambdaLR(optimizer, ...)`、`interval="step"`：

- 前 `warmup_steps`（默认 1000）步线性升温：`(step + 1) / warmup_steps`。
- 之后余弦衰减到 `optimizer.max_steps`：`0.5 * (1 + cos(pi * min(max(step - warmup, 0), span) / span))`，`span = max(1, max_steps - warmup_steps)`。

优化器只收集 `requires_grad` 的参数（`pretrain_module.py:153`），即 student；teacher 天然被排除。`server_full` 下 lr 基准为 3e-4（`scripts/e3_oneclick.py:448`）、weight decay 0.05，并叠加上游 `a100_ddp` 的 `gradient_clip_val: 1.0`。

---

## 4. 一次 run 记录了什么

`server_full` 的产物都落在 `<run-root>/e3_server_<UTC时间戳>/` 下。

| 文件 | 写入者 | 内容 |
|---|---|---|
| `e3_status.json` | `e3_oneclick.py` | `mode`、`created_utc`、`host`、`python`、`repository`、preflight 摘要、`real_dataflow`、`status`（`DATAFLOW_OK` → `COMPLETE` / `TRAINING_FAILED`）、`returncode`、checkpoint 摘要、必要时 `checkpoint_source: resume_input` |
| `command.json` | `e3_oneclick.py` | 实际执行的 Hydra argv（逐项 override） |
| `e3_train.log` | `e3_oneclick.py` | `pretrain.py` 的完整 stdout/stderr |
| `resolved_config.yaml` | `pretrain.py:56` | 完全解析后的 Hydra 配置 |
| `run_metadata.json` | `pretrain.py:62`（`semg_fm/reproducibility.py:58`） | `command`、`manifest{path,sha256}`、`git_commit`、`trainer`、`data`、`effective_global_batch_size`、`environment`（python/platform/torch/CUDA） |
| `manifest.json` | `pretrain.py:57` | 数据 manifest 的 `path` 与 `sha256` |
| `train_metrics.jsonl` | `E3RunDiagnostics` | **每个优化步一行**：`step, task_id, loss, latent_loss, student_gradient_norm, feature_std, learning_rate, gpu_memory_bytes, step_time_s, ema_updates`（`semg_fm/e3_diagnostics.py:118-129`） |
| `diagnostics.json` | `E3RunDiagnostics.on_fit_end` | `train_batches_per_task`、`val_batches_per_task`、`train_batch_total`、`missing_train_tasks`、`student_updated`、`teacher_updated`、`teacher_has_no_grads`、`ema_updates`、`feature_std{last,mean,min,max}`、`steps_recorded`，以及 `global_step`、`elapsed_s`、`precision`、`world_size`（`e3_diagnostics.py:147-171`） |
| `e3_error.json` | `E3RunDiagnostics._guard` | **仅在发散时**：`error: non_finite_training_state`、`non_finite` 列表、`global_step`、`task_id`、当时那行 `observed`、`action`（`e3_diagnostics.py:211-219`） |
| `checkpoints/last.ckpt` | `ModelCheckpoint` | 最近一次，每 1000 步写一次 |
| `checkpoints/epoch=N-step=M.ckpt` | `ModelCheckpoint` | 最优 `val/loss`（`save_top_k=1`，`mode=min`） |

`train_metrics.jsonl` 采取**流式追加**而非内存累积，正是为了在 100k 步生产运行下保持内存有界（`e3_diagnostics.py:191-195`）；行解析集合在 `on_fit_end` 汇总成 `diagnostics.json`。

`local_real` 的 `verify_local_real.py` 用同样的 callback，但额外产出自己的 `verification.json`，并把 5 个 section 的结论（数据、采样比例、模型、训练、checkpoint）写进去。

---

## 5. 可靠性保证

1. **周期性 checkpoint。** `configs/trainer/e3_server.yaml:36` 设 `every_n_train_steps: 1000`，`save_last: true`。崩溃最多损失 1000 步。`save_top_k: 1` 另保留最优 `val/loss`。
2. **checkpoint 内容完整。** `PretrainModule.on_save_checkpoint`（`pretrain_module.py:103-108`）额外写入 `rng_state`（python/numpy/torch/CUDA 的 RNG 状态）。`e3_oneclick.py` 的 `describe_checkpoint`（`scripts/e3_oneclick.py:487`）在报告成功前**断言** checkpoint 同时含 `student.*`、`teacher.*`、`optimizer_states`、`lr_schedulers` 与 `rng_state`，缺任一即报错。
3. **严格 resume + 超参兼容护栏。** `on_load_checkpoint`（`pretrain_module.py:110-116`）逐键比较保存的 `hyper_parameters` 与当前值，任一不同即：

   ```
   resume configuration mismatch for <key>; use unchanged training semantics or start a new experiment
   ```

   被比较的键包括模型几何、mask 设置、各权重、`ema_decay`、`learning_rate`、`warmup_steps`、`max_steps`（见 `save_hyperparameters`，`pretrain_module.py:84-97`）。**任何训练语义变化都会拒绝启动，而不是静默用不同语义训练。** 同一回调还恢复 RNG，并在 `on_train_start` 再次恢复一次（`pretrain_module.py:128-132`），以防构造/`setup` 阶段消耗了随机数。全局步、优化器、LR 调度、EMA 计数器一并还原。
4. **NaN/Inf 硬停止。** `E3RunDiagnostics._guard`（`e3_diagnostics.py:197`）在每步检查 `loss`、`feature_std`、`student_gradient_norm`，每 `param_check_interval`（默认 10）步额外扫描 student 参数是否有限。任一非有限即写 `e3_error.json`（含 **步号与 task_id/语料**）并抛 `RuntimeError` 停止训练——**坏更新绝不被写进 checkpoint**。
5. **来源可追溯。** `run_metadata.json` 记录 git commit SHA、manifest SHA-256、随机种子、完整解析配置与环境（`semg_fm/reproducibility.py:58-83`）。

resume 一个已达步数上限的 run 不会再做任何工作；此时 `e3_status.json` 报 `"checkpoint_source": "resume_input"`（`scripts/e3_oneclick.py:620-629`）。

---

## 6. Preflight 门禁

`scripts/run_e3_server.sh` **先**把 `scripts/server_preflight.py` 作为**独立进程**运行，只有每个 section 都通过才启动训练（`scripts/run_e3_server.sh:91-102`）。四个 section：

| section | 检查 |
|---|---|
| **ENVIRONMENT** | Python ≥ 3.10；CUDA 可用；GPU 名称/算力/显存；**bf16 必须支持**（e3_server 是 bf16-mixed）；显存 < 8 GiB 给出警告；系统 RAM；数据盘与输出盘的剩余空间；输出根可写；git commit/branch/dirty；依赖是否与 `requirements-server.txt` 完全一致（`server_preflight.py:105-175`） |
| **DATA** | 复用 `e3_oneclick.audit` 走全量 manifest 与官方语料区间校验、按语料抽样 HDF5 schema；逐语料 census；train/val **泄漏矩阵**（同一非 Nature record/pair/subject 不得跨 split）；official test 不得被 train/val 触及（`server_preflight.py:182-265`） |
| **DATALOADER** | 真实 `DataModule` + `DataLoader`：五任务覆盖、batch 形状、多 worker 加载、**完整窗口索引的内存开销**；并检查 `floor(windows / batch_size) == 0` 的「饿死」语料（batch 过大导致某语料每轮零 batch）（`server_preflight.py:272-401`） |
| **GPU_SMOKE** | 真实跑若干优化步（默认 `--smoke-steps 5`，**最少 5**，保证五个语料各至少一步；smoke batch 2）：梯度、EMA、峰值显存、速度，以及一次 checkpoint 往返（`server_preflight.py:408-502`） |

**为什么必须是独立进程**：`GPU_SMOKE` 会**真实执行优化器步进**（会产生权重、优化器状态、LR 调度的变化）。把它放在独立进程里、跑完即退出，才能保证生产 run 作为**另一个全新进程**启动时，**绝不可能继承 smoke 的权重、优化器状态或 LR 调度**。`server_preflight.py:19-22` 的 docstring 与 `run_e3_server.sh:7-10` 都明确记录了这一设计。此外，preflight 校验的 batch size 与训练**完全一致**（`E3_BATCH_SIZE`，默认 8），否则「batch 大于某语料窗口数」这类问题在 preflight 中不会暴露。

> preflight 通过 = 「可以安全启动」，**不等于**科学会成功（`server_preflight.py:24-25`）。

---

## 7. 「通过」并不意味着什么

本节是有意写实的边界，避免把工程结果误读为科学结论。

- **合成与 mini 的通过是工程结果，不是表示质量的证据。** `local_synthetic` 用的是带 `.E3_SYNTHETIC_DO_NOT_TRAIN` 标记的合成 fixture（`scripts/e3_oneclick.py:39`），`local_real` 用的是少量真实 mini。它们证明的是「管线能跑通、梯度与 EMA 在动、checkpoint 能恢复」，不是「学到的表示有多好」。任何时候都不得用合成 fixture 做生产训练（`e3_oneclick.py:548-549` 会直接拒绝）。
- **真实语料与 A800 尚未验证。** 合成与 mini 全部通过，仍不能说明完整五语料数据根或目标 GPU（A800/A100 及更新架构）上的一切成立。这些属于部署/验收范畴，见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。
- **采样比例是设计决定，不是既定事实。** `TaskBalancedBatchSampler` 每轮只交错「每任务各一个 batch」，它平衡的是**顺序，不是比例**（`semg_fm/data/samplers.py:109-125`）：每个语料每轮实际贡献 `floor(windows / batch_size)` 个 batch（`samplers.py:91-93`）。因此在真实语料上，**每个语料的优化步占比跟随其窗口数量，而非均等**。`verify_local_real.py` 的 SAMPLING section 专门实测这一比例，而不是假设它（`scripts/verify_local_real.py:229-279`，其 `conclusion` 明确写着 "the sampler balances ORDER, not PROPORTION"）。**语料的混合比例是一个需要刻意决定的量**，不是自动获得的结果。

---

## 参考

- [DATA_LAYOUT.md](DATA_LAYOUT.md) — HDF5 契约、`splits.csv` / `nature_corpus.csv` 列与规则。
- [SERVER_DEPLOY.md](SERVER_DEPLOY.md) — 部署、安装、准备数据、运行、resume 与排障。
- 核心实现：`semg_fm/lightning/pretrain_module.py`、`semg_fm/models/ring_criss_cross.py`、`semg_fm/objectives/{masking,masked_latent,kinematic,ring_equivariance}.py`、`semg_fm/data/{datamodule,window_dataset,samplers,intervals}.py`、`semg_fm/e3_diagnostics.py`。
- 入口与门禁：`pretrain.py`、`scripts/e3_oneclick.py`、`scripts/server_preflight.py`、`scripts/run_e3_server.sh`、`scripts/verify_local_real.py`。
- 配置：`configs/pretrain.yaml`、`configs/trainer/{a100_ddp,e3_server,local_real}.yaml`、`configs/data/{canonical_meta,local_real}.yaml`、`configs/model/ring_criss_cross.yaml`。
