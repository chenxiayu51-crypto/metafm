# Meta sEMG Foundation Model — 五语料 E3 预训练工程

**2026-10-16 服务器五路径部署包：先看 [上传运行说明](START_HERE_服务器.md)。**
该入口自动读取官方下载的 corpus CSV 与随包官方 QWERTY generic YAML，支持安全续跑转换后接原生训练。

`metafmodel` 是唯一正式主仓库，保留 Ring Criss-Cross + EMA Teacher–Student +
Hydra/Lightning 主干。E3 五语料联合预训练是当前唯一正式启动目标。

**本仓库只发布源码。** 真实语料、模型权重和训练输出一律不进 Git；它们位于服务器
外部路径。

---

## 整体架构：两阶段

```
阶段一  自监督预训练（当前唯一正式目标 E3）
        五个语料【官方 train】+【官方 val 仅用于监控】
        → 无标签、无任务头
        → Masked Latent + EMA Teacher
        → 通用 sEMG Encoder（student.*，约 1.6M 参数）

阶段二  下游微调（例如 Pose）
        Encoder + 任务头（20DoF 关节角）
        → 以阶段一的 checkpoint 初始化 Encoder
        → 三种协议 full / partial / probe，支持 1/5/10/100% 标签比例
```

两阶段共用**同一个** `RingCrissCrossEncoder`，不存在"预训练用一套、微调用另一套"
的情况。微调加载时会逐项校验 `d_model / depth / heads / patch_samples / channels`
是否与预训练一致，不一致直接报 `backbone configuration mismatch`。

预训练**只使用官方 train**，官方 val 仅用于 `val/loss` 监控与 checkpoint 选择，
**官方 test 从不参与预训练**。

### ⚠️ 当前数据阻塞（见 [docs/known_issues.md](docs/known_issues.md) P0）

上述两阶段在**代码上已就绪**，但在**你现有数据上都无法跑真实训练**：

- 阶段一需要五个语料的官方 split 与官方 `nature_corpus.csv`；目前缺 QWERTY /
  Gesture / Handwriting 的真实数据，因此 E3 真实五语料训练尚不能启动。
- 阶段二需要官方 **Pose train**；目前手上只有 30 份 **official val**，因此真实
  监督微调与独立研究评测处于 BLOCKED 状态。

在补齐这两批数据之前，跑通的是工程链路（Dataset/DataLoader/训练/checkpoint/
恢复），**不是研究结论**。

---

## 三种运行模式

三者共用**同一套** Encoder、EMA Teacher–Student、Masked Latent 目标、Dataset/
Collate、Optimizer 与 Checkpoint 机制。只有数据规模、batch size、worker 数、
训练步数和输出路径不同。

| 模式 | 入口 | 数据 | 用途 |
|---|---|---|---|
| `local_synthetic` | `python scripts/e3_oneclick.py local` | 自动生成的合成夹具 | 软件链路冒烟，**不涉及真实信号** |
| `local_real` | `python scripts/verify_local_real.py --data-root D:/emg_data/real_mini` | 从服务器导出的真实 Mini | 在真实信号上验证全部训练机制 |
| `server_full` | `bash scripts/run_e3_server.sh /srv/datasets/meta_canonical` | 服务器完整五语料 | 正式基础模型预训练 |

真实 Mini 的导出与使用见 [docs/REAL_MINI_DATASET.md](docs/REAL_MINI_DATASET.md)，
三种模式的详细区别与各自断言的检查见
[docs/LOCAL_VALIDATION.md](docs/LOCAL_VALIDATION.md) 与
[docs/E3_TRAINING_PIPELINE.md](docs/E3_TRAINING_PIPELINE.md)。

---

## 快速开始（Linux GPU 服务器）

### 0. 克隆

```bash
git clone <your-fork-url> /srv/projects/metafmodel
cd /srv/projects/metafmodel
```

### 1. 安装依赖

新建独立虚拟环境，不要升级共享环境里其他项目依赖的 CUDA：

```bash
python3 -m venv .venv
export E3_PYTHON=/srv/projects/metafmodel/.venv/bin/python

bash scripts/setup_server.sh      # 只检查并打印命令，不会自动安装任何东西
```

`setup_server.sh` 会检测驱动并打印与之一致的 PyTorch 安装命令。按它打印的命令
操作，然后重复运行直到出现 `Environment is ready.`。通常为：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server.txt
```

固定版本见 [requirements-server.txt](requirements-server.txt)，是经过同一轮验证的
组合，不要随意改动单个版本。

### 2. 放置数据

目录约定（全部可改，没有硬编码）：

```
/srv/datasets/meta_canonical/     规范化数据根   -> E3_DATA_ROOT
/srv/experiments/e3/              训练输出根     -> E3_RUN_ROOT
```

数据根需要 `splits.csv`、`nature_corpus.csv` 和五个语料的规范化 HDF5。
格式契约见 **[docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)**。

如果手上只有**原始** HDF5：

```bash
bash scripts/prepare_data.sh /srv/datasets/raw /srv/datasets/meta_canonical
```

它会调用仓库自带的五个转换器和 `make_splits`，然后停下来要你提供
`nature_corpus.csv`。

> **它不会替你生成 `nature_corpus.csv`。** 三个 GNI 语料按官方 Meta 时间区间切片，
> 这些区间属于发布协议的一部分；猜测区间等于悄悄在验证集或测试集上训练。文件缺失时
> 脚本明确报错并说明所需格式，而不是编造。emg2pose 的官方 val/test 也不会被改标成
> train，官方 test 不参与预训练。

### 3. 训练前验证

```bash
export E3_DATA_ROOT=/srv/datasets/meta_canonical
export E3_RUN_ROOT=/srv/experiments/e3
bash scripts/verify_server.sh "$E3_DATA_ROOT"
```

只读检查：GPU 与 bf16、依赖、入口文件、数据根、`splits.csv`、`nature_corpus.csv`、
五语料是否都有合法 train/val 窗口、规范化 HDF5 与 manifest 元数据是否一致、真实
DataLoader 是否产出五任务 Batch。只有全部通过才返回 0。

### 4. 启动 E3 预训练

```bash
bash scripts/run_e3_server.sh /srv/datasets/meta_canonical
```

这就是完整命令。它会**先跑服务器预检**（环境、数据、DataLoader、真实数据短训练），
全部通过后才调用本仓库唯一的训练入口 `scripts/e3_oneclick.py server`，后者运行原生
`pretrain.py`。不存在第二套训练逻辑。

预检是**独立进程**：它内部的短训练不会污染正式训练的权重、optimizer 或 scheduler。
预检失败则训练根本不启动。

完整部署说明（含产出物、恢复训练、故障排查）见
**[docs/SERVER_DEPLOY.md](docs/SERVER_DEPLOY.md)**。

---

## 训练配置

| 项 | 值 |
|---|---|
| 输入 | 16 通道 / 2000 Hz |
| 窗口 / 步长 | 2000 ms / 1000 ms |
| patch | 40 samples |
| 模型 | Ring Criss-Cross，`d_model=128`，depth 6，heads 8 |
| 目标 | Masked Latent + EMA Teacher（`ema_decay=0.996`） |
| 运动学辅助损失 | **关闭** |
| 精度 | bf16-mixed（服务器模式无 bf16 则拒绝启动） |
| batch size | 8 |
| 优化器 | AdamW，lr 3e-4，weight decay 0.05，1000 步 warmup + 余弦衰减 |
| 总步数 | 100000 |
| 验证 | 每 2000 步，250 个 batch |

这些是**初始配置，不是已验证的最优超参数**。

### 可靠性

- 每 1000 步写 `last.ckpt`，同时保留最佳 `val/loss` checkpoint。
- Checkpoint 含 Student、Teacher、Optimizer、LR Scheduler、Global Step 与 RNG 状态。
- 恢复训练还原 Global Step、Optimizer、LR Scheduler、EMA 计数与 RNG 流；关键配置
  不一致时**拒绝启动**（`resume configuration mismatch`）。
  注意：**sampler 游标未纳入 checkpoint**，恢复后不会逐样本复现中断前的顺序
  （见 [docs/known_issues.md](docs/known_issues.md) P1）。
- 五语料 Task-balanced Sampler，`diagnostics.json` 记录每个语料实际参与的 batch 数。
- 记录 Student/Teacher 更新状态与特征统计。
- 出现 NaN/Inf 立即停止并写出 `e3_error.json`，不会把坏更新写进 checkpoint。
- 每次运行保存 Git Commit SHA、随机种子、完整配置、数据清单哈希与环境信息。

### 恢复训练

```bash
bash scripts/run_e3_server.sh /srv/datasets/meta_canonical \
  --resume /srv/experiments/e3/e3_server_<timestamp>/checkpoints/last.ckpt
```

---

## 产出物位置

`$E3_RUN_ROOT/e3_server_<UTC 时间戳>/`：

```
e3_status.json          状态（COMPLETE / TRAINING_FAILED）与 preflight 摘要
command.json            实际使用的 Hydra 参数
e3_train.log            完整训练日志
resolved_config.yaml    解析后的完整配置
run_metadata.json       git commit、manifest sha256、环境信息
train_metrics.jsonl     每个 optimizer step 一行
diagnostics.json        每语料 batch 数、Student/Teacher 状态、特征统计
e3_error.json           仅在发散时出现
checkpoints/last.ckpt               最新（每 1000 步）
checkpoints/epoch=N-step=M.ckpt     最佳 val/loss
```

---

## 本地验证（非 Linux、无 A800 时）

**合成冒烟**：生成**合成**夹具（带 `.E3_SYNTHETIC_DO_NOT_TRAIN` 标记），跑通五任务
Dataset/DataLoader、GPU 训练与 checkpoint 写入。服务器模式会拒绝这个标记。

```powershell
$env:META_PYTHON = "path\to\python.exe"
python scripts/e3_oneclick.py local
```

**真实 Mini 验证**（推荐，使用真实信号）：先在服务器导出小数据，再在本地跑：

```powershell
.\scripts\run_local_real.ps1 -DataRoot D:\emg_data\real_mini
```

它会断言五语料数据契约、无泄漏、逐语料真实梯度与 EMA、bf16、表征未坍塌、
N 步真实训练、以及 checkpoint 重载 / 恢复 / Global Step。同时**实测逐语料采样比例**
——见下方重要提示。

> ⚠️ **不要以为任务名里有 "Balanced" 就是五语料等比例。**
> `TaskBalancedBatchSampler` 只平衡**顺序**（每轮每语料一个 batch）；每个语料实际贡献
> `floor(窗口数 / batch_size)` 个 batch，所以它在训练步数中的**占比取决于各语料的窗口数**。
> 本地 Mini 实测：pose 18.8% / typing 43.8% / 其余各 12.5%（等分应为 20%）。
> 真实语料上差异会更大。要改比例必须是有记录、有意识的决定，因为它改变训练数据分布。

> 通过合成或 Mini 验证都只证明**工程链路**，不证明基础模型表征质量。

测试套件：

```bash
python -m pytest tests/ -q
```

---

## 数据流

```
原始 HDF5 → semg_h5 转换器 → 规范化 H5 + manifest.csv
  → 官方 split / Nature corpus 区间 → splits.csv
  → [T,16] 窗口 → task-balanced sampler → Ring Encoder
  → masked student + EMA teacher → masked latent loss
  → Lightning → Student Foundation Checkpoint
```

## 文档索引

- [服务器部署](docs/SERVER_DEPLOY.md)：安装、验证、预检门禁、启动、恢复、故障排查
- [E3 训练流程](docs/E3_TRAINING_PIPELINE.md)：三种模式、一步训练到底算什么、产出物、可靠性
- [本地验证](docs/LOCAL_VALIDATION.md)：合成 vs 真实 Mini、五类断言、采样比例实测
- [真实 Mini 数据集](docs/REAL_MINI_DATASET.md)：从服务器导出的完整方法与迁移
- [数据格式](docs/DATA_LAYOUT.md)：规范化 HDF5 契约、splits.csv、nature_corpus.csv
- [数据契约](docs/data_contract.md)：`intervals.py`、`raw_adapter.py`、`window_dataset.py`
- [模型架构](docs/architecture.md)：`models/ring_criss_cross.py`、`patch_encoder.py`、`ring_attention.py`
- [训练闭环](docs/training_pipeline.md)：`lightning/pretrain_module.py` 与原生入口
- [H5 契约](H5_DATA_CONTRACT.md)、[上游来源](UPSTREAM.md)、[上游 README](docs/upstream_README.md)
- [已知限制](docs/known_issues.md)、[实验记录](docs/experiment_log.md)
- [E3 验收报告](docs/E3_ACCEPTANCE.md)

---

## 许可

本仓库**没有 LICENSE 文件**，因此不构成任何开源授权。`semg_h5/` 与
`semg_fm/models/` 源自上游项目快照（见 [UPSTREAM.md](UPSTREAM.md)）；对外发布或
再分发前请先确认上游许可与授权。第三方依赖各自适用其许可，详见
[docs/SERVER_DEPLOY.md](docs/SERVER_DEPLOY.md) 第 9 节。
