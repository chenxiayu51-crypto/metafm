# Meta sEMG Foundation Model — 五语料 E3 预训练工程

`metafmodel` 是唯一正式主仓库，保留 Ring Criss-Cross + EMA Teacher–Student +
Hydra/Lightning 主干。E3 五语料联合预训练是当前唯一正式启动目标。

**本仓库只发布源码。** 真实语料、模型权重和训练输出一律不进 Git；它们位于服务器
外部路径。

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

这就是完整命令。它调用本仓库唯一的训练入口 `scripts/e3_oneclick.py server`，
后者运行原生 `pretrain.py`，不存在第二套训练逻辑。

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
- 恢复训练严格还原全部状态；关键配置不一致时**拒绝启动**（`resume configuration mismatch`）。
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

## 本地冒烟测试（非 Linux、无 A800 时）

```bash
# Windows PowerShell
$env:META_PYTHON = "path\to\python.exe"
python scripts/e3_oneclick.py local
```

本地模式会生成**合成**夹具（带 `.E3_SYNTHETIC_DO_NOT_TRAIN` 标记）并跑通五任务
Dataset/DataLoader、GPU 训练与 checkpoint 写入。服务器模式会拒绝这个标记。

> 通过冒烟测试不代表研究成功：合成数据的指标不能证明真实泛化，5 步只是工程验收。

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

- [服务器部署](docs/SERVER_DEPLOY.md)：安装、验证、启动、恢复、故障排查
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
