# Meta 16 通道 sEMG 规范化 H5 转换

本仓库把 5 个公开的 Meta/CTRL-labs 16 通道腕部 sEMG 语料转换为统一的单腕 HDF5 契约。这些转换器刻意**不是**生理信号预处理流水线：不做通道重排、滤波、缩放、极性校正、截断或重采样。唯一的表示变更，是把 `time_s` 存为相对首个样本的时间；源路径和预处理记录都保留在每份输出中。

## 上手流程

从零开始跑通整个仓库，依次执行四步；每步命令都可在任意目录（不限于 `D:\Fmodel`）中照抄：

1. **获取原始数据**：按各语料官方发布说明下载 emg2pose、emg2qwerty、GNI 三个语料（许可均为 CC BY-NC-SA / CC BY-NC，禁止再分发）。语料清单与规模见 [`meta_semg_16ch_datasets.md`](meta_semg_16ch_datasets.md)。
2. **转换**：安装依赖后运行 [`semg_h5/`](semg_h5) 下 5 个独立转换器，产出统一的 canonical H5 与 `manifest.csv`（见[转换各个语料](#转换各个语料)与 [`H5_DATA_CONTRACT.md`](H5_DATA_CONTRACT.md)）。
3. **生成划分**：用 `make_splits.py` 为没有官方划分的记录分配确定性 train/val/test 划分，官方划分原样保留。
4. **训练与评测**：`pretrain.py` 自监督预训练 → `finetune.py` 逐任务微调 → `evaluate.py` 留存集评测 → `run_baseline.py` 对比基线。先跑[五分钟 CPU 冒烟](#五分钟-cpu-冒烟)验证环境，正式训练上 A100 见 [`SERVER_A100.md`](SERVER_A100.md) 与 [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)。

## 安装

转换器只用 Python 标准库；执行转换还需要 h5py/pandas/tables，安装训练依赖时一并包含：

```bash
python -m pip install -r requirements-train.txt
```

## 转换各个语料

运行每条命令时，工作目录中应只包含对应语料的原始 H5 文件。输出根目录要与原始输入目录分开。

```bash
# 在仓库根目录执行；RAW_* 为各语料原始 H5 所在目录（见各语料官方发布说明），
# CANONICAL_META 为输出根目录（与新数据分开）。下列路径按需替换。
python semg_h5/convert_emg2pose.py --input-root RAW_EMG2POSE --output-root CANONICAL_META
python semg_h5/convert_emg2qwerty.py --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python semg_h5/convert_gni_discrete_gestures.py --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python semg_h5/convert_gni_handwriting.py --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python semg_h5/convert_gni_wrist.py --input-root RAW_GNI_WRIST --output-root CANONICAL_META
```

每个独立转换器在写出结果前，都会校验源数据的目录布局、16 通道信号形状、样本时间戳以及任务专属标签；不需要额外的校验命令。若某语料提供官方划分 CSV，可用 `--splits-csv SPLITS.csv` 传入；该 CSV 必须包含 `split` 列，以及 `path`、`source_file`、`file` 三者之一。其中的路径相对于 `--input-root`（也可以直接是裸文件名）。`--split train` 会在转换前做过滤。不要生成随机的窗口级划分：下游评测必须保留官方的 session/参与者划分。默认拒绝覆盖已存在的输出 H5 文件；只有在确实要重新生成时才传 `--overwrite`。

## 输出契约

```text
CANONICAL_META/
  manifest.csv
  emg2pose/pose/<subject>/<session>_<side>.h5
  emg2qwerty/typing/<subject>/<session>_left.h5
  emg2qwerty/typing/<subject>/<session>_right.h5
  gni_discrete_gestures/discrete_gesture/...
  gni_handwriting/handwriting/...
  gni_wrist/wrist_angle/...
```

每个 H5 都包含根级溯源属性（`schema_version`、各类 ID、`official_split`、`source_file`、`preprocessing_json`），以及：

```text
/signal/emg       float32 [T, 16]
/signal/time_s    float64 [T], 从 0 开始
/labels           任务专属标签，外加 source_json
/meta/source_json 原始源属性与未折叠的元数据
/meta/quality     sample_count、duration_s、nan_count、channel_rms
```

各类任务的标签刻意保持互不相同：`pose` 有 `joint_angles` 和可选的 `no_ik_failure`；`typing` 有 `keystrokes` 和 `prompts`；`discrete_gesture` 有 `events`；`handwriting` 有 prompt 区间；`wrist_angle` 在 `/labels/targets` 下有一个或多个连续数组。

`manifest.csv` 中每份规范化 H5 占一行，列固定为 `path`、`dataset_id`、`task_id`、`subject_id`、`session_id`、`side`、`pair_id`、`official_split`、`sample_count`、`duration_s`、`source_file` 和 `source_sha256`。推荐数据加载器把这张表作为第一级过滤依据。

## 双腕打字数据的重建

`emg2qwerty` 是唯一的双腕语料。一个源 session 会生成两份规范化文件，二者共享同一个 `pair_id`，一份 `side=left`、一份 `side=right`，且都包含该 session 的打字标签。若要训练双腕模型，请按 `pair_id` 对 manifest 行分组，要求左右恰好成对，然后对齐两者共享的相对时钟。若要训练单腕预训练模型，则把它们当作彼此独立的普通规范化记录，但保留 `pair_id`，以避免同一参与者/session 跨划分泄漏。

## 源数据布局约定

emg2pose 转换器期望 `emg2pose/timeseries` 复合字段 `emg`、`joint_angles` 和 `time`（外加可选的 `no_ik_failure`）。emg2qwerty 转换器期望 `emg2qwerty/timeseries` 字段 `emg_left`、`emg_right` 和 `time`；打字元数据既接受 JSON 属性形式，也接受结构化的 `keystrokes`/`keylogger` 和 `prompts` 数据集。三个 GNI 转换器期望一个 `data` 复合数据集，带有 `data.attrs['task']`、一个 `time` 字段和一个 16 通道 EMG 字段；`prompts` 和 `stages` 若存在，则按 pandas HDF 表读取。

遇到未知的源布局会快速失败，并在错误消息中给出缺失的字段/任务。这是刻意设计：避免在某个发布变体需要显式更新读取器时，静默产出错位的基础模型数据。

## 训练栈（CIRCE-Myo）

本仓库还包含针对上述 5 个转换后语料的基础模型训练栈：

- `pretrain.py` — 自监督预训练（EMA teacher/student masked-latent 蒸馏 + 环等变性）
- `finetune.py` — 逐任务微调，支持 linear-probe / partial / full 协议与 1/5/10/100% 标签预算
- `evaluate.py` — 留存集评测，带内存内鲁棒性变换
- `run_baseline.py` — supervised-TCN、BENDR-like、CBraMod-like、AEMG-like 对比基线
- `make_splits.py` — 确定性的防泄漏 train/val/test 划分

安装训练依赖并运行测试：

```bash
python -m pip install -r requirements-train.txt
python -m pytest tests -q
```

### 五分钟 CPU 冒烟

转换完数据后（或已有任意 canonical `manifest.csv`）。在仓库根目录执行：

```bash
# 0. 定义 canonical 数据根目录的绝对路径（Hydra 训练会把工作目录切到运行
#    目录，因此 data.root / data.manifest_path 必须使用绝对路径）。
#    Git Bash 下 $(pwd -W) 输出 Windows 形式（D:/...），Python 可直接使用。
CANONICAL_META="$(pwd -W)/emg_data_canonical"

# 1. 为没有官方划分的记录分配确定性划分（官方划分原样保留）。
python make_splits.py --manifest "$CANONICAL_META/manifest.csv" --root "$CANONICAL_META" --output "$CANONICAL_META/splits.csv" --seed 0

# 2. 一步预训练（fast_dev_run）。数据路径绝对，运行目录相对即可。
python pretrain.py trainer=cpu_smoke data.root="$CANONICAL_META" data.manifest_path="$CANONICAL_META/splits.csv" hydra.run.dir=runs/pretrain-smoke

# 3. 微调几步并评测（cpu_smoke 未定义的键用 + 前缀追加；换任务只需改 task=）。
python finetune.py trainer=cpu_smoke trainer.fast_dev_run=false +trainer.max_epochs=1 \
  +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 +trainer.enable_checkpointing=true \
  task=pose data.root="$CANONICAL_META" data.manifest_path="$CANONICAL_META/splits.csv" \
  data.num_workers=0 hydra.run.dir=runs/finetune-smoke
# 找到刚写出的检查点（cpu_smoke 没有 save_last，文件名形如 epoch=0-step=4.ckpt），
# 必须使用绝对路径；文件名含 '=' 时要再包一层引号，避免被 Hydra 解析为 override。
CKPT="$(pwd -W)/runs/finetune-smoke/lightning_logs/version_0/checkpoints/epoch=0-step=4.ckpt"
python evaluate.py trainer=cpu_smoke task=pose checkpoint_path="'$CKPT'" \
  data.root="$CANONICAL_META" data.manifest_path="$CANONICAL_META/splits.csv" \
  data.num_workers=0 hydra.run.dir=runs/evaluate-smoke
```

Windows PowerShell 对应写法：`$CanonicalMeta = (Get-Location).Path + '\emg_data_canonical'`。只要数据路径为绝对路径，仓库克隆到任意目录都可直接运行。

每次运行都会把解析后的配置、manifest 摘要、环境快照、检查点写入 `hydra.run.dir` 指定的目录（`outputs/<date>/<time>/` 只是未显式指定时的默认值）；评估还会额外写出 `metrics.json`、`per_subject_metrics.csv` 和 `split_manifest.csv`。完整文档见 `PRETRAINING.md`、`DOWNSTREAM_EVALUATION.md`、`BASELINES.md`、`EXPERIMENT_PROTOCOL.md`、`H5_DATA_CONTRACT.md` 和 `SERVER_A100.md`。

