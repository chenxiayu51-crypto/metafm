# 五类 Meta sEMG：上传、格式检查、转换、训练

本包直接基于同事的 metafmodel 主工程，复用五个 `semg_h5/convert_*.py` 转换器、Canonical 接口及 Ring Criss-Cross / EMA / Hydra / Lightning 训练。服务器五个原始目录已写入 `configs/server/user_paths.json`。

源码请解压到独立目录，例如 `/home/fangyun/metafmodel-server-20261016/`，不要解压到 raw 数据目录或覆盖既有工程。

## 1. 环境

激活专用的 Python 3.12 环境，进入解压后的 `metafmodel`：

```bash
cd /你的解压位置/metafmodel
export E3_PYTHON="$(command -v python)"
bash scripts/setup_server.sh
```

`setup_server.sh` 检查当前 CUDA、GPU 与依赖并打印建议，不自动安装。新环境先安装与驱动相容的 PyTorch 2.8.0 CUDA wheel（具体命令见其输出），再安装其余依赖：

```bash
python -m pip install -r requirements-server.txt
```

不要使用 CPU wheel 启动服务器训练。依赖沿用主工程已经验证的版本，详见 `docs/SERVER_DEPLOY.md`。

## 2. 检查实际下载文件

```bash
bash scripts/run_user_server.sh inspect
```

递归检查五个目录内所有 `.h5` / `.hdf5` 的头部、前后各最多 4096 点及任务标签，原始文件只读。详细报告保存到：

```text
/media/SSD4/personal/sjwu/foundation_model/experiments/metafmodel/data_inspection.json
```

`format_status=PASS` 且 `protocol_status=PASS` 才可转换。`training_coverage_status` 仅判断 train/val 元数据覆盖；真实可用窗口、批次和优化器仍需原生预检。文件中部的数值不在抽样范围内，转换器会继续校验完整记录。

| 数据 | 官方 HDF5 结构 | 划分与处理 |
|---|---|---|
| EMG2Pose | `emg2pose/timeseries`：`emg[16]`、`time`、`joint_angles[20]`，可选 IK mask | 保留 H5 split，并与自动发现的 `emg2pose_metadata.csv` / `metadata.csv` 核对；冲突停止 |
| EMG2QWERTY | `emg2qwerty/timeseries`：`emg_left[16]`、`emg_right[16]`、`time`，prompt/keylogger 元数据 | 使用随包附带的官方 generic YAML；每 session 转成左右腕两条记录，共享 pair 和 split |
| Nature Discrete Gestures | `data` compound dataset，task=`discrete_gestures`，EMG/time，prompts/stages | 自动发现下载目录的 `discrete_gestures_corpus.csv` |
| Nature Handwriting | `data`，task=`handwriting`，EMG/time，prompts/stages | 自动发现 `handwriting_corpus.csv`；官方 `[0,inf)` 按实际源时钟展开 |
| Nature Wrist | `data`，task=`wrist`，EMG/time，连续 `wrist_angles` | 自动发现 `wrist_corpus.csv`，保留连续标签 |

保持 2000 Hz、16 通道，不重新采样、不重排电极、不额外滤波。Canonical 将时钟变成相对秒并保存源时间原点；Nature 区间仍使用源时钟。三份 Nature CSV 仅取存在对应 HDF5 的行，裁到实际源时间范围，不重新分配 split。

**QWERTY 官方 generic YAML 含 837 train / 192 val / 16 test session**。完整下载中的其余 session 属于其他协议用途，会明确列入 `excluded_sources`，不会补入训练。若要使用其他评测协议，需显式替换配置后独立审计，不混用不同用户协议。

Nature 官方下载器将各 `*_corpus.csv` 放在下载输出目录，支持其下嵌套子目录。若只解压数据 tar 而没运行官方完整下载器，可能尚缺 corpus CSV。仅当对应文件缺失时，可按官方下载脚本补齐这些公开元数据，无需重下 HDF5：

```bash
curl -fL https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data/data/handwriting/handwriting_corpus.csv -o /media/SSD4/personal/sjwu/foundation_model/dataset/meta_handwriting/handwriting_corpus.csv
curl -fL https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data/data/wrist/wrist_corpus.csv -o /media/SSD4/personal/sjwu/foundation_model/dataset/meta_wrist/wrist_corpus.csv
curl -fL https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data/data/discrete_gestures/discrete_gestures_corpus.csv -o /media/SSD4/personal/sjwu/foundation_model/dataset/meta_discrete_gestures/discrete_gestures_corpus.csv
```

保留已经下载的原文件。官方文档核对与链接见 `docs/SERVER_FORMAT_CONFIRMATION.md`；QWERTY 固定 commit、SHA256 和许可在 `configs/server/official/`。

## 3. 转换与续跑

```bash
bash scripts/run_user_server.sh prepare
```

新数据输出默认写入：

```text
/media/SSD4/personal/sjwu/foundation_model/dataset/meta_canonical_v1/
  manifest.csv
  splits.csv
  nature_corpus.csv
  preparation_report.json
  .prepare/                  # 所有权、按源事务与恢复信息，请保留
  emg2pose/...
  emg2qwerty/...
  gni_discrete_gestures/...
  gni_handwriting/...
  gni_wrist/...
```

首次输出目录必须为空。若已有其他转换产物，在 JSON 中换一个新 canonical 路径。中断后原命令重跑，完成文件校验源与输出 SHA256 后跳过；左右腕部分提交也可恢复。源文件、生成文件或已提交元数据被修改时停止，不自动覆盖。不要删除 `.prepare`，不要并行运行多个 prepare。原始 H5 和 CSV 都不改写。

转换逐源文件执行，沿用原转换器整文件读入，QWERTY 同时读入双腕。RAM 需容纳最大单个 session 及临时数组，实际峰值高于报告中的源未压缩字节数。Canonical 需要额外磁盘，压缩比依数据而定；逐文件临时余量检查不能保证全量最终空间。续跑也要顺序读取源和输出做哈希校验。

## 4. 原生预检与训练

```bash
# CUDA、依赖、五任务 train/val loader、至少5次真实优化
bash scripts/run_user_server.sh preflight --batch-size 8

# 训练入口也会先预检；先做100步服务器验收
bash scripts/run_user_server.sh train --batch-size 8 --max-steps 100

# 验收后完整预训练，步数按实验预算设置
bash scripts/run_user_server.sh train --batch-size 8 --max-steps 100000

# max-steps 为累计目标优化步数
bash scripts/run_user_server.sh train --batch-size 8 --max-steps 100000 --resume /实际运行目录/checkpoints/last.ckpt
```

预检失败不会启动正式训练。批大小同时用于预检与训练，默认单 GPU。100000 是示例预算，不是已经完成的训练。输出在配置 `run_root` 下，每次正式运行有独立 `e3_server_<UTC时间>/`。checkpoint、loss、诊断和状态路径见实际运行输出及 `docs/SERVER_DEPLOY.md`。

## 5. 验证范围

本地验证了五类微型夹具的检查、官方划分保留、转换和恢复，也对本地真实 Pose/Wrist 文件做了头部与抽样读入。尚未连接你的服务器；不能把本地测试当作服务器全量已通过或训练已完成。以服务器 `data_inspection.json`、`preparation_report.json` 和 GPU preflight 为准。

包不含数据、权重、虚拟环境或历史输出。`SOURCE_RELEASE.json` 保存源码基础 commit、文件清单及 SHA256，`python scripts/verify_release.py` 校验随包文件。源码 ZIP 不带 `.git`，训练 metadata 的 `git_commit` 因此可能为 null；需要 Git 溯源时可在独立目录初始化并提交解压源码，再训练，同时保留 release 清单。
