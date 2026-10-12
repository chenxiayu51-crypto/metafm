# Ring Criss-Cross sEMG 基座模型设计

**日期：** 2026-09-09  
**状态：** 已批准进入实施规划  
**仓库：** `D:\Fmodel`

## 1. 目标

在现有的五个 Meta 16 通道手腕 sEMG H5 转换脚本之上，构建一套可复现的 PyTorch Lightning 与 Hydra 研究栈。该研究栈将支持：

- 面向均匀排布的 16 电极手腕环的拓扑感知自监督预训练；
- 在 emg2pose、emg2qwerty、GNI 离散手势、GNI 手写和 GNI wrist 上的下游评测；
- 有监督 TCN、BENDR 风格、CBraMod 风格和 AEMG 风格的对比方法；
- 公平的 subject/session/pair 划分、低标签迁移和通道扰动鲁棒性评测；
- 无需改动代码即可在 1 到 8 张 A100 GPU 上训练。

文档中将主方法命名为 **CIRCE-Myo**。其骨干是 **Ring Criss-Cross Transformer**，它将时间注意力与环形空间注意力结合起来。研究结论仅限于 Meta 腕带系列内部跨任务、跨人、跨会话以及电极相位扰动下的迁移。在没有外部设备数据的情况下，本项目不会声称具备跨设备泛化能力。

## 2. 范围

### 2.1 纳入范围

- 从五个转换脚本产出的 canonical H5 契约进行惰性加载。
- 确定性的 subject/session/pair 级划分管理。
- 基于时间的切窗，标签由 `time_s` 对齐。
- 多尺度时域与频谱 patch 编码器。
- 带环形相对位置偏置的 Ring Criss-Cross 注意力。
- 连续掩码教师预训练、环形等变性，以及可选的运动学锚定。
- 五个任务适配器以及与任务匹配的指标。
- 四个使用相同数据和下游协议的对比方法。
- 用于 CPU 冒烟测试和 1 到 8 卡 GPU DDP 运行的 Hydra 配置。
- 使用合成的 canonical H5 文件的单元测试与集成测试。

### 2.2 首版实现不包含

- 用于 qwerty 解码的外部语言模型。
- 对 BENDR、CBraMod 或 AEMG 的官方复现声明。
- 多节点训练。代码或许能在 Lightning 多节点设置下工作，但验收测试只针对单节点最多 8 张 GPU。
- 在线服务、移动端部署和量化。
- 跨厂商设备评测。
- 任意通道置换不变性。只允许手腕环几何所能支持的变换。

## 3. 仓库结构

`semg_h5/` 中的五个转换脚本保持独立且不变。新的训练代码隔离在 `semg_fm/` 中。

```text
D:\Fmodel
├── semg_h5/
├── semg_fm/
│   ├── data/
│   │   ├── manifest.py
│   │   ├── splits.py
│   │   ├── window_dataset.py
│   │   ├── task_labels.py
│   │   ├── datamodule.py
│   │   └── samplers.py
│   ├── models/
│   │   ├── patch_encoder.py
│   │   ├── ring_attention.py
│   │   ├── ring_criss_cross.py
│   │   ├── bendr.py
│   │   ├── cbramod.py
│   │   ├── aemg.py
│   │   ├── supervised_tcn.py
│   │   └── heads.py
│   ├── objectives/
│   │   ├── masked_latent.py
│   │   ├── ring_equivariance.py
│   │   ├── kinematic.py
│   │   └── vector_quantization.py
│   ├── lightning/
│   │   ├── pretrain_module.py
│   │   ├── downstream_module.py
│   │   └── baseline_module.py
│   ├── metrics/
│   └── utils/
├── configs/
│   ├── pretrain.yaml
│   ├── finetune.yaml
│   ├── evaluate.yaml
│   ├── baseline.yaml
│   ├── data/
│   ├── model/
│   ├── task/
│   └── trainer/
├── pretrain.py
├── finetune.py
├── evaluate.py
├── run_baseline.py
├── tests/
├── requirements-train.txt
├── PRETRAINING.md
├── DOWNSTREAM_EVALUATION.md
├── BASELINES.md
├── EXPERIMENT_PROTOCOL.md
├── H5_DATA_CONTRACT.md
└── SERVER_A100.md
```

## 4. Canonical 数据契约

每条记录必须包含：

- `/signal/emg`：有限的 `float32 [T, 16]`；
- `/signal/time_s`：单调的 `float64 [T]`；
- 根属性，包括 `dataset_id`、`task_id`、`subject_id`、`session_id`、`side`、`pair_id`、`official_split` 和 `sample_rate_hz`；
- `/labels` 下的任务专属数据集。

loader 在产出窗口之前先校验该契约。它按 worker 惰性打开 H5 文件，且不会在进程间序列化时保留存活的 `h5py.File` 句柄。窗口和 patch 时长以毫秒配置，再依据每条记录的 `sample_rate_hz` 换算。

dataloader 返回统一的字典：

```python
{
    "emg": FloatTensor[T, 16],
    "sample_mask": BoolTensor[T],
    "dataset_id": str,
    "task_id": str,
    "record_id": str,
    "subject_id": str,
    "session_id": str,
    "side": str,
    "pair_id": str,
    "start_s": float,
    "sample_rate_hz": float,
    "target": task_specific_value,
}
```

任务专属的 collate 由 `task_id` 选择。预训练 batch 在源任务上是同质的，以便可选标签具有稳定的 schema。任务均衡 sampler 控制各数据集在长期运行中的贡献比例。

## 5. 划分与泄漏防护

划分发生在切窗之前。loader 优先使用非空的 `official_split` 取值。若某个语料缺少官方划分，则由一个专用命令按受试者分组生成确定性的 split CSV。该文件此后成为所有后续运行的不可变输入。

规则：

- 除非官方协议明确要求受试者适配，否则同一个受试者不得出现在多个划分中；
- 同一个会话不得跨划分；
- 共享 `pair_id` 的 qwerty 左右记录必须留在同一个划分中；
- 低标签子集按完整的受试者/会话组选取，而不是随机窗口；
- 归一化统计量仅从训练记录估计；
- 每次运行都会复制解析后的 split CSV，并保存源 manifest 的 SHA-256 摘要。

## 6. 信号归一化与增强

H5 转换脚本保留源数值。训练预处理在 Hydra 配置中显式声明且可复现。

默认归一化使用从训练语料或记录估计的、在 16 个通道间共享的 robust 标量。默认禁用独立的逐通道归一化，因为它会破坏空间幅值比例。解析后的配置会记录滤波、归一化和增强参数。

物理上支持的增强包括：

- 环形通道平移；
- 连续环扇区 dropout；
- 独立通道 dropout；
- 标量或缓慢变化的增益漂移；
- 加性噪声；
- 短时块 dropout。

禁止任意通道乱序。左右镜像和极性翻转保持禁用，直到转换器元数据能够证明某语料所需的物理变换成立。

## 7. CIRCE-Myo 骨干

### 7.1 多尺度 patch 编码器

输入形状为 `[B, T, 16]`。短时域分支捕获约 20–50 ms 的形态。第二个分支概括约 100–250 ms 的包络或对数功率信息。配置以毫秒指定时长，使采样率不同的记录可以共享同一套模型契约。

融合后的张量形状为 `[B, C=16, N, D]`。实现在整个骨干中保留通道轴。

### 7.2 Ring Criss-Cross 模块

每个模块并行计算时间注意力和空间注意力：

- 时间注意力重塑为 `[B*C, N, D]`，在每个电极上沿时间做注意力；
- 环形注意力重塑为 `[B*N, C, D]`，在每个时间 token 上跨电极做注意力；
- 一个学习到的门控融合两路残差更新。

环形注意力使用可学习的带符号环形相对位置偏置，以 `(c_i - c_j) mod 16` 索引。因此通道 15 和 0 相邻。固定的通道编号不被赋予任何绝对的解剖学含义。

骨干返回：

- `z_eq [B, 16, N, D]`，电极场表示，在环形平移下应等变地变换；
- `z_motion [B, N, D]`，轨道池化后的运动表示，设计为在环形平移下保持稳定。

### 7.3 掩码生成

掩码在 `[16, N]` 的 patch 网格上进行，并且始终输出显式的布尔掩码。可用的掩码类型包括时间段、连续电极弧段、整通道 dropout、局部时空块和稀疏独立 patch。默认配置采样混合类型，使重建既需要时间上下文，也需要跨电极上下文。

## 8. 预训练目标

默认方法使用三个目标：

```text
L = L_masked_latent + lambda_ring * L_ring + lambda_kin * L_kinematic
```

### 8.1 连续掩码教师目标

EMA 教师接收未掩码视图，学生接收掩码视图。损失只在被掩码的位置计算，由教师与学生连续潜变量之间的归一化 Smooth-L1 与余弦距离组合而成。教师端 stop-gradient，并在优化器步之后更新。所有 DDP rank 从相同参数出发，并执行完全一致的 EMA 更新。

### 8.2 环形等变目标

对于随机采样的环形平移 `S_k`，电极场目标是等变的，池化后的运动目标是不变的：

```text
z_eq(S_k x) ~= S_k z_eq(x)
z_motion(S_k x) ~= z_motion(x)
```

实现在计算第一个损失之前先对齐被平移过的电极轴。该目标不会把任意样本或左右手腕当作正样本对。

### 8.3 运动学目标

该目标仅对来自 emg2pose 和 GNI wrist 的训练记录生效。一个小型适配器从 `z_motion` 预测归一化后的连续目标。逆运动学状态无效的 emg2pose 帧会从损失中屏蔽。运动学归一化统计量来自训练划分，评测时恢复原始单位。

包络重建、频谱重建、未来预测和 VQ 目标是可选消融项，而不是默认损失。

## 9. 下游任务

### 9.1 emg2pose

- 数据来源：`/labels/joint_angles` 和可选的 `/labels/no_ik_failure`。
- 任务头：逐帧连续回归。
- 指标：逐关节及总体 MAE、RMSE、Pearson 相关和速度 MAE。
- 无效 IK 帧同时从损失和指标中排除。

### 9.2 GNI wrist

- 数据来源：`/labels/targets` 下的数值数据集。
- 任务头：可配置的多目标回归。
- 指标：每个目标的原始单位 MAE、RMSE 和相关，以及它们的宏平均。

### 9.3 GNI 离散手势

- 数据来源：`/labels/events`，含 `name` 和 `time_s`。
- 任务头：帧级分类，包含背景类。
- 训练目标：以事件为中心、按毫秒配置的容差区间。
- 指标：宏 F1、平衡准确率，以及在声明的容差时间下的事件 F1。

### 9.4 GNI 手写

- 数据来源：`/labels/prompts`，含 `text`、`start_s` 和 `end_s`。
- 任务头：在每个 prompt 区间上做字符级 CTC 解码。
- 指标：字符错误率、归一化编辑距离和 prompt 完全匹配准确率。
- 实现不会编造逐字符的时间标签。

### 9.5 emg2qwerty

- 数据来源：`/labels/keystrokes` 和 `/labels/prompts`。
- 任务头：字符级 CTC 解码器，可选输出用于按键计时的 event。
- 指标：字符错误率、按键事件 F1 和检测延迟。
- 主要的编码器对比不使用外部语言模型。

## 10. 微调协议

每个任务都支持：

- 冻结骨干的线性探针；
- 对骨干末尾配置数量的模块做部分微调；
- 完整微调。

标签预算为受试者/会话组级别的 1%、5%、10% 和 100%。所有方法使用相同的任务头、优化器族、最大更新预算、划分文件与预处理。每个报告配置至少支持三个随机种子。

## 11. 对比方法

### 11.1 有监督 TCN

一个时间卷积编码器，直接在各下游任务上训练，不做自监督预训练。用于确认预训练相对于一个足够强的有监督模型是否有增益。

### 11.2 BENDR 风格基线

卷积潜变量编码器加时间 Transformer，采用掩码连续对比预测。负样本来自同一记录中的其他位置。用于检验增益是否来自不显式使用环形拓扑的普通连续掩码预训练。

### 11.3 CBraMod 风格基线

时域与 FFT patch 编码器接入并行的时间注意力和普通空间注意力。条件位置编码和掩码波形重建遵循 CBraMod 的核心机制。它使用固定通道顺序，不带环形相对位置偏置或等变损失。

### 11.4 AEMG 风格基线

收缩感知编码器、向量量化 codebook 和掩码 token 分类器复现 AEMG 的核心假设。日志包含 codebook 困惑度、code 使用率和死 code 比例。文档会写明与原文的全部差异，且不将其标注为官方实现。

所有方法都提供 small 和 base 两档配置。实验表报告参数量、训练 token 数、步数、全局 batch size 和近似算力，使模型规模或数据预算的差异始终可见。

## 12. 鲁棒性评测

评测时的变换在内存中进行，绝不修改 H5 文件。必须包含的测试有：

- 1、2、4 和 8 个通道的环形平移；
- 一个或多个相邻电极缺失；
- 随机单通道 dropout；
- 增益漂移；
- 按配置信噪比加入的加性噪声；
- 短时信号丢失。

指标按数据集、受试者、扰动类型和严重程度分别报告。合成环形平移实验用于确立算法层面的通道相位鲁棒性。除非源数据包含相应的真实重新佩戴元数据，否则不会把它们描述为真实重佩戴鲁棒性的证据。

## 13. PyTorch Lightning 与 Hydra 运行时

顶层脚本是轻量的 Hydra 入口。配置组用于选择数据、模型、任务、鲁棒性变换和 trainer profile。每次运行都会保存解析后的配置。

默认服务器行为：

- 在 A100 上使用 BF16 混合精度；
- 单卡使用 `strategy=auto`，多卡使用 DDP；
- `devices=auto`，可通过 `trainer.devices=1`、`2`、`4` 或 `8` 覆盖；
- 使用梯度累积以维持配置的有效全局 batch size；
- 在支持处使用确定性随机种子；
- 对 model、optimizer、scheduler、EMA teacher、epoch 和 global step 做 checkpoint；
- 从 `last.ckpt` 重启时不重置 EMA teacher 与 schedule。

运行元数据记录 Git commit、manifest 摘要、split 摘要、依赖版本、可见 GPU 数、world size、每设备 batch size、有效全局 batch size、精度和随机种子。

## 14. 输出

每次运行都会写入一个 Hydra 输出目录，其中包含：

```text
config.yaml
metrics.json
per_subject_metrics.csv
split_manifest.csv
environment.txt
checkpoints/
```

在配置允许时，评测还会额外写出 `predictions.h5`。异构任务的指标保持相互独立；框架不会为回归、分类和序列识别强行合成一个单一的聚合分数。

## 15. 测试策略

测试创建包含 16 通道和确定性标签的紧凑 canonical H5 fixture。生产代码遵循测试驱动开发。

必须覆盖的单元测试包括：

- manifest 校验与惰性 H5 访问；
- 时间到样本的窗口换算与标签对齐；
- 划分泄漏检查，包括 qwerty 的 `pair_id`；
- DDP rank 切分下的任务均衡采样；
- 环形距离与通道滚动行为；
- 通道 15 和 0 的 Ring Attention 边界行为；
- 评测模式下 Ring Criss-Cross 骨干的数值等变性；
- 掩码语义，以及损失只在被掩码位置生效；
- 全部五个任务的 collator、损失和指标；
- EMA teacher 的 checkpoint 状态；
- Hydra 组合与命令行覆盖。

集成测试对 CIRCE-Myo 和每个预训练基线各跑一次 CPU `fast_dev_run`，然后对每个任务跑一条短微调/评测路径。本地测试套件不声称验证 NCCL。`SERVER_A100.md` 定义了针对 1、2、4 和 8 张 GPU 的单独单 batch DDP 冒烟测试。

## 16. 文档

- `README.md`：安装、项目地图和五分钟上手命令。
- `PRETRAINING.md`：CIRCE-Myo 架构、损失、数据混合和续训行为。
- `DOWNSTREAM_EVALUATION.md`：任务标签、指标、划分规则、低标签协议和鲁棒性命令。
- `BASELINES.md`：基线机制、保真度边界和公平对比规则。
- `EXPERIMENT_PROTOCOL.md`：主表、消融、随机种子、预算和报告清单。
- `H5_DATA_CONTRACT.md`：每个转换器输出所必需的字段和校验命令。
- `SERVER_A100.md`：环境搭建，以及 1 到 8 卡 GPU 的启动、续训和排障命令。

## 17. 验收标准

实现满足以下条件时视为完成：

1. 现有五个转换器测试仍然通过，且转换脚本保持独立。
2. 合成的 canonical manifest 能为 CIRCE-Myo、BENDR 风格、CBraMod 风格和 AEMG 风格方法各跑通一次 CPU 预训练步。
3. 合成数据能为全部五个下游任务各跑通一次训练/评测步。
4. 划分测试证明受试者、会话和 qwerty 配对不会跨划分泄漏。
5. 环形等变测试在声明的浮点容差内通过。
6. Hydra 命令无需修改源码即可选择模型、任务、trainer 设备、标签预算和鲁棒性设置。
7. checkpoint 续训能恢复 EMA teacher 和训练步。
8. Markdown 文档中的每条公开命令都由配置组合或冒烟测试覆盖。
9. 完整的本地测试套件通过。
10. A100 服务器冒烟测试说明覆盖 1、2、4 和 8 张 GPU，并清楚区分未验证的多卡行为与本地已验证的行为。

