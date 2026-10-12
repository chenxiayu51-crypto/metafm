# Meta 16 通道腕带 sEMG 公开数据集整理

检索日期：2026-09-04

## 结论先行

目前能够确认由 Meta Reality Labs / CTRL-labs 公开、并使用同一类 **sEMG-RD 16 通道腕带、2 kHz 采样**的数据资产，主要来自 3 个项目，但按任务应拆成 5 个语料：

1. `emg2pose`
2. `emg2qwerty`
3. `generic-neuromotor-interface/discrete_gestures`
4. `generic-neuromotor-interface/handwriting`
5. `generic-neuromotor-interface/wrist`

它们非常适合共同训练一个 16 通道 sEMG 基础编码器，但“同设备”不等于“把 HDF5 直接 concat 即可”。采样率和通道拓扑高度一致，主要差异来自：左右手极性、双腕/单腕结构、振幅尺度、预处理状态、标签时序精度以及小时数统计口径。

按当前仓库统计，5 个任务合计约 **998.2 个数据集标称小时**。若将 `emg2qwerty` 的左右腕拆成两个单腕自监督流，则约有 **1,344.2 单腕小时**，对应约 **96.8 亿个 16 通道时间点**（2 kHz）。这个数不能与论文中的“录制墙钟时间”混用。

## 数据清单

| 项目 / 子任务 | 公开受试者与规模 | 输入结构 | 监督信号 | 适合的基础模型用途 | 当前许可 |
|---|---:|---|---|---|---|
| `emg2pose` | 193 人；751 sessions；25,253 个 HDF5；370.3 h；29 stages | 双腕同步采集，但每个文件为单手 `T×16`；2 kHz；12 bit；40 Hz 高通；左手已经翻转符号；噪声底标准差缩放为 1 | 与 sEMG 对齐的手部关节角；pose regression / tracking | 最强的连续运动与手型监督；也适合单腕无标签预训练 | CC BY-NC-SA 4.0 |
| `emg2qwerty` | 108 人；1,135 sessions；346 h；超过 520 万次按键 | 每个 session 同时包含左、右腕，各 `T×16`，合计 32 通道；2 kHz；40 Hz 高通；双腕按 0.5 ms 对齐 | prompt、按键按下/释放及时间戳 | 双腕时序表征、跨腕对比学习、文本/按键序列解码 | LICENSE 文件为 CC BY-NC-SA 4.0 |
| `discrete_gestures` | 100 人；100 recordings；当前 README 合计 64.0 h（Nature 正文约 63 h） | 优势手单腕 `T×16`；2 kHz；40 Hz 高通 | 9 类事件：食指/中指 press 与 release、拇指 tap、上下左右 swipe；标签由 prompt 对齐算法得到 | 稀疏事件检测、意图分类、静息/动作边界学习 | 当前 LICENSE 为 CC BY-NC 4.0 |
| `handwriting` | 100 人；807 recordings；当前 README 合计 140.7 h（Nature 正文 126 h） | 优势手单腕 `T×16`；2 kHz；40 Hz 高通 | 字母、数字、单词、随机字符串、短语的 prompt 及开始/结束时间 | 长程序列建模与语言条件微调 | 当前 LICENSE 为 CC BY-NC 4.0 |
| `wrist` | 100 人；182 recordings；当前 README 合计 77.2 h（Nature 正文约 74 h） | 优势手单腕 `T×16`；2 kHz；40 Hz 高通 | 屈伸与尺桡偏腕角；公开代码说明腕角已上采样到 2 kHz | 连续低自由度运动回归、动力学表示 | 当前 LICENSE 为 CC BY-NC 4.0 |

### 规模口径说明

- Meta 对 `emg2pose + emg2qwerty` 的官方宣传口径是 301 名参与者、716 h，即 `370 + 346`。但 `emg2pose` 的 370 h 已把同时采集的左右手分别计时，而 `emg2qwerty` 的 346 h 是含两只腕带的 session 时长。因此做模型 token / 信号量预算时，建议另算“单腕小时”。
- `emg2pose`：370.3 单腕小时。
- `emg2qwerty`：346 h × 2 腕 ≈ 692 单腕小时。
- Nature 三任务：当前仓库 64.0 + 140.7 + 77.2 = 281.9 单腕小时；论文数据可用性声明为 63 + 126 + 74 ≈ 263 h。差异可能来自仓库更新或时长统计边界，正式实验应以下载后的 corpus CSV 重算。
- 当前仓库口径合计：370.3 + 692 + 281.9 = **1,344.2 单腕小时**。
- 参与者只能写成 **601 个“数据集—参与者实例”**，不能宣称 601 名独立个体。不同项目的匿名 ID 不可跨库关联，也不能排除原始招募人群存在重叠。

## 为什么可归一

sEMG-RD 的共同硬件基础包括：

- 16 个差分通道，干式镀金电极，围绕腕部排列；
- 通道 0 与 15 之间的缺口位于靠近尺骨茎突一侧；
- 2 kHz 采样，12 bit（`emg2pose` 论文明确给出）；
- 硬件前端约 20–850 Hz 模拟带通，公开数据再做 40 Hz 数字高通；
- 左右腕使用镜像佩戴方式，相同编号对应相近解剖位置，但差分极性相反；
- 官方基线本身使用 circular channel roll 模拟腕带旋转，说明 16 通道应被视为环形拓扑，而不是普通无序特征。

因此，五个任务可以共享同一个 raw-waveform encoder；任务标签则应放入独立 adapter / head，而不是强行统一为一个标签空间。

## 不能直接拼接的关键原因

### 1. 左右手极性

`emg2pose` 明确说明已对左手信号翻转符号。`emg2qwerty` 论文明确说明左右腕差分极性相反，但公开论文和 README 没有同样明确地保证发布文件已完成符号统一。因此不要对所有左腕再次乘以 `-1`；应给每个数据源记录 `polarity_canonicalized`，并通过代码、样本统计和跨腕相关结构验证。

对于自监督预训练，可加入全局符号翻转增强，使 encoder 不依赖任意极性；对于 MUAP 形态研究，则必须保留可追溯的原始极性与变换记录。

### 2. 振幅单位和缩放不同

`emg2pose` 已把噪声底缩放到标准差 1；Nature 方法部分则把设备噪声 2.46 μVrms 作为尺度常数；其他发布文件的数值单位不能仅凭 dtype 推断。因此不宜先对每个通道独立 z-score 后混合——这样会破坏跨通道幅值包络，而该包络恰是运动单位与腕部空间模式的重要信息。

推荐双路表示：

- 波形路：每 session 使用一个跨 16 通道共享的 robust scale（静息段 RMS 或全局 MAD）；
- 幅值路：额外保留 `log(scale)`、每通道 RMS / 噪声估计等慢变量；
- 原始文件永不覆盖，所有 scale、sign、rotation 都写入 manifest。

### 3. 单腕与双腕任务不同

- 自监督预训练：可把 `emg2qwerty` 左右腕拆成两个 16 通道样本，并保留 `pair_id`；可额外做左右腕同步对比或跨腕预测。
- 监督 typing：不能拆散，必须用双腕联合输入，因为按键标签来自两手共同完成的序列。
- `emg2pose`：左右手虽然同步采集，但官方文件按单手拆分；如要恢复双腕 pairing，需要从 user/session/stage/time 元数据显式配对。
- Nature 三任务：均为优势手单腕，不能假设左右手都有对应数据。

### 4. 标签质量不同

- `emg2pose` 与 `wrist` 有外部 motion capture 连续回归目标，时间精度最好。
- `emg2qwerty` 有键盘 keylogger，按键时刻精确。
- `discrete_gestures` 的真实动作时刻不是传感器直接测得，而是从 prompt 经过算法对齐，存在残余时移。
- `handwriting` 主要是 prompt 与起止区间，不是逐笔画轨迹；适合序列监督，但不能当作精确运动学标签。

## 推荐的统一数据协议

### Canonical sample schema

```text
signal              float32 [T, 16]
sample_rate_hz       2000
dataset_id           {emg2pose, emg2qwerty, gni_discrete, gni_handwriting, gni_wrist}
task_id              {ssl, pose, typing, gesture, handwriting, wrist_angle}
subject_id           dataset_id + original_subject_id
session_id           dataset_id + original_session_id
side                 {left, right, dominant_unknown}
pair_id              optional; bilateral synchronous pair
channel_order        explicit 0..15 mapping
polarity_transform   {none, global_negate, unknown}
rotation_offset      integer or unknown
amplitude_scale      applied scalar and estimation method
timestamp            float64 [T] or start_time + sample_index / 2000
labels               task-specific object
official_split       original train / val / test
license_id           source-specific license snapshot
source_file_sha256   checksum
```

### 最小归一流水线

1. 用各仓库官方 loader 读取，不先重写原 HDF5。
2. 校验每段的采样间隔、shape、NaN、饱和、平线、异常 RMS 和丢包。
3. 统一成 `[T,16] @ 2 kHz`，保留真实 timestamp；不要对已高通数据重复施加 40 Hz 高通。
4. 建立左右手极性状态表；未知状态不做不可逆翻转。
5. 采用 session 级共享 robust scale，同时保留原幅值统计作为条件变量。
6. 以环形通道增强处理腕带旋转：训练时做 `roll(k)`；若做反射或 sign flip，必须作为独立增强并记录。
7. 推荐 0.5–2.0 s 的 raw window；例如 1.024 s = 2,048 samples，patch 可设为 20 或 40 samples（10/20 ms）。
8. 数据采样按“任务均衡 + subject 均衡”，否则 `emg2qwerty` 或长 session 会主导 batch。
9. user/session 级切分，绝不能在窗口化之后随机切分。
10. 若要报告官方 benchmark，不得把官方 val/test 的无标签信号用于联合预训练；否则应明确改称 transductive 设置。

## 建议的基础模型研究框架

### 预训练阶段

- 主干：面向环形 16 通道的 Conv stem + Transformer / Conformer / SSM。
- 自监督目标：masked waveform/latent prediction、时频一致性、相邻通道预测。
- 利用 `emg2qwerty` 的双腕同步：同一动作上下文的跨腕对比、跨腕 masked prediction。
- 利用有标签数据加入弱监督锚点：pose velocity、wrist angle、gesture event、keystroke/text。
- 用 dataset embedding 和 slow calibration token 表示设备批次、session 与幅值尺度，但同时用 adversarial 或 invariance loss 抑制数据集捷径。

### 评测矩阵

至少同时报告：

1. held-out user；
2. held-out session / band placement；
3. held-out task linear probe；
4. leave-one-dataset-out transfer；
5. 少样本个体化曲线（0、1、5、15、30 min）；
6. 数据集身份分类准确率，作为残余 domain shift 的诊断指标；
7. 对 channel rotation、polarity、amplitude drift 和缺失通道的鲁棒性。

最有研究价值的主问题可以表述为：**在同一硬件但异任务、异用户、异佩戴、异监督质量的条件下，能否学习一个对腕带环形位移和幅值漂移稳健、同时保留精细运动单位信息的通用 sEMG 表征？**

## 获取入口

### emg2pose

- 项目与 loader：<https://github.com/facebookresearch/emg2pose>
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2024/file/64f8884f6ba3d9ace5bb647c4f917896-Paper-Datasets_and_Benchmarks_Track.pdf>
- 完整数据（约 431 GiB）：<https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar>
- mini（约 600 MiB）：<https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar>
- metadata：<https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv>

### emg2qwerty

- 项目与 loader：<https://github.com/facebookresearch/emg2qwerty>
- 论文：<https://proceedings.neurips.cc/paper_files/paper/2024/file/a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf>
- 数据：<https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz>
- 新一些的训练框架（复用同一数据）：<https://github.com/facebookresearch/fairemg>

截至检索日，`emg2qwerty` GitHub 仓库已于 2026-08-01 归档为只读；S3 数据链接仍由 README 提供。`FairEMG` 是训练框架，不是额外的数据集，统计时不可重复计算。

### Nature / generic neuromotor interface

- 当前项目与下载脚本：<https://github.com/facebookresearch/generic-neuromotor-interface>
- Nature 论文：<https://www.nature.com/articles/s41586-025-09255-w>
- 数据根目录：<https://fb-ctrl-oss.s3.amazonaws.com/generic-neuromotor-interface-data>

官方命令：

```bash
python -m generic_neuromotor_interface.scripts.download_data \
    --task discrete_gestures \
    --output-dir ~/emg_data

python -m generic_neuromotor_interface.scripts.download_data \
    --task handwriting \
    --output-dir ~/emg_data

python -m generic_neuromotor_interface.scripts.download_data \
    --task wrist \
    --output-dir ~/emg_data
```

加 `--small-subset` 可先下载每个任务 3 名参与者的小样本进行 schema 验证。

## 许可与发布注意事项

- `emg2pose`、`emg2qwerty` 的仓库 LICENSE 为 **CC BY-NC-SA 4.0**：署名、非商业、衍生材料相同方式共享。
- `generic-neuromotor-interface` 当前仓库 LICENSE 为 **CC BY-NC 4.0**；但 Nature 正文最初写的是 BY-NC-SA 4.0。正式发布模型或再分发预处理数据前，应保存下载时的 LICENSE commit，并让机构法务确认适用版本。
- 这些数据适合非商业学术研究；不要把基于它们训练的权重默认视为可商用资产。
- 如发布统一后的衍生数据，最保守做法是对整个混合集遵循最严格的 BY-NC-SA 条款，并逐数据源保留 provenance。

## 一句话取舍

- 想做通用 raw encoder：五个任务全部纳入。
- 想做最干净的连续运动监督：优先 `emg2pose + wrist`。
- 想做长上下文与跨腕建模：优先 `emg2qwerty`。
- 想做低延迟事件检测：优先 `discrete_gestures`。
- 想做跨任务基础模型论文：全量自监督 + 五个独立任务头 + leave-one-dataset-out，是最有说服力的设计。

## 主要一手来源

1. Salter et al., *emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation*, NeurIPS 2024 Datasets and Benchmarks.
2. Sivakumar et al., *emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography*, NeurIPS 2024 Datasets and Benchmarks.
3. Kaifosh, Reardon & CTRL-labs at Reality Labs, *A generic non-invasive neuromotor interface for human-computer interaction*, Nature 2025, DOI: 10.1038/s41586-025-09255-w.
4. Meta AI, *Advancing Neuromotor Interfaces by Open Sourcing Surface Electromyography (sEMG) Datasets for Pose Estimation and Surface Typing*：<https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/>

