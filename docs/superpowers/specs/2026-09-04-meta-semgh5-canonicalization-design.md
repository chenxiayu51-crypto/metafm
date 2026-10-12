# Meta sEMG Canonical H5 转换设计

## 目标

把五个公开的 Meta/CTRL-labs sEMG 任务语料转换为统一的、关注信息保真的 per-wrist HDF5 格式，使其可以被同一个通用 dataloader 读取，同时保留任务专属标签和下游评测所需的源基准划分。

## 范围

转换器接收已下载的源数据并生成 canonical 输出。它不下载源压缩包、不训练模型、不创建固定训练窗口、不修改源文件，也不推断缺失标签。

支持的输入为 `emg2pose`、`emg2qwerty`，以及 `generic-neuromotor-interface` 的三个任务：`discrete_gestures`、`handwriting` 和 `wrist`。

## 选定的架构

每个输出 H5 表示一路物理手腕信号流。其公共波形数组始终是形状为 `[T, 16]` 的 `/signal/emg` 和形状为 `[T]` 的 `/signal/time_s`。因此，单手腕输入语料为每条源记录生成一个输出。`emg2qwerty` 从每个双侧会话生成两个输出文件，分别对应左手腕和右手腕，并为二者分配共享的 `pair_id`。

这样原始自监督采样是精确且一致的。打字 dataloader 通过在 `pair_id` 上连接左右两行输出来重建双侧输入；绝不能仅因为时间戳重叠就拼接任意记录。

所有标签都保留在 `/labels` 中，每个 `task_id` 对应一套任务专属布局。所有可变的溯源信息和任务选择信息都同样写入语料级 CSV manifest。通用 dataloader 只需读取公共根属性和 `/signal`；各任务头读取各自对应的标签组。

## Canonical H5 契约

每个输出文件都必须包含以下根属性：

```text
schema_version = "1.0"
dataset_id     = 取 {emg2pose, emg2qwerty, gni_discrete_gestures,
                     gni_handwriting, gni_wrist} 之一
task_id        = 取 {pose, typing, discrete_gesture, handwriting, wrist_angle} 之一
subject_id     = 源数据命名空间下的匿名标识符
session_id     = 源数据命名空间下的 session/录制标识符
side           = "left"、"right"、"dominant" 或 "unknown"
pair_id        = 仅双侧 qwerty 记录非空
sample_rate_hz = 2000
official_split = 源数据的 train/val/test 取值，或 "unknown"
source_file    = 相对输入目录的路径
preprocessing_json = 一个 JSON 对象，描述已知的源数据处理
                     以及本转换器施加的变换
```

每个输出文件都必须包含：

```text
/signal/emg       float32 [T, 16]
/signal/time_s    float64 [T]
/meta/source_json UTF-8 JSON 标量
/meta/quality     float64 或整型标量 dataset，分别记录样本数、
                  时长、NaN 数量和逐通道 RMS
/labels/...       见下方列出的任务专属 dataset
```

转换器必须保留时间戳，并将其平移为该输出记录中第一个样本所对应的相对秒数。原始绝对参考（若存在）保留在 `source_json` 中。信号会被转换为 `float32`，但不做重新滤波、重新缩放、符号翻转、旋转、归一化、重采样或裁剪。`preprocessing_json` 记录源数据声明的处理，以及 `converter_transformations: []`。

## 任务标签契约

| task_id | 必需的 `/labels` 内容 |
|---|---|
| `pose` | `joint_angles [T,D]`，可选的 `no_ik_failure [T]`，以及 `stage` UTF-8 标量 |
| `typing` | `keystrokes` 表，含按键、按下时间和释放时间；`prompts` 表，含文本、开始和结束时间 |
| `discrete_gesture` | 事件表，含手势名称和事件时间；提供时附带 stage 元数据 |
| `handwriting` | prompt 表，含文本、开始和结束时间 |
| `wrist_angle` | 源数据提供的所有连续腕角度数组，与 `time_s` 对齐；提供时附带 stage 元数据 |

事件时间和 prompt 时间使用与 `/signal/time_s` 相同的相对秒时钟表示。未知或缺失的可选源字段以“该可选数据集不存在”来表示，绝不编造数值。

## 文件

```text
semg_h5/
  __init__.py
  canonical.py                    # schema 常量、H5 写入、校验辅助函数
  source_readers.py               # 各源数据形状的读取器与源元数据提取
  manifest.py                     # manifest 行构造与 CSV 原子合并
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
  validate.py                     # 对 canonical 文件/目录树做 CLI 校验
tests/
  test_canonical.py
  test_convert_emg2pose.py
  test_convert_emg2qwerty.py
  test_convert_gni.py
README.md
```

`canonical.py` 是唯一允许写入 canonical H5 路径的模块。五个命令模块只负责把源记录适配为 `CanonicalRecord` 和任务标签载荷。这样可以防止公共字段出现漂移。

## CLI 契约

每个转换器都提供：

```text
python -m semg_h5.convert_<dataset> \
  --input-root INPUT_ROOT --output-root OUTPUT_ROOT \
  [--split train|val|test] [--overwrite]
```

脚本基于带命名空间的源 ID 生成确定性的文件名，并输出/更新 `OUTPUT_ROOT/manifest.csv`。在没有 `--overwrite` 的情况下，它拒绝覆盖已存在的 canonical H5。对于不可读的源文件、格式错误的时间向量、非 16 通道输入和未知的输入布局，它会给出可操作的、指明具体路径的错误。

## 错误处理与数据完整性

- 在抽取源字段之后，要求 EMG 恰好为 16 通道。
- 要求时间戳有限且单调，并且信号与时间的长度一致。
- 在记录级别保留源基准 split 和受试者身份；不创建窗口级随机划分。
- 独立转换 qwerty 的左右两侧，但在分配 `pair_id` 之前校验二者共享的长度/时间戳对齐关系。
- 将所有已知的源预处理保留在元数据中，不做任何未记录的生理变换。
- 在 manifest 中保存相对源根的输入路径和源文件 SHA-256，用于溯源。

## 测试策略

测试使用与各公开 reader 文档所述布局一致的小型合成 H5 fixture。在实现之前，测试会断言共享 writer 拒绝非法的 `[T,C]` 波形并生成合法的 canonical 文件。随后，按数据源划分的测试会断言波形形状、`task_id`、源 split/ID、任务标签转换，以及 qwerty 的左右配对。一个校验 CLI 测试会检查转换后的输出目录树和 manifest 一致性。

## 验收标准

1. 五个转换器 CLI 存在，且无需修改源代码即可使用。
2. 每个输出的 H5 都满足公共契约，并可通过一条命令完成校验。
3. `emg2qwerty` 对每个双侧源会话输出两个相关联的 16 通道文件。
4. 保留官方下游任务所需的充分标签，且不编造标签。
5. manifest 支持按 dataset、task、split、subject、session、side 和 `pair_id` 过滤。
6. 测试创建合成源文件，在实现过程中呈现红—绿行为，并作为整套测试全部通过。

