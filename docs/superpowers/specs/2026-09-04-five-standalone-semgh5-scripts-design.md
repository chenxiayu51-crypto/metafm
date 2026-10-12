# 五个独立的 sEMG H5 转换脚本

## 目标

将 `D:\Fmodel\semg_h5` 精简为恰好五个可独立执行的 Python 脚本，每个对应一个受支持的公开 Meta/CTRL-labs 数据集任务：

1. `convert_emg2pose.py`
2. `convert_emg2qwerty.py`
3. `convert_gni_discrete_gestures.py`
4. `convert_gni_handwriting.py`
5. `convert_gni_wrist.py`

任何脚本都不得从 `semg_h5` 导入其他本地模块。

## 设计

每个保留下来的脚本都内嵌其所需的小型公共函数：源数据布局检查、`[T,16]` sEMG 校验、canonical H5 写入、任务标签序列化、输出路径生成、SHA-256 溯源、CSV manifest upsert、可选的官方 split 查询，以及 argparse CLI。脚本可以导入 `requirements.txt` 中声明的第三方依赖（`h5py`、`numpy`、`pandas`）以及 Python 标准库。

五个脚本保持现有的输出契约：

- `/signal/emg`：`float32 [T,16]`
- `/signal/time_s`：`float64 [T]`，仅平移到相对于本条记录的原点
- 根级溯源属性与 `/meta/quality`
- 各任务标签：pose 角度；qwerty 按键/提示文本；离散手势事件；手写提示区间；wrist 连续目标
- `manifest.csv` 字段，包括任务 ID、官方 split、各类 ID、side、pair ID 和源文件哈希

`convert_emg2qwerty.py` 为每个手腕输出一个 H5，并保留共享的 `pair_id`。三个 GNI 脚本各自校验自己的源任务取值，而不共用同一个 runner。

## 移除与兼容性

删除 `semg_h5` 下所有其他 Python 文件：`__init__.py`、`canonical.py`、`manifest.py`、`source_readers.py`、`gni_conversion.py` 和 `validate.py`。独立脚本必须能够直接运行，例如：

```powershell
python D:\Fmodel\semg_h5\convert_emg2pose.py --input-root RAW --output-root CANONICAL
```

先前的包模块调用方式与独立的校验命令被有意移除。校验在每个脚本写入记录之前执行。README 示例将改为直接执行脚本。

## 验证

测试将按文件路径直接调用每个脚本，对合成的源 H5 fixture 运行。测试将验证每个输出都包含 `[T,16]` sEMG 及其任务专属标签，qwerty 会生成相互关联的左右配对，以及每个脚本都不从 `semg_h5` 导入任何内容。
