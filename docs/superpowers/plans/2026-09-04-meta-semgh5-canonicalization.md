# Meta sEMG Canonical H5 转换实施计划

> **面向 agentic workers：** 必需子技能（REQUIRED SUB-SKILL）：使用 `executing-plans` 逐任务实施本计划。步骤使用复选框语法（`- [ ]`）跟踪进度。

**目标：** 创建五个经过测试的 CLI 转换器，把 Meta/CTRL-labs 公开 sEMG HDF5 语料转换为统一的 per-wrist canonical H5 schema，并附带一份溯源 manifest。

**架构：** 共享模块 `semg_h5.canonical` 负责 schema 校验与所有 canonical H5 写入。各来源专用的 reader 只负责把已记录的源数组与标签抽取为 `CanonicalRecord`。每个输出文件包含一路 `[T,16]` 腕部信号；qwerty 借助 `pair_id` 产生两个相互关联的输出。CSV manifest 镜像选择/溯源字段，使 dataloader 无需打开每个 H5 即可完成过滤。

**技术栈：** Python 3.10+、`h5py`、`numpy`、`pandas`、`pytest`；标准库 `argparse`、`json`、`hashlib`、`csv`、`pathlib`。

---

## 文件结构

```text
semg_h5/
  __init__.py
  canonical.py
  manifest.py
  source_readers.py
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
  validate.py
tests/
  conftest.py
  test_canonical.py
  test_manifest.py
  test_convert_emg2pose.py
  test_convert_emg2qwerty.py
  test_convert_gni.py
README.md
requirements.txt
```

`canonical.py` 是唯一获准在 canonical 文件中创建 `/signal`、`/labels` 和 `/meta` 的代码路径。`manifest.py` 为每个输出 H5 序列化出稳定的一行。`source_readers.py` 提供五个简短 CLI 模块所需的输入布局函数。测试构造模拟已声明布局的最小 H5 fixture，因此运行测试套件无需下载任何公开数据集。

### 任务 1：包骨架与 schema 契约测试

**文件：**
- 新建：`requirements.txt`
- 新建：`semg_h5/__init__.py`
- 新建：`semg_h5/canonical.py`
- 新建：`tests/conftest.py`
- 新建：`tests/test_canonical.py`

- [ ] **步骤 1：编写失败的 canonical 记录测试**

```python
# tests/test_canonical.py
import numpy as np
import pytest

from semg_h5.canonical import CanonicalRecord, validate_record


def test_validate_record_accepts_a_16_channel_monotonic_stream():
    record = CanonicalRecord(
        dataset_id="emg2pose", task_id="pose", subject_id="emg2pose:user_1",
        session_id="emg2pose:session_1", side="left", pair_id="",
        official_split="train", source_file="source.h5", emg=np.zeros((4, 16)),
        time_s=np.array([0.0, 0.0005, 0.001, 0.0015]), labels={}, source_meta={},
        preprocessing={"converter_transformations": []},
    )
    validate_record(record)


def test_validate_record_rejects_a_non_16_channel_stream():
    record = CanonicalRecord.minimal(emg=np.zeros((4, 15)), time_s=np.arange(4) / 2000)
    with pytest.raises(ValueError, match="16 channels"):
        validate_record(record)
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_canonical.py -v`

预期：收集阶段失败，因为 `semg_h5.canonical` 不存在。

- [ ] **步骤 3：创建最小包与校验实现**

```python
# semg_h5/canonical.py
from dataclasses import dataclass
from typing import Any
import numpy as np

SCHEMA_VERSION = "1.0"
SAMPLE_RATE_HZ = 2000

@dataclass
class CanonicalRecord:
    dataset_id: str
    task_id: str
    subject_id: str
    session_id: str
    side: str
    pair_id: str
    official_split: str
    source_file: str
    emg: np.ndarray
    time_s: np.ndarray
    labels: dict[str, Any]
    source_meta: dict[str, Any]
    preprocessing: dict[str, Any]

    @classmethod
    def minimal(cls, emg: np.ndarray, time_s: np.ndarray) -> "CanonicalRecord":
        return cls("unknown", "unknown", "unknown", "unknown", "unknown", "", "unknown",
                   "", emg, time_s, {}, {}, {"converter_transformations": []})

def validate_record(record: CanonicalRecord) -> None:
    if record.emg.ndim != 2 or record.emg.shape[1] != 16:
        raise ValueError("EMG must have shape [T, 16] with exactly 16 channels")
    if record.time_s.ndim != 1 or len(record.time_s) != len(record.emg):
        raise ValueError("time_s must be a length-T vector matching EMG")
    if not np.isfinite(record.emg).all() or not np.isfinite(record.time_s).all():
        raise ValueError("EMG and time_s must be finite")
    if len(record.time_s) > 1 and np.any(np.diff(record.time_s) < 0):
        raise ValueError("time_s must be monotonic")
```

```text
# requirements.txt
h5py>=3.8
numpy>=1.24
pandas>=1.5
pytest>=7.0
```

- [ ] **步骤 4：运行 canonical 测试，确认 GREEN**

运行：`pytest tests/test_canonical.py -v`

预期：2 passed。

- [ ] **步骤 5：提交第一个经过测试的契约**

```bash
git add requirements.txt semg_h5/__init__.py semg_h5/canonical.py tests/test_canonical.py
git commit -m "feat: define canonical sEMG record contract"
```

### 任务 2：canonical H5 写入器与 reader 校验

**文件：**
- 修改：`semg_h5/canonical.py`
- 修改：`tests/test_canonical.py`

- [ ] **步骤 1：编写失败的写入/读取往返测试**

```python
def test_write_record_creates_required_groups_and_relative_clock(tmp_path):
    from semg_h5.canonical import read_canonical_header, write_record
    record = CanonicalRecord.minimal(
        emg=np.arange(64, dtype=np.float64).reshape(4, 16),
        time_s=np.array([10.0, 10.0005, 10.001, 10.0015]),
    )
    output = tmp_path / "record.h5"
    write_record(output, record)
    header = read_canonical_header(output)
    assert header["schema_version"] == "1.0"
    assert header["emg_shape"] == (4, 16)
    assert np.allclose(header["time_s"], [0.0, 0.0005, 0.001, 0.0015])
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_canonical.py::test_write_record_creates_required_groups_and_relative_clock -v`

预期：FAIL，因为 `write_record` 和 `read_canonical_header` 尚未定义。

- [ ] **步骤 3：实现不带任何生理变换的写入器**

实现 `write_record(path, record, overwrite=False)`，行为严格如下：

```python
with h5py.File(path, "w") as h5:
    h5.attrs.update({
        "schema_version": SCHEMA_VERSION,
        "dataset_id": record.dataset_id,
        "task_id": record.task_id,
        "subject_id": record.subject_id,
        "session_id": record.session_id,
        "side": record.side,
        "pair_id": record.pair_id,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "official_split": record.official_split,
        "source_file": record.source_file,
        "preprocessing_json": json.dumps(record.preprocessing, sort_keys=True),
    })
    signal = h5.create_group("signal")
    signal.create_dataset("emg", data=record.emg.astype(np.float32), compression="gzip")
    signal.create_dataset("time_s", data=record.time_s.astype(np.float64) - record.time_s[0])
```

把 `source_json` 写为 UTF-8 标量 dataset，写入 `/meta/quality/sample_count`、`/meta/quality/duration_s`、`/meta/quality/nan_count` 和 `/meta/quality/channel_rms`，然后调用标签写入的分发函数。除 `overwrite=True` 外，拒绝已存在的路径。

- [ ] **步骤 4：运行写入器测试，确认 GREEN**

运行：`pytest tests/test_canonical.py -v`

预期：3 passed。

- [ ] **步骤 5：提交 H5 写入器**

```bash
git add semg_h5/canonical.py tests/test_canonical.py
git commit -m "feat: write validated canonical sEMG H5 records"
```

### 任务 3：标签序列化、manifest 与目录树校验器

**文件：**
- 新建：`semg_h5/manifest.py`
- 新建：`semg_h5/validate.py`
- 新建：`tests/test_manifest.py`
- 修改：`semg_h5/canonical.py`

- [ ] **步骤 1：编写失败的 manifest 与任务标签测试**

```python
def test_manifest_row_exposes_task_selection_and_pairing(tmp_path):
    from semg_h5.manifest import append_manifest_row, read_manifest
    row = {"path": "typing/user_1/session_1_left.h5", "dataset_id": "emg2qwerty",
           "task_id": "typing", "subject_id": "emg2qwerty:user_1",
           "session_id": "emg2qwerty:session_1", "side": "left", "pair_id": "pair_1",
           "official_split": "train", "sample_count": 4, "source_sha256": "abc"}
    append_manifest_row(tmp_path / "manifest.csv", row)
    assert read_manifest(tmp_path / "manifest.csv").to_dict("records") == [row]
```

```python
def test_typing_labels_round_trip_as_structured_tables(tmp_path):
    from semg_h5.canonical import CanonicalRecord, write_record
    record = CanonicalRecord.minimal(np.zeros((4, 16)), np.arange(4) / 2000)
    record.task_id = "typing"
    record.labels = {"keystrokes": [{"key": "a", "press_s": 0.0005, "release_s": 0.001}],
                     "prompts": [{"text": "a", "start_s": 0.0, "end_s": 0.0015}]}
    write_record(tmp_path / "typing.h5", record)
    with h5py.File(tmp_path / "typing.h5") as h5:
        assert h5["labels/keystrokes"][0]["key"].decode() == "a"
        assert h5["labels/prompts"][0]["text"].decode() == "a"
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_manifest.py tests/test_canonical.py::test_typing_labels_round_trip_as_structured_tables -v`

预期：FAIL，因为 manifest 与标签序列化尚不存在。

- [ ] **步骤 3：实现稳定的 manifest 与标签写入器**

实现 `append_manifest_row`：读取已有 CSV，替换 `path` 相同的既有行，按 `path` 排序，并通过 `manifest.csv.tmp` 原子写入。manifest 字段顺序固定为：

```python
MANIFEST_COLUMNS = ["path", "dataset_id", "task_id", "subject_id", "session_id", "side",
                    "pair_id", "official_split", "sample_count", "duration_s",
                    "source_file", "source_sha256"]
```

用 UTF-8 变长字符串与具名数值字段实现 H5 标签表 dataset。仅支持以下标签映射：

```text
typing:             keystrokes(key, press_s, release_s), prompts(text, start_s, end_s)
discrete_gesture:   events(name, time_s), stage(可选 UTF-8 标量)
handwriting:        prompts(text, start_s, end_s)
pose:               joint_angles, no_ik_failure(可选), stage(可选 UTF-8 标量)
wrist_angle:        labels/targets/<源字段> 下的目标数组, stage(可选 UTF-8 标量)
```

实现 `python -m semg_h5.validate --input PATH`：递归校验所有 `*.h5`，检查 manifest 中的路径，并在退出码非零时列出全部无效路径。

- [ ] **步骤 4：运行测试，确认 GREEN**

运行：`pytest tests/test_manifest.py tests/test_canonical.py -v`

预期：所有测试通过。

- [ ] **步骤 5：提交 manifest 与标签支持**

```bash
git add semg_h5 tests/test_manifest.py tests/test_canonical.py
git commit -m "feat: add task labels manifest and canonical validator"
```

### 任务 4：emg2pose source reader 与转换器

**文件：**
- 新建：`semg_h5/source_readers.py`
- 新建：`semg_h5/convert_emg2pose.py`
- 新建：`tests/test_convert_emg2pose.py`

- [ ] **步骤 1：编写失败的 emg2pose 转换测试**

```python
def test_convert_emg2pose_preserves_pose_labels_and_source_split(tmp_path, emg2pose_h5):
    from semg_h5.convert_emg2pose import convert_file
    manifest_row = convert_file(emg2pose_h5, tmp_path / "out", official_split="val")
    with h5py.File(tmp_path / "out" / manifest_row["path"]) as h5:
        assert h5.attrs["dataset_id"] == "emg2pose"
        assert h5.attrs["task_id"] == "pose"
        assert h5.attrs["official_split"] == "val"
        assert h5["signal/emg"].shape == (4, 16)
        assert h5["labels/joint_angles"].shape == (4, 3)
        assert h5["labels/no_ik_failure"].shape == (4,)
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_convert_emg2pose.py -v`

预期：FAIL，因为该模块不存在。

- [ ] **步骤 3：实现已记录的 emg2pose 抽取逻辑**

把 group `emg2pose/timeseries` 读作含 `emg`、`joint_angles` 和 `time` 字段的复合 dataset。若源 group/属性中存在元数据则读取，否则从源相对路径派生带命名空间的 ID。每个源 H5 恰好产出一条 canonical 记录。保留可选的源字段 `no_ik_failure` 与 `stage`；能取到源元数据时据此设置 `side`。CLI 必须枚举输入 `*.h5`，在提供元数据 CSV 时关联出 official split，写入每条记录，追加一行 manifest，并支持 `--split` 过滤。

- [ ] **步骤 4：运行 emg2pose 测试，确认 GREEN**

运行：`pytest tests/test_convert_emg2pose.py -v`

预期：1 passed。

- [ ] **步骤 5：提交 emg2pose 转换器**

```bash
git add semg_h5/source_readers.py semg_h5/convert_emg2pose.py tests/test_convert_emg2pose.py
git commit -m "feat: convert emg2pose into canonical H5"
```

### 任务 5：emg2qwerty 双侧转换器

**文件：**
- 新建：`semg_h5/convert_emg2qwerty.py`
- 新建：`tests/test_convert_emg2qwerty.py`
- 修改：`semg_h5/source_readers.py`

- [ ] **步骤 1：编写失败的双侧配对测试**

```python
def test_convert_qwerty_emits_one_linked_record_per_wrist(tmp_path, qwerty_h5):
    from semg_h5.convert_emg2qwerty import convert_file
    rows = convert_file(qwerty_h5, tmp_path / "out", official_split="train")
    assert len(rows) == 2
    assert {row["side"] for row in rows} == {"left", "right"}
    assert rows[0]["pair_id"] == rows[1]["pair_id"]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5["signal/emg"].shape == (4, 16)
            assert h5["labels/keystrokes"].shape[0] == 1
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_convert_emg2qwerty.py -v`

预期：FAIL，因为 qwerty 模块不存在。

- [ ] **步骤 3：实现已记录的 qwerty 抽取逻辑**

读取 `emg2qwerty/timeseries` 的复合字段 `emg_left`、`emg_right` 和 `time`。从源 group 的元数据 JSON/属性中读取 keylogger 与 prompt 元数据。把 keylogger 的绝对时间戳和 prompt 时间转换到与 `time_s` 相同的记录相对时基。写输出前校验两侧均为 `[T,16]`、长度一致且时间向量对齐。给两条记录相同的确定性标识 `emg2qwerty:<source-session>` 与 `pair_id` `emg2qwerty:<source-session>`。把标签写进两个输出文件，使任一配对的打字记录都能自描述。

- [ ] **步骤 4：运行 qwerty 测试，确认 GREEN**

运行：`pytest tests/test_convert_emg2qwerty.py -v`

预期：1 passed。

- [ ] **步骤 5：提交 qwerty 转换器**

```bash
git add semg_h5/convert_emg2qwerty.py semg_h5/source_readers.py tests/test_convert_emg2qwerty.py
git commit -m "feat: convert bilateral emg2qwerty sessions"
```

### 任务 6：generic-neuromotor-interface source reader 与三个任务 CLI

**文件：**
- 新建：`semg_h5/convert_gni_discrete_gestures.py`
- 新建：`semg_h5/convert_gni_handwriting.py`
- 新建：`semg_h5/convert_gni_wrist.py`
- 新建：`tests/test_convert_gni.py`
- 修改：`semg_h5/source_readers.py`

- [ ] **步骤 1：为每个 GNI 任务编写参数化的失败测试**

```python
@pytest.mark.parametrize(
    ("converter_module", "task_id", "label_path"),
    [
        ("convert_gni_discrete_gestures", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting", "handwriting", "labels/prompts"),
        ("convert_gni_wrist", "wrist_angle", "labels/targets/flexion_extension"),
    ],
)
def test_convert_gni_task_preserves_16ch_stream_and_its_labels(
    tmp_path, gni_h5_by_task, converter_module, task_id, label_path
):
    module = importlib.import_module(f"semg_h5.{converter_module}")
    row = module.convert_file(gni_h5_by_task[task_id], tmp_path / "out", official_split="test")
    with h5py.File(tmp_path / "out" / row["path"]) as h5:
        assert h5.attrs["task_id"] == task_id
        assert h5["signal/emg"].shape == (4, 16)
        assert label_path in h5
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_convert_gni.py -v`

预期：FAIL，因为尚无 GNI 转换器模块。

- [ ] **步骤 3：实现共享的 GNI reader**

把源 `data` 读作复合 dataset，抽取 `time` 及其唯一的 16 通道 EMG 字段。从 `data.attrs["task"]` 读取任务名。若存在 `prompts` pandas HDF 表则读取，`stages` 表同理。标签分发如下：

```text
discrete_gestures: prompts 的 name/time 列 -> labels/events(name, time_s)
handwriting:       prompts 的 prompt/start/end 列 -> labels/prompts(text, start_s, end_s)
wrist:             除 time/emg 外的所有数值数据字段 -> labels/targets/<field>
```

对于 wrist fixture 与输出契约，把源字段名 `flexion_extension` 规范化为 `flexion_extension`；其余源数值目标字段一律在其安全的 ASCII 字段名下保留。要求目标数组长度均为 `T`。当存在唯一匹配的 stage 时，把 `stage` 写为 UTF-8 标量，否则把完整的源 stage 行保留在 `source_json` 中。

- [ ] **步骤 4：实现三个薄 CLI**

每个模块都必须以自己固定的任务字符串调用共享的 `run_gni_conversion(task_name, input_root, output_root, split, overwrite)`。通用 runner 必须拒绝 `data.attrs["task"]` 与被调用任务不一致的源 H5。`dataset_id` 分别写为 `gni_discrete_gestures`、`gni_handwriting` 和 `gni_wrist`；除源数据中存在明确的 side 外，side 一律写 `dominant`。

- [ ] **步骤 5：运行 GNI 测试，确认 GREEN**

运行：`pytest tests/test_convert_gni.py -v`

预期：3 passed。

- [ ] **步骤 6：提交全部三个 GNI 转换器**

```bash
git add semg_h5/convert_gni_discrete_gestures.py semg_h5/convert_gni_handwriting.py semg_h5/convert_gni_wrist.py semg_h5/source_readers.py tests/test_convert_gni.py
git commit -m "feat: convert generic neuromotor interface tasks"
```

### 任务 7：CLI 集成测试与用户文档

**文件：**
- 修改：`tests/test_canonical.py`
- 新建：`tests/test_cli_integration.py`
- 新建：`README.md`

- [ ] **步骤 1：编写失败的端到端 CLI 测试**

```python
def test_converter_cli_writes_a_valid_tree_and_manifest(tmp_path, qwerty_h5):
    result = subprocess.run(
        [sys.executable, "-m", "semg_h5.convert_emg2qwerty", "--input-root", str(qwerty_h5.parent),
         "--output-root", str(tmp_path / "out")], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    validation = subprocess.run(
        [sys.executable, "-m", "semg_h5.validate", "--input", str(tmp_path / "out")],
        text=True, capture_output=True, check=False
    )
    assert validation.returncode == 0, validation.stderr
    assert len(pd.read_csv(tmp_path / "out" / "manifest.csv")) == 2
```

- [ ] **步骤 2：运行测试，确认 RED**

运行：`pytest tests/test_cli_integration.py -v`

预期：FAIL，直到所有转换器都暴露各自的 CLI parser 并写出 manifest。

- [ ] **步骤 3：实现统一的 argparse 入口与 README**

在文档中记录以下确切命令：

```bash
python -m semg_h5.convert_emg2pose --input-root RAW_EMG2POSE --output-root CANONICAL_META
python -m semg_h5.convert_emg2qwerty --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python -m semg_h5.convert_gni_discrete_gestures --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python -m semg_h5.convert_gni_handwriting --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python -m semg_h5.convert_gni_wrist --input-root RAW_GNI_WRIST --output-root CANONICAL_META
python -m semg_h5.validate --input CANONICAL_META
```

README 必须包含 canonical 目录树布局、manifest 列、qwerty 的 `pair_id` 重建示例、不重新滤波/不重新缩放（no-refilter/no-rescale）策略，以及 `--split` 和 `--overwrite` 的语义。

- [ ] **步骤 4：运行集成测试与完整测试套件，确认 GREEN**

运行：`pytest -q`

预期：所有测试通过，无失败项。

- [ ] **步骤 5：提交文档与集成测试覆盖**

```bash
git add README.md tests/test_cli_integration.py
git commit -m "docs: document canonical Meta sEMG conversion workflow"
```

### 任务 8：最终校验与验收审计

**文件：**
- 校验：上述所有新建文件

- [ ] **步骤 1：重新运行一次完整校验**

运行：`pytest -q`

预期：退出码 0 且所有测试通过。

- [ ] **步骤 2：检查各转换器命令的帮助信息**

运行：

```bash
python -m semg_h5.convert_emg2pose --help
python -m semg_h5.convert_emg2qwerty --help
python -m semg_h5.convert_gni_discrete_gestures --help
python -m semg_h5.convert_gni_handwriting --help
python -m semg_h5.convert_gni_wrist --help
python -m semg_h5.validate --help
```

预期：每条命令都退出码 0，并在适用时暴露 `--input-root`、`--output-root`、`--split` 和 `--overwrite`。

- [ ] **步骤 3：审计验收标准**

对照已批准的设计逐项核验：

```text
[ ] 五个转换器模块全部存在
[ ] 所有输出的记录都含有 /signal/emg [T,16] 与 /signal/time_s [T]
[ ] qwerty 输出成对的 left/right 行
[ ] 每个任务都保留了各自必需的标签
[ ] manifest 含有 task、split、各类 ID、side、pair_id 和溯源哈希
[ ] 校验器能捕获格式错误的 canonical H5 与 manifest 不一致
[ ] 完整 pytest 测试套件通过
```

- [ ] **步骤 4：如有需要，提交仅涉及审计的最终文档改动**

```bash
git status --short
```

预期：只剩有意未纳入版本控制的研究产物；没有任何生成的测试 H5 文件或输出 manifest 被跟踪。

## 计划自检

- 规格覆盖：任务 1–3 落实共享契约、标签、manifest 与校验器；任务 4–6 覆盖每个已批准的源数据集；任务 7 记录并实际演练公开接口；任务 8 与每一条验收标准一一对应。
- 占位符扫描：没有遗留不完整或延后实现的行为。
- 类型一致性：每个转换器都产出 `CanonicalRecord`；所有 canonical 写入都经由 `write_record`；manifest 行使用固定的 `MANIFEST_COLUMNS` 列表。
