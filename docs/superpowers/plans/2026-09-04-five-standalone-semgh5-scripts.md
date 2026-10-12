# 五个独立 sEMG H5 转换器实施计划

> **面向 agentic workers：** 必需子技能（REQUIRED SUB-SKILL）：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框语法（`- [ ]`）跟踪进度。

**目标：** 把包形式的转换器实现替换为 `semg_h5` 中恰好五个可直接执行的独立 Python 脚本。

**架构：** 每个转换器都自带读取其源数据布局、校验 `[T,16]` sEMG、写入 canonical H5 记录并向 `manifest.csv` upsert 所需的全部代码。公共机制刻意重复实现，因此五个脚本之间没有任何本地 import。对外的 H5 schema 与各任务专用标签保持不变。

**技术栈：** Python 3.10+、`h5py`、`numpy`、`pandas`、`pytest`，以及标准库 `argparse`、`csv`、`hashlib`、`json`、`pathlib`、`subprocess`。

---

## 文件结构

```text
semg_h5/
  convert_emg2pose.py
  convert_emg2qwerty.py
  convert_gni_discrete_gestures.py
  convert_gni_handwriting.py
  convert_gni_wrist.py
tests/
  conftest.py
  test_standalone_scripts.py
README.md
```

`semg_h5/*.py` 下剩下的每个文件都是面向用户的转换器。测试代码通过 `importlib.util.spec_from_file_location` 按脚本绝对路径加载转换器，因此测试通过即可证明不需要任何包级 import。

### 任务 1：直接执行的回归契约

**文件：**
- 新建：`tests/test_standalone_scripts.py`
- 修改：`tests/conftest.py`

- [ ] **步骤 1：编写按文件路径导入各转换器的失败测试**

```python
SCRIPT_NAMES = (
    "convert_emg2pose.py",
    "convert_emg2qwerty.py",
    "convert_gni_discrete_gestures.py",
    "convert_gni_handwriting.py",
    "convert_gni_wrist.py",
)

def load_script(name):
    path = ROOT / "semg_h5" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_semg_h5_contains_exactly_five_converter_scripts():
    assert sorted(path.name for path in (ROOT / "semg_h5").glob("*.py")) == list(SCRIPT_NAMES)

def test_each_converter_has_no_local_semg_h5_imports():
    for name in SCRIPT_NAMES:
        source = (ROOT / "semg_h5" / name).read_text(encoding="utf-8")
        assert "from ." not in source
        assert "from semg_h5" not in source
```

- [ ] **步骤 2：运行直接执行测试，确认 RED**

运行：`python -m pytest tests/test_standalone_scripts.py -v`

预期：FAIL，因为该目录下仍存在共享模块，且现有的五个转换器使用了相对 import。

- [ ] **步骤 3：使用已有的合成 H5 fixture 补充直接调用脚本的转换测试**

```python
@pytest.mark.parametrize(
    ("script", "fixture_name", "expected_task", "expected_label"),
    [
        ("convert_emg2pose.py", "emg2pose_h5", "pose", "labels/joint_angles"),
        ("convert_emg2qwerty.py", "qwerty_h5", "typing", "labels/keystrokes"),
        ("convert_gni_discrete_gestures.py", "gni_discrete_gestures_h5", "discrete_gesture", "labels/events"),
        ("convert_gni_handwriting.py", "gni_handwriting_h5", "handwriting", "labels/prompts"),
        ("convert_gni_wrist.py", "gni_wrist_h5", "wrist_angle", "labels/targets/flexion_extension"),
    ],
)
def test_standalone_converter_writes_task_contract(script, fixture_name, expected_task, expected_label, request, tmp_path):
    module = load_script(script)
    source = request.getfixturevalue(fixture_name)
    rows = module.convert_file(source, tmp_path / "out", official_split="test")
    rows = rows if isinstance(rows, list) else [rows]
    for row in rows:
        with h5py.File(tmp_path / "out" / row["path"]) as h5:
            assert h5.attrs["task_id"] == expected_task
            assert h5["signal/emg"].shape[1] == 16
            assert expected_label in h5
```

### 任务 2：独立的 emg2pose 与 qwerty 脚本

**文件：**
- 修改：`semg_h5/convert_emg2pose.py`
- 修改：`semg_h5/convert_emg2qwerty.py`

- [ ] **步骤 1：把 `convert_emg2pose.py` 中的所有相对 import 替换为本地辅助函数**

在脚本内嵌 `decode_value`、属性读取、`[T,16]`/单调时间校验、canonical 写入器、pose 标签写入器、SHA-256 计算、manifest upsert、split 查询、输出路径构造、`convert_file`、`run_conversion` 和 `main`。保留已记录的源数据要求：`emg2pose/timeseries` 复合字段 `emg`、`joint_angles`、`time`，以及可选的 `no_ik_failure` 和 `stage`。

- [ ] **步骤 2：运行 pose 独立脚本测试，确认 GREEN**

运行：`python -m pytest tests/test_standalone_scripts.py -k emg2pose -v`

预期：PASS，包含 `/signal/emg [T,16]`、`/labels/joint_angles`、可选的 `/labels/no_ik_failure`，以及一行 manifest 记录。

- [ ] **步骤 3：把 `convert_emg2qwerty.py` 中的所有相对 import 替换为本地辅助函数**

内嵌与前述相同的公共写入器/manifest 机制，并额外处理 qwerty 专有逻辑：`emg2qwerty/timeseries` 的字段 `emg_left`、`emg_right`、`time`；JSON 或结构化形式的 keystroke/prompt 标签；把标签时间戳转换为记录相对秒；以及生成恰好两行输出记录并共用同一 `pair_id`。

- [ ] **步骤 4：运行 qwerty 独立脚本测试，确认 GREEN**

运行：`python -m pytest tests/test_standalone_scripts.py -k emg2qwerty -v`

预期：PASS，左右两条记录齐备、`pair_id` 一致，且两个文件中都含 `/labels/keystrokes`。

- [ ] **步骤 5：提交两个自包含的 Meta 转换器**

```bash
git add semg_h5/convert_emg2pose.py semg_h5/convert_emg2qwerty.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: make Meta converters standalone"
```

### 任务 3：三个独立的 GNI 脚本

**文件：**
- 修改：`semg_h5/convert_gni_discrete_gestures.py`
- 修改：`semg_h5/convert_gni_handwriting.py`
- 修改：`semg_h5/convert_gni_wrist.py`

- [ ] **步骤 1：在每个 GNI 脚本中各自内嵌 GNI 公共机制**

每个脚本只能直接 import Python 标准库、`h5py`、`numpy` 和 `pandas`；读取 `data.attrs['task']`；定位 `time` 和唯一的 16 通道 EMG 字段；写出 canonical H5/manifest；并实现自己的 CLI。每个脚本都必须拒绝与其固定预期任务不符的源数据任务。

- [ ] **步骤 2：实现三组固定标签映射**

```text
convert_gni_discrete_gestures.py: data.attrs['task'] == 'discrete_gestures'; prompts 的 name/time 列 -> labels/events(name, time_s)
convert_gni_handwriting.py:       data.attrs['task'] == 'handwriting'; prompts 的 prompt/start/end 列 -> labels/prompts(text, start_s, end_s)
convert_gni_wrist.py:             data.attrs['task'] == 'wrist'; 除 time/emg 外的数值字段 -> labels/targets/<field>
```

优先尝试 Pandas HDF 表 `prompts` 和 `stages`，若不可用则回退到结构化 HDF5 dataset。仅当某一行能明确给出 stage 元数据时，才把它存为标量 `labels/stage`；完整的 stage 行仍保留在源元数据中。

- [ ] **步骤 3：运行全部 GNI 独立脚本测试，确认 GREEN**

运行：`python -m pytest tests/test_standalone_scripts.py -k gni -v`

预期：三条 PASS，各自带有不同的 task ID 与标签路径。

- [ ] **步骤 4：提交三个自包含的 GNI 转换器**

```bash
git add semg_h5/convert_gni_discrete_gestures.py semg_h5/convert_gni_handwriting.py semg_h5/convert_gni_wrist.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: make GNI converters standalone"
```

### 任务 4：移除包内支撑代码并修订文档

**文件：**
- 删除：`semg_h5/__init__.py`
- 删除：`semg_h5/canonical.py`
- 删除：`semg_h5/manifest.py`
- 删除：`semg_h5/source_readers.py`
- 删除：`semg_h5/gni_conversion.py`
- 删除：`semg_h5/validate.py`
- 修改：`README.md`
- 修改：`tests/test_cli_integration.py`

- [ ] **步骤 1：编写按绝对文件路径调用脚本的失败 CLI 测试**

```python
result = subprocess.run(
    [sys.executable, str(ROOT / "semg_h5" / "convert_emg2qwerty.py"),
     "--input-root", str(qwerty_h5.parent), "--output-root", str(output_root)],
    text=True, capture_output=True, check=False,
)
assert result.returncode == 0, result.stderr
assert len(pd.read_csv(output_root / "manifest.csv")) == 2
```

- [ ] **步骤 2：在删除之前运行 CLI 测试，确认 GREEN**

运行：`python -m pytest tests/test_cli_integration.py -v`

预期：PASS，因为独立 qwerty 脚本不依赖任何包。

- [ ] **步骤 3：只删除这六个具名支撑模块**

使用显式路径，并在删除前核对目标清单。不要删除五个转换器脚本、测试、README、依赖清单、研究报告或文档。

- [ ] **步骤 4：把 README 中的模块调用命令替换为直接路径**

```powershell
python semg_h5\convert_emg2pose.py --input-root RAW_EMG2POSE --output-root CANONICAL_META
python semg_h5\convert_emg2qwerty.py --input-root RAW_EMG2QWERTY --output-root CANONICAL_META
python semg_h5\convert_gni_discrete_gestures.py --input-root RAW_GNI_DISCRETE --output-root CANONICAL_META
python semg_h5\convert_gni_handwriting.py --input-root RAW_GNI_HANDWRITING --output-root CANONICAL_META
python semg_h5\convert_gni_wrist.py --input-root RAW_GNI_WRIST --output-root CANONICAL_META
```

移除 `python -m semg_h5.validate` 的独立文档，并说明每个转换器在写入前都会校验其源数据。

- [ ] **步骤 5：提交恰好五个脚本的目录**

```bash
git add -A semg_h5 README.md tests/test_cli_integration.py tests/test_standalone_scripts.py tests/conftest.py
git commit -m "refactor: retain five standalone sEMG converters"
```

### 任务 5：最终校验

**文件：**
- 校验：`semg_h5/*.py`、测试套件、README

- [ ] **步骤 1：运行完整测试套件**

运行：`python -m pytest -q`

预期：退出码 0。

- [ ] **步骤 2：核验文件数量精确无误且不存在包级 import**

运行：`rg --files semg_h5 -g '*.py'; rg -n 'from \.|from semg_h5' semg_h5`

预期：恰好是这五个具名文件；第二条命令无任何匹配输出。

- [ ] **步骤 3：检查所有 CLI 帮助页**

运行：`python semg_h5/convert_emg2pose.py --help`，并对另外四个脚本重复此操作。

预期：每个命令都退出码 0，并暴露 `--input-root`、`--output-root`、`--split`、`--splits-csv` 和 `--overwrite`。

- [ ] **步骤 4：确认没有生成类测试产物被纳入版本控制**

运行：`git status --short`

预期：`.pytest_tmp`、`__pycache__`、转换得到的 `.h5` 和生成的 `manifest.csv` 均未被跟踪。

## 计划自检

- 规格覆盖：任务 2 和任务 3 把每个转换器都替换为独立实现，同时保留全部任务标签；任务 4 精确删除六个共享模块；任务 1、4、5 验证直接执行能力与五文件边界。
- 占位符扫描：本计划不含任何延后或未明确的实现行为。
- 类型一致性：所有脚本都暴露 `convert_file(source_path, output_root, official_split='unknown', overwrite=False)`；只有 qwerty 返回行列表，其余四个返回单行。
