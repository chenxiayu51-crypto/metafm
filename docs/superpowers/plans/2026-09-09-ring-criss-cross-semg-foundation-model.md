# Ring Criss-Cross sEMG 基座模型实施计划

> **面向 agentic workers：** 必需子技能（REQUIRED SUB-SKILL）：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框语法（`- [ ]`）跟踪进度。

**目标：** 构建一套 PyTorch Lightning 与 Hydra 研究栈，在五个 canonical Meta 16 通道 H5 语料上预训练 CIRCE-Myo，评测五个下游任务，并运行四种公平对比方法。

**架构：** 保持五个独立转换器不变。新增 `semg_fm` 包，包含惰性 canonical-H5 数据层、Ring Criss-Cross 骨干、预训练目标函数、对比骨干、下游头、指标，以及轻薄的 Hydra 入口。每项行为都先由一个失败的 pytest 规定，再以最小实现完成，验证通过后才进入下一次提交。

**技术栈：** Python 3.10+、PyTorch 2.2+、Lightning 2.2+、Hydra 1.3+、h5py、NumPy、pandas、torchmetrics、pytest。

---

## 文件地图

- `semg_fm/data`：manifest 校验、分组划分、惰性 H5 窗口、标签、数据增强、均衡采样、Lightning DataModule。
- `semg_fm/models`：共享 patch 编码器、环形注意力、CIRCE-Myo，以及四个对比编码器。
- `semg_fm/objectives`：掩码、连续教师损失、环形等变性、运动学损失与 VQ 诊断。
- `semg_fm/lightning`：预训练、下游与基线 LightningModule。
- `semg_fm/metrics`：回归、分类/事件、编辑距离与鲁棒性聚合。
- `configs`：面向数据、模型、任务、trainer 与鲁棒性设置的 Hydra 组合。
- `pretrain.py`、`finetune.py`、`evaluate.py`、`run_baseline.py`：公开命令。
- `tests`：合成的 canonical H5 fixture、单元测试与 CPU 快速开发集成测试。
- 仓库根目录的 Markdown 文件：安装、数据契约、预训练、下游评测、基线、实验协议与 A100 运维。

### 任务 1：训练依赖与 Hydra 组合

**文件：**
- 新建：`requirements-train.txt`
- 新建：`semg_fm/__init__.py`
- 新建：`semg_fm/config.py`
- 新建：`configs/pretrain.yaml`
- 新建：`configs/finetune.yaml`
- 新建：`configs/evaluate.yaml`
- 新建：`configs/baseline.yaml`
- 新建：`configs/data/canonical_meta.yaml`
- 新建：`configs/model/ring_criss_cross.yaml`
- 新建：`configs/trainer/cpu_smoke.yaml`
- 新建：`configs/trainer/a100_ddp.yaml`
- 测试：`tests/test_hydra_config.py`

- [ ] **步骤 1：编写失败的配置测试**

```python
from hydra import compose, initialize_config_dir
from pathlib import Path


CONFIG_DIR = str((Path(__file__).parents[1] / "configs").resolve())


def test_pretrain_config_composes_with_four_devices():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer.devices=4"])
    assert cfg.trainer.devices == 4
    assert cfg.model.name == "ring_criss_cross"
    assert cfg.data.channels == 16


def test_cpu_smoke_profile_disables_distributed_training():
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="pretrain", overrides=["trainer=cpu_smoke"])
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.trainer.devices == 1
    assert cfg.trainer.strategy == "auto"
```

- [ ] **步骤 2：运行测试并确认因配置缺失而失败**

运行：`python -m pytest tests/test_hydra_config.py -q`  
预期：FAIL，因为 `configs/pretrain.yaml` 与 `semg_fm.config` 不存在。

- [ ] **步骤 3：实现依赖与配置契约**

`requirements-train.txt` 必须锁定兼容的主/次版本下限：

```text
-r requirements.txt
torch>=2.2
lightning>=2.2
hydra-core>=1.3,<1.4
omegaconf>=2.3,<2.4
torchmetrics>=1.3
pytest>=7.0
```

`semg_fm/config.py` 定义了一个供各入口使用的小型校验器：

```python
def validate_devices(devices: int | str) -> None:
    if devices == "auto":
        return
    if not isinstance(devices, int) or devices < 1 or devices > 8:
        raise ValueError("trainer.devices must be 'auto' or an integer from 1 to 8")
```

预训练默认值必须组合 `data=canonical_meta`、`model=ring_criss_cross` 与 `trainer=a100_ddp`；CPU smoke 使用 `accelerator: cpu`、`devices: 1`、`strategy: auto`、`precision: 32-true` 与 `fast_dev_run: true`。A100 使用 `accelerator: gpu`、`devices: auto`、`strategy: auto` 与 `precision: bf16-mixed`。

- [ ] **步骤 4：运行聚焦测试**

运行：`python -m pytest tests/test_hydra_config.py -q`  
预期：PASS。

- [ ] **步骤 5：提交配置基础**

```powershell
git add requirements-train.txt semg_fm/__init__.py semg_fm/config.py configs tests/test_hydra_config.py
git commit -m "build: add lightning hydra training configuration"
```

### 任务 2：合成 canonical H5 fixture 与 manifest 校验

**文件：**
- 修改：`tests/conftest.py`
- 新建：`semg_fm/data/__init__.py`
- 新建：`semg_fm/data/manifest.py`
- 测试：`tests/test_fm_manifest.py`

- [ ] **步骤 1：编写失败的 manifest 测试**

```python
import pandas as pd
import pytest
from semg_fm.data.manifest import load_manifest


def test_manifest_resolves_records_and_preserves_task_ids(canonical_meta_root):
    frame = load_manifest(canonical_meta_root / "manifest.csv", canonical_meta_root)
    assert set(frame.task_id) == {
        "pose", "typing", "discrete_gesture", "handwriting", "wrist_angle"
    }
    assert frame.resolved_path.map(lambda p: p.exists()).all()


def test_manifest_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"path": ["x.h5"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(path, tmp_path)
```

`canonical_meta_root` 为每个任务创建一个小的、取值有限的 `[T,16]` H5，并创建一个共享同一 `pair_id` 的 qwerty 左右配对。它写入转换器所使用的每种真实的任务专属标签结构。

- [ ] **步骤 2：确认测试因 data 包缺失而失败**

运行：`python -m pytest tests/test_fm_manifest.py -q`  
预期：FAIL，报 `ModuleNotFoundError: semg_fm.data`。

- [ ] **步骤 3：实现 manifest 校验**

```python
REQUIRED_COLUMNS = {
    "path", "dataset_id", "task_id", "subject_id", "session_id", "side",
    "pair_id", "official_split", "sample_count", "duration_s", "source_file",
    "source_sha256",
}


def load_manifest(path, root):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["resolved_path"] = frame["path"].map(lambda value: Path(root) / value)
    absent = [str(p) for p in frame.resolved_path if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"manifest records not found: {absent[:3]}")
    return frame
```

- [ ] **步骤 4：运行聚焦测试**

运行：`python -m pytest tests/test_fm_manifest.py -q`  
预期：PASS。

- [ ] **步骤 5：提交 fixture 与 manifest 加载器**

```powershell
git add tests/conftest.py tests/test_fm_manifest.py semg_fm/data
git commit -m "feat: validate canonical training manifests"
```

### 任务 3：无泄漏的确定性分组划分

**文件：**
- 新建：`semg_fm/data/splits.py`
- 新建：`make_splits.py`
- 测试：`tests/test_fm_splits.py`

- [ ] **步骤 1：编写失败的划分测试**

```python
from semg_fm.data.splits import assign_group_splits, assert_no_leakage


def test_qwerty_pair_never_crosses_splits(manifest_frame):
    assigned = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    pair_rows = assigned[assigned.pair_id == "pair-1"]
    assert pair_rows.split.nunique() == 1
    assert_no_leakage(assigned)


def test_assignment_is_deterministic(manifest_frame):
    a = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    b = assign_group_splits(manifest_frame, seed=17, ratios=(0.7, 0.15, 0.15))
    assert a[["path", "split"]].equals(b[["path", "split"]])
```

- [ ] **步骤 2：确认因划分实现缺失而失败**

运行：`python -m pytest tests/test_fm_splits.py -q`  
预期：FAIL，导入 `semg_fm.data.splits` 时报错。

- [ ] **步骤 3：实现分组分配与 CLI**

当 `pair_id` 非空时使用 `pair_id`；否则使用元组 `(dataset_id, subject_id)`。对 `f"{seed}:{group_key}"` 做 SHA-256 哈希，把前八个字节映射到 `[0,1)`，再按累计比例分配 train/val/test。`assert_no_leakage` 必须拒绝同一受试者分组或同一 pair ID 出现在不同 split 中。`make_splits.py` 读取 manifest，保留非空的官方 split，只为缺失行分配，校验泄漏，并写出 `split_manifest.csv`。

- [ ] **步骤 4：运行划分测试与一次 fixture CLI 检查**

运行：`python -m pytest tests/test_fm_splits.py -q`  
预期：PASS。

- [ ] **步骤 5：提交划分管理**

```powershell
git add semg_fm/data/splits.py make_splits.py tests/test_fm_splits.py
git commit -m "feat: add leakage-safe subject splits"
```

### 任务 4：惰性 H5 窗口数据集与任务标签对齐

**文件：**
- 新建：`semg_fm/data/task_labels.py`
- 新建：`semg_fm/data/window_dataset.py`
- 测试：`tests/test_window_dataset.py`
- 测试：`tests/test_task_labels.py`

- [ ] **步骤 1：编写失败的数据集与对齐测试**

```python
import torch
from semg_fm.data.window_dataset import CanonicalWindowDataset


def test_window_uses_milliseconds_and_returns_time_major_emg(canonical_meta_root):
    ds = CanonicalWindowDataset.from_manifest(
        canonical_meta_root / "manifest.csv", split="train",
        window_ms=200, stride_ms=100,
    )
    item = ds[0]
    assert item["emg"].ndim == 2
    assert item["emg"].shape[1] == 16
    assert item["emg"].shape[0] == 400


def test_pose_invalid_ik_frames_become_target_mask(pose_dataset):
    item = pose_dataset[0]
    assert item["target"].shape[0] == item["emg"].shape[0]
    assert item["target_mask"].dtype == torch.bool
    assert not item["target_mask"][3]


def test_handwriting_returns_prompt_text_without_character_times(handwriting_dataset):
    item = handwriting_dataset[0]
    assert item["target_text"] == "hello"
    assert "character_times" not in item
```

- [ ] **步骤 2：在实现数据集之前确认失败**

运行：`python -m pytest tests/test_window_dataset.py tests/test_task_labels.py -q`  
预期：FAIL，因为 `CanonicalWindowDataset` 与标签适配器尚不存在。

- [ ] **步骤 3：实现惰性访问与五个适配器**

`CanonicalWindowDataset` 只存储记录路径与整数样本边界。`__getitem__` 在上下文管理器中打开 H5 文件，只读取选中的 EMG 切片，校验 `[window_samples,16]`，并调用 `parse_target(task_id, h5, start_s, end_s, start_index, end_index)`。为 pose、wrist 目标、gesture 事件、handwriting 提示文本与 qwerty 击键/提示文本实现适配器。文本字段同时解码 bytes 与 UTF-8 字符串。空目标窗口携带显式的空目标，而不是伪造的标签。

- [ ] **步骤 4：运行聚焦的数据集测试**

运行：`python -m pytest tests/test_window_dataset.py tests/test_task_labels.py -q`  
预期：PASS。

- [ ] **步骤 5：提交 canonical 数据集层**

```powershell
git add semg_fm/data/task_labels.py semg_fm/data/window_dataset.py tests/test_window_dataset.py tests/test_task_labels.py
git commit -m "feat: load aligned canonical semg windows"
```

### 任务 5：物理数据增强、批组装、采样与 DataModule

**文件：**
- 新建：`semg_fm/data/augment.py`
- 新建：`semg_fm/data/samplers.py`
- 新建：`semg_fm/data/datamodule.py`
- 测试：`tests/test_augment.py`
- 测试：`tests/test_datamodule.py`

- [ ] **步骤 1：编写失败的物理增强测试**

```python
import torch
from semg_fm.data.augment import circular_shift, contiguous_channel_dropout


def test_circular_shift_rolls_only_channel_axis():
    x = torch.arange(2 * 5 * 16).reshape(2, 5, 16)
    shifted = circular_shift(x, 1)
    assert torch.equal(shifted[..., 0], x[..., 15])
    assert torch.equal(shifted[:, 0, :], torch.roll(x[:, 0, :], 1, -1))


def test_contiguous_dropout_wraps_across_ring_boundary():
    x = torch.ones(4, 16)
    y, missing = contiguous_channel_dropout(x, start=15, count=2)
    assert missing.tolist() == [0, 15]
    assert y[:, [0, 15]].count_nonzero() == 0
```

- [ ] **步骤 2：确认增强测试失败**

运行：`python -m pytest tests/test_augment.py tests/test_datamodule.py -q`  
预期：FAIL，因为数据增强与 DataModule 模块尚不存在。

- [ ] **步骤 3：实现增强与批处理**

用 `torch.roll` 实现通道滚动，以及模运算下的连续索引、增益漂移、加性噪声与时间块 dropout。对每个破坏性变换返回布尔观测掩码。实现任务同质的批组装，以及一个确定性的任务均衡批采样器，其 `set_epoch` 会改变顺序。`CanonicalDataModule` 依据同一份 split manifest 构建 train/validation/test 数据集，并把 Lightning 的 rank/world-size 信息传给采样器。

- [ ] **步骤 4：运行聚焦的数据管线测试**

运行：`python -m pytest tests/test_augment.py tests/test_datamodule.py -q`  
预期：PASS。

- [ ] **步骤 5：提交分布式数据管线**

```powershell
git add semg_fm/data tests/test_augment.py tests/test_datamodule.py
git commit -m "feat: add physical augmentation and task-balanced loading"
```

### 任务 6：多尺度 patch 编码器与环形注意力

**文件：**
- 新建：`semg_fm/models/__init__.py`
- 新建：`semg_fm/models/patch_encoder.py`
- 新建：`semg_fm/models/ring_attention.py`
- 测试：`tests/test_patch_encoder.py`
- 测试：`tests/test_ring_attention.py`

- [ ] **步骤 1：编写失败的形状与边界测试**

```python
import torch
from semg_fm.models.patch_encoder import MultiScalePatchEncoder
from semg_fm.models.ring_attention import signed_ring_offsets


def test_patch_encoder_preserves_sixteen_channel_axis():
    encoder = MultiScalePatchEncoder(channels=16, d_model=32, patch_samples=40)
    out = encoder(torch.randn(2, 400, 16))
    assert out.shape == (2, 16, 10, 32)


def test_ring_offsets_make_endpoints_neighbors():
    offsets = signed_ring_offsets(16)
    assert abs(int(offsets[0, 15])) == 1
    assert abs(int(offsets[15, 0])) == 1
    assert int(offsets[0, 8]) in {-8, 8}
```

- [ ] **步骤 2：确认因模型缺失而失败**

运行：`python -m pytest tests/test_patch_encoder.py tests/test_ring_attention.py -q`  
预期：FAIL，报导入错误。

- [ ] **步骤 3：实现编码器与环相对注意力**

时间分支对每个通道施加分组时间卷积。频谱分支计算 `log1p(abs(rfft(patch)))` 并投影固定数量的频点。将两个分支拼接后线性投影到 `d_model`。`RingMultiheadAttention` 计算缩放点积注意力，并在 softmax 之前加上一张以带符号环形偏移为索引的可学习偏置表。它接受电极观测掩码，且永远不会关注缺失电极。

- [ ] **步骤 4：运行聚焦的模型测试**

运行：`python -m pytest tests/test_patch_encoder.py tests/test_ring_attention.py -q`  
预期：PASS，包括存在一个缺失通道时输出仍为有限值。

- [ ] **步骤 5：提交 patch 与 ring 原语**

```powershell
git add semg_fm/models tests/test_patch_encoder.py tests/test_ring_attention.py
git commit -m "feat: encode multiscale circular semg patches"
```

### 任务 7：Ring Criss-Cross 骨干与等变性

**文件：**
- 新建：`semg_fm/models/ring_criss_cross.py`
- 新建：`configs/model/ring_criss_cross.yaml`
- 测试：`tests/test_ring_criss_cross.py`

- [ ] **步骤 1：编写失败的骨干测试**

```python
import torch
from semg_fm.models.ring_criss_cross import RingCrissCrossEncoder


def test_backbone_returns_field_and_motion_tokens():
    model = RingCrissCrossEncoder(d_model=32, depth=2, heads=4, patch_samples=40)
    outputs = model(torch.randn(2, 400, 16))
    assert outputs["z_eq"].shape == (2, 16, 10, 32)
    assert outputs["z_motion"].shape == (2, 10, 32)


def test_field_rolls_and_motion_stays_stable_under_channel_shift():
    torch.manual_seed(7)
    model = RingCrissCrossEncoder(d_model=16, depth=1, heads=4, patch_samples=40).eval()
    x = torch.randn(1, 400, 16)
    a = model(x)
    b = model(torch.roll(x, 3, dims=-1))
    assert torch.allclose(torch.roll(a["z_eq"], 3, dims=1), b["z_eq"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(a["z_motion"], b["z_motion"], atol=1e-5, rtol=1e-4)
```

- [ ] **步骤 2：确认骨干测试失败**

运行：`python -m pytest tests/test_ring_criss_cross.py -q`  
预期：FAIL，因为 `RingCrissCrossEncoder` 不存在。

- [ ] **步骤 3：实现并行的时序/环形块**

每个块只做一次归一化，为时间维度上的 `nn.MultiheadAttention` 与通道维度上的 `RingMultiheadAttention` 分别重塑张量，恢复为 `[B,16,N,D]`，并用 `softmax(gate_logits)` 权重融合残差。前馈层逐点作用。避免使用绝对通道 embedding。在通道轨道上做均值池化计算 `z_motion`，并遵守观测掩码。

- [ ] **步骤 4：运行骨干与原语测试**

运行：`python -m pytest tests/test_ring_criss_cross.py tests/test_ring_attention.py -q`  
预期：PASS。

- [ ] **步骤 5：提交 CIRCE-Myo 骨干**

```powershell
git add semg_fm/models/ring_criss_cross.py configs/model/ring_criss_cross.yaml tests/test_ring_criss_cross.py
git commit -m "feat: add ring criss-cross semg encoder"
```

### 任务 8：掩码、连续教师、等变性与运动学损失

**文件：**
- 新建：`semg_fm/objectives/__init__.py`
- 新建：`semg_fm/objectives/masking.py`
- 新建：`semg_fm/objectives/masked_latent.py`
- 新建：`semg_fm/objectives/ring_equivariance.py`
- 新建：`semg_fm/objectives/kinematic.py`
- 测试：`tests/test_pretrain_objectives.py`

- [ ] **步骤 1：编写失败的目标函数测试**

```python
import torch
from semg_fm.objectives.masked_latent import masked_latent_loss
from semg_fm.objectives.ring_equivariance import ring_equivariance_loss


def test_masked_latent_loss_ignores_visible_positions():
    student = torch.zeros(1, 2, 3, 4)
    teacher = student.clone()
    teacher[:, :, 0] = 1
    mask = torch.tensor([[[True, False, False], [True, False, False]]])
    first = masked_latent_loss(student, teacher, mask)
    teacher[:, :, 1:] = 1000
    second = masked_latent_loss(student, teacher, mask)
    assert torch.allclose(first, second)


def test_equivariance_loss_aligns_shifted_field():
    field = torch.randn(2, 16, 3, 8)
    motion = field.mean(1)
    shifted = torch.roll(field, 4, dims=1)
    loss = ring_equivariance_loss(field, motion, shifted, motion, shift=4)
    assert loss.item() < 1e-7
```

- [ ] **步骤 2：确认目标函数测试失败**

运行：`python -m pytest tests/test_pretrain_objectives.py -q`  
预期：FAIL，报导入错误。

- [ ] **步骤 3：实现掩码与损失**

`MaskGenerator` 按配置的时间跨度、环形弧段、通道丢弃、块与随机 patch 概率，返回布尔型的 `[B,16,N]` 网格。`masked_latent_loss` 仅在 mask 为真的位置组合 Smooth-L1 与余弦距离。`ring_equivariance_loss` 在比较前把被平移的场反向滚回，并直接比较池化后的 motion。`kinematic_loss` 仅在目标掩码为真的位置施加 Smooth-L1，并在某个 batch 没有运动学目标时返回可微的零。

- [ ] **步骤 4：运行目标函数测试**

运行：`python -m pytest tests/test_pretrain_objectives.py -q`  
预期：PASS。

- [ ] **步骤 5：提交预训练目标函数**

```powershell
git add semg_fm/objectives tests/test_pretrain_objectives.py
git commit -m "feat: add circe myo pretraining objectives"
```

### 任务 9：Lightning 预训练模块、EMA 状态与公开入口

**文件：**
- 新建：`semg_fm/lightning/__init__.py`
- 新建：`semg_fm/lightning/pretrain_module.py`
- 新建：`semg_fm/registry.py`
- 新建：`pretrain.py`
- 测试：`tests/test_pretrain_module.py`
- 测试：`tests/test_pretrain_smoke.py`

- [ ] **步骤 1：编写失败的 EMA 与 smoke 测试**

```python
import torch
from lightning import Trainer
from semg_fm.lightning.pretrain_module import PretrainModule


def test_ema_teacher_is_saved_and_restored(tiny_pretrain_module, tiny_datamodule, tmp_path):
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(tiny_pretrain_module, datamodule=tiny_datamodule)
    before = {k: v.clone() for k, v in tiny_pretrain_module.teacher.state_dict().items()}
    path = tmp_path / "state.ckpt"
    trainer.save_checkpoint(path)
    restored = PretrainModule.load_from_checkpoint(path)
    for key, value in before.items():
        assert torch.equal(value, restored.teacher.state_dict()[key])


def test_circe_pretraining_fast_dev_run(tiny_pretrain_module, tiny_datamodule):
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(tiny_pretrain_module, datamodule=tiny_datamodule)
    assert trainer.global_step == 1
```

- [ ] **步骤 2：确认预训练测试失败**

运行：`python -m pytest tests/test_pretrain_module.py tests/test_pretrain_smoke.py -q`  
预期：FAIL，因为 Lightning 模块不存在。

- [ ] **步骤 3：实现预训练与 Hydra 入口**

`PretrainModule` 持有 student、冻结的 teacher、掩码生成器、运动学适配器、优化器与调度器。它在 `torch.no_grad()` 下创建未掩码的 teacher 视图，创建 student 视图与平移视图，计算三项损失，并用 `sync_dist=True` 记录每项损失以及任务专属的验证值。`optimizer_step` 只在优化器更新完 student 权重之后执行 EMA。所有 teacher 参数都保存在 `state_dict` 中。`pretrain.py` 校验 devices，通过 Lightning 设置随机种子，实例化由 registry 选定的模型与 DataModule，快照解析后的配置与 manifest 摘要，并调用 `Trainer.fit(ckpt_path=cfg.resume_from)`。

- [ ] **步骤 4：运行预训练测试**

运行：`python -m pytest tests/test_pretrain_module.py tests/test_pretrain_smoke.py -q`  
预期：在 CPU 上 PASS。

- [ ] **步骤 5：提交公开预训练路径**

```powershell
git add semg_fm/lightning semg_fm/registry.py pretrain.py tests/test_pretrain_module.py tests/test_pretrain_smoke.py
git commit -m "feat: train circe myo with lightning"
```

### 任务 10：监督式 TCN 与 BENDR 风格对比方法

**文件：**
- 新建：`semg_fm/models/supervised_tcn.py`
- 新建：`semg_fm/models/bendr.py`
- 新建：`configs/model/supervised_tcn.yaml`
- 新建：`configs/model/bendr.yaml`
- 测试：`tests/test_baseline_tcn_bendr.py`

- [ ] **步骤 1：编写失败的对比测试**

```python
import torch
from semg_fm.models.supervised_tcn import SupervisedTCN
from semg_fm.models.bendr import BendrEncoder


def test_supervised_tcn_returns_temporal_features():
    y = SupervisedTCN(in_channels=16, d_model=32)(torch.randn(2, 400, 16))
    assert y.ndim == 3 and y.shape[0] == 2 and y.shape[-1] == 32


def test_bendr_returns_continuous_context_and_targets():
    out = BendrEncoder(in_channels=16, d_model=32)(torch.randn(2, 400, 16))
    assert out["context"].shape == out["target"].shape
    assert out["context"].ndim == 3
```

- [ ] **步骤 2：确认对比测试失败**

运行：`python -m pytest tests/test_baseline_tcn_bendr.py -q`  
预期：FAIL，模型模块缺失。

- [ ] **步骤 3：实现两个连续基线**

TCN 使用支持因果的残差 Conv1d 块，返回 `[B,N,D]`。BENDR 使用 Conv1d 特征编码器，后接时间 Transformer，并对外暴露未上下文化（uncontextualized）的连续目标与经掩码的上下文特征。其对比损失从同一条记录的其他位置采样负样本，并排除正样本下标。

- [ ] **步骤 4：运行聚焦的基线测试**

运行：`python -m pytest tests/test_baseline_tcn_bendr.py -q`  
预期：PASS。

- [ ] **步骤 5：提交 TCN 与 BENDR 基线**

```powershell
git add semg_fm/models/supervised_tcn.py semg_fm/models/bendr.py configs/model tests/test_baseline_tcn_bendr.py
git commit -m "feat: add supervised tcn and bendr baselines"
```

### 任务 11：CBraMod 风格与 AEMG 风格对比方法

**文件：**
- 新建：`semg_fm/models/cbramod.py`
- 新建：`semg_fm/models/aemg.py`
- 新建：`semg_fm/objectives/vector_quantization.py`
- 新建：`configs/model/cbramod.yaml`
- 新建：`configs/model/aemg.yaml`
- 测试：`tests/test_baseline_cbramod_aemg.py`

- [ ] **步骤 1：编写失败的结构与 VQ 测试**

```python
import torch
from semg_fm.models.cbramod import CBraModEncoder
from semg_fm.models.aemg import AEMGEncoder


def test_cbramod_preserves_channel_time_grid():
    out = CBraModEncoder(d_model=32, depth=1, heads=4, patch_samples=40)(torch.randn(2, 400, 16))
    assert out["tokens"].shape == (2, 16, 10, 32)
    assert out["reconstruction"].shape[:3] == (2, 16, 10)


def test_aemg_reports_codebook_health():
    out = AEMGEncoder(d_model=32, codebook_size=64, patch_samples=40)(torch.randn(2, 400, 16))
    assert out["token_ids"].dtype == torch.long
    assert 0.0 <= float(out["dead_code_fraction"]) <= 1.0
    assert float(out["perplexity"]) >= 1.0
```

- [ ] **步骤 2：确认因基线缺失而失败**

运行：`python -m pytest tests/test_baseline_cbramod_aemg.py -q`  
预期：FAIL，报导入错误。

- [ ] **步骤 3：实现两种核心机制复现**

CBraMod 使用共享的时间/FFT patch 编码器、按普通通道顺序的条件位置卷积、并行的时间注意力与全空间注意力，以及一个波形 patch 重建头。它不使用环相对偏置。AEMG 检测经收缩加权的 patch 且不丢弃低能量 patch，应用带 commitment loss 与 EMA 码本更新的最近邻 VQ，并预测被掩码的 token ID。`codebook_statistics` 计算使用计数、困惑度（perplexity），以及一个 epoch 内未被使用条目的精确占比。

- [ ] **步骤 4：运行聚焦的基线测试**

运行：`python -m pytest tests/test_baseline_cbramod_aemg.py -q`  
预期：PASS。

- [ ] **步骤 5：提交 CBraMod 与 AEMG 基线**

```powershell
git add semg_fm/models/cbramod.py semg_fm/models/aemg.py semg_fm/objectives/vector_quantization.py configs/model tests/test_baseline_cbramod_aemg.py
git commit -m "feat: add cbramod and aemg baselines"
```

### 任务 12：下游头、损失与指标

**文件：**
- 新建：`semg_fm/models/heads.py`
- 新建：`semg_fm/metrics/__init__.py`
- 新建：`semg_fm/metrics/regression.py`
- 新建：`semg_fm/metrics/events.py`
- 新建：`semg_fm/metrics/text.py`
- 新建：`semg_fm/tasks.py`
- 测试：`tests/test_downstream_tasks.py`
- 测试：`tests/test_metrics.py`

- [ ] **步骤 1：编写失败的指标测试**

```python
from semg_fm.metrics.text import character_error_rate
from semg_fm.metrics.events import event_f1


def test_character_error_rate_uses_edit_distance():
    assert character_error_rate(["cat"], ["cut"]) == 1 / 3


def test_event_f1_matches_with_declared_tolerance():
    result = event_f1(pred=[("tap", 1.04)], target=[("tap", 1.0)], tolerance_s=0.05)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
```

为 pose、wrist、gesture、handwriting CTC 与 qwerty CTC 各添加一个 head/loss 形状测试。

- [ ] **步骤 2：确认任务与指标测试失败**

运行：`python -m pytest tests/test_downstream_tasks.py tests/test_metrics.py -q`  
预期：FAIL，模块缺失。

- [ ] **步骤 3：实现任务注册表与指标**

回归头返回逐帧输出，并遵守布尔型目标掩码。Gesture 使用带显式背景类别的逐帧 logits 与确定性事件解码。文本头返回时间主序（time-major）的 CTC 对数概率，并使用随 checkpoint 一并保存的词表。实现原始单位下的 MAE/RMSE/Pearson、macro F1、balanced accuracy、基于容差的一对一事件匹配、Levenshtein CER、归一化编辑距离与精确提示文本准确率。

- [ ] **步骤 4：运行下游单元测试**

运行：`python -m pytest tests/test_downstream_tasks.py tests/test_metrics.py -q`  
预期：PASS。

- [ ] **步骤 5：提交下游契约**

```powershell
git add semg_fm/models/heads.py semg_fm/metrics semg_fm/tasks.py tests/test_downstream_tasks.py tests/test_metrics.py
git commit -m "feat: add five semg downstream task contracts"
```

### 任务 13：微调、评测与鲁棒性入口

**文件：**
- 新建：`semg_fm/lightning/downstream_module.py`
- 新建：`semg_fm/evaluation.py`
- 新建：`finetune.py`
- 新建：`evaluate.py`
- 新建：`configs/task/emg2pose.yaml`
- 新建：`configs/task/gni_wrist.yaml`
- 新建：`configs/task/gni_gesture.yaml`
- 新建：`configs/task/gni_handwriting.yaml`
- 新建：`configs/task/emg2qwerty.yaml`
- 新建：`configs/robustness/none.yaml`
- 新建：`configs/robustness/channel_rotation.yaml`
- 新建：`configs/robustness/channel_dropout.yaml`
- 新建：`configs/robustness/gain_drift.yaml`
- 新建：`configs/robustness/additive_noise.yaml`
- 测试：`tests/test_downstream_smoke.py`
- 测试：`tests/test_evaluation_outputs.py`

- [ ] **步骤 1：编写失败的五任务 smoke 与输出测试**

```python
import json
from semg_fm.evaluation import write_evaluation_outputs


def test_evaluation_writes_machine_readable_outputs(tmp_path):
    write_evaluation_outputs(
        tmp_path,
        metrics={"cer": 0.25},
        per_subject=[{"subject_id": "s1", "cer": 0.25}],
        predictions={"record_id": ["r1"], "text": ["test"]},
    )
    assert json.loads((tmp_path / "metrics.json").read_text())["cer"] == 0.25
    assert (tmp_path / "per_subject_metrics.csv").is_file()
    assert (tmp_path / "predictions.h5").is_file()
```

对全部五种任务配置与三种微调模式（linear probe、partial fine-tune 与 full fine-tune）参数化一个 Lightning `fast_dev_run` 测试。

- [ ] **步骤 2：确认下游集成测试失败**

运行：`python -m pytest tests/test_downstream_smoke.py tests/test_evaluation_outputs.py -q`  
预期：FAIL，因为公开的下游模块与写出逻辑不存在。

- [ ] **步骤 3：实现微调模式与评测输出**

`DownstreamModule` 实例化任务头，按模式冻结骨干参数，计算任务损失，累积逐记录预测，并记录经同步的验证指标。`evaluate.py` 加载 checkpoint，施加一个配置好的内存内鲁棒性变换，执行评测，并写出 `metrics.json`、`per_subject_metrics.csv`、可选的 `predictions.h5`、划分 CSV 副本、解析后的配置与环境信息。旋转取值是整数通道位移；文档把 1、2、4、8 映射到 22.5、45、90 与 180 度。

- [ ] **步骤 4：运行全部下游集成测试**

运行：`python -m pytest tests/test_downstream_smoke.py tests/test_evaluation_outputs.py -q`  
预期：在 CPU 上 PASS。

- [ ] **步骤 5：提交下游命令**

```powershell
git add semg_fm/lightning/downstream_module.py semg_fm/evaluation.py finetune.py evaluate.py configs/task configs/robustness tests/test_downstream_smoke.py tests/test_evaluation_outputs.py
git commit -m "feat: fine tune and evaluate five semg tasks"
```

### 任务 14：统一基线 Lightning 命令与 CPU smoke 套件

**文件：**
- 新建：`semg_fm/lightning/baseline_module.py`
- 新建：`run_baseline.py`
- 测试：`tests/test_baseline_smoke.py`

- [ ] **步骤 1：编写失败的参数化基线 smoke 测试**

```python
import pytest
from lightning import Trainer
from semg_fm.registry import build_baseline_module


@pytest.mark.parametrize("name", ["supervised_tcn", "bendr", "cbramod", "aemg"])
def test_baseline_completes_one_cpu_step(name, tiny_datamodule):
    module = build_baseline_module(name=name, d_model=16, depth=1)
    trainer = Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(module, datamodule=tiny_datamodule)
    assert trainer.global_step == 1
```

- [ ] **步骤 2：确认基线集成失败**

运行：`python -m pytest tests/test_baseline_smoke.py -q`  
预期：FAIL，因为 `build_baseline_module` 与 `BaselineModule` 尚不完整。

- [ ] **步骤 3：实现阶段感知的基线训练**

`run_baseline.py` 接受 `model=<name>` 与 `stage=pretrain|finetune`。监督式 TCN 会以明确的错误拒绝 `stage=pretrain`。BENDR 记录连续对比损失，CBraMod 记录掩码重建损失，AEMG 记录 token 分类、commitment loss、困惑度、使用率与死码占比。微调使用与 CIRCE-Myo 相同的 `DownstreamModule` 头和协议。

- [ ] **步骤 4：运行基线 smoke 测试**

运行：`python -m pytest tests/test_baseline_smoke.py -q`  
预期：四种方法均 PASS。

- [ ] **步骤 5：提交基线命令**

```powershell
git add semg_fm/lightning/baseline_module.py semg_fm/registry.py run_baseline.py tests/test_baseline_smoke.py
git commit -m "feat: run fair semg comparison methods"
```

### 任务 15：可复现性元数据与 checkpoint 续训

**文件：**
- 新建：`semg_fm/utils/__init__.py`
- 新建：`semg_fm/utils/reproducibility.py`
- 修改：`semg_fm/lightning/pretrain_module.py`
- 修改：`pretrain.py`
- 修改：`finetune.py`
- 修改：`evaluate.py`
- 修改：`run_baseline.py`
- 测试：`tests/test_reproducibility.py`
- 测试：`tests/test_checkpoint_resume.py`

- [ ] **步骤 1：编写失败的摘要与续训测试**

```python
from semg_fm.utils.reproducibility import sha256_file, effective_global_batch


def test_effective_batch_records_world_size_and_accumulation():
    assert effective_global_batch(per_device=8, devices=4, accumulate=2) == 64


def test_manifest_digest_changes_when_contents_change(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("a", encoding="utf-8")
    first = sha256_file(path)
    path.write_text("b", encoding="utf-8")
    assert sha256_file(path) != first
```

续训测试训练一步、保存 checkpoint、加载它，并在训练第二步之前断言 `global_step`、teacher 状态与调度器状态相等。

- [ ] **步骤 2：确认可复现性测试失败**

运行：`python -m pytest tests/test_reproducibility.py tests/test_checkpoint_resume.py -q`  
预期：FAIL，缺少工具函数或缺少 EMA/调度器状态。

- [ ] **步骤 3：实现运行快照与精确续训**

向每个输出目录写入解析后的 Hydra YAML、Git commit、manifest 与 split 摘要、Python/PyTorch/Lightning 版本、CUDA 可见性、world size、精度、随机种子与有效全局 batch。把 EMA teacher 与调度器置于 Lightning checkpoint 状态中。把 `ckpt_path` 直接传给 `Trainer.fit`，使续训保留优化器与全局步数。

- [ ] **步骤 4：运行可复现性与续训测试**

运行：`python -m pytest tests/test_reproducibility.py tests/test_checkpoint_resume.py -q`  
预期：PASS。

- [ ] **步骤 5：提交可复现性支持**

```powershell
git add semg_fm/utils semg_fm/lightning/pretrain_module.py pretrain.py finetune.py evaluate.py run_baseline.py tests/test_reproducibility.py tests/test_checkpoint_resume.py
git commit -m "feat: record and resume reproducible semg runs"
```

### 任务 16：用户文档与实验协议

**文件：**
- 修改：`README.md`
- 新建：`PRETRAINING.md`
- 新建：`DOWNSTREAM_EVALUATION.md`
- 新建：`BASELINES.md`
- 新建：`EXPERIMENT_PROTOCOL.md`
- 新建：`H5_DATA_CONTRACT.md`
- 新建：`SERVER_A100.md`
- 测试：`tests/test_documented_commands.py`

- [ ] **步骤 1：编写失败的文档命令测试**

```python
from pathlib import Path


def test_public_documents_include_all_entry_points():
    root = Path(__file__).parents[1]
    text = "\n".join((root / name).read_text(encoding="utf-8") for name in [
        "README.md", "PRETRAINING.md", "DOWNSTREAM_EVALUATION.md", "BASELINES.md", "SERVER_A100.md"
    ])
    for command in ["pretrain.py", "finetune.py", "evaluate.py", "run_baseline.py", "make_splits.py"]:
        assert command in text
```

- [ ] **步骤 2：确认文档测试失败**

运行：`python -m pytest tests/test_documented_commands.py -q`  
预期：FAIL，因为文档不存在。

- [ ] **步骤 3：撰写运维文档**

`README.md` 包含安装说明与一条五分钟的 CPU smoke 命令。`PRETRAINING.md` 说明各形状、掩码、损失与续训。`DOWNSTREAM_EVALUATION.md` 规定每条标签路径、划分规则、指标与鲁棒性命令。`BASELINES.md` 把每种实现标注为核心机制复现，并列出与原文的偏差。`EXPERIMENT_PROTOCOL.md` 固定主表、消融、随机种子、标签预算、参数量报告方式，以及不借助语言模型的 qwerty 对比。`H5_DATA_CONTRACT.md` 与加载器的实际校验逐条对应。`SERVER_A100.md` 给出 1、2、4、8 卡的命令、免 Slurm 启动、续训、NCCL 诊断，以及在保持有效 batch size 前提下的显存不足应对方案。

- [ ] **步骤 4：运行文档与 Hydra 组合测试**

运行：`python -m pytest tests/test_documented_commands.py tests/test_hydra_config.py -q`  
预期：PASS。

- [ ] **步骤 5：提交文档**

```powershell
git add README.md PRETRAINING.md DOWNSTREAM_EVALUATION.md BASELINES.md EXPERIMENT_PROTOCOL.md H5_DATA_CONTRACT.md SERVER_A100.md tests/test_documented_commands.py
git commit -m "docs: document semg foundation model experiments"
```

### 任务 17：完整验证与验收审计

**文件：**
- 修改：`.gitignore`
- 修改：`pytest.ini`（若现有临时目录设置与最终测试套件不兼容）

- [ ] **步骤 1：在修改 ignore 规则前验证转换器独立性**

运行：`python -m pytest tests/test_standalone_scripts.py -q`。如果该测试文件在后续检出中不可用，则改为运行以下精确命令：

```powershell
python semg_h5/convert_emg2pose.py --help
python semg_h5/convert_emg2qwerty.py --help
python semg_h5/convert_gni_discrete_gestures.py --help
python semg_h5/convert_gni_handwriting.py --help
python semg_h5/convert_gni_wrist.py --help
```

预期：所有独立转换器检查均 PASS，且不需要导入 `semg_fm`。

- [ ] **步骤 2：运行完整测试套件**

运行：`python -m pytest -q`  
预期：所有测试 PASS，且没有把警告提升为错误。

- [ ] **步骤 3：运行公开 CPU smoke 命令**

```powershell
python pretrain.py trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python run_baseline.py model=bendr trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python finetune.py task=emg2pose trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
python evaluate.py task=emg2pose trainer=cpu_smoke data.root=tests/fixtures/canonical_meta
```

预期：每条命令退出码为 0，并在其 Hydra 输出目录中写出解析后的配置与指标。

- [ ] **步骤 4：审计仓库状态与生成产物**

运行：`git status --short`  
预期：只出现预期的源码、配置、测试与文档改动。数据集文件、checkpoint、Hydra 输出、`.pytest_cache` 与 Python 字节码均被忽略。已有的未跟踪 `emg_data/` 与 `meta_semg_16ch_datasets.md` 保持不变，除非用户另行要求处理。

- [ ] **步骤 5：提交最终集成修复**

```powershell
git add .gitignore pytest.ini semg_fm configs tests pretrain.py finetune.py evaluate.py run_baseline.py make_splits.py requirements-train.txt README.md PRETRAINING.md DOWNSTREAM_EVALUATION.md BASELINES.md EXPERIMENT_PROTOCOL.md H5_DATA_CONTRACT.md SERVER_A100.md
git commit -m "test: verify semg foundation model research stack"
```

## 完成证据

在汇报完成之前，需采集：

- 最终的 `python -m pytest -q` 摘要；
- 四项预训练 smoke 测试的结果；
- 五项下游 smoke 测试的结果；
- Ring Criss-Cross 等变性容差的结果；
- `git status --short` 显示没有意外的已跟踪改动；
- 一条说明：如果当前机器没有多卡 CUDA 环境，NCCL 与多卡行为仍需按文档执行 A100 服务器 smoke 测试。
