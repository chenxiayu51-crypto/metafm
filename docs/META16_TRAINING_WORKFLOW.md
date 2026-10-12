# Meta16 完整训练流程

## 语料与切分协议

规范数据根目录为 `$CanonicalRoot`（= `$Repo\emg_data_canonical`，`$Repo` 为本仓库根目录）。2026-09-21 的完整审计结果为 28,609 条记录、601 个 subject、14,893 个 session，共 1,396.474 小时。数据、manifest、checkpoint、预测结果和 `$RunsRoot` 运行产物不得提交到 Git。

| 任务 ID | 记录数 | 主要目标 |
| --- | ---: | --- |
| pose | 25,253 | 关节角回归 |
| typing | 2,270 | 文本解码 |
| discrete_gesture | 100 | 手势分类/事件识别 |
| handwriting | 804 | 书写轨迹或事件识别 |
| wrist_angle | 182 | 腕关节角回归 |

`Official` 保留数据发布方给出的 `official_split`；同一 subject 可能按发布协议跨越多个 split。`StrictSubject` 复制 manifest，将官方分配改为 `unknown`，再按 dataset、subject 和 pair 分组进行确定性切分。同一张结果表只能使用一种协议，并必须在表中明确标注。

从 `02_PrepareSplits.ps1` 之后的每个阶段都必须明确选择 split：使用 `-SplitsPath`，或者同时使用 `-Protocol` 与 `-SplitSeed`，两种方式不可混用。以下示例使用协议与种子。输出目录包含切分身份，已有目录默认拒绝复用，只有显式传入 `-Force` 才允许继续。完整命令和失败恢复方式见[转换完成后的详细执行指南](POST_CONVERSION_EXECUTION_GUIDE.md)。

## 阶段门槛

### 1. 环境预检

检查 Python 依赖、训练入口、规范数据根和 `manifest.csv`。

```powershell
powershell -File archive\legacy-powershell-workflow\00_Preflight.ps1
```

通过标准：命令退出码为 0，并打印 PyTorch 版本和 CUDA 可用性。若 CUDA 为 `False`，只能执行 CPU 烟雾测试，不能启动默认的 `a100_ddp` 正式训练。

### 2. 数据审计

检查所有路径、允许的 task/split 取值、语料总量，以及抽样 H5 中形状为 `[time, 16]` 的 `signal/emg`。

```powershell
powershell -File archive\legacy-powershell-workflow\01_AuditCanonical.ps1 -H5Samples 20
```

> 注意：`01_AuditCanonical.ps1` 依赖的 `audit_canonical.py` 与 `semg_fm.data.audit` 只存在于旧架构，本仓库未保留，该阶段当前不可运行。替代做法：直接用 pandas 检查 `$CanonicalRoot\manifest.csv` 的列完整性与路径存在性，并抽签若干 H5 检查 `signal/emg` 形状为 `[time, 16]`。

通过标准：缺失文件和 schema 错误均为 0。输出位于 `$RunsRoot\audits\audit-*.json` 及对应 CSV。

### 3. 生成切分

为两种协议生成带种子的 split 文件。使用 `-Force` 前应先执行 `-DryRun`；任何协议都不会覆盖 `manifest.csv`。

```powershell
powershell -File archive\legacy-powershell-workflow\02_PrepareSplits.ps1 -Protocol Official -Seed 20260916
powershell -File archive\legacy-powershell-workflow\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed 20260916
```

通过标准：打印每个 task 的 train/val/test 数量；StrictSubject 还必须打印 `Strict-subject leakage check: PASS`。输出为 `splits.official.seed-20260916.csv` 和 `splits.strict-subject.seed-20260916.csv`。

### 4. 五任务 CPU 烟雾测试

正式 GPU 实验前，对全部五个任务运行真实数据 fast-dev 流程。

```powershell
powershell -File archive\legacy-powershell-workflow\03_SmokeTest.ps1 -Task All -Protocol Official -SplitSeed 20260916
```

通过标准：五个 Lightning CPU 运行全部完成，并使用 `data.num_workers=0`。输出位于 `runs\smoke\official-seed-20260916\<task>\seed-0`。

### 5. 比较基线

使用 `supervised_tcn`、`bendr`、`cbramod` 或 `aemg` 建立对比基线。

```powershell
powershell -File archive\legacy-powershell-workflow\04_TrainBaseline.ps1 -Model bendr -Stage finetune -Task pose -LabelBudget 1.0 -Protocol Official -SplitSeed 20260916
```

### 6. CIRCE-Myo 预训练

先运行有界 `Pilot`，检查 loss、resolved config 和 manifest 元数据，再在 CUDA 主机启动 `Full`。本次代码交付不会自动运行 100,000 step 的正式训练。

```powershell
powershell -File archive\legacy-powershell-workflow\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1 -Protocol Official -SplitSeed 20260916
powershell -File archive\legacy-powershell-workflow\05_Pretrain.ps1 -Mode Full -MaxSteps 100000 -Devices 1 -Protocol Official -SplitSeed 20260916
```

### 7. 下游微调与独立评估

对每个任务使用 `0.01`、`0.05`、`0.10` 和 `1.0` 标签预算，并分别运行 `probe`、`partial`、`full`。评估时只使用本仓库产生且来源明确的 checkpoint。

```powershell
$PretrainCkpt = '$RunsRoot\pretrain\official-seed-20260916\full\steps-100000\devices-1\accum-1\batch-8\seed-0\checkpoints\last.ckpt'
powershell -File archive\legacy-powershell-workflow\06_Finetune.ps1 -Task pose -FinetuneMode probe -LabelBudget 0.1 -PretrainedCheckpoint $PretrainCkpt -Protocol Official -SplitSeed 20260916
$FinetuneCandidates = @(Get-ChildItem '$RunsRoot\finetune\official-seed-20260916\pose\probe\budget-0.1\seed-0\checkpoint-*\checkpoints\last.ckpt')
if ($FinetuneCandidates.Count -ne 1) { throw "应当恰好找到一个 checkpoint，实际为 $($FinetuneCandidates.Count)" }
$FinetuneCkpt = $FinetuneCandidates[0].FullName
powershell -File archive\legacy-powershell-workflow\07_Evaluate.ps1 -Task pose -CheckpointPath $FinetuneCkpt -Protocol Official -SplitSeed 20260916
```

评估输出包含 resolved config、manifest digest、环境元数据、整体指标、per-subject 指标以及可选预测结果。

### 8. 鲁棒性评估

分别评估 clean、channel rotation、channel dropout、gain drift、additive noise 和 temporal loss 条件。

```powershell
powershell -File archive\legacy-powershell-workflow\08_Robustness.ps1 -Task pose -Condition All -CheckpointPath $FinetuneCkpt -Protocol Official -SplitSeed 20260916
```

每种条件使用独立输出目录。本次代码交付不会自动运行完整的多 seed 鲁棒性网格。

使用 `RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -Protocol Official -SplitSeed 20260916 -DryRun` 可以检查安全的阶段顺序。管道只运行 `-Stages` 明确列出的阶段，不会自动加入昂贵训练阶段。
