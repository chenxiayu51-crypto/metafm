# 五数据集转换完成后的完整执行指南

本指南覆盖五个数据集全部转换完成之后的数据审计、切分、预训练、下游训练、基线对比、评估和鲁棒性实验。转换脚本本身不在本指南范围内。

2026-09-21 的全量审计结果为：28,609 条记录、601 个 subject、14,893 个 session、1,396.474 小时，缺失文件为 0，抽检 200 个 H5 无 schema 错误。当前已经生成并验证两套固定切分：

```text
$CanonicalRoot\splits.official.seed-20260916.csv
$CanonicalRoot\splits.strict-subject.seed-20260916.csv
```

Official 文件保留发布协议，存在 158 个发布方定义的跨 split subject 组，pair 泄漏为 0；StrictSubject 文件的 subject 与 pair 跨 split 数量均为 0。

所有命令从仓库根目录执行。先定义变量（克隆到任意位置均可）：`$Repo = 'D:\Fmodel'`（本仓库根目录）、`$CanonicalRoot = "$Repo\emg_data_canonical"`（canonical 数据根）。数据、检查点和运行结果都在 `$Repo` 下，输出根为 `$RunsRoot`。不要使用先前的用户目录工作区。

## 0. 一次性设置与运行原则

```powershell
cd $Repo
$Protocol = 'Official'       # 或 StrictSubject；同一张结果表只能使用一种协议
$SplitSeed = 20260916
$SplitFile = "$CanonicalRoot\splits.official.seed-$SplitSeed.csv"
```

`Official` 保留数据发布方给出的 `official_split`，允许发布版本身存在同一 subject 跨 split 的情况；`StrictSubject` 将全部记录按 `dataset_id/subject_id/pair_id` 的组独立切分，适合主体泛化报告。选择 `StrictSubject` 时，将上面变量改为：

```powershell
$Protocol = 'StrictSubject'
$SplitFile = "$CanonicalRoot\splits.strict-subject.seed-$SplitSeed.csv"
```

从 `03_SmokeTest.ps1` 起的每个脚本都必须明确收到同一个切分选择：要么传入 `-SplitsPath $SplitFile`，要么同时传入 `-Protocol $Protocol -SplitSeed $SplitSeed`。推荐后一种写法；它使输出目录含有协议和种子，例如 `official-seed-20260916`。使用任意自定义 `-SplitsPath` 时，输出目录会使用 `custom-<hash>`，以防与标准协议实验混淆。

所有训练、评估和对比脚本在目标输出目录已存在时都会停止。先检查已有运行，而不是盲目重跑；只有确定要恢复或重用该目录时才添加 `-Force`。每一步均可先加 `-DryRun`：它会验证参数与输入并打印 Python/Hydra 命令，但不会训练或创建输出。

## 1. 环境预检与数据审计

先安装并验证训练依赖。该命令不改变数据。

```powershell
python -m pip install -r requirements-train.txt
powershell -File archive\legacy-powershell-workflow\00_Preflight.ps1
```

通过标准：两个命令退出码为 0；预检会显示 PyTorch 版本和 CUDA 可用性。若显示 `False`，只能运行 `cpu_smoke`，不得在该环境启动默认使用 `a100_ddp` 的正式预训练、基线、微调和评估。

执行完整审计：

```powershell
powershell -File archive\legacy-powershell-workflow\01_AuditCanonical.ps1 -H5Samples 20
```

> 注意：`01_AuditCanonical.ps1` 依赖的 `audit_canonical.py` 与 `semg_fm.data.audit` 只存在于旧架构，本仓库未保留，该阶段当前不可运行。替代做法：直接用 pandas 检查 `$CanonicalRoot\manifest.csv` 的列完整性与路径存在性，并抽检若干 H5 检查 `signal/emg` 形状为 `[time, 16]`。

通过标准：没有缺失 H5、无 schema 错误，抽样信号为 `[time, 16]`；结果位于 `$RunsRoot\audits\audit-*.json` 与 CSV。

## 2. 生成切分并冻结文件

先显示将执行的严格主体切分命令，再真正运行：

```powershell
powershell -File archive\legacy-powershell-workflow\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed $SplitSeed -DryRun
powershell -File archive\legacy-powershell-workflow\02_PrepareSplits.ps1 -Protocol StrictSubject -Seed $SplitSeed
```

发布方官方协议对应：

```powershell
powershell -File archive\legacy-powershell-workflow\02_PrepareSplits.ps1 -Protocol Official -Seed $SplitSeed
```

输出分别是 `splits.strict-subject.seed-<seed>.csv` 与 `splits.official.seed-<seed>.csv`。`StrictSubject` 必须打印 `Strict-subject leakage check: PASS`；`Official` 会保留原始分配并报告发布版本定义的跨 subject 重叠。若同名切分已存在，脚本会拒绝覆盖；仅在已审阅旧文件且要有意重建时使用 `-Force`。

切分生成后，请把具体文件名、协议、种子记录到实验日志；后续任何阶段不得改用裸 `splits.csv`。

## 3. CPU 烟雾测试

真实 GPU 工作之前，确认每个可用任务能在同一切分上完整走完加载、模型、损失和记录流程：

```powershell
powershell -File archive\legacy-powershell-workflow\03_SmokeTest.ps1 -Task All -Protocol $Protocol -SplitSeed $SplitSeed
```

如只验证已经完成转换的任务，可替换 `All`，例如：

```powershell
powershell -File archive\legacy-powershell-workflow\03_SmokeTest.ps1 -Task typing -Protocol $Protocol -SplitSeed $SplitSeed
```

通过标准：所有指定任务退出码为 0，输出在 `$RunsRoot\smoke\<protocol>-seed-<split-seed>\<task>\seed-<train-seed>`。先以 `-DryRun` 检查命令；改变 `-Seed` 会产生独立目录。只有要有意复用完全相同的目录时才使用 `-Force`。

## 4. CIRCE-Myo 预训练

先完成有界 pilot；检查 loss、保存的 resolved config 和环境元数据后再申请/启用完整训练：

```powershell
powershell -File archive\legacy-powershell-workflow\05_Pretrain.ps1 -Mode Pilot -MaxSteps 200 -Devices 1 -BatchSize 8 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

pilot 输出为：

```text
$RunsRoot\pretrain\<protocol>-seed-<split-seed>\pilot\steps-200\devices-1\accum-1\batch-8\seed-0
```

完整训练必须显式给出步数，不会由脚本自动启动：

```powershell
powershell -File archive\legacy-powershell-workflow\05_Pretrain.ps1 -Mode Full -MaxSteps 100000 -Devices 1 -AccumulateGradBatches 1 -BatchSize 8 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

恢复中断的完整训练：

```powershell
$PretrainCkpt = '$RunsRoot\pretrain\official-seed-20260916\full\steps-100000\devices-1\accum-1\batch-8\seed-0\checkpoints\last.ckpt'
powershell -File archive\legacy-powershell-workflow\05_Pretrain.ps1 -Mode Resume -MaxSteps 100000 -ResumeFrom $PretrainCkpt -Devices 1 -BatchSize 8 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0 -Force
```

脚本会打印 `global_batch_size = Devices × AccumulateGradBatches × BatchSize`。改变该批量大小、协议、切分种子或训练种子时，必须视为新实验。

## 5. 下游微调网格

对每个任务运行标签预算 `0.01, 0.05, 0.10, 1.0`，以及 `probe`、`partial`、`full` 三种微调模式；每一个组合均使用同一个预训练 checkpoint、协议和 split seed。下面是一个 `pose/probe/10%` 示例：

```powershell
$PretrainCkpt = '$RunsRoot\pretrain\official-seed-20260916\full\steps-100000\devices-1\accum-1\batch-8\seed-0\checkpoints\last.ckpt'
powershell -File archive\legacy-powershell-workflow\06_Finetune.ps1 -Task pose -FinetuneMode probe -LabelBudget 0.10 -PretrainedCheckpoint $PretrainCkpt -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

全量模式示例：

```powershell
powershell -File archive\legacy-powershell-workflow\06_Finetune.ps1 -Task typing -FinetuneMode full -LabelBudget 1.0 -PretrainedCheckpoint $PretrainCkpt -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

输出目录模式为 `$RunsRoot\finetune\<protocol>-seed-<split-seed>\<task>\<mode>\budget-<budget>\seed-<train-seed>\checkpoint-<12位路径哈希>`；最后一段由初始化 checkpoint 的绝对路径自动生成。正式报告至少重复 `Seed 0,1,2`；报告每种协议时不得混合随机种子或标签预算。

## 6. 监督基线与其他基线

每个基线应与微调实验有相同 task、协议、split seed、标签预算和训练 seed。例：

```powershell
powershell -File archive\legacy-powershell-workflow\04_TrainBaseline.ps1 -Model supervised_tcn -Stage finetune -Task pose -FinetuneMode full -LabelBudget 1.0 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
powershell -File archive\legacy-powershell-workflow\04_TrainBaseline.ps1 -Model bendr -Stage finetune -Task pose -FinetuneMode full -LabelBudget 1.0 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
powershell -File archive\legacy-powershell-workflow\04_TrainBaseline.ps1 -Model cbramod -Stage finetune -Task pose -FinetuneMode full -LabelBudget 1.0 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
powershell -File archive\legacy-powershell-workflow\04_TrainBaseline.ps1 -Model aemg -Stage finetune -Task pose -FinetuneMode full -LabelBudget 1.0 -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

`supervised_tcn` 只能用于 `Stage finetune`；脚本会拒绝它的预训练模式。需要“预训练→基线微调”时，把该基线的 `-PretrainedCheckpoint` 显式传入。所有基线结果位于 `runs\baselines\<protocol>-seed-<split-seed>\<model>\...`。

## 7. 固定 checkpoint 后的独立评估

不要在尚未确认来源的 checkpoint 上报告结果。评估时显式写出模型 checkpoint 和切分：

```powershell
$TaskCandidates = @(Get-ChildItem '$RunsRoot\finetune\official-seed-20260916\pose\full\budget-1\seed-0\checkpoint-*\checkpoints\last.ckpt')
if ($TaskCandidates.Count -ne 1) { throw "应当恰好找到一个任务 checkpoint，实际为 $($TaskCandidates.Count)" }
$TaskCkpt = $TaskCandidates[0].FullName
powershell -File archive\legacy-powershell-workflow\07_Evaluate.ps1 -Task pose -CheckpointPath $TaskCkpt -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0 -SavePredictions
```

输出为 `runs\evaluation\<protocol>-seed-<split-seed>\<task>\seed-<train-seed>\checkpoint-<12位路径哈希>`，其中最后一段由评估 checkpoint 自动生成；目录中应包含整体和 per-subject 指标、resolved config、manifest digest 与环境元数据。评估前先用 `-DryRun`；它不检查 checkpoint 是否存在，真实执行则会检查。

## 8. 鲁棒性对比

对固定、已评估的 checkpoint，逐条件进行独立运行。一次运行全部六个条件：

```powershell
powershell -File archive\legacy-powershell-workflow\08_Robustness.ps1 -Task pose -Condition All -CheckpointPath $TaskCkpt -Protocol $Protocol -SplitSeed $SplitSeed -Seed 0
```

条件是 `none`、`channel_rotation`、`channel_dropout`、`gain_drift`、`additive_noise` 和 `temporal_loss`。输出分别位于 `runs\robustness\<protocol>-seed-<split-seed>\<task>\<condition>\seed-<train-seed>\checkpoint-<12位路径哈希>`，因此不同种子、checkpoint 或条件的结果不会混写。

## 9. 安全管道与失败恢复

只需编排预检、审计、切分和烟雾测试时，可先干跑：

```powershell
powershell -File archive\legacy-powershell-workflow\RunPipeline.ps1 -Stages Preflight,Audit,Splits,Smoke -Protocol $Protocol -SplitSeed $SplitSeed -DryRun
```

确认后移除 `-DryRun`。`RunPipeline.ps1` 只运行 `-Stages` 明确列出的阶段，绝不自动加入昂贵的 Baseline、Pretrain、Finetune、Evaluate 或 Robustness；包含 `Splits` 时，它会把同一协议/种子生成的精确 split 文件传给后续阶段。

若某阶段因机器中断失败：

1. 保留其输出目录和日志，先定位最后一个有效 checkpoint；不要删除整个 `$RunsRoot`。
2. 对预训练或微调，将 checkpoint 传给 `-ResumeFrom`；对新实验改用新的 `-Seed` 或新的输出配置。
3. 只有确认要重用原输出目录后加 `-Force`，并在实验日志中注明原因。
4. 重新执行 `07_Evaluate.ps1` 和 `08_Robustness.ps1`，确保它们引用的是最终 checkpoint 与同一 split 选择。

每份最终结果表至少记录：数据版本/manifest digest、`Protocol`、`SplitSeed`、训练 seed、任务、标签预算、微调模式、模型 checkpoint、指标和运行目录。
