# Meta sEMG Foundation Model 本轮交付

## 1. 工程状态

唯一主仓库：`C:\Users\18165\Documents\ChatGPT\metafmodel`；MetaEMGFM保留历史成果和数据，不再是正式训练主线。
Git分支codex/local-integration；导入基准bf322d1；报告生成时HEAD `bf322d179a775925f6fb91ffe970f5d0c1615b86`。
正式集成源码最终commit见同目录git_version.txt。无push/远端PR。
上游仅main ZIP SHA固定，未取得upstream commit，未伪造来源版本。
Python 3.12.14、Torch 2.8.0+cu128、CUDA 12.8，
NVIDIA GeForce RTX 4060 Ti / 8.00GiB。
环境共享路径 `C:\Users\18165\Documents\ChatGPT\MetaEMGFM\.venv\Scripts\python.exe`；editable metafmodel0.2.0安装成功、pip check通过、wheel构建通过。
结构：configs、semg_h5、semg_fm、scripts、tests、outputs/verification/pretrain/finetune/evaluation、reports、docs。
data只通过配置引用旧路径，未大规模复制。主干/objectives保持上游代码。

## 2. 代码变更

|文件|改动/原因|验证|
|---|---|---|
|semg_fm/data/intervals.py|官方锚点、Nature半开区间与冲突防护|test_local_integrity、test_make_splits_cli、真实95区间|
|window_dataset.py、datamodule.py、make_splits.py|紧凑区间索引、身份与schema、raw配置、val评测|原窗口/采样测试、test_evaluation_split、统一验收|
|raw_adapter.py、raw_signal.py|原始五任务读取；轻量标签元数据；packet时钟|test_raw_adapter；35文件全信号/标签对比；Windows workers2|
|lightning/pretrain_module.py|RNG安全保存/恢复、训练语义校验|原checkpoint tests、test_local_checkpoint、独立进程10→20|
|local_callbacks.py|原生训练诊断及状态报告，不引入trainer|test_local_callbacks、真实SSL run|
|lightning/downstream_module.py|配置匹配、strict加载报告|test_local_checkpoint、真实student逐tensor相等|
|pretrain.py、finetune.py、evaluate.py|完整可信checkpoint恢复；显式val评测来源|原entry tests；原生CLI合成/真实val|
|configs/data、trainer、pretrain、finetune、local|本地低显存、外部路径、A800准备|Hydra实际CLI，本地PASS；A800 NOT_RUN|
|pose_verification.py、verification_checks.py、local_verification.py、scripts/local_verify.py|分阶段独立状态与原生闭环|test_verify_status、test_pose_verification、最新验收|
|scripts/workspace_audit.py、workspace_quarantine.py|只读审计、hash可恢复隔离|5安全测试，checkpoint全部保持|
|pyproject.toml、run_local.ps1、文档|editable安装、wheel、单命令验收、交付|pip check、package_build.log、清理后入口|
|tests/test_window_dataset.py|修正原fixture同subject跨非官方任务split、typing pair跨split，保留断言|原窗口tests未降低严格性|
|旧MetaEMGFM/README.md|标记历史项目，指向正式主仓库|不更改旧训练实现，旧43测试|

基准原280tests本轮已重新运行；最终主工程312 passed, 36 warnings in 20.53s；历史工程43 passed, 1 warning in 9.48s。
首次basetemp父目录缺失及中间失败均不算PASS；最终以本轮日志为准。

## 3. 数据验证

真实Pose 30录制均official val；真实Wrist
5录制official corpus train。
Pose200ms/100ms有效val 21,048窗。
Wrist 95个train区间，200ms/100ms
35,780窗；2s/1s 3,458窗。
35文件全量signal、relative time、label精确相等；原始SHA256匹配清单。
raw/canonical窗口边界、metadata split、label mask一致；官方split未修改。
五任务转换器保留；QWERTY/手势/手写仅夹具，缺真实文件和官方split/corpus。
详细证据见data_verification.md与`C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121246_587071/data_report.json`。

## 4. Encoder 验证

原Ring Criss-Cross，d_model32/depth2/heads4/patch40，35,140参数。
前反向有限、非零梯度，CUDA预算内。GPU报告为PyTorch allocated峰值，
不含全部驱动/context，时间是单次测量且首轮含初始化，不是吞吐基准。

|输入|z_eq|z_motion|前向s|反向s|峰值allocated MiB|
|---|---|---|---|---|---|
|[2, 400, 16]|[2, 16, 10, 32]|[2, 10, 32]|1.066152|0.309210|20.37|
|[2, 4000, 16]|[2, 16, 100, 32]|[2, 100, 32]|0.016096|0.018010|44.72|

Ring最大绝对误差 4.77e-07；motion invariance
2.38e-07。真实佩戴旋转/左右手环规则未验证。
ring error是无梯度诊断，非优化项；latent是实际主loss，kinematic默认关闭。

## 5. 预训练验证

SMOKE TEST，真实Wrist官方train，batch2/200ms，原生pretrain.py。
第10步loss 0.32353464；新进程恢复到20步loss 0.34405309。
Student更新、Teacher更新、无Teacher梯度，EMA count20。
恢复student/teacher/optimizer/scheduler准确、global_step10继续、RNG保存恢复。
训练semantics hparams校验通过。FP32 AMP NOT_RUN；采样序列不保证复现。
有限loss和feature std不证明预训练有研究收益。无合法val因此无best。
最终checkpoint：`C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\pretrain\smoke_wrist_20261012_121246_587071\resumed\checkpoint_last.ckpt`。

## 6. Pose 验证

实际SSL Student 45 tensors strict加载；missing/unexpected keys空，配置匹配，逐tensor精确。
features [2, 10, 32] → head [2, 10, 20] →
线性插值 [2, 400, 20]；EMG/label/no_ik_failure同sample slice。
真实val forward loss 7.37638378，仅接口诊断。单位未缩放，
弧度约定需官方确认。probe/partial/full均合成数据原生2步、freeze梯度和状态保存恢复PASS。
真实监督微调BLOCKED，未用val训练。真实val评测2batch加载合成head，
评测接口PASS，独立研究评测BLOCKED；不把其MAE/RMSE当泛化成绩。

## 7. 文件清理

初始可读32324文件/13,802,453,218bytes，是ACL限制下限。
仅隔离旧build 56个.py/353,527bytes，映射和SHA可恢复。
所有66原checkpoint按原路径/hash复核保持不变；数据、环境、重要实验保留。
永久删除0；释放空间0。待确认删除只有隔离build候选，无自动删除计划。
清理后312 passed, 36 warnings in 20.53s和旧43 passed, 1 warning in 9.48s；统一验收重新执行，见project_cleanup_summary.md。

## 8. 完整验收表

|模块|状态|范围|
|---|---|---|
|ENVIRONMENT|PASS|本地工程验收|
|DEPENDENCIES|PASS|本地工程验收|
|DATA_FORMAT|PASS|本地工程验收|
|SPLIT_INTEGRITY|PASS|本地工程验收|
|HDF5_CONSISTENCY|PASS|本地工程验收|
|ENCODER_FORWARD|PASS|本地工程验收|
|ENCODER_BACKWARD|PASS|本地工程验收|
|RING_EQUIVARIANCE|PASS|人工环位移，仅数学性质|
|EMA_TRAINING|PASS|本地工程验收|
|CHECKPOINT_RESUME|PASS|状态准确；样本序列/AMP未验收|
|POSE_WEIGHT_LOADING|PASS|本地工程验收|
|POSE_FORWARD|PASS|本地工程验收|
|POSE_TRAINING|BLOCKED|真实监督训练缺official train；合成3模式机制PASS|
|POSE_EVALUATION|PASS|仅2batch真实val接口；没有独立研究成绩|
|WINDOWS_WORKERS_2|PASS|raw/canonical各4窗口与串行完全相同|
|AMP / A800 / DDP|NOT_RUN|本轮FP32单卡|
|真实三类新任务|BLOCKED|缺真实文件/官方协议；夹具PASS|
|独立Pose研究评测|BLOCKED|缺合法真实训练及test|

最新run：`C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121246_587071`；统一状态13 PASS/1 BLOCKED/0 FAIL。

## 9. 遗留问题

P0：合法Pose train/test缺失；All-Meta其他三任务真实数据缺失，阻止完整研究实验。
P1：sampler游标/完整序列复现、AMP/多卡/A800/长训练、2s SSL尚未验证；
单位/环方向协议未确认，旧ACL目录UNKNOWN保留。已修复本輪发现的所有可复现关键代码问题。
P2：representation收益/坍塌、窗口上下文、decoder与label预算消融待真实实验。

## 10. 下一阶段 E0–E3 前置条件

|实验|缺数据/协议|Adapter / Decoder|窗口与服务器|
|---|---|---|---|
|E0 Scratch EMG2Pose|official train +独立test，val保留|现Pose adapter/head20DoF可用；确认radians/IK masks|200ms是smoke；研究可试2s，重新验显存/稳定性|
|E1 Pose-only SSL|Pose official train；合法下游train/test|无需新增adapter；保留head作基准|真实train先短跑；不能用当前val代替train|
|E2 Pose+Wrist SSL|Pose train及下游train/test；Wrist train当前可用，更广corpus/val后续补|现两adapter+同质混合采样，确认标签辅助loss关闭策略|2s Encoder已验，2s SSL和混合批需实跑|
|E3 All-Meta SSL|再补QWERTY/Gesture/Handwriting真实train/val和corpus，pair协议|原五转换器及raw接口已具备；三任务真实格式先验，再按差异最小补丁|需要服务器I/O、采样内存、AMP/DDP与长窗口验收|

A800仅配置准备，可复用Hydra/Lightning源码但不能宣称直接完成迁移；先确认
硬件CUDA兼容性（名称不足以确认），重测环境、bf16、DDP、workers与resume。
没有证据要求立即重写Pose decoder；先用现20DoF head建立真实基准再优化。
