"""Render delivery documents from actual latest acceptance artifacts (no training)."""
import csv
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,text):Path(path).write_text(text,encoding='utf-8')
def dump(value):return '```json\n'+json.dumps(value,indent=2,ensure_ascii=False)+'\n```\n'


def main():
    latest=read(ROOT/'outputs/verification/latest.json');run=Path(latest['path'])
    statuses=read(run/'summary.json');env=statuses['ENVIRONMENT']['details']
    split=read(run/'split_report.json');data=read(run/'data_report.json')
    model=read(run/'model_report.json');ssl=read(run/'ssl_report.json');pose=read(run/'pose_report.json')
    for name in ('environment','data','model','ssl','pose'):
        source={'environment':env,'data':dict(data=data,split=split),'model':model,'ssl':ssl,'pose':pose}[name]
        save(REPORTS/f'{name}_verification.md',f'# {name} verification\n\nSource: `{run}`\n\n'+dump(source))
    with (REPORTS/'file_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
        before=[row for row in csv.DictReader(f) if row['path']=='.']
    with (REPORTS/'post_cleanup/file_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
        after=[row for row in csv.DictReader(f) if row['path']=='.']
    mapping=read(ROOT/'outputs/archive/quarantine_build_20261012/path_mapping.json')
    archived_bytes=sum(row['size'] for row in mapping['files'])
    initial_checkpoints=read(REPORTS/'inventory_details.json')['checkpoints']
    post={row['path']:row for row in read(REPORTS/'post_cleanup/inventory_details.json')['checkpoints']}
    unchanged=all(row['path'] in post and post[row['path']]['sha256']==row['sha256'] for row in initial_checkpoints)
    assert unchanged,'Historical checkpoint path/hash changed'
    test_log=(REPORTS/'final_regression_tests.log').read_text(encoding='utf-8',errors='replace')
    legacy_log=(REPORTS/'legacy_regression_tests.log').read_text(encoding='utf-8',errors='replace')
    tests=re.findall(r'\d+ passed[^\n]*',test_log)[-1]
    legacy_tests=re.findall(r'\d+ passed[^\n]*',legacy_log)[-1]
    assert not any(v['status']=='FAIL' for v in statuses.values())
    cleanup=f'''# 项目整理结果

旧build的{len(mapping['files'])}个可重建Python副本已隔离，{archived_bytes:,} bytes。
来源 `{mapping['source']}`；隔离 `{mapping['destination']}`。
逐文件原路径/目标/SHA256见 `outputs/archive/quarantine_build_20261012/path_mapping.json`。
永久删除：0。释放空间：0 bytes。实际分类候选不能当作删除授权。
原始/Canonical HDF5、manifest/corpus、venv、历史源码、全部原有checkpoint保留。
初始{len(initial_checkpoints)}个checkpoint已按原路径与SHA256再核验：全部不变。

初始可读盘点：{before[0]['file_count']}文件 / {int(before[0]['total_bytes']):,} bytes。
这是下限；初始15扫描错误含14旧目录ACL拒绝和当时尚未建立的主工程。
整理后根统计（含本轮新增实验、文档、Git，不可当作清理净减少量）：
{dump(after)}
初始见file_inventory.csv；清理后见post_cleanup/file_inventory.csv。
metric_files仅表示存在JSON/CSV，不是实验指标有效性证明。
历史test/review/reuse输出及ACL拒绝目录UNKNOWN保留，没有整目录按名字删除。

恢复（原build必须不存在，任何父目录junction/reparse会拒绝）：
```powershell
.\\scripts\\run_local.ps1 test
$pythonExecutable='../MetaEMGFM/.venv/Scripts/python.exe'
& $pythonExecutable scripts/workspace_quarantine.py --restore outputs/archive/quarantine_build_20261012/path_mapping.json
```
恢复先核验所有hash与完整文件列表，不覆盖原路径。
待另行确认永久删除的候选仅此已验证隔离build；本轮没有请求删除。

清理后主工程：{tests}；历史工程：{legacy_tests}。
清理后统一验收：`{run}`，13 PASS / 1 BLOCKED / 0 FAIL / 0 NOT_RUN。
'''
    save(REPORTS/'project_cleanup_summary.md',cleanup)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    changes='''|文件|改动/原因|验证|
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
'''
    table='|模块|状态|范围|\n|---|---|---|\n'
    scopes={'POSE_TRAINING':'真实监督训练缺official train；合成3模式机制PASS',
            'POSE_EVALUATION':'仅2batch真实val接口；没有独立研究成绩',
            'CHECKPOINT_RESUME':'状态准确；样本序列/AMP未验收',
            'RING_EQUIVARIANCE':'人工环位移，仅数学性质'}
    for key,value in statuses.items():table+=f'|{key}|{value["status"]}|{scopes.get(key,"本地工程验收")}|\n'
    table+='|WINDOWS_WORKERS_2|PASS|raw/canonical各4窗口与串行完全相同|\n|AMP / A800 / DDP|NOT_RUN|本轮FP32单卡|\n|真实三类新任务|BLOCKED|缺真实文件/官方协议；夹具PASS|\n|独立Pose研究评测|BLOCKED|缺合法真实训练及test|\n'
    shapes='|输入|z_eq|z_motion|前向s|反向s|峰值allocated MiB|\n|---|---|---|---|---|---|\n'
    for row in model['measurements']:
        shapes+=f'|{row["input"]}|{row["output"]["z_eq"]}|{row["output"]["z_motion"]}|{row["forward_s"]:.6f}|{row["backward_s"]:.6f}|{row["peak_memory_bytes"]/2**20:.2f}|\n'
    last=ssl['resumed']['steps'][-1];first=ssl['first']['steps'][-1]
    final=f'''# Meta sEMG Foundation Model 本轮交付

## 1. 工程状态

唯一主仓库：`{ROOT}`；MetaEMGFM保留历史成果和数据，不再是正式训练主线。
Git分支codex/local-integration；导入基准bf322d1；报告生成时HEAD `{commit}`。
正式集成源码最终commit见同目录git_version.txt。无push/远端PR。
上游仅main ZIP SHA固定，未取得upstream commit，未伪造来源版本。
Python {env['python']}、Torch {env['torch']}、CUDA {env['cuda_version']}，
{env['gpu']} / {env['total_vram_bytes']/2**30:.2f}GiB。
环境共享路径 `{env['executable']}`；editable metafmodel0.2.0安装成功、pip check通过、wheel构建通过。
结构：configs、semg_h5、semg_fm、scripts、tests、outputs/verification/pretrain/finetune/evaluation、reports、docs。
data只通过配置引用旧路径，未大规模复制。主干/objectives保持上游代码。

## 2. 代码变更

{changes}
基准原280tests本轮已重新运行；最终主工程{tests}；历史工程{legacy_tests}。
首次basetemp父目录缺失及中间失败均不算PASS；最终以本轮日志为准。

## 3. 数据验证

真实Pose {sum(r['task']=='pose' for r in data['real'])}录制均official val；真实Wrist
{sum(r['task']=='wrist_angle' for r in data['real'])}录制official corpus train。
Pose200ms/100ms有效val {pose['forward']['valid_windows']:,}窗。
Wrist {split['wrist_train_intervals']}个train区间，200ms/100ms
{split['wrist_windows']:,}窗；2s/1s {split['wrist_2s_windows']:,}窗。
35文件全量signal、relative time、label精确相等；原始SHA256匹配清单。
raw/canonical窗口边界、metadata split、label mask一致；官方split未修改。
五任务转换器保留；QWERTY/手势/手写仅夹具，缺真实文件和官方split/corpus。
详细证据见data_verification.md与`{run}/data_report.json`。

## 4. Encoder 验证

原Ring Criss-Cross，d_model32/depth2/heads4/patch40，{model['parameters']:,}参数。
前反向有限、非零梯度，CUDA预算内。GPU报告为PyTorch allocated峰值，
不含全部驱动/context，时间是单次测量且首轮含初始化，不是吞吐基准。

{shapes}
Ring最大绝对误差 {model['ring_equivariance_max_abs']:.3g}；motion invariance
{model['motion_invariance_max_abs']:.3g}。真实佩戴旋转/左右手环规则未验证。
ring error是无梯度诊断，非优化项；latent是实际主loss，kinematic默认关闭。

## 5. 预训练验证

SMOKE TEST，真实Wrist官方train，batch2/200ms，原生pretrain.py。
第10步loss {first['loss']:.8f}；新进程恢复到20步loss {last['loss']:.8f}。
Student更新、Teacher更新、无Teacher梯度，EMA count20。
恢复student/teacher/optimizer/scheduler准确、global_step10继续、RNG保存恢复。
训练semantics hparams校验通过。FP32 AMP NOT_RUN；采样序列不保证复现。
有限loss和feature std不证明预训练有研究收益。无合法val因此无best。
最终checkpoint：`{ssl['checkpoint']}`。

## 6. Pose 验证

实际SSL Student 45 tensors strict加载；missing/unexpected keys空，配置匹配，逐tensor精确。
features {pose['forward']['features']} → head {pose['forward']['head']} →
线性插值 {pose['forward']['target']}；EMG/label/no_ik_failure同sample slice。
真实val forward loss {pose['forward']['loss']:.8f}，仅接口诊断。单位未缩放，
弧度约定需官方确认。probe/partial/full均合成数据原生2步、freeze梯度和状态保存恢复PASS。
真实监督微调BLOCKED，未用val训练。真实val评测2batch加载合成head，
评测接口PASS，独立研究评测BLOCKED；不把其MAE/RMSE当泛化成绩。

## 7. 文件清理

初始可读{before[0]['file_count']}文件/{int(before[0]['total_bytes']):,}bytes，是ACL限制下限。
仅隔离旧build {len(mapping['files'])}个.py/{archived_bytes:,}bytes，映射和SHA可恢复。
所有66原checkpoint按原路径/hash复核保持不变；数据、环境、重要实验保留。
永久删除0；释放空间0。待确认删除只有隔离build候选，无自动删除计划。
清理后{tests}和旧{legacy_tests}；统一验收重新执行，见project_cleanup_summary.md。

## 8. 完整验收表

{table}
最新run：`{run}`；统一状态13 PASS/1 BLOCKED/0 FAIL。

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
'''
    save(REPORTS/'final_delivery.md',final)
    experiment=f'''# 实验记录（本轮）

来源run `{run}`；基准commit bf322d1，运行当时HEAD尚为导入基准且带集成修改。
最终源码commit在reports/git_version.txt；所有当前run与配置/log仍可追溯。
不把运行metadata中的基准SHA误认为完整集成代码SHA。

1. 上游基准280 tests，正式新增回归最终{tests}；旧工程{legacy_tests}。
2. 35文件hash与全部EMG/time/labels一致，Nature95train区间。
3. 原生真实SSL10步与新进程10→20恢复，checkpoint `{ssl['checkpoint']}`。
4. 三模式合成Pose2步，各自last.ckpt，真实val forward和2batch评测接口。
5. 中间失败：packet短窗口rate误判、NumPy RNG不兼容默认安全load、raw typing
prompt相对时间、标签reader全量EMG、restore ancestor junction/..误判；均用回归固定。
6. 旧build可恢复隔离，hash逐文件验证；清理后全测试与统一验收。

只有smoke/接口结果。真实Pose监督训练BLOCKED，独立性能无结果；原有实验仍保留。
'''
    save(ROOT/'docs/experiment_log.md',experiment)
    print(str(REPORTS/'final_delivery.md'))


if __name__=='__main__':main()
