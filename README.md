# Meta sEMG Foundation Model — 正式研究工程

metafmodel 是唯一正式主仓库，保留同事原始 Ring Criss-Cross + EMA Teacher–Student + Hydra/Lightning。
MetaEMGFM 保存历史验证、真实数据和当前共享环境。本轮没有下载全量数据或复制大HDF5。

本轮验收：主工程312 tests、历史工程43 tests；35份真实Pose/Wrist全量一致；真实Wrist
官方train20步与独立进程checkpoint恢复；Encoder→20DoF Pose接口、合成三种微调协议和
原生评测入口通过。当前30份Pose全为official val，真实监督微调BLOCKED，没有独立研究性能成绩。

## 本地运行

```powershell
cd C:\Users\18165\Documents\ChatGPT\metafmodel
.\scripts\run_local.ps1 verify
.\scripts\run_local.ps1 test
.\scripts\run_local.ps1 pretrain data=local trainer=local_gpu +pretrain=local_wrist trainer.max_steps=10
```

单命令verify生成唯一outputs/verification/local_*与latest.json；各模块分开报告
PASS / FAIL / BLOCKED / NOT_RUN，失败保存日志并继续独立检查。仅FAIL返回非零。
默认环境相邻MetaEMGFM/.venv；可设置META_PYTHON。新机器先安装对应Torch，再
`python -m pip install -e .[test]`。官方数据路径通过configs或环境变量引用。

## 当前数据流和代码

原始HDF5 → 原生semg_h5转换器或RawWindowDataset → 官方split/Nature corpus区间 →
[T,16]窗口 → task_homogeneous_collate [B,T,16] → Ring Encoder → masked student/EMA teacher →
checkpoint.student → DownstreamModule 20DoF head → 插值样本对齐 → masked regression / evaluate。

- [数据契约](docs/data_contract.md)：intervals.py、raw_adapter.py、window_dataset.py、datamodule.py。
- [模型架构与流程图](docs/architecture.md)：models/ring_criss_cross.py、patch_encoder.py、ring_attention.py。
- [训练闭环](docs/training_pipeline.md)：lightning/pretrain_module.py、downstream_module.py、原生三个入口。
- [本地部署与A800准备](docs/local_deployment.md)：服务器硬件/AMP/DDP仍NOT_RUN。
- [本轮完整十部分交付](reports/final_delivery.md)、[实验记录](docs/experiment_log.md)、[已知限制](docs/known_issues.md)。
- [审计](reports/integration_audit.md)、[可恢复清理](reports/project_cleanup_summary.md)、[上游来源](UPSTREAM.md)。

上游转换器仍可直接使用，见[原README](docs/upstream_README.md)及H5_DATA_CONTRACT.md。
新版Nature必须官方corpus，不能按原README示例整文件随机划分。Pose val不能重标train。
仅原始标签/时间/安全划分等必要功能迁移；models与objectives无架构重写。

旧build56个.py已隔离，所有66原checkpoint路径/hash不变；永久删除0，释放0bytes。
测试和日志里出现warning不代表研究成功：20步是工程smoke，synthetic指标不能证明真实泛化。
