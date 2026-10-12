# 主工程整合审计（实施前）

实际扫描日期：2026-10-12。

- 现有验证工程：C:/Users/18165/Documents/ChatGPT/MetaEMGFM。Git master 尚无提交，全部源码未跟踪；不重置或清除。
- 同事源码：third_party/metafmodel，官方 main ZIP SHA256 851c3630b625711f453689f9b71eb47c389b296d8e0588a57c2b2726defb08e8。已有 Pose split 与 manifest dtype=str 两处补丁。没有独立 metafmodel 主仓库或上游 commit ID。
- 计划正式目录：相邻 C:/Users/18165/Documents/ChatGPT/metafmodel，独立源码仓库，数据仍引用旧工程路径；共享现有可用 venv，不复制 CUDA runtime。
- 本轮实测 Python 3.12.14 / torch 2.8.0+cu128 / Lightning 2.6.6 / Hydra 1.3.7 / h5py 3.16.0 / pandas 3.0.6 / numpy 2.5.3 / pytest 9.1.1；CUDA 可用，RTX 4060 Ti。
- 数据、checkpoint 与访问失败目录见 file_inventory.csv 和 inventory_details.json。部分历史 pytest 目录 ACL 拒绝，标 UNKNOWN 保留，不修改系统 ACL。

| 模块 | 已具备功能 | 必要变更 | 优先级 |
|---|---|---|---|
| semg_h5 | 五类独立转换器、标签、溯源、质量统计 | 沿用已验证两处补丁，不重复转换器 | P0 |
| manifest/splits | CSV字段与路径、subject/pair hashing | 官方锚点防冲突、拒绝Nature整文件随机split | P0 |
| window_dataset/datamodule | canonical [T,16]、紧凑索引、任务collate与采样 | Nature corpus区间索引、metadata、raw adapter入口、身份校验 | P0 |
| Ring模型/objectives | 时间/环注意力、gate、时频patch、masked latent | 保留架构，重新测形状/显存/梯度/环误差 | 保持原样 |
| PretrainModule | EMA、优化器、schedule、teacher checkpoint | 可配置诊断callback，RNG恢复状态 | P1 |
| DownstreamModule | 20DoF head、probe/partial/full、strict加载、插值对齐、masked metrics | 明确返回加载报告与实证接口检查；不替换decoder | P1 |
| entrypoints | Hydra/Lightning预训练、微调、评测、baseline | 配置统一输出、本地/A800路径、支持val evaluate接口 | P1 |
| scripts/reports/docs | 原说明与测试 | 统一验收入口、审计与最新报告索引 | P1 |

不迁移早期CNN及其训练器。只迁移split/时间/原始读取等可复用逻辑，原生Lightning负责训练。真实Pose Train有无与val合法性需重新实测；缺失则真实微调BLOCKED，合成训练与真实val只读接口分开。
