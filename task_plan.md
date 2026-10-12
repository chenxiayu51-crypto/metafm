# metafmodel 正式主工程整合 Implementation Plan

**Goal:** 独立固定同事原生Ring/EMA主工程，在真实Wrist官方区间完成10+10步训练恢复与Pose接口闭环，安全整理历史产物。

**Architecture:** 原生semg_fm/semg_h5、Hydra、Lightning保持主线。扩展原CanonicalWindowDataset区间几何并提供原始HDF5适配；配置引用已有真实数据，不复制大型文件。统一local_verify调用原模块和入口，状态PASS/FAIL/BLOCKED/NOT_RUN。

**Tech Stack:** Python 3.12、PyTorch CUDA、Lightning 2.6、Hydra 1.3、HDF5。

## Global Constraints
- 不替换Ring Criss-Cross，不创建另一训练框架；不改官方split。
- Nature corpus半开区间，跨split重叠拒绝，缺corpus禁止训练。
- Pose val不用于训练；真实Pose Train缺失则BLOCKED，合成只验证机制。
- 不永久删除，不复制大数据；所有归档可恢复且SHA256可核验。

## 顺序及验收
- [x] 审计：reports/file_inventory.csv、integration_audit.md、cleanup_plan.md。
- [x] 固定：复制小型上游源码到相邻metafmodel，Git基准commit和codex/local-integration分支；共享现有venv。
- [x] 环境：原280测试重新执行，保存日志和环境报告。
- [x] 数据：先增失败测试Nature无corpus/跨split/边界、官方split/pair/subject、source属性、raw vs canonical窗口与mask；实现intervals.py、raw_adapter.py、window_dataset/datamodule最小扩展，再跑全套。
- [x] 模型：CUDA [2,400,16]及[2,4000,16]前反向、参数/显存/耗时、环位移误差、loss梯度审计。
- [x] SSL：调用原PretrainModule，合法Wrist Train10步保存、子进程退出恢复到20步；optimizer/scheduler/student/teacher/RNG核验，诊断callback记录指标。
- [x] Pose：strict加载实际SSL student，20DoF对齐/mask/单位，probe/partial/full合成梯度、save/load；真实val forward/loss/evaluate只读，真实微调BLOCKED。
- [x] 统一：scripts/local_verify.py、configs/local/local_verify.yaml，各项状态独立报告，最新索引，无覆盖。
- [x] 清理：通过验收后仅隔离已核实可重生成缓存/构建产物；所有checkpoint/真实数据保留；路径/hash/恢复脚本。
- [x] 回归：原及新增测试，统一验收再运行，独立review修复，README/6份docs/最终10部分报告、A800配置准备。

具体实现按测试red-green实施；新训练入口仍是pretrain.py/finetune.py/evaluate.py。后续训练配置trainer callbacks，避免平行训练器。

真实Pose监督训练BLOCKED不等于完成真实训练；所有可执行工程阶段已执行，限制详见最终报告。
