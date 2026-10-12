# 实验记录（本轮）

来源run `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121912_773000`；基准commit bf322d1，最终验收运行HEAD为固定集成commit 74b22f8，源码未再修改。
最终源码commit在reports/git_version.txt；所有当前run与配置/log仍可追溯。
最终训练run_metadata.json中的SHA为74b22f8；中间运行记录基准SHA并带修改，保留用于历史诊断。

1. 上游基准280 tests，正式新增回归最终312 passed, 36 warnings in 20.53s；旧工程43 passed, 1 warning in 9.48s。
2. 35文件hash与全部EMG/time/labels一致，Nature95train区间。
3. 原生真实SSL10步与新进程10→20恢复，checkpoint `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\pretrain\smoke_wrist_20261012_121912_773000\resumed\checkpoint_last.ckpt`。
4. 三模式合成Pose2步，各自last.ckpt，真实val forward和2batch评测接口。
5. 中间失败：packet短窗口rate误判、NumPy RNG不兼容默认安全load、raw typing
prompt相对时间、标签reader全量EMG、restore ancestor junction/..误判；均用回归固定。
6. 旧build可恢复隔离，hash逐文件验证；清理后全测试与统一验收。

只有smoke/接口结果。真实Pose监督训练BLOCKED，独立性能无结果；原有实验仍保留。
