# 2026-10-16 服务器部署源码包本地验证

## 已实际执行

- 当前完整回归：`python -m pytest -q --disable-warnings`，**388 passed, 44 warnings**。
- 新部署模块：16 项测试，覆盖五路径格式、缺失/冲突官方split、Nature区间重叠、官方Handwriting无穷边界、QWERTY YAML排除、五类转换、hash恢复保护、左右腕提交中断、metadata提交中断、非空输出保护及预检/训练相同batch size。
- 本地真实 EMG2Pose Mini 两份文件、Nature Wrist 两份文件，通过 `inspect_file` 头部与前后采样读取；Pose 每份119266点/20角度，Wrist 分别2673104与3267440点/2角度。
- `bash -n` 检查新Linux入口与既有训练/环境入口通过。
- 新Python入口 `--help` 正常。
- 独立代码审查发现并修复metadata覆盖保护问题；复核无剩余明确P1/P2。

## 仍由服务器验收

五个服务器raw目录不可从此Windows会话读取，尚无全量服务器文件清单、完整转换或Linux CUDA训练结果。其他三类的本轮读入证据为官方格式文档/metadata与测试夹具，不是其真实HDF5验收。按 `START_HERE_服务器.md` 执行 inspect、prepare、preflight、100步训练后才可确认服务器闭环。

44个warnings来自现有训练单元测试中的CPU运行、缺少validation loader、worker建议与恢复提示等；测试退出码为0。它们不代表服务器CUDA环境已通过。

源码包不附Git历史。测试中有一项专门要求真实Git checkout的commit溯源；在无`.git`的解压目录跑完整pytest前，应先初始化并提交源码，或单独运行部署测试。包的来源与完整性通过 `SOURCE_RELEASE.json` 和外部归档SHA256记录。
