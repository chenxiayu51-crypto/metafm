# 已知限制

P0：缺官方Pose train/test，真实监督训练与独立研究评测BLOCKED；只存在
30份official val。缺真实QWERTY/Gesture/Handwriting及其官方split/corpus，
不能运行E3真实All-Meta训练。

P1：中途恢复没有sampler游标，样本顺序未复现；正式长训练需扩展sampler
state或只在epoch边界恢复。AMP、长时间稳定性、多卡DDP、A800、2s完整SSL
和真实三类raw Adapter均未验收。radian单位、电极环方向和左右手规则仍需
官方证据。部分旧输出ACL不可读，UNKNOWN保留，不改系统ACL。

P2：20步loss有限只是工程证据；后续要验证latent collapse、下游收益、
窗口上下文和label-budget实验。现有20DoF decoder可用于接口，只有研究结果
证明不足时再改。当前raw label cache按录制缓存轻量表，64项上限；大数据
任务还需I/O性能基准。

已修：Nature整文件泄漏/官方split冲突、事件边界、QWERTY prompt相对时钟、
raw标签全EMG加载、NumPy RNG安全加载、恢复语义配置漂移、严格encoder配置。
