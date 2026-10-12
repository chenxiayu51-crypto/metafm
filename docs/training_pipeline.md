# 原生训练闭环

唯一训练入口是 pretrain.py / finetune.py / evaluate.py（Hydra + Lightning）。
local_verify 只是调用原生入口和生成验收报告，不是另一套 trainer。

SSL：官方 Wrist train → 200ms/100ms、batch2 → Ring Student 的 masked tokens
预测未掩码 EMA Teacher → 原生 optimizer/scheduler → 10步 checkpoint →
退出进程 → 恢复至20步。训练为 smoke，不证明表示质量。

Teacher 无梯度，优化步后 EMA 更新；callback 记录 loss/feature std/gradient/
LR/GPU memory 和训练时间。checkpoint 保存 student/teacher/optimizer/
scheduler/global_step/EMA count/Python NumPy Torch CUDA RNG。NumPy RNG 用
primitive 数据兼容安全权重加载。恢复严格校验训练 hyperparameters；
延长运行仅改变 trainer.max_steps，不改变原 optimizer.max_steps 调度语义。
采样器尚无恢复游标，RNG恢复不等于完整样本序列重现。FP32没有AMP scaler。

Pose：严格加载相同配置 student，创建20DoF temporal head。probe冻结整个
backbone，partial只解冻最后指定blocks，full全解冻。掩码回归去除IK失败样本。
合成训练用三个独立subject的train/val/test，原生入口每模式2步保存，再
strict读取；检查head更新、frozen参数不变、trainable backbone更新。

真实Pose只有val。真实监督训练BLOCKED。评测入口显式
`+data.evaluation_split=val`，不修改manifest；本轮2batch仅检查接口，加载的
head来自合成训练，输出不属于泛化成绩。原生Lightning指标命名仍为test/*，
evaluation_protocol.json明确实际来源split与scope。

SSL输出包含train.log、metrics.json、resolved_config.yaml/config.yaml、
checkpoint_last.ckpt、last.ckpt、environment.json、manifest.json、
data_manifest_hash.json、run_metadata.json。无合法SSL val时不制造best权重。
Pose checkpoint在outputs/finetune，评测metrics和per-subject在outputs/evaluation。
