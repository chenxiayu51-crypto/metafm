# 数据契约

原始文件无需全部转换。`data.format=raw` 或 `canonical` 使用同一个窗口、
collate、Ring 模型和 Lightning 入口。当前真实验证是 Pose 与 Wrist；其他三类
已有原生转换器和 raw 标签元数据解码，本轮仅用上游夹具验收。

|任务|原始节点/字段|Canonical 标签|
|---|---|---|
|pose|emg2pose/timeseries: emg,time,joint_angles,no_ik_failure|joint_angles / no_ik_failure|
|typing|emg2qwerty/timeseries: emg_left,emg_right,time|keystrokes / prompts|
|discrete_gesture|data: emg,time；prompts 表|events|
|handwriting|data: emg,time；prompts 表|prompts|
|wrist_angle|data: emg,time,wrist_angles|targets/wrist_angles|

每窗 emg 为 float32 `[T,16]`、sample_mask 为 bool `[T]`，metadata 包含
dataset_id/task_id/subject_id/session_id/hand_side/source_file/split/
sample_start/sample_end/sampling_rate。16通道单腕；QWERTY 两腕共享 pair_id，
不能跨 split。无幅值归一化、重采样或单位转换。

manifest 按字符串读取，保留前导零和 SHA256。Canonical 必须 schema 1.0、
float32 EMG、float64 time、属性与清单一致，sample_count 匹配。
正式运行前统一验收比较原始文件 hash、全部 signal/time/label；每窗仍核验属性。

官方 split 优先于生成的 split 列。unknown 的 subject/pair 按组划分并锚定
官方数据；模糊官方锚点拒绝。Nature 不能整文件随机划分，必须通过官方
corpus `(dataset,split,start,end)`，按源时间的 `[start,end)` 搜索样本索引，
窗口完整位于单一区间，跨 split/区间重叠拒绝。标签终点也受区间终点限制。
多 Nature 任务可配置 `data.corpus_path={task_id: absolute_csv}`。

Nature 原时间可能是分包时钟；不通过短窗口时间跨度重新推断2kHz采样率。
检查有限、非递减、大于100ms间断及长窗口恒定时钟；原时间保留，不声称
时间戳是逐采样均匀时钟。Pose 逐样本时钟检查严格递增和采样间隔。

HDF5 在每次读取内打开/关闭，不保留句柄；raw 文本/事件任务只缓存有界的
轻量标签元数据，不读取整录制 EMG。官方真实三类数据仍需到齐后实测。
Pose 角度原单位保留，20维已核验；radian约定尚需官方元数据确认。
