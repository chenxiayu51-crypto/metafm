# 2026-10-16 官方发布格式核对

核对的是官方公开文档、下载脚本与 metadata 实例，不是用户服务器磁盘。服务器实读由 `scripts/server_dataset.py inspect` 完成。

1. [Nature README](https://github.com/facebookresearch/generic-neuromotor-interface#dataset-details) 明确 `${TASK_NAME}_corpus.csv` 包含 split 与 relevant stage 起止时刻。[官方下载脚本](https://github.com/facebookresearch/generic-neuromotor-interface/blob/main/generic_neuromotor_interface/scripts/download_data.py) 先解压 full/small tar，再另行下载 CSV。
2. 已读取三项官方 corpus CSV 的实际头部，列为 `start,end,split,dataset_number,user_number,dataset`。Wrist/Gesture 是 epoch 秒区间；**Handwriting 的 `start=0,end=inf` 表示全录音**。部署入口用 H5 的源时间范围求交后交给现有 finite interval loader，原始 CSV 不改，split 不改。
3. [EMG2Pose README](https://github.com/facebookresearch/emg2pose#data) 说明 full/mini tar 均附 `emg2pose_metadata.csv`，split 为 train/val/test。沿用主工程 H5 split 保留，并与 CSV 对照，冲突停止。
4. [QWERTY README](https://github.com/facebookresearch/emg2qwerty) 说明下载为 1135 session 加 metadata.csv。[generic YAML](https://github.com/facebookresearch/emg2qwerty/blob/3200d91eeb952cbed1f278e47d0cc56928334fd1/config/user/generic.yaml) 提供 837/192/16 session；[split生成脚本](https://github.com/facebookresearch/emg2qwerty/blob/3200d91eeb952cbed1f278e47d0cc56928334fd1/scripts/generate_splits.py) 说明 test 用户的 personalization train/val 不属于 generic train。本包读取已发布 YAML，不重跑随机划分算法。

QWERTY YAML 与对应 LICENSE 按固定 commit 放入 `configs/server/official/`。URL、commit 和原始 SHA256 在 `sources.json`，许可在 `EMG2QWERTY_LICENSE`；不包含原始训练数据。

## 数据流与代码

```text
五个 raw 目录（configs/server/user_paths.json）
  -> server_dataset.inspect / inspect_file
  -> 官方 CSV 与 QWERTY generic YAML 交叉核对
  -> server_dataset.prepare（单源 staging / journal / SHA256）
  -> 既有 semg_h5/convert_*.py::convert_file
  -> Canonical HDF5 + manifest.csv
  -> prepare_data.build_splits（非Nature只保留已明确的官方split）
     + 官方Nature区间求交 -> nature_corpus.csv
  -> server_preflight.py
  -> run_e3_server.sh -> e3_oneclick.py server -> pretrain.py
  -> 原生 DataModule / WindowDataset -> Ring encoder / EMA -> checkpoint
```

`inspect` 不遍历全部信号数值，转换时继续调用原转换器完整校验。train/val 元数据覆盖不等同于足够的2秒窗口，原生 GPU preflight 是正式训练前的最终检查。
