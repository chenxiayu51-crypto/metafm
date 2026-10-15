# 真实数据 mini 子集（real mini dataset）

`scripts/export_real_mini.py` 把真实五语料数据根切成一个体积很小、可迁移、且与生产**数据契约完全一致**的子集。
在**持有真实语料的机器**上导出，传到工作站，用来验证整条训练通路——无需下载数百 GB。

---

## 1. 为什么需要 real mini

服务器上的真实数据根有数百 GB（单个 emg2pose session 就有数百 MB）。要验证数据契约、采样、模型前/反向、checkpoint 恢复，不需要把全部语料拉到工作站。

exporter 不复制整条录音，而是把每条被选中的录音**截成一段包含其官方区间的连续窗口**。文件小、能训练，但字段、dtype、标签、时钟语义一个都不变。

本文件的产物是一个自包含的数据根，可直接喂给 `scripts/verify_local_real.py`（见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)）以及仓库自身的 loader。它不是新格式：它是 [DATA_LAYOUT.md](DATA_LAYOUT.md) 契约的一个子集样本。

---

## 2. 用法与参数

```bash
python scripts/export_real_mini.py \
    --data-root /srv/datasets/meta_canonical \
    --output    /srv/datasets/meta_real_mini
```

（上面的 `/srv/...` 与服务器部署布局一致，见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。）

全部参数（均取自 `argparse`，默认值已核对源码）：

| flag | 默认 | 含义 |
|---|---|---|
| `--data-root` | **必填** | 真实 canonical 数据根 |
| `--output` | **必填** | mini 目标目录 |
| `--manifest` | `splits.csv` | `<data-root>` 下的 manifest 文件名 |
| `--corpus` | `nature_corpus.csv` | `<data-root>` 下的官方 Nature 语料文件名 |
| `--records-per-task-per-split` | `2` | 每个 task、每个 split 选多少条源录音 |
| `--segment-seconds` | `30.0` | 导出窗口上限（秒）；该窗口仍必须包含官方区间 |
| `--margin-seconds` | `1.0` | Nature 区间两侧额外保留的上下文（秒） |
| `--seed` | `0` | 选择与洗牌的随机种子 |
| `--overwrite` | 关 | 允许写入非空输出目录（会先 `shutil.rmtree`） |
| `--skip-validation` | 关 | 跳过用项目自身 Dataset 回读的校验 |

参数合法性：`--records-per-task-per-split` 必须 `>= 1`；`--segment-seconds` 必须 `> 0`；`--data-root` 必须是已存在目录；`--output` 不得等于也不得位于 `--data-root` 之内。

五语料与 task 映射（与 [SERVER_DEPLOY.md](SERVER_DEPLOY.md) 一致）：

| `dataset_id` | `task_id` | 是否 Nature |
|---|---|---|
| `emg2pose` | `pose` | 否 |
| `emg2qwerty` | `typing` | 否 |
| `gni_discrete_gestures` | `discrete_gesture` | **是** |
| `gni_handwriting` | `handwriting` | **是** |
| `gni_wrist` | `wrist_angle` | **是** |

---

## 3. 输出布局

```
<mini root>/
  splits.csv                 # 13 列 + 尾部 split 列
  nature_corpus.csv          # 仅保留被选中 Nature 源文件的官方区间行
  <dataset_id>/<task_id>/<subject>/<session>_<side>.h5
  export_metadata.json
  checksums.sha256
```

| 文件 | 内容 |
|---|---|
| `splits.csv` | 与 [DATA_LAYOUT.md](DATA_LAYOUT.md) §2 相同的 13 列，外加尾部 `split` 列。`path` 相对 mini 根（可迁移，不含绝对路径）；`official_split` 源样复制；mini 中 `split` 列**留空**，真实 split 由 `official_split`（非 Nature）或 `nature_corpus.csv`（Nature）决定。按 `path` 排序。 |
| `nature_corpus.csv` | 列 `dataset,split,start,end`；只保留 basename 被选中的行；区间为**源录音**秒、半开 `[start,end)`；按 `(dataset,start)` 排序 |
| `*.h5`（五语料目录） | 保留源录音的相对路径。`signal/emg` 为 `float32 [T,16]`；`signal/time_s` 为 `float64 [T]`（已 re-base）；`meta/source_json` 含移位后的 `source_time_origin_s`；`labels/*` 与原 schema 一致；`meta/quality/{sample_count,duration_s,nan_count,channel_rms}` 重算 |
| `export_metadata.json` | `schema="e3-real-mini-v1"`、`created_utc`、`source_root`、`source_manifest{name,sha256}`、`source_corpus{name,sha256}`、`parameters{records_per_task_per_split,segment_seconds,margin_seconds,seed}`、`records_per_task`（每 task 的 train/val 计数）、`records_exported`、`note` |
| `checksums.sha256` | 除自身外每个文件的 `sha256`，路径相对输出根 |

导出结束会打印 `[mini] exported N records, X MB -> <output>`。

---

## 4. 四条保证

### 保证 1 —— 只用官方 train/val，绝不触碰 test，`official_split` 原样复制

只有 `train` 与 `val` 记录合格；`test` 永不导出（`plan_plain` 里显式 `continue`，Nature 侧只挑 `NEEDED_SPLITS`）。`effective_split()` 镜像 `window_dataset.from_manifest`：**official split 优先于 `split` 列**。导出后 `official_split` 逐字复制到 mini 的 manifest。每个 task 必须同时有 `train` 与 `val`，否则报错。

> emg2pose 官方发布同时带 train/val/test；exporter 会跳过它的 test 文件。

### 保证 2 —— 保留源时钟

截断时令 `shift = signal/time_s[a]`，写出 `relative = time_s[a:b] - shift`，同时把 `meta["source_time_origin_s"] = old_origin + shift`（这行是 `write_record` 里"保持源时钟诚实"的唯一一行）。

于是 `source_time = time_s + origin` 在**每个保留样本上与原样本完全相同**。官方语料区间以源时刻表达、从不改写，因此仍落在相同源时刻上；`intervals.corpus_bounds` 用
`searchsorted(time_s, bound - source_time_origin_s)` 完成映射（见 [DATA_LAYOUT.md](DATA_LAYOUT.md) §3）继续可用。`tests/test_export_real_mini.py::test_export_preserves_the_source_clock` 逐样本验证这一点。

### 保证 3 —— 数据契约不变

采样率、通道顺序、dtype、标签、metadata 保持 converter 写入的确切 schema。**不滤波、不重采样、不重缩放**；文件变小完全靠截断，而非降采样。

### 保证 4 —— 每个源 basename 恰好一条记录

loader 要求 Nature 源 basename 唯一（[DATA_LAYOUT.md](DATA_LAYOUT.md) §3；`intervals.selected_records` 也拒绝重复名）。因此每条源录音在输出里恰好出现一次。当同一源同时提供官方 train 与 val 区间时，它仍只有**一条**记录，其窗口被撑大到**同时包含这两个区间**（`plan_nature` 会挑离所选 train 区间**最近**的 val 区间，尽量缩小跨度）。exporter 还会检查所有被选 Nature 源的 basename 全局唯一，否则报错。

---

## 5. 硬约束：官方区间必须长于一个 2000 ms 窗口

窗口几何为 2000 ms / 1000 ms stride、2 kHz → 一个窗口 4000 采样。若某官方区间短于一个窗口，`_WindowIndex._build_index` 为它算出的 count 为 0，于是**该区间贡献 0 个窗口**。

exporter 绝不静默丢弃这种区间：`nature_window` 若窗口放不下某 split 的官方区间就抛出 `ExportError`，并提示调大 `--segment-seconds` 或 `--records-per-task-per-split`。导出末尾的 `validate()` 用**项目自身 Dataset** 回读 mini，任一 split 的任一 task 窗口数为 0 即报错。`server_preflight.py` 在真实语料上也检查该约束。

---

## 6. 传输到工作站

exporter 结束时打印：

```bash
tar -C <output.parent> -czf <name>.tar.gz <name>
scp -r <user>@<server>:<output> <local>/real_mini
rsync -av --checksum <user>@<server>:<output>/ <local>/real_mini/
```

传输后校验：

```bash
cd <local>/real_mini
sha256sum -c checksums.sha256
```

> **mini 不得提交到 Git。** 它来源于真实数据，应放在仓库之外的 `<local>/real_mini` 之类位置。仓库本身的运行输出约定见 [SERVER_DEPLOY.md](SERVER_DEPLOY.md)。

---

## 7. 失败模式与错误语义

所有 `ExportError` 会向 stderr 打印 `[mini ERROR] <msg>` 并以 **退出码 2** 结束。

| 场景 | 触发的错误信息 |
|---|---|
| 输出位于数据根内 | `output <out> must not live inside the source data root <root>` |
| 输出非空且未加 `--overwrite` | `output directory is not empty: <out> (pass --overwrite)` |
| 缺 corpus | `missing official Nature corpus: <path>` |
| 窗口放不下官方区间 | `<source>: the <N>s window dropped its official <split> interval. Raise --segment-seconds or increase --records-per-task-per-split.` |
| 窗口过小（< 2 采样） | `<source>: window is too small to export (<n> samples)` |
| Nature basename 歧义 | `ambiguous Nature source basename: <name>` |
| Nature 源无官方区间 | `Nature source <name> has no official corpus interval; it cannot be exported without its official split, and inventing one is not allowed` |
| 某 task 缺 train 或 val | `<dataset_id> produced no <split> record; the mini could not train that task. Increase --records-per-task-per-split.` |
| `--records-per-task-per-split < 1` | `--records-per-task-per-split must be at least 1` |
| `--segment-seconds <= 0` | `--segment-seconds must be positive` |
| 数据根不存在 | `data root does not exist: <root>` |
| manifest 缺列 | `<manifest> is missing columns: [...]` |
| manifest 路径逃逸数据根 | `manifest path escapes the data root: <path>` |
| manifest 指向缺失文件 | `manifest references a missing file: <path>` |
| manifest 无该 task 记录 | `the manifest has no records for <dataset_id>` |
| 非 Nature 无法确定官方 split | `<path> has official_split=<v> and no usable 'split' column value, ... Run make_splits first; nothing may be relabelled.` |

`validate()` 阶段（用项目自身 Dataset 回读）的失败信息：

| 场景 | 信息 |
|---|---|
| 某 split 未暴露五 task | `mini <split> exposes [...], expected [...]` |
| 某 task 无窗口 | `mini <split> has no <task_id> windows` |
| EMG 形状不对 | `mini <split>/<task_id> EMG (...) != (4000, 16)` |
| sample_mask 为空 | `mini <split>/<task_id> produced an empty sample mask` |

---

## 8. 相关测试

`tests/test_export_real_mini.py` 用一份贴近真实服务器的 fixture（Nature 带非零 `source_time_origin_s`、emg2pose 带 test、emg2qwerty 左右成对）覆盖：

- `test_export_skips_official_test_and_keeps_five_tasks` —— test 不导出，official_split 原样保留
- `test_export_preserves_the_source_clock` —— `time_s + origin` 逐样本不变
- `test_export_keeps_nature_intervals_inside_the_window` —— 保留区间被记录窗口覆盖
- `test_mini_validates_against_the_real_dataset` —— 用项目 Dataset 回读通过
- `test_mini_ships_checksums_and_metadata` —— checksums 与 metadata 完整，且 manifest 路径不含绝对路径
- 拒绝场景：缺 corpus、非空输出未覆盖、输出在数据根内、窗口过短
