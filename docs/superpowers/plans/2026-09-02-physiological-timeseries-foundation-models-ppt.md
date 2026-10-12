# 生理时序基座模型 PPT 实施计划

> **面向 agentic workers：** 必需子技能（REQUIRED SUB-SKILL）：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框语法（`- [ ]`）跟踪进度。

**目标：** 产出一份经过视觉校验的中文 PPTX，深度解读 17 项有影响力的生理/时序基座模型工作，并把其中的经验转化为一条具体的前臂/腕部 sEMG 运动交互研究方案。

**架构：** 用一份来源 manifest 让每一条事实性论断都能追溯到官方会议、出版社或 arXiv 页面。单个 JavaScript 生成器借助 `@oai/artifact-tool`，把结构化的论文记录转成 16:9 演示文稿：每项工作三页幻灯片，备注中至少包含 500 个中文字符的讲解。交付前由渲染与 lint 环节逐页检查导出的幻灯片，确认可读性并排查版式缺陷。

**技术栈：** JavaScript ES modules、`@oai/artifact-tool`、内置 Node.js 运行时、PowerPoint/PPTX、基于 Poppler 的幻灯片渲染、Python 幻灯片检查。

---

### 任务 1：构建经过核验的来源 manifest

**文件：**
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\sources.md`
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\source-notes.txt`

- [ ] **步骤 1：为 17 篇选定论文各建一条记录**

使用 `D:\Fmodel\docs\superpowers\specs\2026-09-02-physiological-timeseries-foundation-models-ppt-design.md` 中的固定论文池。每条记录需包含标题、会议/年份、模态、数据规模、tokenizer/表示、backbone、目标函数、评测、可借鉴点，以及一个官方来源 URL。把 `Biosignal Fingerprinting / M2AE` 标注为预印本。

- [ ] **步骤 2：把所有来源 URL 记录到交付溯源文件中**

每个来源都使用下面这一确切格式，以便插入 PowerPoint 备注：

```text
LaBraM | ICLR 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html
```

- [ ] **步骤 3：核验论文池确有 17 项互不重复的工作，且所有 URL 均为一手来源**

运行：

```powershell
(Select-String -Path 'D:\Fmodel\deliverables\physio_fm_ppt\sources.md' -Pattern '^## ').Count
```

预期：`17`。

### 任务 2：撰写结构化解读内容与讲者备注文案

**文件：**
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\paper_records.js`

- [ ] **步骤 1：定义论文记录 schema**

每项工作都严格使用下面这一结构：

```js
{
  id: 'labram',
  title: 'LaBraM',
  venue: 'ICLR 2024',
  category: 'EEG / neural foundation model',
  slides: {
    positioning: { title: '问题与定位', bullets: [], diagram: [] },
    method: { title: '技术路线拆解', bullets: [], diagram: [] },
    evidence: { title: '证据与迁移判断', bullets: [], diagram: [] }
  },
  notes: { positioning: '', method: '', evidence: '' },
  sources: ['https://example.org']
}
```

- [ ] **步骤 2：每篇论文恰好写三组幻灯片内容**

每条记录的内容需回答：(a) 该工作为何存在、它填补了哪一表示空白；(b) 输入、patch/tokenizer/codebook（如有）、backbone、自监督损失、适配路径；(c) 实证证据、局限，以及对手前臂/腕部 sEMG 运动交互而言的 `可借鉴` 与 `不应照搬` 之处。

- [ ] **步骤 3：用中文撰写讲者备注并附上来源**

把每篇论文的三段备注拼接起来。去掉来源块后的文本至少要有 500 个中文字符。每段备注以如下内容结尾：

```text
[Sources]
- https://official-primary-source.example
[/Sources]
```

- [ ] **步骤 4：补充结尾对比章节的记录**

添加数据、tokenization、目标函数、评测、设计建议与投稿路线图这几类记录。设计建议须优先考虑连续多尺度时间编码、掩码重建、EMG 到未来运动学/IMU 的预测、选择性对比一致性，以及第二阶段的 VQ codebook 消融。

### 任务 3：生成 PowerPoint 演示文稿

**文件：**
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\build_deck.mjs`
- 新建：`D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx`

- [ ] **步骤 1：加载内置 artifact 运行时并启动 artifact 操作**

在第一条创建命令之前，先运行一次规定的操作标记：

```powershell
& $skillDir\container_tools\mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format pptx
```

用 `load_workspace_dependencies` 设置 `RUNTIME_NODE`、`RUNTIME_NODE_MODULES` 和 `RUNTIME_BIN_DIR`；在 `deliverables\physio_fm_ppt` 下创建一个 `node_modules` junction，指向 `RUNTIME_NODE_MODULES`。

- [ ] **步骤 2：实现一套可复用的 16:9 幻灯片系统**

在 `build_deck.mjs` 中实现 `addChrome(slide, section, page)`、`addPaperSlide(slide, record, kind)`、`addNotes(slide, text, urls)` 和 `addPipeline(slide, nodes)`。使用深海军蓝背景，以低饱和青色作为主强调色，暖橙色仅用于标注项目相关性的提示框。幻灯片标题字号设为 36 pt 及以上，正文 16 pt 及以上。每一页都带页码和章节标签。

- [ ] **步骤 3：按以下顺序生成全部所需幻灯片**

生成 5 页横版引导页、51 页论文页（3 × 17）和 6–10 页综述页。在 EEG/异构信号、可穿戴/通用时序、腕部 sEMG/运动交互和项目方案之间，插入清晰的章节分隔页。每页只保留一条逻辑主张，并使用紧凑的 PowerPoint 原生流程图，而不是复制论文插图。

- [ ] **步骤 4：导出并保存演示文稿**

运行：

```powershell
& $env:RUNTIME_NODE 'D:\Fmodel\deliverables\physio_fm_ppt\build_deck.mjs'
```

预期：在 `D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx` 生成 60–66 页的 PPTX。

### 任务 4：渲染、检查并修正演示文稿

**文件：**
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\rendered\`
- 新建：`D:\Fmodel\deliverables\physio_fm_ppt\slides_test_report.txt`

- [ ] **步骤 1：运行幻灯片结构检查**

运行：

```powershell
python 'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\presentations\26.813.12317\skills\presentations\scripts\slides_test.py' 'D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx'
```

预期：无文本溢出或无效幻灯片错误。

- [ ] **步骤 2：把所有幻灯片渲染为 PNG 并生成拼图**

运行：

```powershell
python 'C:\Users\User\.codex\plugins\cache\openai-primary-runtime\presentations\26.813.12317\skills\presentations\scripts\render_slides.py' 'D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx' --output_dir 'D:\Fmodel\deliverables\physio_fm_ppt\rendered'
```

检查拼图和每一页幻灯片图像。在 `build_deck.mjs` 中修正裁切、对比度不足、非预期重叠和留白失衡问题，然后重新生成并重复检查，直到全部通过。

- [ ] **步骤 3：核验内容数量标准**

检查：60–66 页幻灯片；17 个互不重复的论文章节；每项工作恰好三页深读幻灯片；所有记录都含来源块；每篇论文的三段备注合计都达到 500 中文字符阈值；无复用论文插图。把结果记录到 `slides_test_report.txt`。

### 任务 5：交付经过核验的成果物

**文件：**
- 修改：`D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx`
- 修改：`D:\Fmodel\deliverables\physio_fm_ppt\slides_test_report.txt`

- [ ] **步骤 1：确认最终产物存在且非空**

运行：

```powershell
Get-Item 'D:\Fmodel\deliverables\生理时序基座模型顶会论文深读_2026-09.pptx' | Select-Object Name,Length,LastWriteTime
```

预期：`Length` 大于零。

- [ ] **步骤 2：随范围与校验摘要一并交付演示文稿**

在报告中说明：该演示文稿覆盖 17 项工作，区分了同行评审论文与被标注的预印本，包含带来源链接的讲者备注，并已完成渲染与 lint 检查。交付回复中只给出最终 PPTX 的链接。

## 自检

1. **规格覆盖：** 任务 1–2 落实固定的 17 篇论文池、三页幻灯片格式、详细方法、备注与来源溯源。任务 3 落实 16:9 视觉系统与 60–66 页演示文稿。任务 4 覆盖视觉与结构 QA。任务 5 覆盖交付物与交接。设计规格中的每项要求都已分配。
2. **占位符扫描：** 本计划不含 `TODO`、`TBD`、“similar to”或延后实现的措辞。路径、命令、预期结果与内容 schema 均已明确。
3. **类型一致性：** `paper_records.js` 是论文各字段的唯一来源；`build_deck.mjs` 按任务 2 定义的 schema 消费 `record`、`kind`、备注与来源 URL。任务 3–5 的输出路径与 QA 路径保持一致。
