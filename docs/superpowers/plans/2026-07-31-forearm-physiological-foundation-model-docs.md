# 前臂生理基座模型研究包实施计划

> **面向 agentic workers：** 必需子技能（REQUIRED SUB-SKILL）：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实施本计划。步骤使用复选框语法（`- [ ]`）跟踪进度。

**目标：** 构建一个以原始文献为依据的中文研究包，深入讲解 EEG 及其他生理基座模型，并把其中的经验落实为具体的前臂可穿戴生理基座模型设计。

**架构：** 研究包将证据收集与综合写作分离。结构化文献矩阵与 BibTeX 文件构成证据层；八份聚焦的 Markdown 文档消费这些证据，并相互交叉链接。推荐模型是肌肉中心、因子化的多模态编码器，以通用时间序列模型作为基线，离散神经肌肉 tokenization 留待后续工作。

**技术栈：** Markdown、CSV、BibTeX、PowerShell 校验，以及来自出版商页面、会议论文集、PubMed、arXiv 与官方仓库的一手学术来源。

---

## 文件结构

- `docs/research/README.md`：研究包索引、阅读路径与证据状态图例。
- `docs/research/00_研究地图与入门指南.md`：入门地图、术语与推荐学习顺序。
- `docs/research/01_EEG基座模型现状与方法.md`：EEG 模型分类体系与详细的模型对比。
- `docs/research/02_其他生理与时间序列基座模型.md`：ECG、PPG、可穿戴多模态与通用时间序列基座模型。
- `docs/research/03_EMG与前臂感知研究现状.md`：EMG 与非 EMG 的前臂感知全景。
- `docs/research/04_公开数据集与可用资源.md`：数据集、许可证、模态、标签与适用性。
- `docs/research/05_前臂生理基座模型方案.md`：科学问题界定、架构、目标函数与训练阶段。
- `docs/research/06_评测基准与实验矩阵.md`：下游任务、split 协议、指标、基线与消融。
- `docs/research/07_分阶段研究路线图.md`：最小可行研究、论文级研究与长期扩展。
- `docs/research/literature_matrix.csv`：每条经过核实的工作或数据集占一行。
- `docs/research/references.bib`：可直接导入文献管理器的引文。

本计划中的 git 提交必须使用所列出的提交信息。仓库当前缺少作者身份信息；不要自行编造。只有用户提供仓库本地 Git 作者姓名与邮箱之后，提交步骤才可执行。

### 任务 1：确立研究包契约与术语台账

**文件：**
- 新建：`docs/research/README.md`

- [ ] **步骤 1：创建研究包索引**

撰写 `docs/research/README.md`，包含以下章节：

```markdown
# 前臂可穿戴生理基座模型研究包

## 研究范围
本研究包讨论 EEG 与其他生理信号基座模型，并据此设计面向前臂可穿戴神经肌肉信号的通用表示模型。运动解码、疲劳、力量、身份识别和信号质量检测均为下游任务，而不是模型定义本身。

## 阅读路径
1. 初学者：00 → 01 → 03 → 05 → 06 → 07
2. 模型研究者：01 → 02 → 05 → 06
3. 数据与实验负责人：03 → 04 → 06 → 07

## 证据状态
- 同行评审：已在期刊或正式会议发表。
- 预印本：仅有 arXiv、bioRxiv、medRxiv 或审稿中版本。
- 数据资源：数据集主页或数据论文。
- 推断：由多个来源综合得到的项目判断，不归因于单篇论文。

## 术语表
| 规范术语 | 首次定义 | 使用规则 |
|---|---|---|
| 生理基座模型 | physiological foundation model | 泛指生理信号上的大规模预训练与迁移模型 |
| 前臂生理基座模型 | forearm physiological foundation model | 本项目的研究对象，不提前创造模型缩写 |
| 表面肌电 | surface electromyography (sEMG) | 首次展开，后续使用 sEMG |
| 高密度表面肌电 | high-density surface electromyography (HD-sEMG) | 不与普通多通道 sEMG 混用 |
| 跨受试者 | cross-subject | 与跨会话、跨设备分别报告 |
| 冻结探测 | frozen probing | 编码器完全冻结，仅训练读取头 |
| 少样本适配 | few-shot adaptation | 必须报告每类样本数或标注比例 |

## 检索边界
检索截止日期：2026-07-31。优先引用原始论文、正式出版页面、官方数据集页面和官方代码仓库。
```

- [ ] **步骤 2：校验必需章节与术语**

运行：

```powershell
rg -n "研究范围|阅读路径|证据状态|术语表|检索边界|sEMG|HD-sEMG" docs/research/README.md
```

预期：每个必需章节与两个 EMG 术语都至少出现一次。

- [ ] **步骤 3：检查 Markdown 空白字符**

运行：

```powershell
git diff --check -- docs/research/README.md
```

预期：无任何输出。

- [ ] **步骤 4：提交研究包契约**

```powershell
git add -- docs/research/README.md
git commit -m "docs: define physiological foundation model research package"
```

### 任务 2：搭建经过核实的证据层

**文件：**
- 新建：`docs/research/literature_matrix.csv`
- 新建：`docs/research/references.bib`

- [ ] **步骤 1：创建文献矩阵表结构**

使用这个精确的 CSV 表头：

```csv
record_id,category,title,year,publication_status,venue,signal_modalities,pretraining_data,model_scale,input_representation,architecture,pretraining_objectives,adaptation_modes,downstream_tasks,generalization_axes,open_code,open_weights,primary_url,doi_or_identifier,key_contribution,key_limitation,relevance_to_forearm,last_verified
```

为下列分组填充经过核实的种子条目：

- EEG：BENDR、BIOT、LaBraM、EEGPT、CBraMod，以及至少一个 2026 年的 EEG 基座模型基准。
- 生理/时间序列：Apple 可穿戴 PPG/ECG 基座模型、PaPaGei 与 MOMENT。
- EMG：Meta 的通用非侵入式神经运动接口、AEMG、emg2pose 与 emg2qwerty。
- 前臂数据集：NinaPro、Hyser/HD-sEMG、HD-FW KIN，以及一个重复会话数据集。

对每一行，核实发表状态与主 URL。确实无法获取的元数据字段留空，不要猜测填写。

- [ ] **步骤 2：使用可复现的检索式**

按以下检索式族执行来源检索，并把所有保留的来源记录进矩阵：

```text
"EEG foundation model" AND (pretraining OR benchmark OR generalization)
(BENDR OR BIOT OR LaBraM OR EEGPT OR CBraMod) EEG
(ECG OR PPG OR wearable biosignal) AND "foundation model"
(surface EMG OR sEMG) AND (foundation model OR self-supervised pretraining)
(forearm OR wrist) AND (sEMG OR mechanomyography OR force myography OR ultrasound OR electrical impedance tomography)
(sEMG OR electromyography) AND (cross-subject OR cross-session OR electrode shift OR fatigue OR biometrics)
```

以出版商/会议论文集/PubMed/arXiv 页面作为证据。只有在原始论文可得时，才用综述文章做发现性检索。

- [ ] **步骤 3：创建经过核实的 BibTeX 条目**

创建 `docs/research/references.bib`，为 Markdown 研究包中引用的每篇论文各写一条条目。尽量使用基于 DOI 的键名，否则使用稳定的 `firstauthor_year_shorttitle` 键名。每条条目必须包含 `title`、`author`、`year`，以及 `doi`、`url` 或稳定的会议标识符三者之一。

- [ ] **步骤 4：校验矩阵结构**

运行：

```powershell
$rows = Import-Csv docs/research/literature_matrix.csv
$required = @('record_id','category','title','year','publication_status','primary_url','key_contribution','key_limitation','last_verified')
$missing = $required | Where-Object { $_ -notin $rows[0].PSObject.Properties.Name }
if ($missing.Count -gt 0) { throw "Missing columns: $($missing -join ', ')" }
if ($rows.Count -lt 16) { throw "Expected at least 16 verified seed records; found $($rows.Count)" }
```

预期：无输出且退出码为 0。

- [ ] **步骤 5：校验证据完整性**

运行：

```powershell
$rows = Import-Csv docs/research/literature_matrix.csv
$bad = $rows | Where-Object { -not $_.record_id -or -not $_.title -or -not $_.publication_status -or -not $_.primary_url -or -not $_.last_verified }
if ($bad.Count -gt 0) { $bad | Format-Table record_id,title; throw "Incomplete required evidence fields" }
```

预期：没有字段不完整的行。

- [ ] **步骤 6：校验 BibTeX 覆盖度**

运行：

```powershell
$entries = (rg -c '^@' docs/research/references.bib).Split(':')[-1]
if ([int]$entries -lt 12) { throw "Expected at least 12 BibTeX entries; found $entries" }
```

预期：至少 12 条条目。

- [ ] **步骤 7：提交证据层**

```powershell
git add -- docs/research/literature_matrix.csv docs/research/references.bib
git commit -m "docs: add verified biosignal foundation model evidence"
```

### 任务 3：撰写详细的 EEG 基座模型综述

**文件：**
- 新建：`docs/research/01_EEG基座模型现状与方法.md`

- [ ] **步骤 1：起草统领全文的论点**

以如下限定范围的论点开篇：

```markdown
EEG 基座模型的主要进展并非简单扩大参数规模，而是逐步解决跨数据集预训练中的格式异构：通道数量与位置不同、采样率和窗口长度不同、任务标签体系不同，以及信号噪声和缺失通道普遍存在。对前臂肌肉基座模型最有价值的经验，因此是输入统一、空间—时间建模、自监督目标与严格迁移评测，而不是照搬某个模型的规模。
```

- [ ] **步骤 2：补充方法分类体系**

为下列内容分别建立独立章节：

1. EEG 数据与格式异构。
2. 原始波形、频谱和离散 token 表示。
3. 卷积、Transformer、空间—时间分解和多尺度架构。
4. 对比预测、掩码重建、离散 token 预测和自回归目标。
5. 冻结探测、少样本适配、参数高效微调和完整微调。
6. 跨数据集、跨受试者、跨通道布局和临床迁移评测。

- [ ] **步骤 3：补充详细的模型档案**

对 BENDR、BIOT、LaBraM、EEGPT 和 CBraMod，报告：

- 发表状态与年份；
- 输入表示与通道处理方式；
- 骨干架构；
- 预训练语料与目标函数；
- 适配协议；
- 下游任务覆盖范围；
- 开源代码/权重状态；
- 最具迁移价值的思路；
- 主要局限。

附一张对比表，行是上述五个模型，列覆盖上述全部九个字段。

- [ ] **步骤 4：用基准结果修正模型宣称**

使用 2026 年的一手基准来源说明：冻结线性探针可能低估迁移能力、也可能暴露迁移薄弱；专用模型依然具备竞争力；通道适配依赖架构；参数量更大并不保证跨受试者泛化更好。将这些结论标注为基于证据的提醒，而非普遍规律。

- [ ] **步骤 5：每个主题以一条前臂迁移经验收尾**

添加一个名为 `## 对前臂肌肉基座模型的可迁移经验` 的收尾章节，覆盖：

- 灵活的通道 tokenization；
- 显式的设备与导联布局（montage）元数据；
- 空间与时间因子化；
- 重建、预测与对齐目标的混合；
- 冻结、少样本与完整微调的评测；
- 与紧凑专家基线的对比。

- [ ] **步骤 6：校验覆盖情况**

运行：

```powershell
rg -n "BENDR|BIOT|LaBraM|EEGPT|CBraMod|掩码|自回归|冻结探测|少样本|跨受试者|前臂" docs/research/01_EEG基座模型现状与方法.md
```

预期：每个术语都出现在实质性正文或对比表中。

- [ ] **步骤 7：提交 EEG 综述**

```powershell
git add -- docs/research/01_EEG基座模型现状与方法.md
git commit -m "docs: review EEG foundation model methods"
```

### 任务 4：综述其他生理与时间序列基座模型

**文件：**
- 新建：`docs/research/02_其他生理与时间序列基座模型.md`

- [ ] **步骤 1：按信号与建模问题组织综述**

使用以下章节：

- ECG 基座模型与自监督模型。
- PPG 基座模型。
- 可穿戴多模态模型。
- 通用时间序列基座模型。
- 跨模态迁移与缺失模态学习。
- 可迁移与不可迁移到前臂信号的经验。

- [ ] **步骤 2：对比代表性系统**

包含 Apple 可穿戴 PPG/ECG 工作、PaPaGei 与 MOMENT 的已核实档案。只有在能核实一手论文时，才补充经过同行评审的多模态生理模型。对比受试者级对比学习、形态学感知目标、掩码自编码、共享潜空间与模态专属编码器。

- [ ] **步骤 3：区分可迁移与不可迁移的前提假设**

明确说明：

- PPG/ECG 具有相对稳定的生理形态，而 sEMG 受电极位置与肌肉募集影响强烈。
- 受试者级正样本对可能保留与身份相关的生理信息，但会与身份不变的下游迁移相冲突。
- 通用时间序列模型提供了强架构基线，但可能忽略传感器物理与解剖结构。
- 当同步信号观测同一生理过程的不同环节时，多模态对齐最有用。

- [ ] **步骤 4：校验覆盖情况与证据链接**

运行：

```powershell
rg -n "ECG|PPG|Apple|PaPaGei|MOMENT|多模态|缺失模态|前臂" docs/research/02_其他生理与时间序列基座模型.md
```

预期：所有术语均出现，且每个被点名的模型都有一手来源链接。

- [ ] **步骤 5：提交跨模态综述**

```powershell
git add -- docs/research/02_其他生理与时间序列基座模型.md
git commit -m "docs: review physiological and time-series foundation models"
```

### 任务 5：梳理 EMG 与非 EMG 前臂感知

**文件：**
- 新建：`docs/research/03_EMG与前臂感知研究现状.md`

- [ ] **步骤 1：定义生理链条**

按各模态所观测的对象来组织：

```text
neural drive → muscle electrical activation → mechanical deformation/force → limb kinematics
                 sEMG/HD-sEMG          MMG/FMG/ultrasound/EIT       IMU/mocap
```

说明 EIT 主要观测阻抗分布，超声观测结构形变，而 IMU/运动学属于运动结果，而非直接的肌肉信号。

- [ ] **步骤 2：综述 sEMG 与 HD-sEMG**

覆盖运动单位动作电位、幅值与频谱特征、电极布局、串扰、皮肤阻抗、归一化、高密度空间图、运动单位分解与可穿戴约束。

- [ ] **步骤 3：综述前臂替代模态**

对 MMG、FMG/pFMG、超声/超声肌图/回声肌图、EIT 与 IMU，报告测量原理、时间与空间分辨率、可穿戴可行性、对贴放位置与运动的敏感性、深度敏感性、功耗/算力影响，以及在预训练中可能承担的角色。

- [ ] **步骤 4：综述 EMG 基座模型的前身与当前工作**

覆盖自监督表示学习、域适应、快速重校准、Meta 的人群规模通用神经运动接口，以及 2026 年的 AEMG 预印本。用明确标准区分任务专属的大型监督解码器与基座模型：广泛的预训练数据、可复用的编码器、多种适配设置与多样化的下游任务。

- [ ] **步骤 5：分析核心域偏移**

为下列每一项建立一个小节：

- 受试者解剖结构与肌肉募集；
- 会话与重新佩戴带来的漂移；
- 电极旋转、平移与通道失效；
- 设备与采样率差异；
- 姿态、负载与收缩强度的混杂；
- 疲劳与时间非平稳性；
- 身份信息与隐私。

- [ ] **步骤 6：校验模态与风险覆盖情况**

运行：

```powershell
rg -n "sEMG|HD-sEMG|MMG|FMG|超声|EIT|IMU|疲劳|身份|电极|跨会话|AEMG" docs/research/03_EMG与前臂感知研究现状.md
```

预期：每个模态术语与域偏移术语都出现。

- [ ] **步骤 7：提交前臂感知综述**

```powershell
git add -- docs/research/03_EMG与前臂感知研究现状.md
git commit -m "docs: map EMG and forearm sensing research"
```

### 任务 6：编写公开数据集与资源指南

**文件：**
- 新建：`docs/research/04_公开数据集与可用资源.md`

- [ ] **步骤 1：定义数据集表结构**

使用以下列：

```markdown
| 数据集 | 受试者/会话 | 模态与通道 | 采样率 | 同步标签 | 主要任务 | 泛化价值 | 许可/访问 | 主要限制 |
```

- [ ] **步骤 2：补充经过核实的数据集**

包含 NinaPro、Hyser/HD-sEMG、emg2pose、emg2qwerty、HD-FW KIN、一个人口多样性数据集与一个重复会话数据集。只有在受试者数量、模态与访问条款已从数据论文或官方页面核实之后，才添加其他数据集。

- [ ] **步骤 3：在不虚构精度的前提下评定适用性**

对每个数据集，针对下列用途给出分档适用性——高、中或低：

- 无标签预训练；
- 运动解码；
- 力/运动学；
- 疲劳；
- 身份；
- 跨会话迁移；
- 跨设备迁移。

对每一项非显而易见的评级说明理由。在没有经验证的加权方案之前，不要构造数值化的综合评分。

- [ ] **步骤 4：提出首个公开数据语料库方案**

定义三个层级：

1. 广泛的 sEMG 表示预训练。
2. 同步的 sEMG–运动学/力 生理锚定。
3. 重复会话与人口统计学压力测试。

对每个层级，列出候选数据集、必需的前处理元数据与许可证兼容性顾虑。

- [ ] **步骤 5：校验数据集覆盖情况**

运行：

```powershell
rg -n "NinaPro|Hyser|emg2pose|emg2qwerty|HD-FW KIN|疲劳|身份|跨会话|许可" docs/research/04_公开数据集与可用资源.md
```

预期：所有被点名的数据集与用例均出现。

- [ ] **步骤 6：提交数据集指南**

```powershell
git add -- docs/research/04_公开数据集与可用资源.md
git commit -m "docs: catalogue public forearm biosignal datasets"
```

### 任务 7：细化前臂生理基座模型设计

**文件：**
- 新建：`docs/research/05_前臂生理基座模型方案.md`

- [ ] **步骤 1：陈述任务与不做的主张**

使用如下表述框架：

```markdown
目标是学习可在多个前臂生理任务间迁移的表示，而不是建立一个将所有肌肉与所有传感模态统一起来的全身模型。第一阶段以 sEMG 为核心部署输入，IMU、运动学和力量作为优先生理锚点；MMG/FMG、超声和 EIT 按数据可得性逐步接入。
```

- [ ] **步骤 2：细化输入契约**

在概念层面定义信号张量、通道掩码、时间戳与元数据。要求显式给出采样率、通道身份或相对位置、设备、会话与受试者标识。说明哪些标识可以进入编码器，哪些仅用于划分/审计，以及在元数据缺失时部署行为如何。

- [ ] **步骤 3：细化模型模块**

为每个模块记录动机、机制与消融作用：

- 模态专属信号适配器；
- 通道与空间元数据编码器；
- 多时间尺度共享编码器；
- 动态生理状态分支；
- 身份/解剖分支；
- 设备/会话干扰分支；
- 任务专属头；
- 可选的跨模态教师编码器。

- [ ] **步骤 4：细化训练阶段**

分四个阶段：

1. 单模态的掩码与预测式 sEMG 预训练。
2. 通道 dropout、采样率变化与设备条件化鲁棒性。
3. 与运动学、力、IMU 及可用肌肉模态的跨模态对齐。
4. 用动态、身份与设备/会话目标进行因子化，并配合泄漏探针。

- [ ] **步骤 5：细化各目标的作用与失效检查**

对时间掩码、通道掩码、潜变量预测、跨模态对齐、跨会话一致性与因子化，分别说明：

- 它应当改善哪种表示；
- 它可能引入哪种捷径；
- 哪个消融实验检验其必要性；
- 哪个探针能检测其失效。

- [ ] **步骤 6：定义最小可行配置与扩展配置**

最小可行配置：

- 仅 sEMG 的适配器；
- 紧凑的多尺度编码器；
- 时间掩码与通道掩码；
- 运动、疲劳与身份探针；
- 跨受试者与跨会话划分。

扩展配置：

- 同步的 IMU/运动学/力对齐；
- 因子化潜变量分支；
- MMG/FMG 或超声教师模态；
- 参数高效个性化；
- 端侧部署测量。

- [ ] **步骤 7：校验模块覆盖情况**

运行：

```powershell
rg -n "模态适配器|多时间尺度|动态生理|身份|设备/会话|掩码|跨模态|泄漏|最小可行|扩展配置" docs/research/05_前臂生理基座模型方案.md
```

预期：每个架构组件与训练组件都出现。

- [ ] **步骤 8：提交模型方案**

```powershell
git add -- docs/research/05_前臂生理基座模型方案.md
git commit -m "docs: design forearm physiological foundation model"
```

### 任务 8：定义评测基准与实验矩阵

**文件：**
- 新建：`docs/research/06_评测基准与实验矩阵.md`

- [ ] **步骤 1：先定义 split 协议，再定义指标**

明确以下内容：

- 受试者不相交的 split（subject-disjoint）；
- 会话不相交的 split（session-disjoint）；
- 设备不相交的 split（device-disjoint）；
- 数据集不相交的迁移（dataset-disjoint transfer）；
- 在数据允许时，对受试者与任务同时留出（joint subject-and-task holdout）。

要求先划分再切窗，并且只用训练数据拟合预处理。

- [ ] **步骤 2：定义下游任务与指标**

采用：

- 手势/动作：macro-F1、balanced accuracy 与每类召回率；
- 运动学/力：MAE、RMSE、相关系数与时延感知误差；
- 疲劳：受试者不相交的 AUROC 或 macro-F1 加校准，当存在连续真值时补充依赖协议的回归指标；
- 身份验证：ROC-AUC 与等错误率（equal error rate）；
- 身份识别：top-1 与 macro-F1；
- 信号质量/异常：AUROC、AUPRC 与误报率。

- [ ] **步骤 3：定义适配模式**

对比冻结探针、少样本适配、参数高效适配、完整微调与从零训练的专家模型。对每种模式报告可训练参数量、校准数据与算力开销。

- [ ] **步骤 4：定义基线族**

包含：

- 手工 EMG 特征配线性模型或树模型；
- 紧凑 CNN 或时间卷积专家模型；
- 从零训练的 Transformer；
- 通用时间序列预训练模型；
- 受 EEG 启发的灵活通道模型；
- 所提出的因子化模型。

- [ ] **步骤 5：定义消融与压力测试**

测试移除通道元数据、多尺度建模、跨模态对齐、身份分支、设备/会话分支以及每个目标函数。压力测试必须包含通道丢失、电极旋转/偏移、幅值缩放、加性噪声、重采样、长时漂移与教师模态缺失。

- [ ] **步骤 6：定义泄漏检查与阴性对照**

包含打乱标签对照、数据集身份探针、窗口相邻性检查、受试者模板泄漏检查、针对疲劳的时间顺序混杂检验，以及针对疲劳分类的动作强度对照。

- [ ] **步骤 7：构建主张—证据矩阵**

增加一张表，把每条拟提出的主张映射到所需证据。至少包含：

```markdown
| 主张 | 必要证据 | 失败判据 |
|---|---|---|
| 表征可跨任务迁移 | 至少三个不同任务族的冻结或少样本提升 | 仅完整微调有效 |
| 表征可跨人泛化 | 受试者完全隔离的比较 | 随机窗口划分才有提升 |
| 模型具有设备鲁棒性 | 设备隔离或受控设备扰动 | 设备识别探针主导表示 |
| 表征包含疲劳信息 | 控制动作与强度后的疲劳预测 | 仅按时间顺序即可预测 |
| 身份与动态状态可分解 | 子空间交叉探针和消融 | 动态分支仍高精度识别身份 |
```

- [ ] **步骤 8：校验评测完备性**

运行：

```powershell
rg -n "subject-disjoint|session-disjoint|device-disjoint|macro-F1|RMSE|equal error rate|冻结|少样本|从零训练|通道缺失|电极|泄漏|失败判据" docs/research/06_评测基准与实验矩阵.md
```

预期：所有协议、指标族与风险控制措施均出现。

- [ ] **步骤 9：提交基准设计**

```powershell
git add -- docs/research/06_评测基准与实验矩阵.md
git commit -m "docs: define forearm foundation model benchmark"
```

### 任务 9：撰写路线图与入门研究地图

**文件：**
- 新建：`docs/research/07_分阶段研究路线图.md`
- 新建：`docs/research/00_研究地图与入门指南.md`

- [ ] **步骤 1：定义 Phase 0——复现与审计**

Phase 0 必须在两个公开 sEMG 数据集上复现代表性基线，核实受试者/会话划分，并建立统一的前处理与评测框架。只有在泄漏检查与从零训练的基线都稳定之后，才可通过。

- [ ] **步骤 2：定义 Phase 1——最小可行基座模型**

Phase 1 使用仅 sEMG 的预训练，配合时间掩码与通道掩码。在跨受试者与跨会话划分下评测运动、疲劳与身份探针。只有当预训练在冻结或少标签设定下改善了至少两个任务族，且不依赖随机窗口划分时，该阶段才算成功。

- [ ] **步骤 3：定义 Phase 2——生理锚定与因子化**

Phase 2 加入同步的运动学、力或 IMU 对齐，并分离动态、身份与设备/会话子空间。成功要求迁移或鲁棒性得到改善，并具备来自泄漏探针与分支消融的证据。

- [ ] **步骤 4：定义 Phase 3——多模态与部署扩展**

Phase 3 评估将 MMG/FMG、超声或 EIT 作为额外输入或仅训练时使用的教师模态，评估参数高效个性化、缺失模态行为与端侧时延/内存。在 Phase 2 通过其判据之前，本阶段是可选的。

- [ ] **步骤 5：补充按资源规模分档的计划表**

给出三档计划，不假设具体硬件：

- 轻量：公开数据集、紧凑模型、仅 sEMG；
- 标准：多个公开数据集、同步生理锚点、因子化模型；
- 进取：新的纵向采集、多模态教师与部署研究。

对每档计划列出交付物、依赖关系、主要风险与停止/继续判据。

- [ ] **步骤 6：最后撰写入门地图**

在 `00_研究地图与入门指南.md` 中，把已完成的研究包综合为：

- 生理基座模型是什么、不是什么；
- EEG 的核心经验；
- 前臂感知模态；
- 推荐的论文阅读顺序；
- 推荐的动手复现顺序；
- 十个高价值研究问题；
- 与 `README.md` 交叉链接的术语表；
- 指向文档 01–07 的链接。

- [ ] **步骤 7：校验路线图与交叉链接**

运行：

```powershell
rg -n "Phase 0|Phase 1|Phase 2|Phase 3|退出|成功|停止|轻量|标准|进取" docs/research/07_分阶段研究路线图.md
rg -n "01_EEG|02_其他生理|03_EMG|04_公开数据集|05_前臂生理|06_评测|07_分阶段" docs/research/00_研究地图与入门指南.md
```

预期：所有阶段、决策判据与文档链接均出现。

- [ ] **步骤 8：提交路线图与研究地图**

```powershell
git add -- docs/research/00_研究地图与入门指南.md docs/research/07_分阶段研究路线图.md
git commit -m "docs: add forearm foundation model research roadmap"
```

### 任务 10：整合、审计并定稿研究包

**文件：**
- 修改：`docs/research/README.md`
- 修改：`docs/research/00_研究地图与入门指南.md`
- 修改：`docs/research/01_EEG基座模型现状与方法.md`
- 修改：`docs/research/02_其他生理与时间序列基座模型.md`
- 修改：`docs/research/03_EMG与前臂感知研究现状.md`
- 修改：`docs/research/04_公开数据集与可用资源.md`
- 修改：`docs/research/05_前臂生理基座模型方案.md`
- 修改：`docs/research/06_评测基准与实验矩阵.md`
- 修改：`docs/research/07_分阶段研究路线图.md`
- 修改：`docs/research/literature_matrix.csv`
- 修改：`docs/research/references.bib`

- [ ] **步骤 1：审计需求覆盖情况**

确认研究包回答了：

1. 当前 EEG 与生理基座模型使用了哪些方法。
2. 哪些经验可以迁移到前臂肌肉感知，哪些不能。
3. 所提出的因子化多模态模型如何组织与训练。
4. 实验如何区分生理迁移与数据集捷径。

把缺失的链接或说明补到应负责的文档中，而不是重复大段内容。

- [ ] **步骤 2：审计术语一致性**

运行：

```powershell
rg -n "surface EMG|sEMG|HD-sEMG|生理基座模型|前臂生理基座模型|冻结探测|少样本适配" docs/research/*.md
```

预期：缩写在各独立文档中首次出现时均已展开，且规范术语保持一致。

- [ ] **步骤 3：审计无依据的主张与发表状态**

搜索绝对化的新颖性与普适性措辞。除非一手证据直接支持该表述，否则改为限定范围的措辞。正文中引用的每篇预印本都必须标注 `预印本`，每个模型表都必须包含发表状态。

运行：

```powershell
rg -n "首次|唯一|全面|彻底|总是|从不|state-of-the-art|SOTA" docs/research/*.md
```

预期：每处匹配要么作为原作者的主张标注归属，要么被限定范围，要么删除。

- [ ] **步骤 4：审计本地 Markdown 链接**

运行：

```powershell
$files = Get-ChildItem docs/research -Filter *.md
$broken = @()
foreach ($file in $files) {
  $content = Get-Content -Raw $file.FullName
  foreach ($m in [regex]::Matches($content, '\]\((?!https?://|#)([^)]+\.md)(?:#[^)]+)?\)')) {
    $target = Join-Path $file.DirectoryName $m.Groups[1].Value
    if (-not (Test-Path -LiteralPath $target)) { $broken += "$($file.Name) -> $($m.Groups[1].Value)" }
  }
}
if ($broken.Count -gt 0) { $broken; throw "Broken local Markdown links" }
```

预期：无失效链接。

- [ ] **步骤 5：审计一手来源 URL**

把文档 01–04 中每一个被点名的论文与数据集与 `literature_matrix.csv` 对照。补上任何缺失的记录与 BibTeX 条目。确保保留的每一行矩阵都有主 URL 且核验日期为 2026-07-31。

- [ ] **步骤 6：运行全局文件与空白检查**

运行：

```powershell
$expected = @(
  'README.md',
  '00_研究地图与入门指南.md',
  '01_EEG基座模型现状与方法.md',
  '02_其他生理与时间序列基座模型.md',
  '03_EMG与前臂感知研究现状.md',
  '04_公开数据集与可用资源.md',
  '05_前臂生理基座模型方案.md',
  '06_评测基准与实验矩阵.md',
  '07_分阶段研究路线图.md',
  'literature_matrix.csv',
  'references.bib'
)
$missing = $expected | Where-Object { -not (Test-Path -LiteralPath (Join-Path 'docs/research' $_)) }
if ($missing.Count -gt 0) { throw "Missing deliverables: $($missing -join ', ')" }
git diff --check -- docs/research
```

预期：所有交付物均存在，且 `git diff --check` 报告无错误。

- [ ] **步骤 7：更新研究包索引**

为每个交付物在 `docs/research/README.md` 中添加一句说明与相对链接。基于 `literature_matrix.csv` 说明最终来源总数、同行评审论文数、预印本数与数据资源数。

- [ ] **步骤 8：提交整合后的研究包**

```powershell
git add -- docs/research
git commit -m "docs: finalize forearm physiological foundation model research package"
```

- [ ] **步骤 9：汇报完成证据**

在最终交接中记录：

- 确切的文件链接；
- 文献矩阵行数；
- 同行评审/预印本/数据集的数量；
- 所运行的校验命令及其结果；
- 尚未解决的证据缺口（如有）；
- 来自文档 07 的推荐首个实验。
