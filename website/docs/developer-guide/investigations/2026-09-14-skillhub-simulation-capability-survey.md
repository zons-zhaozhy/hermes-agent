# SkillHub 拟真能力调研报告（造数 / 拟人浏览器 / 拟人审查 / 提示词工程）

日期：2026-09-14
背景：实际工作需要 hermes 搞定①模拟真实数据②模拟真人审查分析③模拟真人操作浏览器。调研 SkillHub（skills.sh + lobehub）可装/可内化的资产。

## 搜索面（skills.sh 注册表，实测）

| 关键词 | 结果质量 |
|---|---|
| fake data generation | 0 结果 |
| synthetic data | 全部 1-install 杂项（canopy/kaizen/rse-plugins），无方法论 |
| browser automation | 15+ 结果，头部=vercel agent-browser(159)/midscene(未标)/borghei qa(82) |
| prompt engineering | 头部=wshobson/agents 系(213+87+98)，源头仓已在本地 |
| persona simulation | 多为 coval 专用域；有价值=论文评审模拟/客户面板模拟/合成用户 |

hub 通道实测：search/inspect 单次分钟级甚至挂死（inspect 两笔 8 分钟未返回即放弃，改 gh API 直读源仓原文，全部成功）。

## 候选评估（全部读了 SKILL.md 原文，非凭 description）

### 方向一：模拟真实数据 —— hub 无可取，本地已重度覆盖
本地已有体系：bank-demo-dataset-fabrication（以假乱真标准+分层架构+交叉一致性）、oracle-free-mock-bank-seed、cdc_aml-upstream-simulation-etl（先读指标 SQL 原文再造数+码值对齐）、mysql-partition-bulk-data-ops（造数纪律）、indic-batch-execution（码值族/金额族/时间分布族改值法）。
结论：不装不引，方向已自研成熟。

### 方向二：拟人浏览器操作 —— 不装，内化两个模式
1. vercel-labs/agent-browser（159 installs）：Rust CLI 驱动浏览器，能力层与 hermes 现有 browser_exec（Browser Use CLI + CDP）同层重叠。不引入。
2. web-infra-dev/midscene-skills（字节 web-infra 官方）：核心增量=**拟真操作层**——
   - LLM 视觉定位动作（aiTap/aiHover/aiQuery），按屏幕坐标+语义双驱动，不依赖测试专钩子；
   - 动作间注入人类级延迟与滚动节奏（requestAnimationFrame 节拍），对抗机器人特征检测；
   - 数据驱动而非脚本驱动：一套流程吃多份账号/数据表。
   绑定 midscene 工具链，不装；但「拟人节奏+视觉驱动+数据驱动」三件套是本地缺的模式。
3. borghei/claude-skills qa-browser-automation（82 installs）：**每个交互四条阴影路径**（happy/nil输入/空输入/上游错误）、五断点响应式（320/768/1024/1440/1920）、P0-P4 分级+0-100 健康评分基线棘轮。绑定 Chrome MCP，不装；四阴影路径+评分基线值得并入本地 E2E 技能。

### 方向三：拟人审查分析 —— 本次最有价值发现，模式内化
三个高质量 skill 的共性范式（均与领域解耦，可直接泛化）：
1. shaishavmaisuria/research-paper-lifecycle-skills simulate-reviewers：venue 校准评审团（严苛度按目标场合标定）+ persona 驱动弱点猎杀 + rubric 量化评分 + 决策风险带输出；**每个事实带四档置信标签**（verified-live/corroborated/inferred-from-family/needs-verification），「无来源的数字当 bug 处理」。
2. OneWave-AI/claude-skills prospect-panel-simulator：**冷态拟人**——模拟对象必须低上下文、低信任、手里已有替代品（「慈善阅读的模拟对象毫无用处」）；审查漏斗=3秒扫标题→快速浏览→具体异议（用对象的原话）→信任检查（AI味/过度承诺）→裁决+真实概率。
3. pedroromeroluna/ai-first-product-skills synthetic-users：**第一律「合成验证不了任何东西」**（开场与收尾强制声明）+ 每个拟人特质显式标注 (evidence)/(assumption)，无标注不进档案 + **强制至少一名怀疑者**（只用过就放弃的人/有商务异议的人/根本没这问题的人）+「我喜欢」必须附带成本否则视为引导性缺陷。
本地对照：llm-prompt-asset-governance 已有「模拟真人评审关卡」单点，decision-framework 有四层协作，但上述「冷态+证据标注+怀疑者强制+漏斗」的完整范式未成体系。

### 方向四：提示词工程 —— 已覆盖
hub 装机量第一的 wshobson/agents 系 prompt 技能，源头仓 MIT 内容已在本地（prompt-engineering + prompt-engineering-patterns，含 source 标注）。无增量。

## 结论

| 方向 | 处置 |
|---|---|
| 模拟真实数据 | 本地已覆盖，不动 |
| 拟人浏览器 | 不装（工具链绑定）；拟人节奏/四阴影路径/数据驱动内化进新 skill |
| 拟人审查 | 不装（专用域）；冷态拟人+证据标注+怀疑者+漏斗四范式内化进新 skill |
| 提示词工程 | 本地已覆盖，不动 |

内化落点：~/.hermes/skills/engineering-methodology/human-simulation-playbook/（新建，已查重无同类）。
