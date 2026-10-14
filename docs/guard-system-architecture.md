# 门禁体系架构（分层 · 段预算纪律 · 重注入通道）

> 本文件是 `plugins/guards` 与 `plugins/discipline` 两个套件的架构依据文档——两个套件
> 的模块 docstring 均指向本文件。数据口径：2026-10-14 实测。
> 已查重：全库 search_files guard-system-architecture 命中的仅本文件与两个套件 docstring 的引用。

## 一、分层

| 层 | 套件 / 插件 | 职责 | 子守卫 |
|---|---|---|---|
| L0 | `guards.tool_blacklist` | 必败调用机械黑名单——拦的不是「错」，而是「注定失败」 | — |
| L2 | `guards` | 写入质量护栏：写前功课 + 写动作合规 | pre_write / four_axis / duplicate_check / source_write / coding_standards / diff_debt / casebook_gate / batch_write / scope_gate / groundbreaking_gate |
| L3 | `discipline` | 执行纪律护栏：怎么跑命令、怎么面对错误 | no_guessing / db_safety / error_discipline / curl_safety / log_first / tool_safety / no_bypass / no_pushback / test_discipline + 注入通道 rule_reinjection / section_reinjection |
| 独立 | scientific-programming-guard；terminology-guard；enterprise-quality-gate；requirement-interview | 专项规范（写码预算、用词一致、企业级对表、需求歧义） | — |

裁定原则：同一层同性质才收拢进同一套件；跨层不合并。子模块 register 逐个调用，
单个失败仅告警不连坐。

## 二、两条注入通道（事前）

1. **system prompt 稳定段**（`ctx.register_system_prompt_section(id, renderer)`）
   - 会话起点渲染并冻结，属缓存前缀；段 id 仅小写字母/数字/下划线。
   - 预算：`MAX_SYSTEM_PROMPT_SECTIONS` = 32 段、`MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS` = 8000 字符；
     超预算的段被跳过并告警（fail-open，不阻断会话）。
   - 渲染器返回空串即该段不注入（子代理/批处理会话按平台排除，段体诱导空转）。
2. **重注入通道**（`pre_llm_call`，追加到当前用户消息，不动缓存前缀）
   - `rule_reinjection`：`.hermes-rules.md` 摘要，首注 + 内容变更即注 + 每 10 轮补注。
   - `section_reinjection`：段库版本指纹变更即注 + 每 10 轮补注 + **首轮不注入**
     （起点已由 system prompt 段承载同一内容，首轮再注入=纯重复）。

## 三、段预算纪律（用户裁定：只给「高频 × 可预判」的加段）

准入判据——两条同时满足：

- **高频**：实测拦截次数靠前（近期会话可数，非猜测）。
- **可预判**：判据一句话说得清，且能落成数字阈值 / 命名模板 / 豁免条件。

不入段的反例：

- 需读现场上下文才能判的（例如 error_discipline 的「连续错误未诊断」要结合工具序列）。
- 覆盖率驱动的（「每个子护栏都该有段」不是理由）。
- 模型已会的一般工程常识（注入即消耗每会话字符预算）。

段文本三要素，缺一即与拦截判据脱钩：**判据 + 数字阈值 + 豁免 + 后果**。

## 四、段文本单一事实源

- 段库：`prompt_sections/sections.json`，形如 `{"<id>": "<段文本>"}`；键即注册 id（1:1，禁另立映射表）。
- 读取与对账：`plugins/prompt_sections.py`（`load_section` / `fingerprint` / `section_ids` / `render_all`）。
  渲染器每次调用读盘——段修改对新会话立即生效。
- 现有段（2026-10-14 实测共 9 段渲染，其中文件型段 6 个）：
  `guards_preflight` 773 / `tool_blacklist` 404 / `db_safety` 222 / `test_discipline` 272 /
  `scientific_programming` 396 / `diff_debt` 216 字符；连同 terminology_consistency、
  enterprise_quality_gate、requirement_interview，段总量约 3868 字符（上限 8000）。
- 段库缺失或键名不符：`load_section` 告警一次并返回空串（该段本轮不注入）；
  指纹为空时重注入通道整体跳过（不注入半份内容）。

## 五、事前注入与事后拦截的关系

同一判据两处落地，**禁两套口径**：段文本写什么，检错逻辑就查什么；改判据必须同批改段文本，
否则段从提示退化为误导。回归测试以关键词 + 数字锁段文本：
`tests/plugins/test_guards_preflight_section.py`（段库键齐备、渲染内容含阈值、
重注入节奏：首轮不注入 / 指纹变更即注入 / 第 10 轮补注）。

## 六、生效条件（实测坑）

- 段在会话起点冻结：改段对新会话生效；对在跑的会话靠 `section_reinjection` 按指纹补注。
- 长驻进程持有旧模块：插件代码改动需新进程才加载。实测：进程 11:15 启动、`diff_debt`
  段 16:15 提交 → 同一会话渲染日志始终 4 段，而同刻新起的 cron 进程 5 段。
  结论：改门禁后须开新会话/新进程才全量生效，长期在跑的会话是注入盲区。
