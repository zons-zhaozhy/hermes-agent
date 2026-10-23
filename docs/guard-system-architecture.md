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

## 七、拦截密度治理（2026-10-24 数据驱动收敛）

数据画像（outcomes.db 全量）：plugin_block 47,800 次 / 调用 515,337 次（9.3%）。
按工具：patch 18,451 / terminal 11,642 / write_file 7,526（合计 79%）；按护栏：
diff_debt 17,517 / four_axis 5,356 / tool_safety-read 1,897 / pre_write 2,673。

四大误拦源与根修（commit 3722a5969c）：

| 误拦源 | 根因 | 根修 |
|---|---|---|
| diff_debt | 工具结果已自带 diff 回显的写仍被记欠账——义务双重征收（回显+正文两份） | 结果含 diff 形态（`_RESULT_DIFF_RE`，无锚定子串）不记账；空路径欠账不记（无法出示对照的死锁对象） |
| tool_safety Check2 | 读是幂等无副作用操作，连续失败 3 次永久拦；样本全在 TCC 权限区，拦截指引 execute_code open() 同样必败 | 留日志放行；无进展循环由 block_escalation 观测层兜底 |
| four_axis | marker 时效 600s < 单任务写链时长（含跑测+多笔 patch），中途过期重拦 | 1800s；仍保留时效 + gate 每 turn 重写刷新 |
| source_write | `_SAFE_PREFIXES` 缺 `~/.hermes/cache/`（TMPDIR 官方临时区，24h 清理） | 加入豁免，与 /tmp 同语义 |

### 两条体系级不变量（本轮确立）

1. **读工具零拦截**：read_file/search_files/skill_view 等只读工具任何护栏不得 block。
   读失败无副作用、无破坏面，拦截的代价（每拦=一轮 API 白跑）恒大于收益。
   观测型检测（重试循环、空结果）一律降为日志。
2. **欠账必须可清偿**：任何「记账→拦截」结构的门禁，记账时必须保证存在可出示的
   清偿证据形态（如 diff 对照）。无法清偿的欠账（空路径、对照上下文已丢失）记入
   即死锁——记账前判定，而非拦截后补救。此原则扩展适用一切 fail-open 出口设计：
   开口 fail-open + 加锁 fail-open = 只进不出死锁。

### 拦截密度预算（后续守卫改动对表）

单一守卫的拦截中「误拦」（无真实风险损失的拦截）占比应趋零；误拦 > 真拦的守卫
优先重审判定面而非加豁免标记。每次守卫修改后跑
`sqlite3 ~/.hermes/outcomes.db "SELECT tool_name, error_type, COUNT(*) FROM tool_outcomes WHERE status='blocked' GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20;"`
对表拦截构成变化。

### 反方审查五修（2026-10-24，四修的证伪复核）

反方审查子代理对四修逐条证伪，五条高置信缺陷全部本人读码复核属实后修复：

| # | 缺陷 | 根因 | 根修 |
|---|---|---|---|
| D1 | 四修声称的「block_escalation 失败观测兜底层」不存在——`_on_post_tool_call` 只处理 blocked/ok，error 状态零分支；且 read_file 指纹恒 ()（不计数），读通道结构性失明 | 放行政策成立但兜底承诺是虚构的 | `_on_post_tool_call` 补 error 分支（与 blocked 同指纹计数，成功清账）；`_intent_fingerprint` 补 read_file/search_files 指纹（工具名+路径）——error 无进展循环现被 DB 观测并升级提示（输出层，非拦截，不变量保持） |
| D2 | 回显豁免「结果非模型可控」断言对 skill_manage 为假——其 result 含 frontmatter description 前 120 字符（模型自写），塞 `@@ -1` 子串即可静音欠账 | 豁免未按工具的 result 构造方分型 | `_RESULT_DIFF_RE` 豁免限定 write_file/patch（框架 diff 字段）；skill_manage 一律记账 |
| D3 | 段库「write_file/patch 正常调用不产生欠账」与新文件矛盾——新文件 old_content 空恒 diff=None，照记欠账 | 文本把「修改已有文件」误写成全称 | 段库改精确：修改已有文件（有 diff 回显）豁免；新文件仍须正文贴 git add -N + git diff（编辑纪律原义） |
| D4 | cache 豁免前缀 import 期 `expanduser("~/.hermes/")` 冻结默认 home——profile 会话 TMPDIR 不在冻结前缀内，误拦原样存在（PR #3575 同类） | 插件路径未与 `get_hermes_home()` 同源解析 | `_hermes_cache_prefix()` 每次调用经 `get_hermes_home()` 现算（延迟解析，four_axis marker 同模式） |
| D5 | 段库承诺「terminal 格式化器改文件欠账」但 diff_debt 的 `_WRITE_TOOLS` 不含 terminal——承诺无实现 | 段文本超出实现范围 | 段库删该表述，欠账工具集合与 `_WRITE_TOOLS` 严格一致 |

复核方法备忘：子代理结论逐条读码验证后才修（D1 三处查证：block_escalation error
无分支、_intent_fingerprint 恒 ()、error_discipline 仅执法 terminal）；其中 item 3
（four_axis TTL）与 item 5（不变量冲突）证伪不成立——marker 每 turn 清除+turn 内
重写实测成立，DevilAdvocate 拦读已于 10-22 修复（afb2cee4e6）。
