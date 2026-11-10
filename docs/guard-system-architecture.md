# 门禁体系架构（分层 · 段预算纪律 · 重注入通道）

> 本文件是 `plugins/guards` 与 `plugins/discipline` 两个套件的架构依据文档——两个套件
> 的模块 docstring 均指向本文件。数据口径：2026-10-14 实测。
> 已查重：全库 search_files guard-system-architecture 命中的仅本文件与两个套件 docstring 的引用。

## 一、分层

| 层 | 套件 / 插件 | 职责 | 子守卫 |
|---|---|---|---|
| L0 | `guards.tool_blacklist` | 必败调用机械黑名单——拦的不是「错」，而是「注定失败」 | — |
| L2 | `guards` | 写入质量护栏：写前功课 + 写动作合规 | pre_write / four_axis / duplicate_check / source_write / coding_standards / diff_debt / casebook_gate / batch_write / scope_gate / groundbreaking_gate / slow_search / db_target_audit / plain_vocab / experience_gate（2026-10-26 实测共 17 子模块） |
| L3 | `discipline` | 执行纪律护栏：怎么跑命令、怎么面对错误 | no_guessing / db_safety / error_discipline / curl_safety / log_first / tool_safety / no_bypass / no_pushback / blame_shield / failure_preflight / declare_act_guard / negative_conclusion_guard / test_discipline + 注入通道 rule_reinjection / section_reinjection / workflow_router（2026-11-10 实测共 16 子模块） |
| 独立 | scientific-programming-guard；terminology-guard；enterprise-quality-gate；requirement-interview | 专项规范（写码预算、用词一致、企业级对表、需求歧义） | — |

裁定原则：同一层同性质才收拢进同一套件；跨层不合并。子模块 register 逐个调用，
单个失败仅告警不连坐。

## 一之一、Guard taxonomy 与健康策略（2026-11-05）

Bundled guard 的 `plugin.yaml` 显式声明两个 additive 字段：

- `gate_suites`: `intent / execution / output / completion / integrity` 的列表。一个 guard 可跨
  阶段；taxonomy 只做归属与后续重组依据，运行时 suite 归属仍以真实声明的 Gate Plane hook
  与注册状态为准，元数据不能隐藏可执行面。
- `gate_policy`: `required` 表示工程基线；缺席、加载失败或空注册使相关 suite `unknown`。
  `advisory` 表示专项/辅助能力；刻意 disabled 只在 `inactive_advisory` 诊断道展示，不拉低
  健康，但一旦 enabled，加载失败或空注册仍进入 `missing` 并使 suite `unknown`。
- `gate_kind`: `enforcement` 表示五个运行决策 hook 之一的执法面；`observer` 表示
  `post_llm_call` / `post_tool_call` / session lifecycle 的非阻断观测面；`manual` 表示
  CLI/离线审计，不得计入 runtime active guard。缺失或非法时按 hooks 保守推导，无 hook
  即 `manual`。required manual audit 没有新鲜审计证据时保持 `unknown`，不能凭插件导入
  成功伪装健康。

Gate Plane 证据面覆盖 enforcement 与 observer hooks。observer 回调失败或超时不阻断轮次，
但必须记录 `unknown` 并降低 integrity 健康；`engine-invariants` 不在钩子内部吞掉基础设施
异常，而由插件调度器统一隔离并留痕。

基线 owner：`read-think-gate`、`guards`、`discipline`、`block_escalation`、
`finish_guard`、`guard-sentinel`、`engine-invariants`。
旧/第三方 `tags: [guard]` 插件未声明 metadata 时按保守策略处理：由 hook 推导 suite/kind
且视为 `required`。后续行为归并只允许按本 taxonomy 分批迁入既有 owner suite；先迁/补契约测试，
再改实现，不留兼容壳。

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

## 二之一、输出面纪律（2026-11-10）

事故实录：状态问答的第一版正常人话答案被 `finish_guard` 以「请示式/根治/待拍板」
命中而隐藏续跑；审计与哨兵再把 UNKNOWN / 门禁缺席追加到最终回复，形成第二层
官话污染。裁定：用户答案正文属于交付内容，不是插件公告板。

- `transform_llm_output` 只允许语义改写（脱敏、过滤、明确要求的格式转换），不允许
  追加守卫纪律、健康状态或审查提示；策略诊断写日志。
- 影响下一轮 agent 行为的提醒走 request-scoped `pre_llm_call`；不能执行时走
  `pre_tool_call(block)` / `pre_verify(continue)`，不得伪造用户可见文本。
- `pre_verify` 续跑必须挂在执行欠账上：有改动/有本会话 active TODO 才触发；
  零编辑、无 TODO 的状态问答与解释直接放行。
- 审计 judge 不可用是插件健康诊断：记录 WARNING、进入会话熔断，不改写答案，
  也不向下一轮注入内部 UNKNOWN。

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
- 现有段（2026-10-26 实测更新，其中文件型段 6 个）：
  `guards_preflight` 1177 / `tool_blacklist` 404 / `db_safety` 451 / `test_discipline` 272 /
  `scientific_programming` 396 / `diff_debt` 249 字符；连同 terminology_consistency、
  enterprise_quality_gate、requirement_interview，段总量约 4600 字符（上限 8000）。
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
| source_write | `_SAFE_PREFIXES` 缺 `~/.hermes/cache/`（TMPDIR 官方临时区，24h 清理） | 加入豁免，与 /tmp 同语义 | <!-- no-tmp: ok — 表格叙述引用检测语义,非代码写 /tmp -->

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

## 八、拦截效率重构（2026-11-03 数据驱动，用户令：门禁要高效/零误拦/不重复拦）

数据画像（outcomes.db 全量 + 近 7/30 天）：

- 全期 747,072 次调用被拦 56,975 次（7.6%）；近 30 天 7,149 个 turn 被拦、
  平均每被拦 turn 被拦 4.76 次、最高单 turn 221 次——效率损失主因是
  「拦截→浪费一整轮 API→重试→再拦」循环，不是回调链本身（纯 Python 判定毫秒级）。
- 近 7 天最大拦截源=SlowSearchGuard 1,385 次（18%）；其次 diff 欠账闸门、DBSafety。
- pre_tool_call 回调超时（默认 30s，fail-closed 即拦）30 天 84 次——护栏自身
  故障变成用户误拦。

本轮五修（commit 同批）：

| # | 误拦/低效源 | 根因 | 根修 |
|---|---|---|---|
| E1 | SlowSearchGuard 近 7 天 1,385 次拦截 | 修正完全可机械推导却走拦截通道 | block → modify：自动加 `timeout 120` 包裹 + `--exclude-dir` 四件套后放行；保护强度不变（最坏仍被 120s 截断），拦截归零 |
| E2 | casebook_gate 把 `2>/dev/null` 当写目标（实测同因 2 次拦截） | 角色解析前未剥重定向形态 token（fd 前缀 shlex 不拆词） | `_segment_write_targets` 前置 `_is_redirect_token` 剥离 |
| E3 | casebook_gate「拦一次即放行」承诺落空（纯重试永再拦） | 拦截不记 probe，承诺无实现 | 拦截即记 `nagged-once` probe——兑现 docstring「每会话最多拦一次」 |
| E4 | read_think_gate 只读 terminal 命令（ls/sqlite3 查询/git log）不计入调查证据（实测拦 1 轮） | `READ_ONLY_INVESTIGATION_TOOLS` 枚举白名单不含 terminal | gate 新增 `_terminal_is_investigation` 唯一判定（解释器/`./`可执行/`.sh`/写形态之外=取证）；devil-advocate `_terminal_passes` 改复用同判定（单一事实源） |
| E5 | armed 冻结面过宽：任何重大决策冻结整会话逼派反方子代理（用户 1103 点名子代理慢、必要时才用） | armed 无二级回退性判定 | 新增 `_is_irreversible` judge——不可逆/对外/生产级才 armed；可回滚重大决策降级为非阻断提醒 |
| E6 | 回调超时 30s fail-closed（30 天 84 次误拦窗口） | 默认超时过长 | config `plugins.hook_callback_timeout: 30→5`（窗口缩 6 倍；超时 fail-closed 为上游行为不改核心） |

新不变量（守卫改动对表）：

3. **修正优先于拦截**：拦截消息里教的所有机械修正（加参数/换命令/包裹
   timeout），若可从被拦载荷直接推导，必须走 modify 自动改写而非 block——
   一次拦截=一轮 API 白跑。拦截仅保留给「修正不可机械推导」的形态
   （如缺调查证据、缺反方审查）。
4. **拦截承诺必须可兑现**：拦截消息里写的每个解锁通道（含「直接重试即放行」）
   必须有对应实现路径；写承诺前先核豁免判定的实际作用域与状态记录点。

### 反方审查对账（2026-11-03 独立只读子代理，10m12s/30 次调用）

反方根因排序（采纳）：① 同 turn 第 2 次及以上被拦 **42,429 次**（56,977
次拦截中）＝纯浪费的重复轮——绝对主因；② 拦截消息反复入史挤上下文＝①的
派生；③ 回调链串行开销可忽略（单护栏解析实测 0.1-0.4ms，35 注册点全串行
≈5-20ms/调用，比 LLM 往返低 3 个数量级）。反方反对「重构总线/砍护栏/
优化回调链」——与本轮实施一致（总线未动、护栏未砍、回调链未改）。

反方发现 → 处置：

| # | 反方发现 | 处置 |
|---|---|---|
| D1 | 只读 terminal 不计调查（gate.py:162-173） | 已修（同批 E4） |
| D2 | SQL 比较符 `price > 9.99` 判写 | 已修（同批 E7）：数字字面量+剥引号后不计重定向目标 |
| D4 | `open(` 只读形态计写特征 | 已修（同批 E8）：open( 仅写模式字面量在场才计写特征 |
| D5 | 相对路径无条件拼 OntoX 根恒真（跨仓误触发） | 已修（同批 E9）：相对目标按命令 workdir 解析、无 workdir fail-open |
| D6/D9 | `2>/dev/null` 当写目标；有界 grep 一刀切 | 已修（同批 E2/E1） |
| D3 | 四轴 marker 未齐时不续期 TTL（推断） | **证伪**：未齐状态的正确解锁=补齐 4 轴（len==4 分支必写盘续期）；只重贴已凑轴不刷新 TTL 是有意防中间进度刷 TTL，被拦状态=证据真缺，非死锁（gate.py:874-880 注释即此设计） |
| D7 | `>out.tsv` 粘连形态漏检 | 记录：漏拦形态、引导层+每会话一次兜底，暂不修 |
| D10 | DBSafety 只读 SELECT 首查必拦 | 保留：先查 schema 是 DB 纪律设计意图（禁猜表名），有 `# dbtarget-ok` 豁免 |
| D11 | armed 判定通道故障时冻结残留 | 部分缓解（E5 收窄触发面）；judge 通道故障不 armed 已实现 |
| D12 | 钩子内同步 LLM judge 延迟 | 记录：devil judge 仅重大决策消息触发（hash 去重），非每工具调用 |
| D13/D15 | block_escalation 意图指纹聚合面；四轴双状态源 | 独立工程项，见路线图 |
| D14 | finish_guard L2 续跑交互税 | 有 `_MAX_PRE_VERIFY_NUDGES` 上限兜底，保留观察 |

### 双保险闭环：拦截账本 → 动态提示段（2026-11-03 用户定调落地）

「事后拦截不如事先明确提示」——事前提示降发生率、事后拦截兜底，两层都要；
缺的是数据回流：拦截账本里的教训从未进入提示段。已落地
`plugins/guards/dynamic_tips.py`：guards_preflight 段尾追加「高频拦截教训」
动态段（近 7 天 blocked 消息原文聚合 → `[护栏名]` 前缀提取 → ≤5 条
带修复指引原文，≤600 字符）。去映射版端到端实测首刷输出：
SlowSearchGuard 1,305 次/patch-first 228 次/护栏聚合 227 次/
PreWriteGuard 158 次/SourceCodeWriteGuard 124 次。

设计约束：7 天缓存周期内输出恒定（不破坏会话缓存前缀；越期由既有
section_reinjection 版本指纹补注——动态段文本已并入补注指纹与注入体，
在跑会话同达）；只读连接、任何数据面故障 fail-open 空段或旧缓存兜底。
2026-11-03 用户点名根治静态映射表：动态段不再维护签名→正解映射，直接
消费拦截消息原文（护栏拦截文案自带修复指引是既有约定）——新增子护栏
零同步成本；无 `[护栏名]` 前缀的消息（回调超时等基础设施噪音）不进段。
层级优先级保持不变：modify 自动改写 > 事前提示 > 事后拦截。

### C3 解析器收敛完成（2026-11-05）

唯一事实源 `plugins/guards/terminal_semantics.py`（Redirect 数据类 +
parse_redirects/redirect_writes_file/redirect_file_targets/
looks_like_path/sed_writes_file/is_read_command,语义基底取六套中最全的
block_escalation 实现）。
消费方已切换 4 处：read_think_gate/gate._terminal_writes_file（窄口径
→宽口径,`echo hi > out` 不再漏判）、source_write._extract_redirect_target
（`>>` append 与多重重定向漏检修复）、casebook_gate._redirect_targets
（fd 粘连形态提取+shlex 引号异常清除）、block_escalation._redirects_write
（委托,数字字面量排除对齐）。配套 test_terminal_semantics.py 原 9 例。
2026-11-05 第二批：`patch-first` 已改为消费 `sed_writes_file`，只读
`sed -n ...p`、复合命令中的写 sed、以及 execute_code 内嵌 sed 使用同一
判据。第三批：`block_escalation._READ_CMDS` 迁入 owner 的
`READ_COMMANDS/is_read_command`，并先经 `sed_writes_file` 排除 `sed -i`
等写形态，消除「升级计数把 sed -i 误判只读」的交界漏洞。owner 回归增至
12 例，block_escalation 相关回归 18 例。

### 行为归并前置基线（2026-11-05）

五个此前无直接契约的 guard 已建立行为测试：`lazy-blocker`（5 例）、
`patch-first`（5 例）、`persistence-enforcer`（7 例）、
`pre-response-review`（5 例）、`quality_auditor`（5 例）。本批修复
`patch-first` 复合/内嵌 sed 漏拦、`pre-response-review` profile 越界与
静默降级、`quality_auditor` 同名工具统计误崩；另两项目前为基线确认，
不得据此直接按文件名合并，后续重组仍需按 suite owner 与行为等价迁移。

### 状态源与拦截原因裁定（2026-11-05）

两个原“独立工程项”已按语义拆开，而不是把不同事实证明成同一个状态：

- `guards.session_policy` 是会话→platform 与 cron 豁免的唯一事实源；
  `four_axis` 与 `diff_debt` 只消费该判定，unified cron 的 `local-*`
  会话与 legacy `cron_*` 会话走同一规则。
- `read-think-gate` 继续独占 pre-write 四轴证据 marker；`diff_debt` 继续
  独占 post-write diff 欠账。两者是先后两道不同义务，不得合并成一个
  boolean，也不得在 completion-owner 迁移中互相搬运。
- `diff_debt` 在 session reset/end 只清本会话欠账与本批证据位，避免长驻
  网关进程保留已结束会话状态；TTL 仍只作兜底而非生命周期事实源。
- `block_escalation` 的 blocked 指纹包含稳定原因族：runtime guardrail code、
  插件拦截文案护栏名，或归一化首行；同一目标被不同原因拦截不合并计数，
  成功落地仍按目标交集清账。

### 已知路线图（独立工程项，非本轮范围）

- required/profile 基线收敛：五个 required owner 缺席必须启用或经行为迁移
  显式替换，不得降级刷绿。
- 物理归并仍按小批次执行：`patch-first`→`guards` 已完成（行为迁入
  `guards.patch_first`，独立 manifest/hook surface 删除，owner 注册契约
  在位）；`blame-shield`→`discipline` 已完成（行为迁入
  `discipline.blame_shield`，独立 manifest/hook surface 删除，owner 注册契约
  在位）；`lazy-blocker`→`finish_guard` 已完成（行为迁入
  `finish_guard.lazy_blocker`，独立 manifest/hook surface 删除，owner 注册契约
  在位）；`completion-boundary-audit`→`finish_guard` 已完成（行为迁入
  `finish_guard.completion_boundary_audit`，独立 manifest/hook surface 删除，
  owner 注册契约在位）；`failure_preflight`→`discipline` 已完成（行为迁入
  `discipline.failure_preflight`，独立 manifest/hook surface 删除，owner
  注册契约在位）；`declare_act_guard`→`discipline` 已完成（行为迁入
  `discipline.declare_act_guard`，独立 manifest/hook surface 删除，owner
  注册契约在位，manifest 补记 `post_llm_call` 预热 hook）；
  `negative-conclusion-guard`→`discipline` 已完成（行为迁入
  `discipline.negative_conclusion_guard`，独立 manifest/hook surface 删除，
  owner 注册契约在位，旧目录内写死 home 路径的自测脚本一并移除）。
