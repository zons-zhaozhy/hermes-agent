# 门禁体系全量审计报告（2026-10-27）

审计范围：Hermes Agent fork 全部启用插件中具备拦截能力的 40+ 个门禁，重点为状态冻结型与 guards 聚合 16 子护栏。
审计方法：五维判据矩阵（冻结面/解除可达性/豁免出口/判定成本/可观测性）+ state.db 全量取证（agent.log 为多进程分流残缺视图，实测偏差 19 vs 77 次）。

## 一、起因与慢回合定量

用户报「回复要几分钟」。实测（agent.log 当日全量）：

| 指标 | 数值 |
|---|---|
| API 调用次数 | 1,154 次 |
| in-token p50 / p90 / max | 110,885 / 182,739 / 199,985 |
| 基线系统提示词（含工具 schema） | 31,621 token（21 工具 42,191 字符 ≈ 10,547 token） |
| 单调用平均耗时（zai/glm-5.3） | 17.8s，max 204.8s |
| 每回合 API 调用 p50 / p90 / max | 10 / 31 / 114 |
| 四轴副防线拦截（state.db 实测） | 77 次，≈18s/轮 ≈ 23 分钟拦截税 |
| RateLimitError / APIConnectionError | 9 / 3 次（每次触发重试叠加延迟） |
| 并发会话 / hermes 进程 | 20 / 31 |

慢 = 上下文大（基线 3.1 万 token，长任务会话滚到 11-18 万）× 每回合调用多（p50=10）× 单调用慢（17.8s）。门禁误拦是放大器不是主因。

## 二、根因与修复（四 commit，均含红→绿回归）

### 根因 1：marker 续期条件死区 — commit 93d23a64ee
- 症状：四轴证据贴齐后 30 分钟 TTL 到期仍被拦（「marker missing or expired」当日 19 次可见日志，state.db 全量 77 次）。
- 根因：gate.py:839 续期条件 `if progressed`——四轴凑齐后写工具不再产生新增量，条件恒 False，marker 到期即死。
- 修法：续期条件含完成态 `progressed or len(axes) == 4`。

### 根因 2：双防线空 session_id 桶错配 — commit 8d82fe9cbe
- 症状：副防线拦了 77 次，主防线日志零扫描记录——拦截条件结构上无法满足。
- 根因：read_think_gate/__init__.py:144 pre_tool_batch 用裸 `session_id` 参数，网关空传时与 pre_llm_call:118 的 `_ACTIVE_SESSION_ID` 回退落到不同桶。
- 修法：新增 `effective_session_id()` 单一实现（:82-94），写读两端同源解析；副防线 marker 桶同源；拦截日志全部带 session；60 秒重复拦截升级提示（_LAST_BLOCK）。

### 根因 3：DevilAdvocateAudit armed 全类冻结误伤取证 — commit ebff19ee3a
- 症状：armed 期 terminal/browser_exec/glm-vision 只读取证全被拦（16:57-16:58 三次，17:25 主会话自身一次）。
- 根因：__init__.py 冻结面=「枚举只读白名单+默认拒绝」，白名单不含 browser_exec/glm 系工具家族，新取证工具必被误拦。
- 修法：语义化冻结面——_READONLY_TOOL_PREFIXES 工具家族前缀（glm_vision/gitnexus/glm_zread/glm_web_search/glm_web_reader）+terminal 按命令细分（复用 read_think_gate._terminal_writes_file 与落地模式表，git push/commit/tag/apply/rebase/reset 仍拦）；execute_code/patch/write_file 落地通道保持拦截。
- 已知权衡：armed 期 browser_exec 放行理论可被用于浏览器内执行决策动作——风险接受，待反方审查复核。

### 根因 4（并行会话发现）：压缩后会话 id 漂移桶错配 — commit 1ef9936d3a
- 症状：上下文压缩后四轴证据贴三轮仍被拦。
- 根因：压缩后写端（pre_tool_batch）登记新会话 id 桶，读端（pre_tool_call）仍收旧 id 读旧桶。
- 修法：读端未命中时回退 `_ACTIVE_SESSION_ID` 活跃桶（只认写端刚验证过的桶，不全桶扫描防跨会话借用）。

## 三、guards 16 子护栏深审结论（全部健康）

- pre 侧纪律型（先读后写/四轴/查重/批量红线/病例库/新目录树/源码直写/废弃术语/体验完备/技能剪枝）：拦截面全在写工具与写命令，只读工具仅作已读记账，无状态死锁形态。
- post 侧状态依赖核对：仅 diff_debt 依赖 `status=="ok"`（与框架词表一致）；其余均无状态单调用判定。
- 当日计数定性：finish_guard 112 次命中 = 31 提醒注入 + 15 强制续跑 + 14 L4 走查（提醒型，健康）；source_write 76 次多数为 escape hatch 放行记录（豁免通道正常）。
- 其余拦截型插件（modify_guard/decision_tree/patch-first/pipeline-guard/discipline 系列）：零同步 LLM 判定成本，豁免面在位。

## 四、反方审查轮（deleg_231c7b7f，20 次 API 调用 570s）与第五批修复 e11c50dd72

反方子代理独立审查三个修复后输出 20 项发现（A1-A8/B1-B6/C1-C6），定性：8 项真缺陷（A1-A6/B1/B2）+C1/C3 同类病已修（commit e11c50dd72，8 文件 +287/-83，tests/plugins 全量 1,963 通过 0 失败）；并行会话同窗再修 2 项（1ef9936d3a 桶漂移回退、54ad90e39f 跨会话借用可观测性 WARNING）。修法要点：

| 项 | 修法 |
|---|---|
| A1 browser_exec 整工具放行=armed 旁路 | 移出只读表（code 参数=任意 Python，能力等价 execute_code） |
| A2 `python3 -c`/heredoc/管道喂码绕过写检测 | _terminal_writes_file 增解释器执行形态判写（heredoc 豁免撤销） |
| A3/A4 git 子串无 token 边界+非 git 落地通道缺失 | git 子命令 token 化（-C 全局选项跳过）+crontab/defaults/osascript/npm publish/DB DML 落地表 |
| A5 gitnexus 前缀族含写语义工具 | _PREFIX_FAMILY_WRITE_EXCEPTIONS（cypher/rename/group_sync） |
| A6 armed 判定 fail-open | fail-closed；正门（delegate_task）前置 try 外保证异常路径可达，豁免通道不经此函数 |
| B1 _ACTIVE_SESSION_ID 单全局多会话抢占 | ContextVar 化（写读两端+four_axis 配套），asyncio task/线程隔离 |
| C1 调查表 browser_* 旧工具名锈化 | 换 browser_exec/vision_analyze/session_search |
| C3 沉淀表漏 execute_code/browser_exec | 补齐（代码写文件旁路沉淀判定） |

## 五、遗留观察项与结构债

1. scope_gate 判定模型：当日 51 次判定、9 次本地判定超时回退主模型（每次 ≈20s 超时 + 一次 glm-5.3 调用）——成本项未修，量级远小于已修项。
2. CodingStandardsGuard 正则铁律误伤一次性只读探针（反方子代理实测命中一次）——枚举式判定漂移的新证据，待定性。
3. 装载依赖：全部修复需进程重启装载。网关已于 17:31 重启（pid 70881/70883/70884，晚于当时全部 commit）；e11c50dd72 之后的会话需再次重启装载。CLI 长会话若早于 commit 启动仍按旧代码执法。
4. C2 结构债（未修，反方定性「病灶结构全库复制」）：≥5 份互不一致的只读工具枚举 + ≥8 份互不一致的写工具枚举散布各插件，无单一 owner。本轮五处修复仍是在各插件就地补枚举——与「枚举必然漂移」的结构矛盾未解。收敛方案：新建 plugins/guards/tool_classification.py 单一 owner（只读家族/写语义/落地通道三类判定集中），全库消费方 import 使用 + AST 门禁声明语义归属（违单一事实源纪律须门禁配套）。涉及 8+ 插件改造，属方向级工程。
5. 风险接受项（反方定性，不修）：B3 nosession 共享桶（既有行为）；B4/B5 可观测性轻微损失；A7 landing 子串保守误拦方向；C4 block_escalation 未列只读命令计写（2 次窗口）；C5 SlowSearchGuard 有界只读探针误拦（有豁免通道）；C6 字面量枚举漂移风险现后果轻微。

## 六、复现与验证命令

- 拦截全量：`sqlite3 ~/.hermes/state.db "select ... from messages where content like '%four-axis guard: blocking%'"`（先 .schema 确认列名）
- 装载判定：`ps -eo pid,lstart,command | grep hermes` 对照 commit 时刻
- 回归：`scripts/run_tests.sh tests/plugins/`（四轴 + devil-advocate 共 58 用例）
- 同款结构扫描：`grep -rn "_READONLY\|allowlist\|默认拒绝" plugins/`
