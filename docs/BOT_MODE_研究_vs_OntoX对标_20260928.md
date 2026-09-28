# Hermes Bot Mode（bot 组团能力）研究与 OntoX 对标报告

日期：2026-09-28　取证方式：上游 git 对象 + 官方文档（website/docs/user-guide/bot-mode.md, 381 行全文）+ gitnexus 索引实测。运行时行为未实测（未起 Desktop/未建群/未跑 peer dm）。

## 一、Hermes Bot Mode 能力全景

### 身份模型（全部设计的根基）
Bot = Hermes profile（~/.hermes/profiles/<name>/），有自己的模型/记忆/技能/SOUL.md/凭据/头像。零新原语：`hermes -p <bot> chat` 打开同一 agent，routines 就是 `hermes cron list` 里的 `[bot:<name>]` 任务。群聊身份 = (profile, 标题恰为 "Bot Chat") 的 session，UNIQUE(title) 索引即注册表，官方明文拒绝一切 session-id 指针（五轮加固教训，不开放重审）。

### 群聊房间（组团核心，2026-08-17 起落地，08-30 架构跃迁）
- 2–6 Bot 一房；一条消息触发最多 3 轮串行成员回合；@谁谁回，不 @ 全员可说，全员沉默即收敛
- 防失控三件套：单发 10 消息/3 轮硬顶；沉默令牌（[SILENT]/NO_REPLY）；stop 指令必须紧贴 @mention（代码块/引用内不算）
- 每成员独立持久 room session（标题 `Group: <room> · <thread>`）；每回合携带自上次发言以来全部房间消息（约 200 条/32K 字符窗口，单条截 8K）
- **关窗不停房**：同网关房间由网关侧 durable driver（gateway/hosted_room_driver.py，SQLite 租约+任务状态机，24 列任务表、终态保留 30 天）驱动；房间状态存 shared-state.db；事件日志复制 + 栅栏式权威接管（groups.replica_state/groups.promote/groups.demote，幂等 lineage）
- 跨机器房间：每成员回合跑在自己机器/网关上，跨连接信使（Desktop 持双 socket 接力）投递

### bot-to-bot 私聊（message_agent 工具，2026-08-21）
- fire-and-forget 异步 DM：目标校验（profile 名 > 花名 > @tag，歧义拒绝不猜）、服务端强制归属前缀、16000 字上限
- 回复经后台完成通知送达；不可收通知的面（api_server/one-shot）回退 reply_delivery=poll
- 11 种类型化失败码端到端贯穿（provider_auth_or_access / context_overflow / target_busy / delivery_timeout…），发送 agent 按码分支
- 安全重试：瞬时失败同 session 重跑一次；上下文溢出先压缩再跑；鉴权/配额永不自动重试；被中断回合不自动重放
- 跨机器无桌面路径：hermes peer add/dm/run/status/stop（对端 api_server + 强 API_SERVER_KEY，幂等键防重复起跑）

### 时间线（git log 实测）
- 08-17/18 爆发起点（群聊创建/线程/多群/peer/跨连接/message_agent）
- 08-21~24 可靠性波（@mention 只识别不投递、失败类型化、TTL、turn 锁）
- 08-30 架构跃迁（durable 权威+重放、跨网关传输、无桌面运行）
- 09-04~16 打磨波（sender 贯穿、a2a_key 作者、on_room_member_activity 插件钩子）
- 09-17~26 生态波（bot-forge v0.2→v0.12：create_team 一句话组团/教学/分享/健康检查/回滚；mybot.farm 市场）

### headless 开启法（文档转述，未实测）
1. `hermes -p <bot> chat -c "Bot Chat" --create-if-missing`
2. 任一 profile 的 profile.yaml 写 `ui_meta: { hermes-bots: {}}` 标记 Bot-Mode-managed

## 二、OntoX 现状对标（gitnexus ontoX 索引实测）

| 维度 | Hermes Bot Mode | OntoX Loom（实测：dag_loader.py） |
|---|---|---|
| agent 本质 | 会话型 agent（profile，永续对话，有记忆人格） | DAG 结构化步骤执行节点（_execute_agent 出边仅 VersionedWorkspace.get/put + EventBus.emit，无会话循环） |
| agent 间通信 | message_agent 异步 DM + 群聊轮转 + @mention | 无——节点间只有 workspace 版本化数据传递 |
| 编排拓扑 | 房间轮转（去中心化，agent 自主决定发言） | 中心化 DAG（register_agent/create_agent，_rerun_affected 按依赖重跑） |
| 故障语义 | 11 种类型化失败码端到端，按码分支重试 | step 级成功/失败，saga 编排器有补偿语义 |
| 持久化 | 房间=事件日志+租约，权威死后栅栏接管 | run/step 状态落 PG，重跑走受影响面重算 |
| 生态 | bot-forge/mybot-farm 插件市场 | 场景 YAML 仓库（apps/*/scenarios/） |

结论：两者是**两种范式**而非差距——Loom 的 DAG 是「管线确定性」优先（适合 ETL/批量 LLM/合规跑批），Bot Mode 是「agent 自主性」优先（适合研讨/分工/巡检对话）。OntoX 若要「组团」，正确路径不是给 Loom 加会话，而是新增会话型 agent 层（类似 Hermes profile），DAG 节点可引用会话型 agent 作为「判断题升级作文题」的出口。

可借鉴三条（按价值排序）：
1. **agent 即 profile 的身份模型**——OntoX 已有 Auth 用户/租户体系，会话型 agent 可挂为一种「系统租户/用户」实体，复用现有隔离
2. **房间事件日志+租约驱动器**——「关窗不停房」的架构解，映射到 OntoX=场景跑批的 durable 执行思想已有（step 状态机），缺的是多 agent 轮转的收敛判据（沉默收敛+硬顶）
3. **类型化失败码端到端**——OntoX 已有双层信封 {ok,data,message}，可在 message 层加 reason 码族供上层 agent 分支（llm_batch_failure_diagnosis skill 已有定性规则可机器化）

## 三、同步计划（重叠面实测：1260 文件 > 300 阈值 → 强制分批）

本地 HEAD=67c21c85c1 落后 upstream/main 1368 提交。定制重灾区 5 文件中 3 个被上游再改：agent/agent_init.py、agent/conversation_loop.py、model_tools.py（cli.py/goals.py/tool_executor.py 本批未被触及）。目录分布：tests=328、agent=47、tools=44、gateway=19。

| 批次 | 合并目标 | 覆盖内容 | 冲突预期 |
|---|---|---|---|
| 1 | upstream/main@2026-08-31 | 8 月 bot-mode 全波 + 8-30 架构跃迁 | agent_init/conversation_loop/model_tools 首轮冲突，集中解决 |
| 2 | upstream/main@2026-09-15 | 9 月上旬打磨波（sender 贯穿/插件钩子/桌面 UX） | 少量 |
| 3 | upstream/main 当前顶点 | 9 月下旬生态波 + bot-forge 目录项 | 少量 |

每批工序（依 docs/UPSTREAM_SYNC_LEDGER.md 合并规程）：merge → 冲突逐文件对账账本 → 合并后三验（imports + read_think_gate 测试）→ 触桌面端加跑 vitest（按改动面选目录）→ 回退猎查（diff --stat 对照官方修复符号）→ commit+push。全量 scripts/run_tests.sh 收尾一次（约 1 小时/20 worker，需与并发会话错峰）。

执行注记：批次 1 具备执行条件；因全量测试占满机器且用户有多会话并发惯例，开跑前先查 SESSION_BOARD.md 错峰。
