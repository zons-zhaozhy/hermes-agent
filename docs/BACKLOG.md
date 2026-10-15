# Hermes fork 维护 Backlog

来源：2026-09-28 深夜 multi-bot 运维会话（bot_mode_dm 修复 e394e62c02、periodic_scheduler 锁序修复 05c4ff8e7a 的收尾盘点）。

## 待办

### 0. [cron] 429 硬限识别短路（skipped-quota）
- 优先级：高（429 五小时上限窗口第 6 次复发：2026-10-09 16:45–20:05 共 69 笔 failed 横扫 6 job，10-08 档首次提出仍未实现）
- 问题：配额窗口内调度器每 5–15 分钟照常重派每个 job，进程引导期首次模型调用即 `RuntimeError: HTTP 429: 已达到 5 小时的使用上限。您的限额将在 <时刻> 重置` 直死——agent 层退避无机会执行（失败在 agent 循环外），executions.db 被污染 70 笔/日失败信号。
- 方案：`cron/scheduler.py` 派发前查该 job 最近 failed 错误：含「已达到 5 小时的使用上限」且带未来重置时刻 → 本轮记 `skipped-quota` 不派发（复用 note_cron_skipped 观测），重置时刻过后自动恢复派发。错误消息内重置时刻已可解析，无需外部状态。
- 验收：注入含重置时刻的 429 failed 记录 → 下轮该 job skipped 不新增 failed；重置时刻过后首轮回派正常；executions.db 配额窗口内 failed 笔数从 ~70 → 0。
- 阻塞：scheduler.py 属核心文件（上游同步纪律），动手前需评估上游冲突面；且 gateway 在线不可重启，须挑无人窗口。

### 1. [Lark] SDK logger 桥接 gateway 根 logger
- 优先级：中（观测面缺陷，已造成哨兵一次误报）
- 问题：lark-oapi SDK 内部 logger 未接入 gateway 根 logger 配置，其 INFO 级输出被 stdout 块缓冲吞掉，keepalive 断开/重连期间零日志——19:20 飞书「入站死亡」误报的根源之一（另一根源=判活依赖日志解析而非控制面 status）。
- 方案：按 #73779 已有的 SDK 接线先例，把 lark-oapi logger 挂到 root（propagate=True + 合理 level），使 _try_connect 失败必打 ERROR 的设计真正可见。
- 验收：飞书链路断开时 errors.log/info 日志可见 _try_connect ERROR 行；重连成功可见 on_reconnected 打点。
- 备注：重试看门狗立项已驳回（#73779+#113662 已有既有看门狗，重复建设）。

### 2. telemetry 面板：「.dev_logs/oms.log 未按 profile 隔离」事实核查
- 优先级：低（哨兵 09-27 通报，未复核）
- 内容：哨兵称 .dev_logs/oms.log 无 profile 前缀，多 profile 共用一文件。需读 OMS dev_start 逻辑核实后决定是否修。

### 3. [aux] 既有失败测试：custom 端点支付失败误跳过不同 URL 的候选
- 优先级：中（测试红灯，非本 fork 引入）
- 现象：tests/agent/test_auxiliary_client.py::TestAuxUnhealthyCache::test_custom_billing_failure_keeps_distinct_endpoint_eligible 失败——hosted 自定义端点返回 402 后，fallback 的 local 端点未被尝试，调用直接抛出主错误。
- 证据：① git stash 暂存本次改动后复跑为同样的 1 failed / 239 passed，基线一致；② fork 与 upstream 在 agent/backend_identity.py 无差异，agent/auxiliary_client.py 的 18 行差异全在 _ladder_provider_fallback 的 request timed out 分支（fork 定制 ef562cb74a），payment 路径与 upstream 逐字相同；③ 运行时 _resolve_fallback_entry 零调用，并出现「all fallbacks exhausted」告警。
- 疑似根因：_chain_failure_scope（auxiliary_client.py:7905）在 _custom_health_base_url(resolved_provider, route.base_info) 返回空时退化为 None，_failed_backend_skip 随即落到 FailureScope.CREDENTIAL；same_credential_surface（backend_identity.py:87）按 provider 名比较，custom == custom 即判同一凭据面并跳过候选，于是不同 URL 的候选被误跳过。
- 方案候选：payment 作用域下改用按 base_url 比较的 same_endpoint 语义，而非 provider 名比较。
- 阻塞：backend_identity.py / auxiliary_client.py 属核心文件（上游同步纪律），改动会与上游该语义分叉；建议先向上游报 issue/PR（需授权）。

## 已结案（防重题）

- 429/1302 整点齐射 → 4 任务错峰已落地（:04/:07/:10/:11 分配表，哨兵回读核实）
- bot_mode_dm 投递挂死（管道 EOF 等待）→ e394e62c02，退出码权威范式
- periodic_scheduler 锁序死锁（65848 teardown）→ 05c4ff8e7a，dispatch 移出 _cond
- 飞书「入站死亡」 → 误报，链路 24 秒自愈，不重启（控制面 status 权威）

## 2026-09-28 追加（ruamel 第三雷翻案后）

### 2. 1302 账户限速二期：LLM 预算上限 + provider 分流
- 优先级：高（上调，哨兵附议：今晚两次「重调试 turn+整刻」撞顶）
- 方向：整刻错峰已落地（:04/:07/:10/:11），但重调试 turn（red-first 测试连跑）+cron 并发仍会撞 zai 账户限速；需 LLM 调用预算上限+跨 provider 分流兜底

### 3. pm replay 触发链精确化
- 优先级：低。已实锤「污染→replay 清空 site-packages」机制，但 23:02 replay 的确切调用方（哪次 pm ensure/verify）未定位；PM replay 发生时应有防污染告警
