# 上游同步防冲突账本（SSOT）

最后核验: 2026-11-03 · main 合并 upstream c314285f07 (77 提交, 零冲突) c4a3246345, 落后 0
机器账本核验: 2026-11-06 baseline ed1afb082eb3e46cf2edae9d8f68fb67b09ee4df,
官方文件被 fork 改动 294 项(见第五节 YAML SSOT; fork-guard 实测 missing=0/stale=0)
1101 轮: photon/adapter.py 上游 runtime-record 迁 profile 根目录(33 行)与 fork 定制 4 行自动合并共存, 双方意图保留
(网络通道: SSH443/HTTPS 直连均被重置时, remote.upstream.url 临时切 gh-proxy.com 代理通道完成 lazy fetch, 完毕恢复)

> 注意: `git rev-list --count main..upstream/main` 在官方导入完整历史后不再可信
> (仓库现有 11 个根提交, upstream 可达 ~39.9k vs fork 可达 ~206)。真实落后量用
> `git log --format='%ci' <merge-base>..upstream/main | awk '$1>="日期"'` 按日期计数。

## 〇、macOS 本机全量测试的既有失败底账（0924 实测, 0926 翻案大部分, 勿再当新缺陷排查）

本机全量 `scripts/run_tests.sh` 有约 27 文件持续失败, 已定性为 macOS 环境限制+上游测试
缺标记, 非代码缺陷、非同步引入。判据: 同批文件在 premerge 树(merge 前代码)同构失败;
f799fd8578 在官方 Linux CI "Python tests / Run tests" = success。

**0926 翻案**: 其中 13 失败(holographic/photon/meta_ai/openrouter/kanban)非环境限制,
是 sync 05cb467300 误回退官方修复(占位符/剥离层/secret scope/凭据隔离/游标语义),
8027e3ef2d+68627aa528 两批共恢复 11 文件官方实现, tests/plugins/ 全目录 1750/1750
零失败。剩余失败=下述 Linux 语义/二进制/时序类, 维持原判。

- Linux 语义断言裸跑 macOS（上游缺 linux_only 标记, 违反 AGENTS.md「用标记禁裸 skipif」）:
  test_scratch_dir(setgid 位, chmod 504≠1528)/test_voice_mode(PulseAudio socket)/
  test_cross_vm_fs_wal_refusal(virtiofs/9p)/test_bot_desktop_resources(runtime.py:450
  内存读取被 is_supported_host() 仅 Linux 门控)/test_cua_no_overlay(CuaDriver.app)
- 本机缺二进制/凭据类: test_wake_word(需 pip install ai-edge-litert) 等。
  test_vault_key_storage(1027 定性): 无条件 import keyring(pyproject:304 keychain extra,
  CI tests.yml:75 的 9-extras 矩阵含 keychain; 本地默认 test-env=DEFAULT_TEST_EXTRAS=["all"]
  pm/testenv.py:19 不含) → 本地非激活 shell 跑 run_tests.sh 每次 hermes_sync 回退 ["all"]
  (scripts/_activation.sh:86 需 __HERMES_ACTIVATED 才跳过)。已建 CI 同款 9-extras gen
  (gen-34379eea, ensure_testenv 直连) 17 passed 实证测试本身绿; 需跑时:
  gen-*/venv/bin/python -m pytest tests/agent/test_vault_key_storage.py。
  ~~test_holographic/test_openrouter/image_gen~~ → 0926 已翻案: 此前 13 失败非
  环境限制, 是 sync 05cb467300 误回退官方修复(占位符/剥离层/secret scope),
  8027e3ef2d 已恢复官方实现, image_gen+memory+photon 625/625 绿
- 时序抖动类(满载 20 worker 下偶发, 单跑绿): test_slash_worker_mcp_discovery(subprocess
  +MCP 发现窗口)
- 上游自身回归(1015 实测, upstream/main 纯净树同构失败 1✗/5✓, 非本地定制所致, 不代改):
  test_auxiliary_client.py::TestAuxUnhealthyCache::test_custom_billing_failure_keeps_distinct_
  endpoint_eligible——hosted 402 后 fallback 链误判 exhausted; 根因在 b40998bc3c4 之后
  should_skip_candidate/FailureScope 演进, 待上游修
- 收集超时类: test_run_agent.py 等 3 大文件, 单跑 --collect-only 4.62s 正常, 系 runner
  并行限时形态
- tests-js vitest (1021 实测, merge 前基线同构失败, 勿当同步引入): node-deps×2+
  product-builders×2 为 macOS /var→/private/var 符号链接归一差异(be4fb4c4f9 基线
  worktree 对照 2 failed 同构); generate-icons 缺 PIL、update-ui spawn 链缺 ruamel=
  子进程解释器环境; channel-smoke/desktop-builder/web-compiles 为满载超时。
  typecheck 曾报 @types/plist 缺失=merge 后 node_modules 未重装, npm install 即愈
- 1021 merge 566faf26fc 后全量(proc 63100, 满载 10 worker+4 会话并发):
  ① memory 簇 101 失败(target_contract 64/memory_tool 20/patch_action 12/
  write_approval 5)【已翻案修复 d5625c69f4】: 非本次 merge 引入(merge 对
  memory_tool.py 零改动), 系历史 merge 将该文件覆盖回两参旧版(逆向 40+/68-),
  而 store.py:555 与测试已为 upstream 三参契约——git checkout 0b8dee00fa
  恢复后 171+69 全绿;
  ② worktree 簇 13 失败: 并发/环境敏感, 空闲单跑 29 passed 0 failed, 基线
  worktree 同构失败, 勿当代码缺陷;
  ③ 17 文件收集失败+1 文件全过但退出非零: 满载假红(test_web_server 单跑
  192 全过), 空闲复跑即绿。
- 处置: 不修(修 macOS 语义=打乱上游); 上游侧理想修法=补 linux_only 标记, 本地不代改

## 一、本地独有面（= 唯一可能冲突的地方, 已压到最小）

| 文件 | 差异 | 冲突风险 | 处置 |
|---|---|---|---|
| plugins/** + tests/plugins/** fork 纯新增 | 100+ 文件纯新增命名空间（blame-shield、block_escalation、guards/ 等） | 零（官方不存在该命名空间） | 长期保留 |
| plugins/ 内 fork 改官方文件 | platforms/photon 与 telegram adapter、image_gen/__init__、model-providers/zai、security-guidance/patterns（官方存在且持续演进） | 中（官方 10 月已连续触碰 platforms 适配器；photon 1101 轮实测可共存） | 上游触碰时逐行裁决 |
| agent/tool_executor.py | 78 行: _run_pre_tool_batch_hooks/_first_batch_block/_extract_block_message + 两调用点; 发射器契约已改为"callback 崩溃/超时 fail-closed(见 plugins_dispatch), 仅 dispatch 传输层 fail-open" | 中（官方改 dispatch 逻辑时撞） | 见收窄路线 |
| agent/auxiliary_client.py | 13 行新增(ef562cb74a): judge 超时后 _mark_provider_unhealthy 短TTL隔离(慢端点暂离池, 同429隔离类) | 中（官方改 aux 降级/健康逻辑时撞） | 等官方 aux 侧等价机制后清零; 曾漏记账本, 0924补 |
| hermes_cli/plugins.py | VALID_HOOKS + pre_tool_batch 注释更新为 policy/fail-closed 契约 | 低（hook 名单追加式） | 等 upstream 化 |
| hermes_cli/plugins_dispatch.py | pre_tool_batch 加入 _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS + _hook_timeout_block_message 泛化 | 低（分类集追加式） | 与 pre_tool_call 同机制 |
| agent/turn_facade_lease.py | 4 行: 等待上限 1800s→300s | 低 | 上游改同参数时让位 |
| agent/error_classifier.py | 16 行: reasoning 词形拒绝(422 自愈, 76d800c136) + 中文配额 429→billing 长隔离(cd46e86651) | 低 | 2026-10-30 补记(1030 报告漏记项) |
| gateway/run_inbound.py | 9f21fa12a8 定制行 | 低 | 2026-10-30 补记(1030 报告漏记项) |
| hermes_cli/config_defaults.py | 9d75c43c33 定制行 | 低 | 2026-10-30 补记(1030 报告漏记项) |
| tools/delegate_tool.py | b0148fb64e 定制行 | 低 | 2026-10-30 补记(1030 报告漏记项) |
| tools/file_operations_common.py | 1e2164cc6e 定制行 | 低 | 2026-10-30 补记(1030 报告漏记项) |
| gateway/session_viewer_automation.py | viewer_driven 过期接管机制（STALE_VIEWER_AUTOMATION_SECONDS 活性窗口 + 失活注册清理，gateway 收回 post-turn hooks）；2026-10-26 起与上游 registry 参数融合：过期机制仅对 "viewer_automation" 注册表生效，"viewer_heartbeat" 保持上游订阅即活语义 | 中（上游持续演进该文件，10-20/10-26 两轮冲突） | 10-26 已融合，配套测试 tests/gateway/test_session_viewer_automation.py |
| tests/agent/test_read_think_gate_*.py | 2 文件纯新增 | 零 | 长期保留 |
| tests/agent/test_auxiliary_client.py | 457da1e3f2: 对齐 timeout 短TTL隔离定制（auxiliary_client.py 配套测试） | 零（测试文件） | 与 aux 定制同命运 |
| tests/hermes_cli/test_pre_tool_batch_fail_closed.py | 5e0f4214ce: pre_tool_batch fail-closed 配套测试 | 零（测试文件） | 与 tool_executor 定制同命运 |
| tests/cron/test_unreachable_retry.py | b929b65d16 配套测试 | 零（测试文件） | 与 cron 定制同命运 |
| cron/unreachable_retry.py | b929b65d16: 恢复 unreachable ladder 的 offline 摘要+quota 429 transient 判定 | 中（官方改 retry 逻辑时撞） | 跟踪官方 retry 侧等价机制 |
| pm/index_config.py (1015 新增) | 1 行: FORWARDED_UV_SETTINGS 白名单补 UV_OFFLINE(传输层设置转发缺口, 被墙网络下 uv 对 direct-url 依赖强制在线 revalidate 而缓存已齐) | 低（追加式） | 值得提 upstream PR |
| ~~gateway/run_notifications.py + tests/gateway/test_requeue_dead_letter.py (6b9786e161)~~ | (1021 退役) requeue 死信定制已被官方 #140899 等价机制吸收(_refusal_retry_at 指数退避 hold, 事件不丢, 官方 tui_gateway 侧 test_notification_refusal_backoff 锁定退避契约) | 已清零 | 已同步件 |
| ~~tools/bot_mode_dm.py + tests/tools/test_bot_*.py 3 件~~ | (1015 退役) _turn_child 排水定制随上游投递架构重构退役——上游 _wait_live_dm/admit 架构不再 capture_output 跑 turn, 本地修的管道继承卡死问题面消失, 且本地修复线概念(_LIVE_WAIT_MAX_SECONDS/relaunch/late replies)已全部上游化 | 已清零 | 已同步件 |
| scripts/skill_sha_drift.py | b101a20a2d/74c996fef6: skill-drift-check 依赖脚本（基线切换恢复件+噪音修复） | 低（scripts/ 追加式） | 长期保留 |
| scripts/paradigm_triage.py | cba41a4730: 范式级信号分诊件——信号词预筛+Ollama qwen2.5:0.5b 打分(0-9)→top 队列供日学习优先消化;判断题走小模型(SystemOne 纪律) | 低（scripts/ 追加式） | 长期保留（每日 12:40 cron 范式分诊·日跑消费） |
| .hermes-rules.md | fork 根纪律文件（官方无此文件） | 零 | 长期保留；md5 与 HEAD 对齐由三问审计第4条核验 |
| apps/desktop/src/plugins/hermes-bots/cron-detail.test.tsx + apps/desktop/src/store/voice-prefs.test.ts (3811501b22) | locale/spy 修正（中文环境特有） | 零 | 值得提 upstream PR |
| lefthook.yml | lint python 钩子去 `--config ./ruff.pre-commit.toml`（1c571cc47f, 门禁唯一真相=ruff.toml, 官方 hook 自带该参数会在下次 merge 复活第二份真相） | 中（上游改 lefthook 时撞） | merge 时保留去 --config 形态; 若上游也收敛则清零 |
| ruff.pre-commit.toml | 已删除（1c571cc47f, 上游自述"duplicate required subset"的复制版, 与 ruff.toml 漂移致 pre-commit 报未触碰行错误） | 高（上游文件删除, 每次 merge 必现 deleted-theirs） | merge 时维持删除; 上游若修 hook 再评估 |
| pyproject.toml [tool.ruff] 段 | 已删除 L748-818 死段（1c571cc47f, ruff.toml 优先级完全遮蔽该段=死配置） | 中（上游改 [tool.ruff] 时撞） | merge 时若上游段有实质变更需回评; [tool.ty.rules] 保持上游 |
| .gitignore (+.codegraph/) / docs/review/ (6 文件 fork 专有审查文档) | 追加式 | 零 | 长期保留 |
| batch_runner.py + hermes_cli/cli_commands_mixin.py + plugins/platforms/feishu/feishu_comment.py + tui_gateway/methods_prompt.py + tests/test_agent_close_lifecycle.py (aabc8f6427, 0926 cherry-pick) | 官方 94f3dbec9b one-shot AIAgent close() 泄漏修复（-x 保留官方作者; cherry-pick 前基线后→已同步件） | 低（与官方同源, 下次 merge 自动对齐） | 已同步件 |
| hermes_cli/goals.py + hermes_cli/goal_command.py + gateway/run_goals.py + hermes_cli/cli_loops_mixin.py + evals/postmortem/review_probes/goal_scope_probe.py + tests/hermes_cli/test_goals.py + tests/gateway/test_goal_continuation_drain.py (0926 补记) | goals/judge 定制线: judge 对话语义注入(recent_history 摘要 24 条/2400 字上限)+contract verification draft 六字段+一键 gate add 桥+goal_command 三通道吞异常根治 | 中（官方改 goals/judge 框架时撞） | 官方 goals 若提供等价 judge 上下文注入则外迁 |
| agent/retry_utils.py (0926 补记) | 4 行: retry 参数微调 | 低 | 上游改同参数时让位 |
| tools/mcp_tool_config.py + tests/tools/test_mcp_tool.py (0926 补记) | MCP stdio env 标量归一为字符串(540014b918): YAML 标量 int/float/bool 转 str, 非标量响亮拒绝 | 低（追加式分支） | 值得提 upstream PR |
| tui_gateway/prompt_turn.py (0926 补记) | 3 行: prompt 轮次微调 | 低 | 上游改同参数时让位 |
| pyproject.toml + uv.lock (0926 补记) | 依赖钉版差异（fork 追加依赖） | 低（追加式） | 每次 pm lock 后重新对账 |
| tools/browser_tool_lifecycle.py + browser_use_cli.py + browser_tool.py + tests/test_browser_inflight_reaper.py (1023 移除) | f9203bf53a inflight 保护: >120s 长调用被闲置回收器误杀的根治; 官方 68500d9ff9 删除 agent-browser 与旧 lifecycle, 载体消失, 1023 同步时跟随官方删除; 官方新架构 cleanup_inactive_sessions 仍无执行中豁免, 缺口仍在待移植 | 已消除(载体删除) | 945898555e 已移植(见下条) |
| tools/browser_session_lifecycle.py + tools/browser_session.py + tools/browser_use_cli.py + tests/tools/test_browser_session_inflight.py (945898555e, 1029 补记账本) | 长调用 inflight 豁免移植到新会话层 janitor: browser_exec 经 inflight_session_scope 注册执行中点, 防长于 browser.inactivity_timeout 的调用被闲置回收器中途杀浏览器(官方新架构 cleanup_inactive_sessions 仍无执行中豁免) | 中（官方改 janitor/inactivity 逻辑时撞） | 值得提 upstream PR |
| tools/browser_use_cli.py + tests/tools/test_browser_use_cli.py (1029 新增) | browser_exec 超时后回收 harness daemon: 真实失败签名=harness IPC 5s response_timeout(helpers.py:52, 非 Hermes 300s), 单条 CDP 请求在 daemon 侧无超时等待(daemon.py:769) 且 stale-session 恢复路径在 _session_state_lock 内调 attach_first_page(daemon.py:782) → 请求卡住则锁不释放, 后续同类请求全阻塞(用户报「daemon 挂死」症状); 处置=识别 stderr 含 _IPCResponseTimeout/「waiting for the daemon」即 `--reload` 回收该 daemon, 下次调用重新 spawn | 中（官方改 browser_exec 超时分支或 daemon 回收原语时撞） | 值得提 upstream PR（根治应在上游 daemon 侧给 send_raw 加超时） |
| agent/coding_context.py + tests/agent/test_coding_context.py (1028 新增) | Fork 定制: _NON_CODING_SKILL_CATEGORIES deny-list 扩至 28 类(官方 18 + 自定义 10: engineering-methodology/frontend/product/business/autonomous-ai-agents/devops/software-development/ontox-ops/aml/wechat-ops), focus 会话技能索引降级 names-only, 省 ~15K 字符/缓存未命中轮 (28,325→13,333, 技能段减半); 非仓库 cwd(从 $HOME 启动 CLI)同名降级、非编码面(telegram)保持全量. **测试契约 fork 化**: 官方 test_skill_demotion_gated_on_focus 原断言 devops/software-development 保留(官方 coding-adjacent 语义)→ fork 改断言其降级(收益实测 devops 3234 字符/software-development 2347), github/data-science 因收益微小(<80)仍保留; upstream sync 时该测试断言冲突以 fork 语义为准, 官方若更新 deny-list 语义让位 | 中（官方改 deny-list/技能索引/focus 判定时撞） | 官方若给自定义类加等级机制或改 focus 判定则让位; 配套: 24 workspace 副本已同步; 测试 29/29 绿 |

### 同步历史（1105 补建）

| 日期 | merge | 量级 | 冲突裁决 | 三验 |
|---|---|---|---|---|
| 2026-11-03 | c4a3246345（他轮已记） | — | — | — |
| 2026-11-04 | 776618988d4（漏记补建） | 371 文件/15572+/7616- | test_vision_routing 取上游 _seed_models_dev 助手；uv.lock 取 fork 侧（双侧校验均被 github 直连阻断，待网络恢复 pm repair） | 133 passed + imports OK + 8/8 enabled |
| 2026-11-05 | 98134a247cb | 468 文件/13421+/2266- | session_bot.py 融合：fork 离 loop 读结构 + 上游 Principal→SessionActor 改名（顺带根修 fork 测试 test_goal_pause 的旧类名断链） | 125 passed + imports OK + 8/8 enabled |
| 2026-11-06 | 2db2fa2872 | 830 文件（上游 11-05 单日 577 提交） | 零冲突（干跑预判=实测） | 125 passed(worktree PYTHONPATH 验证) + imports OK + 插件面异常见下 |
| 2026-11-06 补注 | — | — | 插件面异常详情（补 1106 轮「见下」缺失）：deep-test-gate 与 outcome-collector 由 enabled 翻转为 disabled；定性=profile 运行时状态翻转（仓库 plugin.yaml 无启用语义改动），非合并损伤；1107 复核仍 disabled，未擅动待用户确认知情 | — |
| 2026-11-07 | 3a2fd4d338 | 499 文件 | plugins.py 取 upstream（PluginContext 抽至新模块 plugins_context.py，fork 的 PluginManualGateCtxMixin 第二基类移植入新模块，MRO 验证）；hosted_room_driver.py 互补并集（fork 防退避行+upstream maintain 块）；连带修复上游 render.tsx 重复导入 2 错误解 lefthook 拦截 | 125 passed + imports OK + 插件仍 6/8（同 1106 补注） |
| 2026-11-07 | d8b75d4068（1108 执行） | 795 文件 | 6 冲突：webhook_ingress 融合（fork 离 loop _committed + 上游 live 参数）；run_bootstrap 融合（fork recovery 模块导入 + 上游 config 重载逻辑）；run_startup 取 fork（函数外迁至 run_startup_recovery.py）；run_startup_recovery 移植上游演进体（返回 list[str]/bool 驱动重载）；hermes_state 融合（上游 pinned 连接快路径 + fork owner_loop_db_guard 守卫）；test_code_health_cli 双 hunk 并集；e2e spec 取上游重写版 | 125 passed + imports OK + 插件仍 6/8（同前） |
| 2026-11-10 | 3070a6a766（他会话收口，账本补记） | 1310 文件/1796 提交 | 6 冲突由他会话按本账本裁决哲学解决：aux 两 hunk import 并集；gateway/run.py 状态交替并集（拆两行）；lefthook 当时取 just ruff；weixin 融合（上游 describe_proxy_error 脱敏 + fork park_on_not_ready 暂存，注释中文化）；surfaces.json 重测实测值；uv.lock 取 fork 基底 | 未由本会话执行；1111 轮补记 |
| 2026-11-11 | cdb494051d | 1154 文件/1624 提交 | 6 冲突：lint.yml CI 步骤并集（fork owner-loop-io + 上游 profile-create-entry）；run_turn.py 取我方（方法已迁 run_turn_prepare.py）+ 移植上游 _cleanup_adapter 参数入迁出副本（消费方 run_turn_bubble_cleanup.py:43 实证必要）；scripts/check 守卫并集；surfaces.json/delegate_task.json 取我方实测值（fork 工具面）；test_code_health_cli 并集 | 166 passed + imports OK + 插件 6/8 |

## 二、防冲突铁律（同步操作规程）

1. 频率: 每日 fetch upstream; 官方日均 ~50-250 提交, 拖越久冲突面越大。
2. 合并方式: git merge upstream/main（保留双方历史）; 禁 rebase（重写历史→force push 灾难）。
3. 合并前: git status 必须干净; 有未提交改动先 commit。
4. 冲突处置判据: 冲突文件属"本地独有面"→本地侧保留; 官方文件被本地改过（不该存在）→取官方侧并按插件层重新外置。
5. 合并后三验:
   scripts/run_tests.sh tests/agent/test_read_think_gate_wiring.py tests/agent/test_read_think_gate_classifier_switch.py
   python3 -c "import agent.tool_executor, hermes_cli.plugins; print('imports OK')"
   桌面: npx vitest run src/app/capabilities/
6. 推送: origin main 即时推; 临时同步分支用完即删。

## 三、收窄路线（进一步降冲突面, 按需执行）

- tool_executor 78 行→0: 上游已有 pre_tool_call/post_llm_call 通用 hook; 若官方未来提供等价
  batch hook, 本地 78 行立即删除改用官方面。跟踪: diff hermes_cli/plugins.py::VALID_HOOKS。
- plugins.py 8 行: 等 pre_tool_batch 上游 PR 合入后清零。
- turn_facade_lease 4 行: 非关键语义（本地等待体验偏好）, 上游改同参数时本地让位。
- 桌面 2 测试修正: 提 upstream PR（jsdom29 Storage Proxy spy 语义 + locale 无关断言）, 合入后本地清零。

## 四、同步命令序列（复制即用）

git fetch upstream main
git status --short                    # 必须为空
git merge upstream/main --no-edit
# 冲突 → 按第二节第4条判据处置
scripts/run_tests.sh tests/agent/test_read_think_gate_wiring.py tests/agent/test_read_think_gate_classifier_switch.py
python3 -c "import agent.tool_executor, hermes_cli.plugins; print('imports OK')"
git push origin main
git rev-list --count main..upstream/main   # 应为 0

### 1102 矫正记录

- b37661ba13（1101 轮账本提交）实际包含 54 文件（409+/213-）：1101 轮 `git stash apply` 冲突时 git 将无冲突恢复文件全部暂存，随后账本 commit 连带提交了归属会话当时的工作区 WIP。提交信息「docs(ledger)」与内容不符。内容本体为归属会话完整工作（其后继 6 提交在其上持续精炼：f5b688e7f9/3bc0a387a2/7b17134e94/5d148be8ee/2cc3d2f9d9/0985aa13bc），保留不重写。教训入技能：stash apply 冲突后必须 `git status` 区分暂存态，提交前核对 `git diff --cached --stat`。

### 1103 补记 5 项（交集内文件, 族谱行截断展开致未具名, 归属已 git log 复核）

| 文件 | 归属 |
|---|---|
| agent/agent_init.py | 1101 WIP（经 b37661ba13 入库） |
| gateway/relay/media.py | 1101 WIP（经 b37661ba13 入库） |
| gateway/run_adapters.py | 1101 WIP（经 b37661ba13 入库） |
| gateway/session_controls.py | 7b34e3a917（SQLite 移出 owner loop） |
| tests/hermes_cli/test_cli_modern_layout.py | 1101 WIP（经 b37661ba13 入库） |

### 1102 补记 4 项（1102 对账报告实证漏记，归属复核 git log 已验）

| 文件 | 定制内容 | 归属提交 |
|---|---|---|
| agent/turn_finalizer.py | finalize_turn 4 行定制 | 1101 WIP（经 b37661ba13 入库） |
| gateway/session_ingress.py | SQLite 移出 owner loop 诊断 6 行 | 7b34e3a917 |
| gateway/slash_commands_model.py | 切换持久化失败响亮可见 4 行 | 72344632e2 |
| tools/terminal_tool_guards.py | block_escalation 错误/阻塞分流 10 行 | ce9faf8ce0 |

### 1103 同步记录

- merge c4a3246345 (77 提交, 74 文件 3428+/328-, 零冲突)。预交集 2 文件
  (turn_facade_lease/delegate_tool) 均自动合并, 定制签名逐项核对存活:
  tool_executor/_run_pre_tool_batch_hooks×3, session_viewer_automation×4,
  coding_context/_NON_CODING_SKILL_CATEGORIES, LEASE_WAIT_SECONDS=300,
  browser inflight_session_scope, delegate_tool 限制行。
- 合并前工作区 2 未提交文件(session_ancillary.py 锁外移 + useLongRunToolCharms.ts
  节奏改进)以保护性提交 2c2edcf6d9 入库(上游未触碰两文件, 零冲突)。
- 三验: read_think_gate 16/16 绿(run_tests.sh), imports OK, desktop vitest
  144/144 绿(capabilities+updates-overlay.report)。
- workspace 副本: 现存 8 份(2 installs), merge 改动包(agent/gateway/hermes_cli/
  pm/tools)rsync 同步(副本单向落后坐实, 0 副本独有文件), 逐环境 venv 中立目录
  env -i 探针 8/8 通过(解析到各自 workspace 副本且含新模块 pm/carried_extras)。
  注意: 探针须从中立 cwd 跑——python -c 会把 cwd 注入 sys.path, 在 fork 仓内跑
  会假绿为 fork 直连。

### 1106 补记 21 项（fork-guard fresh audit 红灯复核）

| 文件 | 定制内容 | 归属提交 |
|---|---|---|
| gateway/run_turn_runner.py | shared clarify 注册写入 owner loop 下沉 | b232236990 |
| gateway/run_bootstrap.py | READY 发布等待 off-loop hosted readiness | a8e085bea7 |
| gateway/session_detached.py | detached 启动恢复与 lineage 读下沉 | 3fbbff1a15/882593041f2 |
| gateway/session_envelope.py | native/automation route preflight 的 admission/routing 读下沉 | 9adf243b69 |
| gateway/session_local_recovery.py | 本地恢复、adoption 与 close receipt 读下沉 | 3fbbff1a15/b31509c203 |
| gateway/session_route_recovery.py | detached mount plan 与 lineage 读下沉 | 882593041f2 |
| gateway/session_settlement_recovery.py | settlement 恢复读与 publication 下沉 | 4a73fc2498 |
| gateway/session_viewer_close.py | viewer close 的 busy/idle/retirement 读下沉 | 4a73fc2498 |
| hermes_state_local.py | 本地 receipt 配置内容寻址去重，兼容旧 inline 读 | 4e1abf0bfb |
| hermes_state_local_lineage.py | branch/reset 读路径适配 config blob 解码 | 4e1abf0bfb |
| hermes_state_mutation_branch.py | branch 事务写读适配 config blob 编解码 | 4e1abf0bfb |
| hermes_state_mutation_prepared.py | prepared mutation 快照/提交适配 config blob | 4e1abf0bfb |
| hermes_state_mutation_reset.py | reset 事务 receipt 解码适配 config blob | 4e1abf0bfb |
| tests/gateway/test_bot_chat_unadoptable_local_title.py | Bot Chat 标题读下沉契约 | 4a73fc2498 |
| tests/gateway/test_local_session_orphan_end.py | 本地 orphan/adoption 恢复契约 | 3fbbff1a15 |
| tests/gateway/test_viewer_close_lifecycle.py | viewer close busy 检查契约 | 4a73fc2498 |
| tests/gateway/fixtures/operator_cross_surface_peer.py | READY async 化调用契约 | a8e085bea7 |
| tests/gateway/test_native_http_auth.py | READY async 化调用契约 | a8e085bea7 |
| tests/gateway/test_runtime_bootstrap_lifecycle.py | READY async 化调用契约 | a8e085bea7 |
| tests/gateway/test_runtime_listener_supervision.py | READY async 化与监听失败清理契约 | a8e085bea7 |
| tests/gateway/test_runtime_token_identity.py | READY async 化调用契约 | a8e085bea7 |

### 1106 补记 3 项（workflow skills 提交后复核）

| 文件 | 定制内容 | 归属提交 |
|---|---|---|
| skills/software-development/requesting-code-review/SKILL.md | 评审改为需求/架构/安全/证据四线核查，显式禁止自动委托、自动修复与自动提交 | c6d5dab042 |
| skills/software-development/systematic-debugging/SKILL.md | 根因调试改为证据、模式、假设、修复、纵深防御五阶段，并接入 Hermes 测试运行器 | c6d5dab042 |
| skills/software-development/test-driven-development/SKILL.md | TDD 改为纵向行为切片与仓库 canonical runner，消除裸框架调用示范 | c6d5dab042 |

### 1107 补记 2 项（gateway 拨号重试提交后复核）

| 文件 | 定制内容 | 归属提交 |
|---|---|---|
| hermes_cli/gateway_connection.py | 传输类首次拨号失败在调用 deadline 内有界重试；网关已判定的不可用不盲等 | 87da001684 |
| tests/hermes_cli/test_gateway_connection_reconnect.py | 区分 resume 的 owner 拒答与传输超时：前者下次调用重拨，后者同调用重拨 | 87da001684 |

## 五、机器账本（SSOT）

> 叙述表格只解释意图；本 YAML 块是 `fork-guard audit` 唯一机器事实源。
> 口径：`git diff --name-only <merge-base>..HEAD` ∩ `<merge-base>` 树中已存在路径。
> 今后改动 baseline 已存在文件，必须在同一变更中更新本清单。

```yaml
version: 1
baseline: b3746e35ddd32079697728e2df57a9d78f1e9e32
official_paths:
- .github/workflows/lint.yml
- .gitignore
- AGENTS.md
- acp_adapter/gateway_server.py
- acp_adapter/tools.py
- agent/agent_init.py
- agent/auxiliary_client.py
- agent/auxiliary_reasoning_floor.py
- agent/codex_runtime.py
- agent/coding_context.py
- agent/context_compressor.py
- agent/context_compressor_telemetry.py
- agent/conversation_compression.py
- agent/curator.py
- agent/curator_backup.py
- agent/display.py
- agent/error_classifier.py
- agent/periodic_scheduler.py
- agent/prepared_compaction.py
- agent/prompt_builder.py
- agent/relay_runtime.py
- agent/retry_utils.py
- agent/skill_utils.py
- agent/system_prompt.py
- agent/tool_dispatch_helpers.py
- agent/tool_executor.py
- agent/tool_guardrails.py
- agent/tool_result_classification.py
- agent/turn_context.py
- agent/turn_facade_lease.py
- agent/turn_finalizer.py
- agent/turn_stop_gates.py
- agent/turn_tool_round.py
- apps/desktop/e2e/boot-failure.spec.ts
- apps/desktop/e2e/boot.spec.ts
- apps/desktop/e2e/interim-messages.spec.ts
- apps/desktop/e2e/onboarding.spec.ts
- apps/desktop/e2e/sidebar-states.spec.ts
- apps/desktop/e2e/unread-dot-restart.spec.ts
- apps/desktop/e2e/warm-resume-jitter.spec.ts
- apps/desktop/electron/connection-config.test.ts
- apps/desktop/scripts/gen-share-codes.ts
- apps/desktop/src/plugins/hermes-bots/cron-detail.test.tsx
- apps/desktop/src/store/voice-prefs.test.ts
- cron/cancellation.py
- cron/scheduler.py
- cron/scheduler_authority.py
- cron/scheduler_delivery.py
- cron/scheduler_delivery_room.py
- cron/scheduler_liveness.py
- cron/scheduler_ownership.py
- cron/scheduler_provider.py
- cron/scheduler_script.py
- cron/scheduler_tick.py
- cron/unreachable_retry.py
- evals/postmortem/review_probes/goal_scope_probe.py
- gateway/config.py
- gateway/config_loader.py
- gateway/control_socket.py
- gateway/host_attach.py
- gateway/hosted_rooms_gateway_import.py
- gateway/platforms/api_server.py
- gateway/platforms/webhook_ingress.py
- gateway/relay/media.py
- gateway/run.py
- gateway/run_adapters.py
- gateway/run_adapters_recovery.py
- gateway/run_api.py
- gateway/run_bootstrap.py
- gateway/run_goals.py
- gateway/run_inbound.py
- gateway/run_notifications.py
- gateway/run_notifications_update.py
- gateway/run_profile_reconcile.py
- gateway/run_runtime.py
- gateway/run_runtime_init.py
- gateway/run_startup.py
- gateway/run_turn.py
- gateway/run_turn_hygiene.py
- gateway/run_turn_prepare.py
- gateway/run_turn_runner.py
- gateway/run_turn_runner_approval_settle.py
- gateway/session_a2a.py
- gateway/session_ancillary.py
- gateway/session_api_turn.py
- gateway/session_authority.py
- gateway/session_bot.py
- gateway/session_busy_controls.py
- gateway/session_controls.py
- gateway/session_cron.py
- gateway/session_detached.py
- gateway/session_envelope.py
- gateway/session_group_controls.py
- gateway/session_hosted_rpc.py
- gateway/session_hosted_service.py
- gateway/session_ingress.py
- gateway/session_local_inventory.py
- gateway/session_local_migration.py
- gateway/session_local_recovery.py
- gateway/session_route_recovery.py
- gateway/session_settlement_recovery.py
- gateway/session_viewer_automation.py
- gateway/session_viewer_close.py
- gateway/slash_commands.py
- gateway/slash_commands_goals.py
- gateway/slash_commands_model.py
- gateway/status.py
- hermes_cli/_parser.py
- hermes_cli/_startup_fast.py
- hermes_cli/banner.py
- hermes_cli/cli_chat_turn_mixin.py
- hermes_cli/cli_init_mixin.py
- hermes_cli/cli_loops_mixin.py
- hermes_cli/cli_model_switch_mixin.py
- hermes_cli/cli_modern_layout.py
- hermes_cli/cli_stream_mixin.py
- hermes_cli/cli_subagent_monitor.py
- hermes_cli/cli_tui_mixin.py
- hermes_cli/config_defaults.py
- hermes_cli/desktop_update_verify.py
- hermes_cli/doctor_state.py
- hermes_cli/doctor_tools.py
- hermes_cli/gateway.py
- hermes_cli/gateway_client.py
- hermes_cli/gateway_connection.py
- hermes_cli/gateway_launchd.py
- hermes_cli/gateway_migrate.py
- hermes_cli/gateway_profile_lifecycle.py
- hermes_cli/goal_command.py
- hermes_cli/goals.py
- hermes_cli/hooks.py
- hermes_cli/kanban_db_dispatch.py
- hermes_cli/main.py
- hermes_cli/observability/shared_metrics_prepared_compaction.py
- hermes_cli/plugin_validate.py
- hermes_cli/plugins.py
- hermes_cli/plugins_cmd.py
- hermes_cli/plugins_cmd_listing.py
- hermes_cli/plugins_context.py
- hermes_cli/plugins_dispatch.py
- hermes_cli/plugins_ledger.py
- hermes_cli/plugins_manifest.py
- hermes_cli/post_update.py
- hermes_cli/stderr_timestamp.py
- hermes_cli/subcommands/plugins.py
- hermes_cli/update_cmd_windows.py
- hermes_cli/update_pause_record.py
- hermes_cli/version_info.py
- hermes_cli/worktree_sweep.py
- hermes_state.py
- hermes_state_dbfile.py
- hermes_state_local.py
- hermes_state_local_lineage.py
- hermes_state_maintenance.py
- hermes_state_mutation_branch.py
- hermes_state_mutation_prepared.py
- hermes_state_mutation_reset.py
- hermes_state_readpool.py
- locales/af.yaml
- locales/ar.yaml
- locales/de.yaml
- locales/en.yaml
- locales/es.yaml
- locales/fr.yaml
- locales/ga.yaml
- locales/hu.yaml
- locales/it.yaml
- locales/ja.yaml
- locales/ko.yaml
- locales/pt.yaml
- locales/ru.yaml
- locales/tr.yaml
- locales/uk.yaml
- locales/zh-hant.yaml
- locales/zh.yaml
- plugins/image_gen/openai/__init__.py
- plugins/model-providers/zai/__init__.py
- plugins/platforms/dingtalk/adapter.py
- plugins/platforms/dingtalk/gateway_env.py
- plugins/platforms/dingtalk/plugin.yaml
- plugins/platforms/feishu/adapter.py
- plugins/platforms/photon/adapter.py
- plugins/platforms/weixin/adapter.py
- plugins/security-guidance/patterns.py
- pm/index_config.py
- pm/shell.py
- pyproject.toml
- ruff.pre-commit.toml
- run_agent.py
- scripts/check
- scripts/ci/prompt_surface.py
- scripts/install_root_lint_baseline.json
- scripts/run_tests_parallel.py
- scripts/sandbox/pick-release-tags.sh
- skills/software-development/requesting-code-review/SKILL.md
- skills/software-development/systematic-debugging/SKILL.md
- skills/software-development/test-driven-development/SKILL.md
- tests/agent/test_auxiliary_client.py
- tests/agent/test_auxiliary_error_classification.py
- tests/agent/test_auxiliary_reasoning_floor.py
- tests/agent/test_coding_context.py
- tests/agent/test_curator_backup.py
- tests/agent/test_error_classifier.py
- tests/agent/test_periodic_scheduler.py
- tests/agent/test_pre_llm_block_hook.py
- tests/agent/test_prepared_compaction.py
- tests/agent/test_skill_utils.py
- tests/agent/test_subagent_lifecycle.py
- tests/ci/workflow_steps.py
- tests/cron/test_cron_inactivity_timeout.py
- tests/cron/test_cron_recover_pause.py
- tests/cron/test_cron_script.py
- tests/cron/test_lifecycle_guard_profile_scan.py
- tests/cron/test_misfire_catchup.py
- tests/cron/test_unreachable_retry.py
- tests/fixtures/prompt_surface/prompts/cli-code.claude.txt
- tests/fixtures/prompt_surface/prompts/cli.claude.txt
- tests/fixtures/prompt_surface/prompts/cli.gemini.txt
- tests/fixtures/prompt_surface/prompts/cli.gpt.txt
- tests/fixtures/prompt_surface/prompts/cli.hermes.txt
- tests/fixtures/prompt_surface/prompts/cron.claude.txt
- tests/fixtures/prompt_surface/prompts/desktop.claude.txt
- tests/fixtures/prompt_surface/prompts/discord.claude.txt
- tests/fixtures/prompt_surface/prompts/telegram.claude.txt
- tests/fixtures/prompt_surface/surfaces.json
- tests/fixtures/prompt_surface/tools/delegate_task.json
- tests/fixtures/prompt_surface/tools/skills_list.json
- tests/fixtures/prompt_surface/tools/write_file.json
- tests/fixtures/provider_wire_snapshot.json
- tests/gateway/fixtures/operator_cross_surface_peer.py
- tests/gateway/fixtures/tui_desktop_routes_peer.py
- tests/gateway/test_api_admission_off_loop.py
- tests/gateway/test_api_run_steer_before_adopt.py
- tests/gateway/test_bootstrap_deferred_integrity_scan.py
- tests/gateway/test_bot_chat_unadoptable_local_title.py
- tests/gateway/test_busy_followup_guard_release.py
- tests/gateway/test_dingtalk.py
- tests/gateway/test_dingtalk_reconnect_circuit_breaker.py
- tests/gateway/test_feishu_card_action_gateway_authz.py
- tests/gateway/test_feishu_profile_scope_hops.py
- tests/gateway/test_gateway_trust_env.py
- tests/gateway/test_goal_continuation_drain.py
- tests/gateway/test_hosted_mux_runtime.py
- tests/gateway/test_hosted_rooms_gateway_import.py
- tests/gateway/test_local_session_orphan_end.py
- tests/gateway/test_loop_liveness_watchdog.py
- tests/gateway/test_multiplex_hot_serve.py
- tests/gateway/test_multiplex_residue_parity.py
- tests/gateway/test_native_http_auth.py
- tests/gateway/test_readiness.py
- tests/gateway/test_resume_display_history.py
- tests/gateway/test_run_runtime_hot_serve_cleanup.py
- tests/gateway/test_runtime_bootstrap_lifecycle.py
- tests/gateway/test_runtime_listener_supervision.py
- tests/gateway/test_runtime_token_identity.py
- tests/gateway/test_session_authority_adopted_stop.py
- tests/gateway/test_session_client_env.py
- tests/gateway/test_session_hosted_daemon.py
- tests/gateway/test_session_steer_before_adopt.py
- tests/gateway/test_session_viewer_automation.py
- tests/gateway/test_startup_recovery_parks_profile.py
- tests/gateway/test_status.py
- tests/gateway/test_viewer_close_lifecycle.py
- tests/gateway/test_watch_notification_reset_pin.py
- tests/hermes_cli/test_banner.py
- tests/hermes_cli/test_cli_gateway_model_route.py
- tests/hermes_cli/test_cli_goal_interrupt.py
- tests/hermes_cli/test_cli_goal_parked_resume.py
- tests/hermes_cli/test_cli_modern_layout.py
- tests/hermes_cli/test_cli_user_message_preview.py
- tests/hermes_cli/test_desktop_swap_file_lock_retry.py
- tests/hermes_cli/test_desktop_update_verify.py
- tests/hermes_cli/test_doctor_journal_modes.py
- tests/hermes_cli/test_doctor_wal_checkpoint_guard.py
- tests/hermes_cli/test_gateway_automation.py
- tests/hermes_cli/test_gateway_connection_reconnect.py
- tests/hermes_cli/test_gateway_disconnect_outcome.py
- tests/hermes_cli/test_gateway_multiplex_status.py
- tests/hermes_cli/test_gateway_runtime_ensure.py
- tests/hermes_cli/test_goals.py
- tests/hermes_cli/test_inventory_pricing.py
- tests/hermes_cli/test_kanban_db.py
- tests/hermes_cli/test_lazy_secrets_dispatch.py
- tests/hermes_cli/test_model_options_account_usage.py
- tests/hermes_cli/test_plugin_config_state_bridge.py
- tests/hermes_cli/test_plugin_manifest_v2.py
- tests/hermes_cli/test_plugin_prompt_sections.py
- tests/hermes_cli/test_plugin_validate.py
- tests/hermes_cli/test_plugins.py
- tests/hermes_cli/test_plugins_cmd_activation_keys.py
- tests/hermes_cli/test_post_update.py
- tests/hermes_cli/test_real_home_tripwire.py
- tests/hermes_cli/test_relay_shared_metrics_runtime.py
- tests/hermes_cli/test_session_recovery_lost_and_found.py
- tests/hermes_cli/test_sessions_held_store_gate.py
- tests/hermes_cli/test_shared_metrics_harness.py
- tests/hermes_cli/test_shared_metrics_prepared_compaction.py
- tests/hermes_cli/test_source_check.py
- tests/hermes_cli/test_startup_fast_guards.py
- tests/hermes_cli/test_stderr_timestamp.py
- tests/hermes_cli/test_subagent_monitor_gateway.py
- tests/hermes_cli/test_uninstall_full_wipe_sweep.py
- tests/hermes_cli/test_update_autostash.py
- tests/hermes_cli/test_update_finish.py
- tests/hermes_cli/test_update_fleet_restart_pending.py
- tests/hermes_cli/test_update_interrupted_pull.py
- tests/hermes_cli/test_update_target_identity.py
- tests/hermes_cli/test_update_zip_webapp_preserve.py
- tests/hermes_cli/test_user_providers_model_switch.py
- tests/hermes_cli/test_version_command.py
- tests/hermes_cli/test_version_info.py
- tests/hermes_cli/test_worktree_snapshot_prune.py
- tests/hermes_state/test_cross_vm_fs_wal_refusal.py
- tests/hermes_state/test_deleted_wal_generation_guard.py
- tests/hermes_state/test_guest_durability_barriers.py
- tests/home_io_guard.py
- tests/plugins/test_a2a_plugin.py
- tests/plugins/test_security_guidance_plugin.py
- tests/pm/test_declined_extras.py
- tests/pm/test_shell_discovery.py
- tests/scripts/test_code_health_cli.py
- tests/scripts/test_run_tests_parallel.py
- tests/scripts/test_run_tests_parallel_stdio.py
- tests/scripts/test_run_tests_parallel_timeout_scaling.py
- tests/scripts/test_source_driver.py
- tests/skills/test_document_env_blocks.py
- tests/tools/conftest.py
- tests/tools/test_bot_desktop_resources.py
- tests/tools/test_browser_launcher.py
- tests/tools/test_browser_use_cli.py
- tests/tools/test_budget_config.py
- tests/tools/test_connector_bridge_wiring.py
- tests/tools/test_delegate.py
- tests/tools/test_mcp_tool.py
- tests/tools/test_mcp_tool_session_expired.py
- tests/tools/test_process_registry.py
- tests/tools/test_skills_tool.py
- tests/tools/test_terminal_pipeline_status.py
- tests/tools/test_tool_result_storage.py
- tests/tui_gateway/test_backend_heartbeat.py
- tests/tui_gateway/test_compaction_status.py
- tests/tui_gateway/test_legacy_submit_forwards_to_owner.py
- tests/tui_gateway/test_multi_client_fanout.py
- tests/tui_gateway/test_profiles_list_canonical_session.py
- tests/tui_gateway/test_profiles_list_session_field_cache.py
- tests/tui_gateway/test_profiles_list_ui_meta_cache.py
- tests/tui_gateway/test_protocol.py
- tools/approval_detection.py
- tools/browser_session.py
- tools/browser_session_lifecycle.py
- tools/browser_use_cli.py
- tools/budget_config.py
- tools/code_execution_tool.py
- tools/delegate_tool.py
- tools/delegate_tool_progress.py
- tools/file_operations.py
- tools/file_operations_common.py
- tools/file_tools.py
- tools/mcp_tool_config.py
- tools/mcp_tool_errors.py
- tools/mcp_tool_handlers.py
- tools/mcp_tool_registration.py
- tools/mcp_tool_transport.py
- tools/memory_tool_store.py
- tools/process_registry.py
- tools/process_registry_checkpoint.py
- tools/skill_manager_batch.py
- tools/skills_tool.py
- tools/terminal_tool.py
- tools/terminal_tool_guards.py
- tools/tool_labels.py
- tools/tool_result_storage.py
- tools/tool_search_validation.py
- tools/transcription_local.py
- tui_gateway/hosted_room_driver.py
- tui_gateway/launch_profile_policy.py
- tui_gateway/prompt_turn.py
- tui_gateway/session_compression.py
- tui_gateway/ws.py
- ui-tui/scripts/bench-native-history.tsx
- ui-tui/scripts/bench-streaming-md.tsx
- ui-tui/scripts/i18n-export-en.ts
- ui-tui/src/app/useLongRunToolCharms.ts
- uv.lock
- website/docs/developer-guide/context-compression-and-caching.md
- website/docs/developer-guide/plugins/index.md
- website/docs/reference/cli-commands.md
- website/docs/reference/mcp-config-reference.md
- website/docs/reference/skills-catalog.md
- website/docs/user-guide/cli.md
- website/docs/user-guide/configuration.md
- website/docs/user-guide/features/curator.md
- website/docs/user-guide/features/hooks.md
- website/docs/user-guide/features/mcp.md
- website/docs/user-guide/features/plugins.md
- website/docs/user-guide/messaging/dingtalk.md
- website/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review.md
- website/docs/user-guide/skills/bundled/software-development/software-development-systematic-debugging.md
- website/docs/user-guide/skills/bundled/software-development/software-development-test-driven-development.md
- website/sidebars.ts
```
