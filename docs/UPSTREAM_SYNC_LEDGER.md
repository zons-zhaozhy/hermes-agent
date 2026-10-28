# 上游同步防冲突账本（SSOT）

最后核验: 2026-10-21 · main 站上 upstream/main 0b8dee00fa (590 提交合并), 落后 0
重叠面 13 文件, 唯一冲突 gateway/run_notifications.py: 本地 6b9786e161 requeue 死信
vs 官方 #140899 指数退避 hold——官方等价机制已上游化, 取官方侧, 本地定制+配套测试退役(见下)
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
| plugins/** + tests/plugins/** | 113 文件纯新增 | 零（官方不存在的命名空间, 官方永不改） | 长期保留 |
| agent/tool_executor.py | 78 行: _run_pre_tool_batch_hooks/_first_batch_block/_extract_block_message + 两调用点; 发射器契约已改为"callback 崩溃/超时 fail-closed(见 plugins_dispatch), 仅 dispatch 传输层 fail-open" | 中（官方改 dispatch 逻辑时撞） | 见收窄路线 |
| agent/auxiliary_client.py | 13 行新增(ef562cb74a): judge 超时后 _mark_provider_unhealthy 短TTL隔离(慢端点暂离池, 同429隔离类) | 中（官方改 aux 降级/健康逻辑时撞） | 等官方 aux 侧等价机制后清零; 曾漏记账本, 0924补 |
| hermes_cli/plugins.py | VALID_HOOKS + pre_tool_batch 注释更新为 policy/fail-closed 契约 | 低（hook 名单追加式） | 等 upstream 化 |
| hermes_cli/plugins_dispatch.py | pre_tool_batch 加入 _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS + _hook_timeout_block_message 泛化 | 低（分类集追加式） | 与 pre_tool_call 同机制 |
| agent/turn_facade_lease.py | 4 行: 等待上限 1800s→300s | 低 | 上游改同参数时让位 |
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
| tools/browser_tool_lifecycle.py + browser_use_cli.py + browser_tool.py + tests/test_browser_inflight_reaper.py (1023 移除) | f9203bf53a inflight 保护: >120s 长调用被闲置回收器误杀的根治; 官方 68500d9ff9 删除 agent-browser 与旧 lifecycle, 载体消失, 1023 同步时跟随官方删除; 官方新架构 cleanup_inactive_sessions 仍无执行中豁免, 缺口仍在待移植 | 已消除(载体删除) | 移植 inflight 语义到 browser_session_lifecycle janitor 后提 upstream |
| agent/coding_context.py (1028 新增) | Fork 定制 1: _NON_CODING_SKILL_CATEGORIES deny-list 扩 5 自定义类(engineering-methodology/frontend/product/business/autonomous-ai-agents), 仓库内 coding/focus 会话技能索引降级 names-only, 省 ~5.1K 字符/轮 (26,464→21,397); 官方契约样例 github/devops/software-development/data-science 保持全描述. Fork 定制 2: compact_skill_categories() 对显式 focus + 交互编码面(cli/tui/acp/desktop) + 非仓库 cwd 也降级(用户从 $HOME 启动 CLI 的实际场景), 非编码面(telegram 等)保持全量 | 中（官方改 deny-list/技能索引逻辑时撞） | 官方若给自定义类加等级机制或改 focus 判定则让位; 配套: 24 workspace 副本已同步; 测试 test_skill_demotion_gated_on_focus + test_focus_demotes_outside_workspace_on_coding_surface 锁定契约 |

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
