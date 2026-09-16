# 上游同步合并报告 — 2026-09-12

合并提交：33e2ccf50c（merge refs/remotes/official/main d76856cc69）
前置盘点：docs/investigations/upstream-sync-gap-20260912.md（1861 提交/8 天窗口）
backup 分支：backup/pre-sync-20260912（f09645a262）

## 规模

- 官方 main：d20a8e4475 (09-05) → d76856cc69 (09-12)，1877 commits
- 2880 files changed, +173796 / -24639
- 冲突文件 15 个，全部逐块语义裁定后手工解决

## 冲突裁定记录（15 文件）

| 文件 | 块数 | 裁定 | 理由 |
|---|---|---|---|
| agent/context_compressor.py | 3 | 全取官方 | 官方 0f4587e336 (09-06) 从架构上删除 projection baseline 机制（_pending_request_rough_tokens/note_request_omits_usage 等全链删除，-101 行），用 usage-anchor 单请求延迟判定替代；fork 的 ratio 修复 (c985b5fc40, 09-01) 所依赖的记账 API 已无调用方 |
| tests/agent/test_context_compressor.py | 1 | 取官方 | ratio 两用例随机制删除；native checkpoint 用例保留 |
| agent/model_metadata.py | 1 | 取官方 | 官方 09-09 全量 GLM 目录（5.3=1310720 等 13 键）取代 fork 的 4 键精简版；glm-5.2=1M 两版一致 |
| tests/agent/test_model_metadata.py | 1 | 融合 | 保留 fork 的 GLM 回归类（不变量化：turbo≤202752），新增 5.3=1310720 断言，官方 OpenRouter variant 类保留 |
| agent/turn_recovery.py | 1 | 取官方+保留floor | 官方 a227484906 Retry-After 扩展到 5xx+body.retry_after；fork 的 min_wait_seconds 下限逻辑 (3935df8361) 在官方侧代码之后独立保留，零冲突 |
| tools/file_tools.py | 1 | 融合 | fork 指纹门（a1f99cd8ef）+ 官方 rewrite_hint（a1ffb27ef8，fork 修复被官方吸收）顺序串联 |
| hermes_cli/commands_platforms.py | 1 | 并集 | _SLACK_VIA_HERMES_ONLY = fork 的 audit + 官方的 login（两命令在 commands.py 均存在） |
| hermes_cli/goals.py | 4 | 融合 | 官方 active_delegations（8477292f0f WAIT-on-delegation）+ fork tool_calls_summary/turn_reasons 双通道并存；模板三形态全保留双 block |
| hermes_cli/cli_loops_mixin.py | (合并后补) | fork 语义找回 | evaluate_after_turn 补传 tool_calls_summary（8afb2ca1ea 反空口白话死循环修复在官方 66366d3dab 拆分中丢失）|
| gateway/run_goals.py | (合并后补) | fork 语义找回 | 同上：_run_post_turn_hooks 从 agent_result["messages"] 提取 tool_calls_summary 传入 judge |
| tui_gateway/prompt_turn.py | (合并后补) | fork 语义找回 | 同上：session["history"] 提取传入 |
| tools/checkpoint_manager.py | 14 | 逐块 | 官方三修复吸收：e70db09f51 (_resolve_checkpoint_base per-call profile 解析)、d77df6674a+ f8c9e93dad (-z NUL 字面路径)；fork 保留：lock-contention janitor+retry (13bcaf7b6c)、staging 熔断、SnapshotFailedError、旧架构 prune 完整体；官方重构版重复定义 prune_checkpoints 丢弃；补 _store_has_head 定义 |
| tests/gateway/test_scale_to_zero.py | 2 | 取官方 | short_sock_dir fixture（官方 sun_path 104B 修复）配套改名 |
| tests/gateway/test_buzz_adapter.py | 1 | 取官方 | while 循环版保证路径必超 900 bound（fork 固定 3x150 在浅 tmp_path 下不足） |
| tests/gateway/test_systemd_notify.py | 1 | 取官方 | linux_only 集中标记替代 fork 手写 probe（conftest.py:1167 已注册） |
| tests/hermes_cli/conftest.py | 1 | 并集 | no_real_launchd (fork, 3 文件消费) + isolated_update_runtime (官方新) 双 fixture 共存 |
| tests/hermes_cli/test_cmd_update.py | 1 | 取官方 | _patch_gateway_discovery 降为 isolated_update_runtime pass-through |
| apps/desktop/electron/remote-lifecycle.ts | 1 | 取官方 | mutexPath 裸嵌两版一致（fork 09e9161872 已被官方等效吸收），仅注释差异；终态与官方 diff=0 |

## 验证

- 10 个直接改动的 py 文件：ast.parse 全过 + import 全过 [实测]
- 窄集 1（checkpoint_manager + path_roundtrip + context_compressor）：217 passed, 0 failed [实测]
- 窄集 2（goal_gates + model_metadata + cmd_update + update_yes_flag + scale_to_zero + buzz + systemd）：413 passed, 0 failed [实测]
- 窄集 3（file_tools + goals + plugins/）：85 passed, 1 flaky（hindsight prefetch 时序，单测重跑通过，与合并无关 [实测]）
- 全量套件：后台运行中，结果见 /tmp/full_suite_result.txt

## 已知保留差异（fork 语义，下次同步勿再丢）

1. checkpoint lock-contention janitor + retry + staging 熔断 + SnapshotFailedError
2. goal judge 的 tool_calls_summary/turn_reasons 证据通道（CLI/gateway/TUI 三处调用点）
3. write_file 指纹门（expected_fingerprint 严格拒绝 + registry 警告）
4. agent.rate_limit.min_wait_seconds 429 等待下限
5. _SLACK_VIA_HERMES_ONLY 含 audit
