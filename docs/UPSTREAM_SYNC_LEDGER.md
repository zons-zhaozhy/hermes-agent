# 上游同步防冲突账本（SSOT）

最后核验: 2026-09-24 · main 站上 upstream/main f799fd8578, 落后 0

> 注意: `git rev-list --count main..upstream/main` 在官方导入完整历史后不再可信
> (仓库现有 11 个根提交, upstream 可达 ~39.9k vs fork 可达 ~206)。真实落后量用
> `git log --format='%ci' <merge-base>..upstream/main | awk '$1>="日期"'` 按日期计数。

## 一、本地独有面（= 唯一可能冲突的地方, 已压到最小）

| 文件 | 差异 | 冲突风险 | 处置 |
|---|---|---|---|
| plugins/** + tests/plugins/** | 113 文件纯新增 | 零（官方不存在的命名空间, 官方永不改） | 长期保留 |
| agent/tool_executor.py | 78 行: _run_pre_tool_batch_hooks/_first_batch_block/_extract_block_message + 两调用点; 发射器契约已改为"callback 崩溃/超时 fail-closed(见 plugins_dispatch), 仅 dispatch 传输层 fail-open" | 中（官方改 dispatch 逻辑时撞） | 见收窄路线 |
| agent/auxiliary_client.py | 13 行新增(ef562cb74a): judge 超时后 _mark_provider_unhealthy 短TTL隔离(慢端点暂离池, 同429隔离类) | 中（官方改 aux 降级/健康逻辑时撞） | 等官方 aux 侧等价机制后清零; 曾漏记账本, 0924补 |
| hermes_cli/plugins.py | VALID_HOOKS + pre_tool_batch 注释更新为 policy/fail-closed 契约 | 低（hook 名单追加式） | 等 upstream 化 |
| hermes_cli/plugins_dispatch.py | pre_tool_batch 加入 _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS + _hook_timeout_block_message 泛化 | 低（分类集追加式） | 与 pre_tool_call 同机制 |
| agent/turn_facade_lease.py | 4 行: 等待上限 1800s→300s | 低 | 上游改同参数时让位 |
| tests/agent/test_read_think_gate_*.py | 2 文件纯新增 | 零 | 长期保留 |
| tests/agent/test_auxiliary_client.py | 457da1e3f2: 对齐 timeout 短TTL隔离定制（auxiliary_client.py 配套测试） | 零（测试文件） | 与 aux 定制同命运 |
| tests/hermes_cli/test_pre_tool_batch_fail_closed.py | 5e0f4214ce: pre_tool_batch fail-closed 配套测试 | 零（测试文件） | 与 tool_executor 定制同命运 |
| tests/cron/test_unreachable_retry.py | b929b65d16 配套测试 | 零（测试文件） | 与 cron 定制同命运 |
| cron/unreachable_retry.py | b929b65d16: 恢复 unreachable ladder 的 offline 摘要+quota 429 transient 判定 | 中（官方改 retry 逻辑时撞） | 跟踪官方 retry 侧等价机制 |
| scripts/skill_sha_drift.py | b101a20a2d/74c996fef6: skill-drift-check 依赖脚本（基线切换恢复件+噪音修复） | 低（scripts/ 追加式） | 长期保留 |
| .hermes-rules.md | fork 根纪律文件（官方无此文件） | 零 | 长期保留；md5 与 HEAD 对齐由三问审计第4条核验 |
| apps/desktop/src/plugins/hermes-bots/cron-detail.test.tsx + apps/desktop/src/store/voice-prefs.test.ts (3811501b22) | locale/spy 修正（中文环境特有） | 零 | 值得提 upstream PR |
| .gitignore / docs/review/ | 追加式 | 零 | 长期保留 |

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
