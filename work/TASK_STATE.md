# TASK_STATE — Z.AI 1302 transient throttle cooldown

Updated: 2026-11-12 18:52 Asia/Shanghai

## Goal & acceptance
- Goal: diagnose and fix "named custom provider 'zai' has no selectable API key" after Z.AI `429 / code 1302`, on BOTH auxiliary and main routes.
- Acceptance: behavior contracts prove `1302` is a short transient throttle, remains eligible for credential rotation, and does not persist a one-hour account-wide bench from either path; targeted tests and project checks pass.

## Constraints & decisions
- Do not print, copy, reset, or rotate user credentials; `auth.json` inspection is metadata-only.
- Preserve the existing dirty `work/TASK_STATE.md`; this section is additive.
- Recent fail-closed behavior (`14ad1d2a2e`) is correct: never send `no-key-required` to the remote endpoint.

## Environment / repo facts
- branch/upstream: `main...origin/main`, both at `14ad1d2a2e`; only `work/TASK_STATE.md` was modified before this task.
- `HERMES_HOME` unset; live profile is `/Users/stan/.hermes`, provider `zai`, model `glm-5.3`.
- `hermes auth list zai` → two keys, both `rate-limited (429)`, ~40m remaining at 23:21.
- Live pool metadata: both rows marked `429/1302` at 2026-11-11 23:02:19; no `last_error_reset_at`.
- Main and auxiliary routes all point at Z.AI; configured fallbacks are also Z.AI (`glm-5.3-flash`, `glm-5.2`), so same-provider cooling cannot recover immediately.
- `auxiliary.transient_retries: 0`, so aux failures rotate/bench the pool on the first Z.AI throttle.

## Key files & findings
- `hermes_cli/runtime_provider_custom.py:817` — emits the reported fail-closed `AuthError` when the pool yields no selectable key.
- `hermes_cli/runtime_provider_custom.py:577` — custom pool resolution skips every exhausted row and returns `None`.
- `agent/error_classifier.py:1260` — `1302` currently classifies as `rate_limit` but gets no reset hint.
- `agent/credential_pool.py:377` — multi-entry 429 default cooldown is 3600s; sole-entry transient cooldown is 60s.
- `agent/agent_runtime_helpers.py:1020` — aux path rotates on first 429 because `auxiliary.transient_retries` is 0.
- `agent/auxiliary_client.py:3597` — auxiliary pool recovery calls `mark_exhausted_and_rotate`; concurrent aux calls can therefore bench both Z.AI rows.
- `agent/auxiliary_client.py:3594` — auxiliary recovery now preserves the classifier's `reset_at` hint while retaining the original exception text.
- `tests/agent/test_auxiliary_error_classification.py:127` — existing `1302` test only asserts `rate_limit`, not the harmful one-hour cooldown.
- 2026-11-12 follow-up: live main-route `1302` at 18:28 still benched both keys for 3600s (55m remaining at 18:33), proving `cb6cad0c6c` covered aux only. Root cause: `agent/turn_recovery.py::_pool_error_context` returned the raw `extract_api_error_context` unchanged, dropping `classified.error_context.reset_at`.
- Fix: `_pool_error_context` now merges `classified.error_context` over the raw context (usage-window 402 branch unchanged); contract at `tests/agent/test_zai_rate_limit_primary_retry.py::test_main_route_1302_carries_short_reset_into_credential_pool`.

## Verification log
- `hermes auth list zai` → exit 0 — both key labels show `rate-limited (429)` with ~40m left (2026-11-11, masked labels only).
- metadata-only `jq` over `credential_pool.zai` → exit 0 — both rows are `429/1302`, marked at 23:02:19, no reset timestamp.
- `rg`/log inspection `/Users/stan/.hermes/logs/gateway.error.log` → matched `429 code 1302` from concurrent main/aux requests at 23:02; subsequent cron was blocked.
- `env -u UV_PYTHON_PREFERENCE scripts/run_tests.sh tests/agent/test_auxiliary_error_classification.py -q` → exit 1 before tests — isolated env rebuild failed because the configured index omitted `pilk==0.2.4` upload time; runner advises `https://pypi.org/simple`.
- `env -u UV_PYTHON_PREFERENCE UV_INDEX_URL=https://pypi.org/simple scripts/run_tests.sh tests/agent/test_auxiliary_error_classification.py -q` → exit 1, RED as expected — 9 passed, 1 failed: `KeyError: 'reset_at'`.
- Added the pool-boundary assertion to the RED contract, then reran it before the production edit → exit 1 — `KeyError: 'reset_at'`, proving classification alone did not reach the credential pool.
- `env -u UV_PYTHON_PREFERENCE UV_INDEX_URL=https://pypi.org/simple scripts/run_tests.sh tests/agent/test_auxiliary_error_classification.py -q` → exit 0 — 11 passed; includes a real temp-`HERMES_HOME` credential pool where only the failed Z.AI key is exhausted with a ≤60s reset window.
- Related regressions (`test_error_classifier.py`, `test_auxiliary_error_policy.py`, `test_auxiliary_client.py`, `test_credential_pool.py`, `test_credential_pool_routing.py`, `test_zai_rate_limit_primary_retry.py`) → exit 0 — 608 passed.
- `env -u UV_PYTHON_PREFERENCE UV_INDEX_URL=https://pypi.org/simple python scripts/check` → exit 0 — 18 checks ok; health reports 0 blocking, 0 advisory.
- 2026-11-12 RED: `scripts/run_tests.sh tests/agent/test_zai_rate_limit_primary_retry.py -q` → exit 1 — new main-path contract failed `ctx.get("reset_at") is None`; pool received only `{'reason': '1302', 'message': ...}`.
- 2026-11-12 GREEN: same file → exit 0 — 2 passed after `_pool_error_context` merge fix.
- 2026-11-12 related regressions (6 files: zai primary retry, codex token replay, route verdicts, billing carrythrough, pool revert, aux classification) → exit 0 — 44 passed, 0 failed.
- 2026-11-12 `env -u UV_PYTHON_PREFERENCE UV_INDEX_URL=https://pypi.org/simple python scripts/check` → exit 0 — 18 checks ok; health 0 blocking, 0 advisory.

## Current status & next steps
- Done (aux, `cb6cad0c6c`): reproduced the pool-wide bench, traced classifier + aux recovery, added the short `1302` Z.AI hint, preserved it across aux recovery, verified against a temp home.
- Done (main, 2026-11-12): RED→GREEN contract proving the main route hands the classifier's 60s `reset_at` to the credential pool; regressions + `scripts/check` green; diff reviewed (2 task files only).
- Done: committed `1c9bb90a5ae` ("fix(agent): carry Z.AI 1302 short cooldown on main route"), pushed to `origin/main` (`bf1872fa82..1c9bb90a5a`); pre-commit/pre-push `scripts/check` both green. Only `work/TASK_STATE.md` remains intentionally uncommitted.

## Open risks / blockers / unverified
- Need to verify whether the Z.AI API documents an exact 1302 reset interval; absent that, use a conservative short cooldown consistent with existing transient-throttle behavior.
- Existing live cooldown will not disappear from the current auth store merely because code is fixed; recovery options are wait, add an independent-account key, or switch provider.
- Running gateway (started 11:42 on 2026-11-12) already imports the classifier fix; the main-route fix requires the gateway to be restarted onto the new commit before it takes effect for new 1302 marks.
- Broader unrelated issues observed: configured fallback chain is entirely same-provider Z.AI, and some background auxiliary fallbacks pass GLM model names to DeepSeek; not part of this atomic fix unless required by tests.
- User-facing error text to preserve: `named custom provider 'zai' has no selectable API key for https://open.bigmodel.cn/api/coding/paas/v4; its credential pool may be cooling down after a rate limit. Check hermes auth list zai or configure providers.zai.api_key/key_env/key_cmd. Placeholder authentication was not sent remotely.`

---

# TASK_STATE — Goal consumption governors

Updated: 2026-11-11 11:42 Asia/Shanghai

## Goal & acceptance
- Goal: prevent the 2026-11-11 failure mode where 3 open-ended goals consumed the 5h quota in ~2h (measured 1,090 zai calls / 118.5M input tokens).
- Acceptance: per-goal API-call budget and concurrent-goal cap behave through `evaluate_after_turn`; new tests plus goal regressions and `python scripts/check` pass; only the two task files are committed and pushed.

## Constraints & decisions
- Do not re-investigate the 00:40–02:40 usage; diagnosis is accepted.
- `goals.max_api_calls` default 120, `goals.max_concurrent` default 2, 0 disables each.
- `done`/`blocked` verdicts are decided before governors, so completion is never stolen.
- Newest goals park when concurrency exceeds the cap; old goals keep slots. Only goals with recent `created_at`/`last_turn_at` count (1h window), avoiding dormant stale active rows.
- Legacy goals with no baseline lazily anchor at first evaluation; `/goal resume(reset_budget=True)` re-anchors. Quota auto-resume must use `reset_budget=False`.

## Files
- `hermes_cli/goal_consumption.py` — NEW topical sibling: governor config, SessionDB counter scan, concurrency scan, parking/budget mixin.
- `hermes_cli/goals.py` — state baseline field, manager kwargs/anchoring, continuation call sites; facade is 2,338 lines vs 2,345 baseline.
- `hermes_cli/goals_status.py` — owner status/introspection moved into the existing status sibling.
- `tests/hermes_cli/test_goal_consumption_governors.py` — 10 behavior tests.
- `website/docs/user-guide/configuration.md`, `website/docs/user-guide/features/goals.md`, `website/docs/reference/slash-commands.md` — config-key documentation.

## Verification so far
- RED: detached worktree at HEAD + only the new test file → 10 failed / 0 passed, exit 1 (`GoalManager` lacked governor kwargs).
- GREEN + regression: 24 goal-related files → 226 passed, 0 failed, 2 skipped, exit 0.
- Docs: `tests/ci/test_docs_drift.py` → 2 passed, exit 0.
- Target ruff: four Python task files + docs → all checks passed.
- `git diff --check` → exit 0.
- Health: task files clean; full health run still reports only the unrelated staged `agent/auxiliary_client.py` line-count issue.

## Current status
- Committed `9d308cea2e4` (`feat(goals): add consumption governors`).
- Isolated commit check: `env -u UV_PYTHON_PREFERENCE python scripts/check --commit HEAD` → 18 checks ok.
- Pushed to `origin/main`; pre-push hook also ran scripts/check successfully.
- Working tree still contains the separate staged aux work and this scratch state file; no task source files remain modified.

## Risks / notes
- Default 120 calls may pause legitimate long-running work; user can raise `goals.max_api_calls` or set 0.
- Tests must use `scripts/run_tests.sh`; user environment exports `UV_PYTHON_PREFERENCE=only-managed`, which must be unset.

---

# TASK_STATE — 2026-11-13 repo housekeeping (commit/push/merge)

Checkpoint: /Users/stan/code/ai/github/fork/hermes-agent/work/TASK_STATE.md

## Goal & acceptance
- Goal: 用户指令「该提交的提交、该推送的推送、该合并的合并」——清空本仓未推送/未合并遗留。
- Acceptance: 工作区干净；本地 `main == origin/main`；PR #11 状态 MERGED；`codex/doctor-actionable-summary` 并入 main；门禁检查通过。

## Progress (2026-11-13 23:32 Asia/Shanghai)
- 起点：工作区 0 变更；`main` ahead `origin/main` 2774 / behind 0。
- `codex/doctor-actionable-summary`（5 commits：gateway off-loop ×4 + doctor actionable summary）merge-tree 干净，待并入。
- `codex/fix-feishu-preauth`（PR #11，2 commits）merge-tree 冲突：仅 `plugins/platforms/feishu/adapter.py` 内容冲突；`tests/gateway/test_feishu.py` 自动合并。该分支把 webhook 安全拆到 `adapter_webhook_security.py` 并改为「先认证后配额」。
- `oauthlib-security-floor` detached worktree `9d431491a3` 已是 main 祖先，无遗留；`codex/read-think-gate-auth-flow` 为旧 main 指针，已并入。
- 推送门禁：`core.hooksPath=/Users/stan/.hermes/githooks/pre-push` 中继回仓内 `.git/hooks/pre-push`（lefthook → scripts/check）。
- 23:40 milestone：`codex/doctor-actionable-summary` 已并入 main（merge `99657f6ebdf`，pre-commit `scripts/check` 19 checks ok）。
- 23:4x milestone：期间 origin/main 被外部推进到 `7c36f81d1f`（原先 2774 个未推送提交已不在待办内；main 当时 ahead 6/behind 0）。
- 23:5x milestone：`codex/fix-feishu-preauth` 冲突已解（adapter.py 保留双 mixin：WebhookSecurity + SyntheticEvents；handler 采用 PR #11 的先认证后配额实现，HEAD 同类内联修复被其超集替代），merge commit `1671c35d680`。
- 验证：`env -u UV_PYTHON_PREFERENCE UV_INDEX_URL=https://pypi.org/simple scripts/run_tests.sh tests/gateway/test_feishu_webhook_security.py tests/gateway/test_feishu.py tests/gateway/test_local_create_staging_off_loop.py tests/hermes_cli/test_doctor_actionable_summary.py -q` → exit 0，4 files，80 passed / 0 failed / 7 skipped。
- 23:42 push 结果：本地 `git push origin main` 的 ref 更新输给并发自动推送（remote rejected: cannot lock ref），但 fetch 后核实 `main == origin/main == 1671c35d680`（0 ahead / 0 behind）；该次 push 的 pre-push `scripts/check` 已全绿（19 checks ok，health 0/0）。
- PR #11 实测：state=MERGED，mergeCommit=1671c35d680，mergedAt=2026-11-13T15:42:08Z。

## Next action
- 提交本检查点收尾 commit 并推送；成功判据 exit 0 或（若再遇并发推送竞态）fetch 后 `main == origin/main` 且工作区干净；失败则保留未推送 commit 并如实汇报。
