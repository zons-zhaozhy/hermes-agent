# TASK_STATE — GLM rate-limit retry precedes paid fallback

Updated: 2026-11-11 03:08 Asia/Shanghai

## Goal & acceptance
- Goal: Z.AI GLM ordinary 429 retries the primary GLM first; 5-hour quota-window 429 never silently switches to DeepSeek.
- Acceptance: red/green tests for Chinese quota classification and primary retry ordering; relevant suites and `scripts/check` pass; user config has no DeepSeek fallback; commit pushed.

## Constraints & decisions
- Preserve existing contract: Chinese 1308 remains `billing` for auxiliary quarantine, but gains `usage_window` no-fallback context.
- Ordinary Chinese 1302 remains `rate_limit`; Z.AI coding GLM route should consume configured API retries before chain switching.
- User config is outside this Git repo and must not be committed.

## Environment / repo facts
- Branch `main` tracks `origin/main`; baseline `d76f2d6de4`; repo clean at investigation start.
- `/Users/stan/.hermes/config.yaml` now contains only `zai glm-5.3-flash -> zai glm-5.2`; DeepSeek was removed; agent retry count is 3.

## Key files & findings
- `agent/error_classifier.py:1273-1290` — Chinese 1308 currently classifies as ordinary billing because Chinese reset prose is not a transient signal.
- `agent/turn_recovery.py:1897-1934` — rate-limit/billing eager fallback runs before the ordinary retry budget.
- `agent/retry_utils.py:67-118` — reset grammar parser does not understand Chinese absolute timestamps.
- `agent/retry_utils.py:143-174` — existing Z.AI Coding overload special case and retry schedule.
- `tests/agent/test_auxiliary_error_classification.py:97-118` — current 1308/1302 behavior contract.
- Unrelated parallel worktree edits appeared after task start: `cli.py`, `hermes_cli/cli_queue_priority.py`, `hermes_cli/cli_chat_turn_mixin.py`, `hermes_cli/cli_session_dock.py`, and their queue test. Do not stage.

## Verification log
- `TZ=Asia/Shanghai date '+%H:%M:%S'; git status --short --branch` → exit 0 — branch synced, clean baseline. (2026-11-11)
- `HERMES_PYTHON=$PWD/.venv/bin/python scripts/run_tests.sh ...5 fallback/classifier files` → exit 1 — red tests confirmed Chinese reset missing, 1308 lacked usage-window context. (2026-11-11)
- `HERMES_PYTHON=$PWD/.venv/bin/python scripts/run_tests.sh ...5 fallback/classifier files` → exit 0 — 28 tests passed after fix. (2026-11-11)
- Config readonly probe → exit 0 — fallback chain `[zai/glm-5.3-flash, zai/glm-5.2]`, no DeepSeek, retries 3. (2026-11-11)
- Classifier probe → exit 0 — 1308 `billing/nonretryable/no-fallback/usage_window/reset_at`; 1302 `rate_limit/retryable`. (2026-11-11)
- `HERMES_PYTHON=$PWD/.venv/bin/python scripts/run_tests.sh ...9 related files` → exit 0 — 236 tests passed. (2026-11-11)

## Current status & next steps
- Done: model fallback fix committed and pushed in `841ca1a63a` (ancestor of current `main`).
- Done: user fallback chain contains only Z.AI GLM models; sticky DeepSeek goal session ended naturally and is idle on `glm-5.3`.
- Done: gateway restarted at 02:41:36 onto source containing `841ca1a63a`; RPC healthy and no DeepSeek sessions.
- Done: launchd false-success restart bug fixed, tested, committed, and pushed as `19cd4f0623`.

## Open risks / blockers / unverified
- The restart-recovery fix affects the next `hermes gateway restart` CLI invocation; no gateway process restart is needed solely to load it.
- Full `tests/hermes_cli/test_gateway_service.py` was not run because an unrelated first test triggers repeated real plugin discovery on this host and stalls; the launchd-scoped subset plus six related files passed, and full `scripts/check` passed.
