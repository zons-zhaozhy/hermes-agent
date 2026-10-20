"""A running browser_exec must not be reaped as an idle session (#fix 2026-10-20).

The harness daemon talks CDP directly to the browser, so a long exec's traffic never
passes through ``_update_session_activity``: the janitor used to read that wall-clock
silence as idle and kill the Chromium mid-call (deterministic crash at exactly the
inactivity timeout). These tests pin the two guards:

* ``inflight_exec`` — the janitor skips task-ids whose exec is running, on every exit path.
* the activity touch on exit — a finished exec leaves a normally-idle (reapable) session.

They are lifecycle invariants, not change-detectors: the recording fake fails the
assertions if any code path forgets the guard again.
"""

from __future__ import annotations

import time
from typing import Iterator, List

import pytest
from _pytest.monkeypatch import MonkeyPatch

import hermes_bootstrap
from tools import browser_tool_lifecycle as lifecycle
from tools.browser_tool_origin import origin as _bt


@pytest.fixture()
def janitor(monkeypatch: MonkeyPatch) -> Iterator[list[str]]:
    """Recording janitor environment: capture teardowns, never touch a real browser."""
    calls: list[str] = []

    def _fake_cleanup(task_id: str) -> None:
        calls.append(task_id)

    monkeypatch.setattr(lifecycle, "cleanup_browser", _fake_cleanup)
    monkeypatch.setattr(lifecycle, "_human_holds_shared_browser", lambda task_id: False)
    _bt._session_last_activity.clear()
    _bt._session_inflight.clear()
    _bt._session_owner_homes.clear()
    _bt._cleanup_failures.clear()
    yield calls
    _bt._session_last_activity.clear()
    _bt._session_inflight.clear()
    _bt._session_owner_homes.clear()
    _bt._cleanup_failures.clear()


def _stale(task_id: str) -> None:
    """Book the session as idle far beyond the inactivity timeout."""
    _bt._session_last_activity[task_id] = time.time() - (_bt.BROWSER_SESSION_INACTIVITY_TIMEOUT + 999)


def test_inflight_session_is_skipped_by_janitor(janitor: list[str]) -> None:
    _stale("inflight-task")
    with lifecycle.inflight_exec("inflight-task"):
        lifecycle._cleanup_inactive_browser_sessions()
    assert "inflight-task" in _bt._session_last_activity  # 期望: 执行中的会话保留
    assert janitor == []  # 期望: 未触发任何回收


def test_stale_session_without_inflight_is_reaped(janitor: list[str]) -> None:
    _stale("stale-task")
    lifecycle._cleanup_inactive_browser_sessions()
    assert janitor == ["stale-task"]  # 期望: 无执行中的闲置会话被回收一次
    assert "stale-task" not in _bt._session_last_activity  # 期望: 记账清理


def test_inflight_exec_clears_guard_and_touches_activity_on_exit(janitor: list[str]) -> None:
    _stale("exit-task")
    t0 = time.time()
    with lifecycle.inflight_exec("exit-task"):
        assert "exit-task" in _bt._session_inflight  # 期望: 执行中已登记
    assert "exit-task" not in _bt._session_inflight  # 期望: 退出后登记清空
    assert _bt._session_last_activity["exit-task"] >= t0  # 期望: 活跃时间戳刷新为退出时刻


def test_inflight_exec_clears_guard_on_exception(janitor: list[str]) -> None:
    _stale("boom-task")
    with pytest.raises(RuntimeError):
        with lifecycle.inflight_exec("boom-task"):
            raise RuntimeError("exec failed")
    assert "boom-task" not in _bt._session_inflight  # 期望: 异常路径同样清空登记


def test_exec_finish_then_idle_is_reapable(janitor: list[str]) -> None:
    # The whole point of touching activity on exit: after a long exec finishes, the
    # session must become normally reapable again (no immortality leak).
    with lifecycle.inflight_exec("long-task"):
        _bt._session_last_activity["long-task"] = time.time() - (
            _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT + 999)
    lifecycle._cleanup_inactive_browser_sessions()
    assert janitor == []  # 期望: 刚执行完(touch 后)未超时，不回收
    _bt._session_last_activity["long-task"] = time.time() - (
        _bt.BROWSER_SESSION_INACTIVITY_TIMEOUT + 999)
    lifecycle._cleanup_inactive_browser_sessions()
    assert janitor == ["long-task"]  # 期望: 真闲置超时后正常回收
