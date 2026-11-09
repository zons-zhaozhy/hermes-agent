"""Hook budgets: a degrading callback is not retried, and one turn's budget is bounded.

Contract under test (``hermes_cli/plugins_dispatch.py``):

* ``pre_tool_call`` / ``pre_tool_batch`` / ``pre_verify`` are fail-closed: a callback that cannot
  decide blocks, and ``pre_verify`` buys it one automatic retry past the 60 s suppression window
  (a transient stall in a guard's auxiliary judge must not destroy the work unit).
* ``pre_llm_call`` degrades instead: a timed-out callback contributes an explicit UNKNOWN note
  naming itself and does not end the turn (see ``test_policy_hook_fail_closed.py``). A degrading
  callback is not retried — it has nothing to protect, so a retry only stretches the turn.
* The two are orthogonal: a registration that opted into ``veto_on_timeout=True`` keeps the retry
  and the block; its neighbours are unaffected.
* One ``pre_llm_call`` gets ONE total budget: with 20+ registered guards, per-callback budgets
  alone let a degraded endpoint queue many minutes of wall clock before the turn starts. Past the
  turn budget, callbacks degrade unstarted instead of being started with a fresh budget.
* The abandoned-worker cap still bounds thread pile-up: at the cap the callback is skipped without
  being invoked, and a veto-grade block says it is cooling down (wait) rather than timed out.
"""

from __future__ import annotations

import threading
import time

from hermes_cli.plugins import PluginManager


def _manager(monkeypatch, timeout: float = 0.05) -> PluginManager:
    """A manager whose hook budget is short enough to time out inside a test."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: timeout)
    return PluginManager()


def _hung_callback(release: threading.Event, calls: list[int], result=None):
    """A callback that stalls its worker until *release*, counting every invocation."""
    def callback(**_kwargs):
        calls.append(1)
        if len(calls) == 1 or result is None:
            release.wait(2.0)
            return None
        return result
    return callback


def _veto_grade(manager: PluginManager, hook_name: str, callback) -> None:
    manager._record_hook_policy(hook_name, callback, veto_on_timeout=True, policy="required")


def test_degrading_pre_llm_call_callback_is_not_retried(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["pre_llm_call"] = [_hung_callback(release, calls, {"context": "late"})]
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
    finally:
        release.set()

    assert len(calls) == 1, "retrying a degrading guard buys latency, not a decision"
    assert len(results) == 1 and "action" not in results[0] and "UNKNOWN" in results[0]["context"]


def test_veto_grade_callback_retries_once_and_can_still_pass(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    callback = _hung_callback(release, calls, {"context": "retry-ok"})
    _veto_grade(manager, "pre_llm_call", callback)
    try:
        manager._hooks["pre_llm_call"] = [callback]
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
    finally:
        release.set()

    assert results == [{"context": "retry-ok"}], "a successful retry must not leave a block behind"
    assert len(calls) == 2, "the first attempt timed out; exactly one retry follows"


def test_veto_grade_second_timeout_fails_closed_and_names_the_callback(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    callback = _hung_callback(release, calls)
    _veto_grade(manager, "pre_llm_call", callback)
    try:
        manager._hooks["pre_llm_call"] = [callback]
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
    finally:
        release.set()

    assert len(results) == 1 and results[0]["action"] == "block"
    message = results[0]["message"]
    assert "pre_llm_call" in message and "callback" in message
    assert "自动重试一次" in message, "the user must know the retry already happened"
    assert len(calls) == 2


def test_veto_grade_abandoned_worker_cap_reports_cooling_down(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    callback = _hung_callback(release, calls)
    _veto_grade(manager, "pre_llm_call", callback)
    try:
        manager._hooks["pre_llm_call"] = [callback]
        messages = [
            manager.invoke_hook("pre_llm_call", session_id="s", turn_id=f"turn-{index}")[0]["message"]
            for index in range(3)
        ]
    finally:
        release.set()

    # Turn 1: attempt + retry. Turn 2: attempt only (the per-callback worker cap is reached).
    assert len(calls) == 3
    assert "冷却" in messages[2], "at the cap the block must say 'wait', not 'retry again'"
    assert all("pre_llm_call" in message for message in messages)


def test_pre_verify_timeout_is_still_retried(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["pre_verify"] = [_hung_callback(release, calls, {"action": "continue",
                                                                       "message": "not done"})]
        results = manager.invoke_hook("pre_verify", session_id="s", turn_id="t")
    finally:
        release.set()

    assert results == [{"action": "continue", "message": "not done"}]
    assert len(calls) == 2


def test_one_turn_budget_stops_starting_callbacks(monkeypatch) -> None:
    """20+ registered guards must not each queue a fresh budget behind a degraded endpoint."""
    manager = _manager(monkeypatch, timeout=0.1)
    release = threading.Event()
    callbacks = [_hung_callback(release, [], None) for _ in range(6)]
    try:
        manager._hooks["pre_llm_call"] = callbacks
        started = time.monotonic()
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert elapsed < 2.0, "the turn budget bounds the whole hook, not each callback separately"
    assert len(results) == len(callbacks), "every guard's absence is still reported this turn"
    assert all("action" not in result and "UNKNOWN" in result["context"] for result in results)


def test_fail_open_hook_callback_is_not_retried(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["post_tool_call"] = [_hung_callback(release, calls, {"context": "ignored"})]
        results = manager.invoke_hook("post_tool_call", session_id="s", tool_call_id="c")
    finally:
        release.set()

    assert results == [], "an advisory hook that stalled contributes nothing and blocks nothing"
    assert len(calls) == 1, "retrying a fail-open callback buys latency, not a decision"


def test_async_veto_grade_callback_is_retried_once_and_can_still_pass(monkeypatch) -> None:
    import asyncio

    manager = _manager(monkeypatch)
    calls: list[int] = []

    async def callback(**_kwargs):
        calls.append(1)
        if len(calls) > 1:
            return {"context": "async-retry-ok"}
        await asyncio.Event().wait()  # cancelled by the hook budget on the first attempt

    _veto_grade(manager, "pre_llm_call", callback)
    manager._hooks["pre_llm_call"] = [callback]
    results = asyncio.run(manager.ainvoke_hook("pre_llm_call", session_id="s", turn_id="t"))

    assert results == [{"context": "async-retry-ok"}]
    assert len(calls) == 2


def test_async_degrading_callback_is_not_retried(monkeypatch) -> None:
    import asyncio

    manager = _manager(monkeypatch)
    calls: list[int] = []

    async def callback(**_kwargs):
        calls.append(1)
        await asyncio.Event().wait()

    manager._hooks["pre_llm_call"] = [callback]
    results = asyncio.run(manager.ainvoke_hook("pre_llm_call", session_id="s", turn_id="t"))

    assert len(calls) == 1
    assert len(results) == 1 and "action" not in results[0] and "UNKNOWN" in results[0]["context"]
