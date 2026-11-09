"""A stalled policy-hook callback must not veto the turn without one retry.

Contract under test (``hermes_cli/plugins_dispatch.py``):

* ``pre_llm_call`` / ``pre_tool_call`` / ``pre_verify`` / ``pre_tool_batch`` are fail-closed:
  a callback that cannot decide blocks. The block used to fire on the FIRST timeout, and the
  follow-up retry was skipped by the 60s suppression window — so a transient stall in a guard's
  auxiliary judge (DNS/connect hang) destroyed the user's turn twice in a row with a message
  that named only the hook.
* ``pre_llm_call`` and ``pre_verify`` decide once per unit of user-visible work, so a timeout
  there costs the whole turn: those callbacks get one automatic retry past the suppression
  window, and the block names the callback/plugin. Per-tool-call gates and advisory (fail-open)
  hooks are untouched — a skip is tolerated there, so retrying would only add latency.
* The abandoned-worker cap still bounds thread pile-up: at the cap the callback is skipped
  without being invoked, and the block says it is cooling down (wait) rather than timed out
  (retry).
"""

from __future__ import annotations

import threading

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


def test_timed_out_policy_callback_retries_once_and_can_still_pass(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["pre_llm_call"] = [_hung_callback(release, calls, {"context": "retry-ok"})]
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
    finally:
        release.set()

    assert results == [{"context": "retry-ok"}], "a successful retry must not leave a block behind"
    assert len(calls) == 2, "the first attempt timed out; exactly one retry follows"


def test_second_timeout_fails_closed_and_names_the_callback(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["pre_llm_call"] = [_hung_callback(release, calls)]
        results = manager.invoke_hook("pre_llm_call", session_id="s", turn_id="t")
    finally:
        release.set()

    assert len(results) == 1 and results[0]["action"] == "block"
    message = results[0]["message"]
    assert "pre_llm_call" in message and "callback" in message
    assert "自动重试一次" in message, "the user must know the retry already happened"
    assert len(calls) == 2


def test_abandoned_worker_cap_stops_retrying_and_reports_cooling_down(monkeypatch) -> None:
    manager = _manager(monkeypatch)
    release = threading.Event()
    calls: list[int] = []
    try:
        manager._hooks["pre_llm_call"] = [_hung_callback(release, calls)]
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


def test_async_policy_callback_is_retried_once_and_can_still_pass(monkeypatch) -> None:
    import asyncio

    manager = _manager(monkeypatch)
    calls: list[int] = []

    async def callback(**_kwargs):
        calls.append(1)
        if len(calls) > 1:
            return {"context": "async-retry-ok"}
        await asyncio.Event().wait()  # cancelled by the hook budget on the first attempt

    manager._hooks["pre_llm_call"] = [callback]
    results = asyncio.run(manager.ainvoke_hook("pre_llm_call", session_id="s", turn_id="t"))

    assert results == [{"context": "async-retry-ok"}]
    assert len(calls) == 2
