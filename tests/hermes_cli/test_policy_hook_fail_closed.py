"""Intent and completion policy hooks cannot silently fail open — and cannot fail CLOSED by accident.

A guard that raises or exhausts its callback budget made no decision. The right answer differs by
what that decision protects:

* ``pre_verify`` decides whether a completion claim may be accepted. Its silence must keep the turn
  open (fail closed on the claim), because accepting an unverified claim is the failure it exists to
  prevent.
* ``pre_llm_call`` decides what context a turn gets, and optionally vetoes it. Its normal output is
  ADDITIVE, and every registration in this tree already answers "no verdict" with fail-open
  (``guards.scope_gate``, ``declare_act_guard``, a ``read_think_gate`` reset that returns None). A
  budget expiry is not a verdict about the turn, so it must not be manufactured into one: the turn
  proceeds with a loud UNKNOWN note naming the guard and its missing context, and denial stays where
  it is load-bearing and cheap (``pre_tool_call`` / ``pre_verify``) — or on a registration that
  explicitly opted into ``veto_on_timeout=True``.

Regression: 2026-11-09, a degraded judge endpoint made five serial ``pre_llm_call`` guards each burn
a 30 s budget (plus a 15 s retry) and ended two consecutive turns with
``blocked_by_plugin_pre_llm_call`` / ``api_calls=0``.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_cli.plugins import PluginManager


def _opt_into_veto(manager: PluginManager, hook_name: str, callback) -> None:
    """Declare a registration veto-grade, as ``ctx.register_hook(..., veto_on_timeout=True)`` does."""
    manager._record_hook_policy(hook_name, callback, veto_on_timeout=True, policy="required")


def test_pre_verify_failure_returns_named_block() -> None:
    manager = PluginManager()

    def broken_guard(**_kwargs):
        raise RuntimeError("judge unavailable")

    manager._hooks["pre_verify"] = [broken_guard]
    results = manager.invoke_hook("pre_verify", session_id="policy-failure", turn_id="turn-1")

    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert "pre_verify" in results[0]["message"]
    assert "broken_guard" in results[0]["message"]
    assert "judge unavailable" in results[0]["message"]


def test_pre_verify_timeout_returns_named_block(monkeypatch) -> None:
    import threading

    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    hold = threading.Event()

    def hung_guard(**_kwargs):
        hold.wait(timeout=10.0)

    manager = PluginManager()
    manager._hook_timeout_suppression_seconds = 0.0
    manager._hooks["pre_verify"] = [hung_guard]
    try:
        results = manager.invoke_hook("pre_verify", session_id="policy-timeout", turn_id="turn-1")
    finally:
        hold.set()

    assert len(results) == 1 and results[0]["action"] == "block"
    assert "hung_guard" in results[0]["message"], (
        "the block names the callback, not only the hook: the operator must be able to tell "
        "which guard hung and whether the automatic retry already happened"
    )


def test_pre_llm_call_failure_degrades_to_named_unknown_note() -> None:
    manager = PluginManager()

    def broken_guard(**_kwargs):
        raise RuntimeError("intent judge unavailable")

    manager._hooks["pre_llm_call"] = [broken_guard]
    results = manager.invoke_hook("pre_llm_call", session_id="degrade-error", turn_id="turn-1")

    assert len(results) == 1
    assert "action" not in results[0], "a slow or broken context guard must not delete the turn"
    note = results[0]["context"]
    assert "UNKNOWN" in note and "pre_llm_call" in note
    assert "broken_guard" in note and "intent judge unavailable" in note, (
        "the note names the guard and its error: the failure stays visible to the agent, which "
        "has to cover the self-check it did not receive"
    )
    assert "pre_tool_call" in note, "the note says where denial still applies"


def test_pre_llm_call_timeout_degrades_to_named_unknown_note(monkeypatch) -> None:
    import threading

    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    hold = threading.Event()

    def hung_guard(**_kwargs):
        hold.wait(timeout=10.0)

    manager = PluginManager()
    manager._hook_timeout_suppression_seconds = 0.0
    manager._hooks["pre_llm_call"] = [hung_guard]
    try:
        results = manager.invoke_hook("pre_llm_call", session_id="degrade-timeout", turn_id="turn-1")
    finally:
        hold.set()

    assert len(results) == 1 and "action" not in results[0]
    note = results[0]["context"]
    assert "UNKNOWN" in note and "hung_guard" in note
    assert "无判定" in note, "the note states the degradation contract, not a silent pass"


def test_degradation_is_recorded_as_unknown_gate_evidence(monkeypatch) -> None:
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05)
    import threading

    hold = threading.Event()

    def hung_guard(**_kwargs):
        hold.wait(timeout=10.0)

    manager = PluginManager()
    manager._hook_timeout_suppression_seconds = 0.0
    manager._hooks["pre_llm_call"] = [hung_guard]
    try:
        manager.invoke_hook("pre_llm_call", session_id="degrade-evidence", turn_id="turn-1")
    finally:
        hold.set()

    event = manager.gate_plane.events("degrade-evidence")[0]
    assert event.hook == "pre_llm_call"
    assert event.status.value == "unknown", "a degraded guard is never reported as a pass"


def test_advisory_gate_absence_contributes_nothing() -> None:
    manager = PluginManager()

    def broken_advisory(**_kwargs):
        raise RuntimeError("advisory judge unavailable")

    manager._record_hook_policy(
        "pre_llm_call", broken_advisory, veto_on_timeout=False, policy="advisory")
    manager._hooks["pre_llm_call"] = [broken_advisory]
    results = manager.invoke_hook("pre_llm_call", session_id="advisory", turn_id="turn-1")

    assert results == [], "an advisory gate owes no turn-level claim when it has no verdict"
    assert manager.gate_plane.events("advisory")[0].status.value == "unknown"


def test_opted_in_veto_still_blocks_the_turn() -> None:
    manager = PluginManager()

    def broken_veto_gate(**_kwargs):
        raise RuntimeError("authorization judge unavailable")

    _opt_into_veto(manager, "pre_llm_call", broken_veto_gate)
    manager._hooks["pre_llm_call"] = [broken_veto_gate]
    results = manager.invoke_hook("pre_llm_call", session_id="veto", turn_id="turn-1")

    assert len(results) == 1 and results[0]["action"] == "block"
    assert "broken_veto_gate" in results[0]["message"]


def test_opt_in_is_per_registration_not_per_plugin() -> None:
    """One veto-grade callback must not promote its neighbours: the opt-in is a registration flag."""
    manager = PluginManager()

    def broken_veto_gate(**_kwargs):
        raise RuntimeError("authorization judge unavailable")

    def broken_context_gate(**_kwargs):
        raise RuntimeError("context judge unavailable")

    _opt_into_veto(manager, "pre_llm_call", broken_veto_gate)
    manager._hooks["pre_llm_call"] = [broken_context_gate, broken_veto_gate]
    results = manager.invoke_hook("pre_llm_call", session_id="mixed", turn_id="turn-1")

    assert len(results) == 2
    assert "action" not in results[0] and "context" in results[0]
    assert results[1]["action"] == "block"


def test_async_pre_llm_call_failure_degrades() -> None:
    manager = PluginManager()

    async def broken_async_guard(**_kwargs):
        raise RuntimeError("async judge unavailable")

    manager._hooks["pre_llm_call"] = [broken_async_guard]
    results = asyncio.run(manager.ainvoke_hook("pre_llm_call", session_id="async-degrade"))

    assert len(results) == 1 and "action" not in results[0]
    assert "async judge unavailable" in results[0]["context"]


def test_async_pre_verify_failure_is_fail_closed() -> None:
    manager = PluginManager()

    async def broken_async_guard(**_kwargs):
        raise RuntimeError("async judge unavailable")

    manager._hooks["pre_verify"] = [broken_async_guard]
    results = asyncio.run(manager.ainvoke_hook("pre_verify", session_id="async-policy"))

    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert "async judge unavailable" in results[0]["message"]


def test_veto_on_timeout_is_a_noop_for_an_observer_registration() -> None:
    """The flag is a semantic claim about a policy hook; an observer has nothing to veto."""
    manager = PluginManager()

    def observer(**_kwargs):
        return None

    _opt_into_veto(manager, "post_tool_call", observer)
    manager._hooks["post_tool_call"] = [observer]
    assert manager.invoke_hook("post_tool_call", session_id="s", tool_call_id="c") == []


@pytest.mark.parametrize("hook_name", ["post_llm_call", "post_tool_call"])
def test_observer_failure_is_nonblocking_but_unknown(hook_name: str) -> None:
    manager = PluginManager()

    def broken_observer(**_kwargs):
        raise RuntimeError("observer backend unavailable")

    manager._hooks[hook_name] = [broken_observer]
    results = manager.invoke_hook(
        hook_name,
        session_id="observer-failure",
        tool_call_id="call-1",
    )

    assert results == []
    event = manager.gate_plane.events("observer-failure")[0]
    assert event.hook == hook_name
    assert event.status.value == "unknown"
    assert event.detail == "RuntimeError: observer backend unavailable"


def test_async_observer_failure_is_nonblocking_but_unknown() -> None:
    manager = PluginManager()

    async def broken_observer(**_kwargs):
        raise RuntimeError("async observer unavailable")

    manager._hooks["post_llm_call"] = [broken_observer]
    results = asyncio.run(manager.ainvoke_hook(
        "post_llm_call", session_id="async-observer",
    ))

    assert results == []
    event = manager.gate_plane.events("async-observer")[0]
    assert event.hook == "post_llm_call"
    assert event.status.value == "unknown"
    assert event.detail == "RuntimeError: async observer unavailable"
