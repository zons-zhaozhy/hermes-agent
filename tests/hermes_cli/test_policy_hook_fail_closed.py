"""Intent and completion policy hooks cannot silently fail open.

A guard that raises or exhausts its callback budget made no decision. For
``pre_llm_call`` that must veto the turn before inference, and for
``pre_verify`` it must keep the turn open; treating either failure as an empty
success lets the exact work the guard protects proceed unaudited.
"""

from __future__ import annotations

import asyncio

import pytest

from hermes_cli.plugins import PluginManager


@pytest.mark.parametrize("hook_name", ["pre_llm_call", "pre_verify"])
def test_policy_failure_returns_named_block(hook_name: str) -> None:
    manager = PluginManager()

    def broken_guard(**_kwargs):
        raise RuntimeError("judge unavailable")

    manager._hooks[hook_name] = [broken_guard]
    results = manager.invoke_hook(hook_name, session_id="policy-failure", turn_id="turn-1")

    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert hook_name in results[0]["message"]
    assert "broken_guard" in results[0]["message"]
    assert "judge unavailable" in results[0]["message"]


@pytest.mark.parametrize("hook_name", ["pre_llm_call", "pre_verify"])
def test_policy_timeout_returns_named_block(monkeypatch, hook_name: str) -> None:
    import threading

    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05
    )
    hold = threading.Event()

    def hung_guard(**_kwargs):
        hold.wait(timeout=10.0)

    manager = PluginManager()
    manager._hook_timeout_suppression_seconds = 0.0
    manager._hooks[hook_name] = [hung_guard]
    try:
        results = manager.invoke_hook(hook_name, session_id="policy-timeout", turn_id="turn-1")
    finally:
        hold.set()

    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert hook_name in results[0]["message"]
    assert "hung_guard" in results[0]["message"], (
        "the block names the callback, not only the hook: the operator must be able to tell "
        "which guard hung and whether the automatic retry already happened"
    )


@pytest.mark.parametrize("hook_name", ["pre_llm_call", "pre_verify"])
def test_async_policy_failure_is_fail_closed(hook_name: str) -> None:
    manager = PluginManager()

    async def broken_async_guard(**_kwargs):
        raise RuntimeError("async judge unavailable")

    manager._hooks[hook_name] = [broken_async_guard]
    results = asyncio.run(manager.ainvoke_hook(hook_name, session_id="async-policy"))

    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert "async judge unavailable" in results[0]["message"]


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
