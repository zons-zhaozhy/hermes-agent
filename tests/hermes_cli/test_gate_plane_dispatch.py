"""Policy-hook dispatch feeds the profile-owned Gate Plane evidence ledger."""

from __future__ import annotations

import asyncio

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_dispatch import _HOOK_SKIPPED


def _events(manager: PluginManager, hook: str) -> list[dict]:
    return [event.as_dict() for event in manager.gate_plane.events() if event.hook == hook]


def test_sync_pass_and_block_results_are_recorded_without_payloads() -> None:
    manager = PluginManager()

    def allow(**kwargs):
        assert kwargs["args"] == {"secret-command": "must-not-enter-ledger"}

    def block(**_kwargs):
        return {"action": "block", "message": "unauthorized write"}

    manager._hooks["pre_tool_call"] = [allow, block]
    results = manager.invoke_hook(
        "pre_tool_call",
        args={"secret-command": "must-not-enter-ledger"},
        tool_call_id="call-1",
        session_id="session-1",
    )

    assert results == [{"action": "block", "message": "unauthorized write"}]
    events = _events(manager, "pre_tool_call")
    assert [
        {key: value for key, value in event.items() if key != "callback"}
        for event in events
    ] == [
        {
            "hook": "pre_tool_call",
            "session_id": "session-1",
            "call_id": "call-1",
            "status": "pass",
            "detail": "",
            "duration_ms": events[0]["duration_ms"],
        },
        {
            "hook": "pre_tool_call",
            "session_id": "session-1",
            "call_id": "call-1",
            "status": "block",
            "detail": "action=block",
            "duration_ms": events[1]["duration_ms"],
        },
    ]
    assert events[0]["callback"].endswith(".allow")
    assert events[1]["callback"].endswith(".block")
    assert "must-not-enter-ledger" not in str(manager.gate_plane.events())


def test_exception_records_unknown_and_degrades_the_directive() -> None:
    """The evidence ledger keeps the failure; the turn keeps going with a loud UNKNOWN note."""
    manager = PluginManager()

    def broken(**_kwargs):
        raise RuntimeError("judge backend unavailable")

    manager._hooks["pre_llm_call"] = [broken]
    results = manager.invoke_hook(
        "pre_llm_call", session_id="session-2", turn_id="turn-2",
    )

    assert len(results) == 1 and "action" not in results[0]
    assert "judge backend unavailable" in results[0]["context"]
    event = _events(manager, "pre_llm_call")[0]
    assert event["status"] == "unknown"
    assert event["detail"] == "RuntimeError: judge backend unavailable"
    assert event["call_id"] == "turn-2"


def test_skipped_timeout_path_records_unknown(monkeypatch) -> None:
    manager = PluginManager()

    def hung(**_kwargs):
        return None

    manager._hooks["pre_verify"] = [hung]
    monkeypatch.setattr(
        manager,
        "_run_hook_callback_bounded",
        lambda *_args, **_kwargs: _HOOK_SKIPPED,
    )
    results = manager.invoke_hook(
        "pre_verify", session_id="session-3", turn_id="turn-3",
    )

    assert results[0]["action"] == "block"
    assert _events(manager, "pre_verify")[0]["status"] == "unknown"
    assert _events(manager, "pre_verify")[0]["detail"] == "callback skipped or timed out"


def test_async_pass_and_exception_are_recorded() -> None:
    manager = PluginManager()

    async def allow(**_kwargs):
        return None

    async def broken(**_kwargs):
        raise RuntimeError("async judge unavailable")

    manager._hooks["pre_llm_call"] = [allow]
    assert asyncio.run(manager.ainvoke_hook(
        "pre_llm_call", session_id="async-1", turn_id="async-turn-1",
    )) == []
    assert _events(manager, "pre_llm_call")[0]["status"] == "pass"

    manager._hooks["pre_verify"] = [broken]
    results = asyncio.run(manager.ainvoke_hook(
        "pre_verify", session_id="async-2", turn_id="async-turn-2",
    ))
    assert results[0]["action"] == "block"
    assert _events(manager, "pre_verify")[0]["status"] == "unknown"
    assert _events(manager, "pre_verify")[0]["detail"] == "RuntimeError: async judge unavailable"


def test_composed_transform_records_every_callback_once() -> None:
    manager = PluginManager()

    def first(response_text: str, **_kwargs):
        return response_text + "\n[first]"

    def second(response_text: str, **_kwargs):
        return response_text + "\n[second]"

    manager._hooks["transform_llm_output"] = [first, second]
    results = manager.invoke_hook(
        "transform_llm_output",
        response_text=" confidential answer ",
        session_id="transform-1",
        turn_id="transform-turn-1",
    )

    assert results == [" confidential answer \n[first]\n[second]"]
    events = _events(manager, "transform_llm_output")
    assert [event["callback"].rsplit(".", 1)[-1] for event in events] == ["first", "second"]
    assert all(event["status"] == "pass" for event in events)
    assert "confidential answer" not in str(manager.gate_plane.events())
