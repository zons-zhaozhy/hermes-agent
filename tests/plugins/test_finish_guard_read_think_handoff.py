"""finish_guard must honor ReadThinkGate's user-authorization handoff."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from plugins._shared_state import get_session_state
from plugins.finish_guard import _gap_streaks, _on_pre_tool_call, _on_pre_verify
from plugins.read_think_gate_host import _STATE_NAMESPACE


@pytest.fixture(autouse=True)
def _clean_handoff_state(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    get_session_state("handoff", _STATE_NAMESPACE).clear()
    _gap_streaks.clear()
    yield
    get_session_state("handoff", _STATE_NAMESPACE).clear()
    _gap_streaks.clear()


def test_gate_publishes_waiting_for_user_authorization() -> None:
    from plugins.read_think_gate_host import ReadThinkGate, ReadThinkGateConfig

    gate = ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {"use_llm_intent_classifier": False, "complexity_adaptive": False}
        ),
        session_id="handoff",
    )
    gate.reset_for_turn("只分析这个问题，先不要修改")
    state = get_session_state("handoff", _STATE_NAMESPACE)

    result = gate.check_batch(
        "尚未获得用户授权。",
        ["terminal"],
        [{"command": "printf done > /tmp/result.txt"}],
    )

    assert result is not None
    assert state["awaiting_user_authorization"] is True


def test_waiting_for_authorization_allows_clarify() -> None:
    get_session_state("handoff", _STATE_NAMESPACE)[
        "awaiting_user_authorization"
    ] = True

    result = _on_pre_tool_call(
        session_id="handoff",
        tool_name="clarify",
        args={"questions": [{"question": "是否继续执行该真实设备任务？"}]},
    )

    assert result == {}


def test_waiting_for_authorization_stops_pre_verify_nudge() -> None:
    from plugins.finish_guard.todo_debt import _update_ledger

    _update_ledger(
        "handoff", [{"id": "w1", "content": "执行任务", "status": "in_progress"}]
    )
    get_session_state("handoff", _STATE_NAMESPACE)[
        "awaiting_user_authorization"
    ] = True

    result = _on_pre_verify(
        session_id="handoff",
        attempt=0,
        final_response="需要用户确认是否继续执行。",
        changed_paths=[],
    )

    assert result == {}
