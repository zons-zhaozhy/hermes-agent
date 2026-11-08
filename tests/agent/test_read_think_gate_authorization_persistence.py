"""ReadThinkGate authorization continuity across operational turns.

A background completion is machinery input, not a human authorization decision. It must
not erase an earlier modification authorization merely by becoming the latest turn text.
Operational feedback in the same task is likewise not a new read-only task.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plugins._shared_state import get_session_state
from plugins.read_think_gate_host import (
    _STATE_NAMESPACE,
    ReadThinkGate,
    ReadThinkGateConfig,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


def _strict_gate(tmp_path: Path, tag: str) -> ReadThinkGate:
    return ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": False,
                "min_read_only_calls": 2,
                "min_reasoning_chars": 200,
                "max_reasoning_rounds": 6,
            }
        ),
        session_id=f"auth-{tag}-{tmp_path.name}",
    )


def test_internal_background_notice_preserves_modification_authorization(
    tmp_path: Path,
) -> None:
    gate = _strict_gate(tmp_path, "internal")
    gate.reset_for_turn("请修复解析器并运行测试")

    gate.reset_for_turn(
        "[IMPORTANT: Background process proc_1 completed normally (exit code 0).]",
        internal=True,
    )

    assert gate.user_intent == "modify"
    assert gate._modification_authorized is True


def test_plugin_maps_internal_display_kind_to_gate_input(tmp_path: Path) -> None:
    import plugins.read_think_gate as plugin

    gate = _strict_gate(tmp_path, "plugin")
    gate.reset_for_turn("请修复解析器并运行测试")
    sid = gate._session_id
    plugin._GATES[sid] = gate
    try:
        plugin.pre_llm_call(
            user_message="[IMPORTANT: Background process proc_2 exited (exit code 7).]",
            session_id=sid,
            platform="local",
            turn_display_kind="internal_notification",
        )
    finally:
        plugin._GATES.pop(sid, None)

    assert gate.user_intent == "modify"


def test_operational_feedback_keeps_task_authorization(tmp_path: Path) -> None:
    gate = _strict_gate(tmp_path, "feedback")
    gate.reset_for_turn("请修复解析器并运行测试")

    gate.reset_for_turn("啥情况你要及时汇报反馈，不能一直闷头干活")

    assert gate.user_intent == "modify"


def test_operational_feedback_survives_a_read_only_semantic_verdict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import plugins.read_think_gate_host as host

    gate = ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": True,
            }
        ),
        session_id=f"auth-semantic-{tmp_path.name}",
    )
    monkeypatch.setattr(
        host,
        "_classify_intent_via_llm",
        lambda message, history=None: host.INTENT_READ_ONLY,
    )
    gate.reset_for_turn("请修复解析器并运行测试")

    gate.reset_for_turn("进度")

    assert gate.user_intent == "modify"
    assert gate._modification_authorized is True


def test_terse_allow_authorizes_the_proposed_modification(tmp_path: Path) -> None:
    gate = _strict_gate(tmp_path, "allow")

    gate.reset_for_turn(
        "允许",
        [{"role": "assistant", "content": "方案：执行 quick336 并记录结果。"}],
    )

    assert gate.user_intent == "modify"


def test_clarified_answer_clears_an_active_authorization_handoff(
    tmp_path: Path,
) -> None:
    gate = _strict_gate(tmp_path, "waiting")
    gate.reset_for_turn("请修复解析器并运行测试")
    gate._awaiting_user_authorization = True
    gate._clarify_context = "方案：继续修复解析器并运行相关测试。"

    gate.note_user_clarification("按方案执行")

    assert gate._awaiting_user_authorization is False
    assert gate._modification_authorized is True
    assert get_session_state(gate._session_id, _STATE_NAMESPACE)[
        "awaiting_user_authorization"
    ] is False


def test_new_read_only_request_does_not_inherit_authorization(
    tmp_path: Path,
) -> None:
    gate = _strict_gate(tmp_path, "new-task")
    gate.reset_for_turn("请修复解析器并运行测试")

    gate.reset_for_turn("只分析另一个模块的性能问题，不要修改")

    assert gate.user_intent == "read_only"
    assert gate._modification_authorized is False


def test_explicit_no_modify_feedback_revokes_authorization(tmp_path: Path) -> None:
    gate = _strict_gate(tmp_path, "revoke")
    gate.reset_for_turn("请修复解析器并运行测试")

    gate.reset_for_turn("只汇报当前状态，不要修改")

    assert gate.user_intent == "read_only"
    assert gate._modification_authorized is False


def test_collect_pre_llm_call_passes_turn_display_kind() -> None:
    from agent.turn_context import _collect_pre_llm_call_context

    agent = SimpleNamespace(
        session_id="auth-dispatch",
        _persist_disabled=False,
        model="test-model",
        platform="local",
        _parent_session_id="",
        _user_id="uid:1",
    )
    with patch("hermes_cli.lifecycle.invoke_hook", return_value=[]) as invoke:
        _collect_pre_llm_call_context(
            agent,
            effective_task_id="task",
            turn_id="turn",
            original_user_message="[IMPORTANT: Background process done.]",
            messages=[],
            conversation_history=[],
            turn_display_kind="internal_notification",
        )

    assert invoke.call_args.kwargs["turn_display_kind"] == "internal_notification"
