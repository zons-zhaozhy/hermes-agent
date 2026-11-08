"""ReadThinkGate clarify-authorization contracts.

The gate resolves user intent once per turn (``pre_llm_call`` →
``reset_for_turn``). A user who authorizes a modification mid-turn through the
clarify form therefore changed nothing in that snapshot: the write stayed
blocked until a fresh chat message restarted the turn (2026-11-08 live
session, two consecutive blocks after an explicit「修，按此方案执行」answer).

Expected verdicts derived from the gate lifecycle contract, not from the
implementation: a clarified user answer is user input on the tool channel, so
it must be able to grant authorization — while an unanswered form, a
read-only-flavored answer, or a malformed payload must not.
"""

from __future__ import annotations

import json
from pathlib import Path

from plugins.read_think_gate_host import ReadThinkGate, ReadThinkGateConfig


def _strict_gate(tmp_path: Path, session_tag: str) -> ReadThinkGate:
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
        session_id=f"clarify-{session_tag}-{tmp_path.name}",
    )


def _record_complete_investigation(gate: ReadThinkGate, target: Path) -> None:
    gate._read_only_count = 3
    gate._investigation_evidence = [
        f"read_file: {target}",
        "search_files(callers): callers",
        "search_files(tests): tests",
        "terminal(readonly): git log -S target",
    ]
    gate._files_read = {str(target)}
    gate._four_axis_found = {"影响面", "原意图", "根因", "风险"}
    gate._satisfied = False
    gate._reasoning_rounds = 0


def _open_read_only_turn_with_clarify(gate: ReadThinkGate) -> None:
    """Read-only opener; the assistant then asks via clarify with a proposal."""
    gate.reset_for_turn("帮我看看这个文件为什么返回错误，先不要动手")
    gate.check_batch(
        "已定位根因。修复方案：把 value 改为 2 并补测试，现在向你确认是否执行。",
        ["clarify"],
        None,
    )


def test_clarified_answer_authorizes_the_pending_write(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path, "auth")
    _open_read_only_turn_with_clarify(gate)

    gate.note_user_clarification("按方案执行")
    _record_complete_investigation(gate, target)

    result = gate.check_batch(
        "影响面、原意图、根因、风险均已核实；验证命令为 scripts/run_tests.sh。"
        "调用方覆盖 app.main，现有测试覆盖成功和失败路径，最坏风险可回滚。" * 3,
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is None
    assert gate.is_satisfied


def test_read_only_clarified_answer_keeps_the_write_blocked(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path, "deny")
    _open_read_only_turn_with_clarify(gate)

    gate.note_user_clarification("先等等，我再想想")
    _record_complete_investigation(gate, target)

    result = gate.check_batch(
        "x" * 300,
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is not None
    assert "用户意图" in result


def test_plugin_intake_parses_answered_clarify_result(tmp_path: Path) -> None:
    import plugins.read_think_gate as rtg_plugin

    gate = _strict_gate(tmp_path, "intake")
    sid = f"intake-{tmp_path.name}"
    rtg_plugin._GATES[sid] = gate
    try:
        _open_read_only_turn_with_clarify(gate)

        rtg_plugin.on_post_tool_call(
            tool_name="clarify",
            result=json.dumps(
                {
                    "responses": [
                        {"status": "skipped", "user_response": None},
                        {"status": "answered", "user_response": "按方案执行"},
                    ]
                }
            ),
            session_id=sid,
            platform="cli",
        )

        assert gate._modification_authorized is True
    finally:
        rtg_plugin._GATES.pop(sid, None)


def test_plugin_intake_ignores_malformed_and_unanswered_results(
    tmp_path: Path,
) -> None:
    import plugins.read_think_gate as rtg_plugin

    gate = _strict_gate(tmp_path, "malformed")
    sid = f"malformed-{tmp_path.name}"
    rtg_plugin._GATES[sid] = gate
    try:
        _open_read_only_turn_with_clarify(gate)

        rtg_plugin.on_post_tool_call(
            tool_name="clarify", result="not-json{", session_id=sid, platform="cli"
        )
        rtg_plugin.on_post_tool_call(
            tool_name="clarify",
            result=json.dumps(
                {
                    "outcome": "cancelled",
                    "responses": [{"status": "skipped", "user_response": None}],
                }
            ),
            session_id=sid,
            platform="cli",
        )
        rtg_plugin.on_post_tool_call(
            tool_name="read_file",
            result=json.dumps({"responses": [{"status": "answered", "user_response": "修"}]}),
            session_id=sid,
            platform="cli",
        )

        assert gate._modification_authorized is False
    finally:
        rtg_plugin._GATES.pop(sid, None)
