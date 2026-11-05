"""quality-auditor 插件统计与注入边界直接契约。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins import quality_auditor
from plugins.quality_auditor import auditor


def _call(name: str) -> dict:
    return {"function": {"name": name, "arguments": "{}"}}


def test_duplicate_tool_names_count_calls_and_deduplicate_names() -> None:
    history = [
        {"role": "assistant", "tool_calls": [_call("old_tool")]},
        {"role": "user", "content": "audit this turn"},
        {
            "role": "assistant",
            "tool_calls": [_call("terminal"), _call("read_file"), _call("terminal")],
        },
    ]

    assert quality_auditor._extract_tool_stats(history) == (
        3,
        ["terminal", "read_file"],
    )


def test_empty_and_malformed_history_return_zero_stats() -> None:
    assert quality_auditor._extract_tool_stats([]) == (0, [])
    assert quality_auditor._extract_tool_stats([{"role": "user"}]) == (0, [])
    assert quality_auditor._extract_tool_stats(
        [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": "bad"}]
    ) == (0, [])


def test_short_response_never_imports_or_fires_auditor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("trivial response must not reach the auditor")

    monkeypatch.setattr(auditor, "fire_quality_audit", forbidden)
    quality_auditor.on_post_llm_call(
        assistant_response="x" * 49,
        user_message="question",
        session_id="quality-short",
        model="test-model",
        conversation_history=[{"role": "user", "content": "question"}],
    )


def test_post_llm_call_passes_current_turn_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def capture(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(auditor, "fire_quality_audit", capture)
    quality_auditor.on_post_llm_call(
        assistant_response="x" * 50,
        user_message="audit",
        session_id="quality-stats",
        model="test-model",
        conversation_history=[
            {"role": "user", "content": "audit"},
            {
                "role": "assistant",
                "tool_calls": [_call("terminal"), _call("terminal")],
            },
        ],
    )

    assert calls == [
        {
            "user_message": "audit",
            "assistant_response": "x" * 50,
            "session_id": "quality-stats",
            "model": "test-model",
            "tool_call_count": 2,
            "tool_names": ["terminal"],
        }
    ]


def test_feedback_injection_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("missing session must not read feedback")

    monkeypatch.setattr(auditor, "get_last_audit_feedback", forbidden)
    assert quality_auditor.on_pre_llm_call(session_id="") is None

    monkeypatch.setattr(
        auditor, "get_last_audit_feedback", lambda session_id: "fix tests"
    )
    assert quality_auditor.on_pre_llm_call(session_id="quality-feedback") == {
        "context": "fix tests"
    }
