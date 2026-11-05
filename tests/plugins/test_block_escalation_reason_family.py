"""Block-escalation fingerprints must distinguish guard reason families."""

from __future__ import annotations

import json
import pathlib
import sqlite3

import pytest

import plugins.block_escalation as be


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(be, "_db_path", tmp_path / "be.db")
    monkeypatch.setattr(be, "_escalated", {})
    conn = sqlite3.connect(tmp_path / "be.db")
    try:
        be._ensure_schema(conn)
    finally:
        conn.close()


def _rows(tmp_path: pathlib.Path) -> list[tuple[str, int]]:
    conn = sqlite3.connect(tmp_path / "be.db")
    try:
        return conn.execute(
            "SELECT fingerprint, count FROM block_streaks ORDER BY fingerprint"
        ).fetchall()
    finally:
        conn.close()


def test_different_plugin_guard_reasons_do_not_escalate_same_target(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(tmp_path, monkeypatch)
    args = {"path": "/tmp/report.md", "content": "x"}
    reasons = (
        "[PreWriteGuard] 预调查证据不足，读取目标与影响面后再写入。",
        "[ToolSafety] 写入参数包含不安全路径，先核对路径。",
    )
    for message in reasons:
        be._on_post_tool_call(
            tool_name="write_file", status="blocked", args=args,
            session_id="reasons", error_type="plugin_block",
            error_message=message,
        )

    assert [count for _, count in _rows(tmp_path)] == [1, 1]
    assert be._on_transform_llm_output("继续", session_id="reasons") is None


def test_same_guard_reason_survives_volatile_details(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(tmp_path, monkeypatch)
    args = {"path": "/tmp/report.md", "content": "x"}
    for attempt in (1, 2):
        be._on_post_tool_call(
            tool_name="write_file", status="blocked", args=args,
            session_id="same-reason", error_type="plugin_block",
            error_message=f"[PreWriteGuard] 第 {attempt} 次预调查不足。",
        )

    assert [count for _, count in _rows(tmp_path)] == [2]
    output = be._on_transform_llm_output("继续", session_id="same-reason")
    assert output is not None
    assert be.ESCALATION_MARKER in output


def test_runtime_guardrail_codes_split_fingerprints(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(tmp_path, monkeypatch)
    args = {"path": "/tmp/report.md", "content": "x"}
    for code in ("repeated_exact_failure_block", "idempotent_no_progress_block"):
        result = json.dumps({
            "error": "blocked",
            "guardrail": {"action": "block", "code": code},
        }, ensure_ascii=False)
        be._on_post_tool_call(
            tool_name="write_file", status="blocked", args=args,
            session_id="codes", error_type="guardrail_block",
            error_message="blocked", result=result,
        )

    assert [count for _, count in _rows(tmp_path)] == [1, 1]
    assert be._on_transform_llm_output("继续", session_id="codes") is None
