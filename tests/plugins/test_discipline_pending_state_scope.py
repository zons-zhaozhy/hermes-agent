"""Discipline tool-event reminders stay bound to the session that caused them."""

from __future__ import annotations

from plugins.discipline import curl_safety, log_first


def test_http_pending_reminder_is_not_consumed_by_another_session() -> None:
    curl_safety._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "curl -I https://example.invalid"},
        session_id="http-owner",
        task_id="http-owner",
    )

    foreign = curl_safety._on_pre_llm_call(
        session_id="http-other",
        task_id="http-other",
        user_message="你还在工作吗",
    )
    owner = curl_safety._on_pre_llm_call(
        session_id="http-owner",
        task_id="http-owner",
        user_message="继续处理这个请求",
    )

    assert foreign == {}
    assert "HTTP 请求安全提醒" in owner.get("context", "")


def test_error_pending_reminder_is_not_consumed_by_another_session() -> None:
    log_first._on_post_tool_call(
        tool_name="terminal",
        result={"exit_code": 1, "output": "boom"},
        session_id="error-owner",
        task_id="error-owner",
    )

    foreign = log_first._on_pre_llm_call(
        session_id="error-other",
        task_id="error-other",
        user_message="我今天有点累",
    )
    owner = log_first._on_pre_llm_call(
        session_id="error-owner",
        task_id="error-owner",
        user_message="继续修复这个错误",
    )

    assert foreign == {}
    assert "错误诊断强制提醒" in owner.get("context", "")


def test_json_string_error_creates_owner_scoped_reminder() -> None:
    log_first._on_post_tool_call(
        tool_name="terminal",
        result='{"exit_code": 3, "output": "missing file"}',
        session_id="json-error-owner",
    )

    out = log_first._on_pre_llm_call(
        session_id="json-error-owner",
        user_message="继续修复这个错误",
    )

    assert "错误诊断强制提醒" in out.get("context", "")
