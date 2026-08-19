"""Tests for dao-recurring-error-guard plugin.

行为契约：
- classify_error：有限类别词命中归一化；无命中/空文本 → None
- 错误计数：同类指纹累计；不同工具或不同类别分开计
- 第 3 次且未沉淀 → pre_llm_call 注入提醒（含指纹与次数）
- 沉淀动作（write_file/patch/git commit）后 → 静默
- env 开关 → 静默
"""

import importlib

import pytest

from plugins._shared_state import clear_session


@pytest.fixture(autouse=True)
def _clean_state():
    clear_session("s1")
    yield
    clear_session("s1")


@pytest.fixture
def plugin():
    return importlib.import_module("plugins.dao-recurring-error-guard")


def _err(sid="s1", tool="terminal", text="ImportError: no module named foo", status="error"):
    return dict(session_id=sid, tool_name=tool, args={}, result={"error": text}, status=status)


def test_classify_known(plugin):
    assert plugin.classify_error("ModuleNotFoundError: x") == "ImportError"
    assert plugin.classify_error("HTTP 403 Forbidden") == "Forbidden-403"
    assert plugin.classify_error("ERROR 1064 (42000)") == "SQL-Syntax-1064"


def test_classify_unknown_and_empty(plugin):
    assert plugin.classify_error("weird failure") is None
    assert plugin.classify_error("") is None


def test_reminder_after_third_error(plugin):
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    out = plugin.on_pre_llm_call(session_id="s1")
    assert out is not None
    assert "道生法" in out["context"]
    assert "ImportError" in out["context"]


def test_no_reminder_below_threshold(plugin):
    plugin.on_post_tool_call(**_err())
    plugin.on_post_tool_call(**_err())
    assert plugin.on_pre_llm_call(session_id="s1") is None


def test_fingerprints_accumulate_separately(plugin):
    plugin.on_post_tool_call(**_err(tool="terminal"))
    plugin.on_post_tool_call(**_err(tool="web_extract", text="ReadTimeout after 30s"))
    assert plugin.on_pre_llm_call(session_id="s1") is None


def test_codification_silences(plugin):
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    plugin.on_post_tool_call(
        session_id="s1", tool_name="write_file", args={"path": "fix.sh"},
        result={}, status="success",
    )
    assert plugin.on_pre_llm_call(session_id="s1") is None


def test_git_commit_counts_as_codification(plugin):
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    plugin.on_post_tool_call(
        session_id="s1", tool_name="terminal", args={"command": "git commit -m fix"},
        result={}, status="success",
    )
    assert plugin.on_pre_llm_call(session_id="s1") is None


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("DAO_RECURRING_ERROR_GUARD_DISABLE", "1")
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    assert plugin.on_pre_llm_call(session_id="s1") is None
