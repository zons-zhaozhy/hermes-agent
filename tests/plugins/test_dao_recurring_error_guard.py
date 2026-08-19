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
def _clean_state(tmp_path, monkeypatch):
    # 隔离持久层：HERMES_HOME 指向 tmp（conftest 已设 HERMES_HOME，但
    # get_hermes_home 可能读 override——显式指回 tmp 确保零写 ~/.hermes）
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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


def test_persist_across_sessions(plugin, tmp_path):
    """跨会话累计：同指纹（terminal×ImportError）s1 计 3 次后，s2 计 1 次 → 累计 4。"""
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    clear_session("s1")
    plugin.on_post_tool_call(
        session_id="s2", tool_name="terminal",
        args={"command": "x"},
        result={"error": "ImportError: no module named bar"},
        status="error",
    )
    counts = plugin._load_persist()
    assert counts.get("terminal×ImportError", {}).get("n") == 4
    clear_session("s2")


def test_persist_window_decay(plugin):
    """14 天窗口外条目衰减：写旧 ts → load 返回空。"""
    plugin._save_persist({"terminal×Timeout": {"n": 9, "ts": 1}})
    assert plugin._load_persist() == {}


def test_persist_corrupt_fail_open(plugin):
    """持久文件损坏 → fail-open 返回空 dict，不抛异常。"""
    plugin._persist_path().parent.mkdir(parents=True, exist_ok=True)
    plugin._persist_path().write_text("NOT-JSON{", encoding="utf-8")
    assert plugin._load_persist() == {}


def test_reminder_includes_total(plugin):
    """提醒文本含跨会话累计。"""
    for _ in range(3):
        plugin.on_post_tool_call(**_err())
    out = plugin.on_pre_llm_call(session_id="s1")
    assert out is not None
    assert "跨会话累计" in out["context"]


def test_bump_persist_concurrent_multiprocess(plugin):
    """并发安全：N 个真实子进程并发 bump 同一指纹 → 计数恰好=N（零丢失）。"""
    import os
    import subprocess
    import sys

    n_proc = 12
    code = (
        "import os,sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "os.environ.setdefault('HERMES_HOME', sys.argv[1])\n"
        "import importlib\n"
        "g = importlib.import_module('plugins.dao-recurring-error-guard')\n"
        "g._bump_persist_count('terminal×ImportError')\n"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, os.environ["HERMES_HOME"]])
        for _ in range(n_proc)
    ]
    for pr in procs:
        assert pr.wait() == 0
    counts = plugin._load_persist()
    assert counts["terminal×ImportError"]["n"] == n_proc
