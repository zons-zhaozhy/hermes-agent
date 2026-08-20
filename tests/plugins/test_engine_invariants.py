"""Tests for engine-invariants plugin (不变量注册表检测层).

行为契约：
- 完好序列 → 全部不变量零违反
- 孤儿 tool 结果 / 未应答 tool_call / 连续 user → 对应项计违反
- 违反只计数+记日志，不改消息（返回 None，history 原样）
- 单项 checker 异常 → checker_error 条目，不影响其他项
- env 开关 → 直接跳过；空历史 → 跳过
- post_llm_call 钩子签名兼容 turn_finalizer 传的全部 kwargs
"""
import importlib

import pytest

plugin = importlib.import_module("plugins.engine-invariants")
from plugins._shared_state import clear_session


@pytest.fixture(autouse=True)
def _reset():
    clear_session("s1")
    yield
    clear_session("s1")


@pytest.fixture
def hooks():
    captured = {}

    class Ctx:
        @staticmethod
        def register_hook(name, fn):
            captured[name] = fn

    plugin.register(Ctx())
    return captured


OK_PAIR = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "t"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    {"role": "assistant", "content": "done"},
]

ORPHAN_RESULT = [
    {"role": "user", "content": "hi"},
    {"role": "tool", "tool_call_id": "ghost", "content": "x"},
]

UNANSWERED = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
                                          "function": {"name": "t"}}]},
    {"role": "assistant", "content": "moved on"},
]

CONSEC_USER = [
    {"role": "user", "assistant": "x", "content": "a"},  # role 用错也不炸
    {"role": "user", "content": "b"},
]


def test_ok_sequence_zero_violations():
    res = plugin.run_all_invariants(OK_PAIR)
    assert res["tool_pairing"] == []
    assert res["role_alternation"] == []


def test_orphan_tool_result_detected():
    res = plugin.run_all_invariants(ORPHAN_RESULT)
    assert any("orphan_tool_result[ghost]" in v
               for v in res["tool_pairing"])


def test_unanswered_tool_call_detected():
    res = plugin.run_all_invariants(UNANSWERED)
    assert any("unanswered_tool_call[c1]" in v
               for v in res["tool_pairing"])


def test_consecutive_user_detected():
    res = plugin.run_all_invariants(CONSEC_USER)
    assert any("consecutive_user@" in v for v in res["role_alternation"])


def test_dup_call_id_detected():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok2"},
    ]
    res = plugin.run_all_invariants(msgs)
    assert any("dup_tool_call_id[c1]" in v for v in res["tool_pairing"])


def test_hook_counts_and_does_not_modify(hooks):
    post = hooks["post_llm_call"]
    snapshot = [dict(m) for m in ORPHAN_RESULT]
    # turn_finalizer 实参签名
    post(conversation_history=ORPHAN_RESULT, session_id="s1",
         task_id="t1", turn_id="u1", user_message="q",
         assistant_response="a", model="m", platform="cli")
    st = plugin.get_session_state("s1", "engine_invariants")
    assert st["violations_tool_pairing"] >= 1
    # 消息列表原样（只检测不修复）
    assert ORPHAN_RESULT == snapshot


def test_hook_disabled_by_env(hooks, monkeypatch):
    monkeypatch.setenv("ENGINE_INVARIANTS_DISABLE", "1")
    post = hooks["post_llm_call"]
    post(conversation_history=ORPHAN_RESULT, session_id="s1")
    st = plugin.get_session_state("s1", "engine_invariants")
    assert "violations_tool_pairing" not in st


def test_hook_empty_history_skipped(hooks):
    post = hooks["post_llm_call"]
    post(conversation_history=[], session_id="s1")
    st = plugin.get_session_state("s1", "engine_invariants")
    assert not st


def test_checker_error_isolated(monkeypatch):
    def boom(msgs):
        raise RuntimeError("checker blew up")

    monkeypatch.setitem(plugin.INVARIANTS, "tool_pairing", boom)
    res = plugin.run_all_invariants(OK_PAIR)
    assert any(v.startswith("checker_error:RuntimeError")
               for v in res["tool_pairing"])
    # 其他项不受影响
    assert res["role_alternation"] == []
