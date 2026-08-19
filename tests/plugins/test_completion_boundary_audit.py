"""Tests for completion-boundary-audit plugin (LLM judge 版, 注入范式).

行为契约：
- judge 判 True（完成声明+无边界）→ 用户可见回复零改动（返回 None），
  记 pending_reminder，下一轮 pre_llm_call 注入红牌（一次消费）
- judge 判 False / None → 透传 None，不记状态
- 短回复（<80字符）→ 直接跳过不调 judge
- 同一回复 hash 去重；judge_calls 上限 30；红牌上限 3
- env 开关 → None；钩子异常 → None
"""
import importlib

import pytest

plugin = importlib.import_module("plugins.completion-boundary-audit")
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


@pytest.fixture
def hook(hooks):
    return hooks["transform_llm_output"]


@pytest.fixture
def inject(hooks):
    return hooks["pre_llm_call"]


LONG_DONE = "x" * 100 + " 全部测试已通过，任务完成。"
LONG_DONE_2 = "y" * 100 + " 已修复并验证完成。"
LONG_WITH_BOUNDARY = "z" * 100 + " 已完成，但未验证生产环境。"


def test_judge_true_marks_and_injects(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    # 用户可见回复零改动
    assert hook(LONG_DONE, session_id="s1") is None
    # 下一轮注入红牌
    out = inject(session_id="s1")
    assert out is not None and "反面检查" in out["context"]
    # 注入一次即消费
    assert inject(session_id="s1") is None


def test_judge_false_no_inject(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: False)
    assert hook(LONG_DONE, session_id="s1") is None
    assert inject(session_id="s1") is None


def test_judge_none_fail_open(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: None)
    assert hook(LONG_DONE, session_id="s1") is None
    assert inject(session_id="s1") is None


def test_short_reply_skipped(hook, monkeypatch):
    called = []
    monkeypatch.setattr(plugin, "needs_boundary_audit",
                        lambda t: called.append(t) or True)
    assert hook("好的完成了", session_id="s1") is None
    assert called == []


def test_dedup(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    assert hook(LONG_DONE, session_id="s1") is None
    # 第二次同文本不重复触发（无新标记）
    assert hook(LONG_DONE, session_id="s1") is None
    # 只有一次注入
    assert inject(session_id="s1") is not None
    assert inject(session_id="s1") is None


def test_max_reminders(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    marked = sum(
        1 for c in "abcde"
        if hook(f"{c*100} 已全部完成。", session_id="s1") is None
        and plugin._state("s1").get("pending_reminder")
    )
    # 前 3 次触发标记，之后达到红牌上限
    outs = [inject(session_id="s1") is not None for _ in range(5)]
    assert sum(outs) >= 1
    st = plugin._state("s1")
    assert int(st.get("count", 0)) <= 3


def test_env_disable(hook, inject, monkeypatch):
    monkeypatch.setenv("COMPLETION_BOUNDARY_AUDIT_DISABLE", "1")
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    assert hook(LONG_DONE, session_id="s1") is None
    assert inject(session_id="s1") is None


def test_hook_exception_passthrough(hook, inject, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit",
                        lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    assert hook(LONG_DONE, session_id="s1") is None
    assert inject(session_id="s1") is None
