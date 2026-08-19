"""Tests for completion-boundary-audit plugin (LLM judge 版).

行为契约：
- judge 判 True（完成声明+无边界）→ 回复尾部追加红牌
- judge 判 False / None → 透传 None
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
def hook():
    captured = {}

    class Ctx:
        @staticmethod
        def register_hook(name, fn):
            captured[name] = fn

    plugin.register(Ctx())
    return captured["transform_llm_output"]


LONG_DONE = "x" * 100 + " 全部测试已通过，任务完成。"
LONG_DONE_2 = "y" * 100 + " 已修复并验证完成。"
LONG_WITH_BOUNDARY = "z" * 100 + " 已完成，但未验证生产环境。"


def test_judge_true_appends(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    out = hook(LONG_DONE, session_id="s1")
    assert out is not None and "反面检查" in out and out.startswith("x")


def test_judge_false_passthrough(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: False)
    assert hook(LONG_DONE, session_id="s1") is None


def test_judge_none_fail_open(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: None)
    assert hook(LONG_DONE, session_id="s1") is None


def test_short_reply_skipped(hook, monkeypatch):
    called = []
    monkeypatch.setattr(plugin, "needs_boundary_audit",
                        lambda t: called.append(t) or True)
    assert hook("好的完成了", session_id="s1") is None
    assert called == []


def test_dedup(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    assert hook(LONG_DONE, session_id="s1") is not None
    assert hook(LONG_DONE, session_id="s1") is None


def test_max_reminders(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    outs = [hook(f"{c*100} 已全部完成。", session_id="s1") for c in "abcde"]
    assert sum(o is not None for o in outs) == 3


def test_env_disable(hook, monkeypatch):
    monkeypatch.setenv("COMPLETION_BOUNDARY_AUDIT_DISABLE", "1")
    monkeypatch.setattr(plugin, "needs_boundary_audit", lambda t: True)
    assert hook(LONG_DONE, session_id="s1") is None


def test_hook_exception_passthrough(hook, monkeypatch):
    monkeypatch.setattr(plugin, "needs_boundary_audit",
                        lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    assert hook(LONG_DONE, session_id="s1") is None
