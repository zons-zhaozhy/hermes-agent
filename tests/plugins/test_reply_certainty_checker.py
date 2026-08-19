"""Tests for reply-certainty-checker (judge 版).

行为契约：
- judge 判 True（未验证模糊断言）→ 用户可见回复零改动 + 下轮注入提醒
- judge 判 False / None → 全程静默
- 短回复(<20字符) → 不调 judge 直接 False
- judge_calls 超 30 → 不再调
- 注入一次后标记消费 → 第二轮 None
- env 开关 → None
- 钩子异常 → fail-open None
"""
import importlib
import os

import pytest

plugin = importlib.import_module("plugins.reply-certainty-checker")
from plugins._shared_state import clear_session

import plugins._llm_judge as lj

SID = "test-rc"


class _Resp:
    def __init__(self, val):
        m = type("M", (), {"content": '{"uncertain": %s}' % ("true" if val else "false")})()
        c = type("C", (), {"message": m})()
        self.choices = [c]


@pytest.fixture(autouse=True)
def _reset():
    clear_session(SID)
    os.environ.pop("REPLY_CERTAINTY_CHECKER_DISABLE", None)
    yield
    clear_session(SID)
    os.environ.pop("REPLY_CERTAINTY_CHECKER_DISABLE", None)


def _hooks():
    captured = {}
    plugin.register(type("Ctx", (), {"register_hook": staticmethod(lambda n, f: captured.__setitem__(n, f))})())
    return captured


def _mock_judge(monkeypatch, val, calls=None):
    def fake(task, system, text, true_key=None, timeout=None):
        if calls is not None:
            calls.append(task)
        return val
    monkeypatch.setattr(plugin, "llm_judge_bool", fake)


def test_flagged_reply_unchanged_and_next_turn_injected(monkeypatch):
    h = _hooks()
    _mock_judge(monkeypatch, True)
    long = "x" * 30 + " 这个方案大概率能在生产环境正常工作。"
    assert h["transform_llm_output"](long, session_id=SID) is None  # 用户回复零改动
    out = h["pre_llm_call"](session_id=SID)
    assert out and "确定性检查" in out["context"]


def test_clean_reply_silent(monkeypatch):
    h = _hooks()
    _mock_judge(monkeypatch, False)
    assert h["transform_llm_output"]("y" * 40 + " 测试通过 [实测]", session_id=SID) is None
    assert h["pre_llm_call"](session_id=SID) is None


def test_fail_open_silent(monkeypatch):
    h = _hooks()
    def boom(*a, **k): raise RuntimeError("aux down")
    monkeypatch.setattr(plugin, "llm_judge_bool", boom)
    assert h["transform_llm_output"]("z" * 50, session_id=SID) is None
    assert h["pre_llm_call"](session_id=SID) is None


def test_short_reply_skips_judge(monkeypatch):
    h = _hooks()
    calls = []
    _mock_judge(monkeypatch, True, calls)
    assert h["transform_llm_output"]("好的", session_id=SID) is None
    assert calls == []  # judge 未被调用


def test_reminder_consumed_once(monkeypatch):
    h = _hooks()
    _mock_judge(monkeypatch, True)
    h["transform_llm_output"]("x" * 50, session_id=SID)
    assert h["pre_llm_call"](session_id=SID) is not None
    assert h["pre_llm_call"](session_id=SID) is None  # 已消费


def test_judge_call_cap(monkeypatch):
    h = _hooks()
    calls = []
    _mock_judge(monkeypatch, True, calls)
    for _ in range(35):
        h["transform_llm_output"]("x" * 50, session_id=SID)
    assert len(calls) == 30


def test_env_disable(monkeypatch):
    h = _hooks()
    _mock_judge(monkeypatch, True)
    os.environ["REPLY_CERTAINTY_CHECKER_DISABLE"] = "1"
    assert h["transform_llm_output"]("x" * 50, session_id=SID) is None
    assert h["pre_llm_call"](session_id=SID) is None


def test_hook_exception_fail_open(monkeypatch):
    h = _hooks()
    monkeypatch.setattr(plugin, "_state", lambda sid: (_ for _ in ()).throw(RuntimeError("state boom")))
    assert h["transform_llm_output"]("x" * 50, session_id=SID) is None
