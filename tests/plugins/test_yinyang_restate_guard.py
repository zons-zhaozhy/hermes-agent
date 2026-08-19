"""Tests for yinyang-restate-guard plugin.

行为契约：
- 质疑信号词命中 → 注入复述守则；普通消息/空 → None
- 会话内第 4 次质疑起静默（上限 3）
- env 开关 → None
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
    return importlib.import_module("plugins.yinyang-restate-guard")


def test_challenge_detected(plugin):
    assert plugin.is_challenge("你这个说法不对") is True
    assert plugin.is_challenge("你是不是瞎扯") is True


def test_plain_message_not_challenge(plugin):
    assert plugin.is_challenge("帮我看看这个文件") is False
    assert plugin.is_challenge("") is False


def test_reminder_injected_on_challenge(plugin):
    out = plugin.on_pre_llm_call(session_id="s1", user_message="结论错了")
    assert out is not None
    assert "复述" in out["context"]


def test_no_injection_on_plain(plugin):
    assert plugin.on_pre_llm_call(session_id="s1", user_message="继续") is None


def test_max_three_reminders(plugin):
    for i in range(5):
        out = plugin.on_pre_llm_call(session_id="s1", user_message="又错了")
        if i < 3:
            assert out is not None
        else:
            assert out is None


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("YINYANG_RESTATE_GUARD_DISABLE", "1")
    assert plugin.on_pre_llm_call(session_id="s1", user_message="错了") is None
