"""Tests for completion-boundary-audit plugin.

行为契约（不测正则实现细节，只测语义边界）：
- 含完成声明 + 无边界声明 + 长回复 → 触发追加
- 已披露边界（未验证/未覆盖/风险等）→ 透传
- 短回复/无完成声明 → 透传
- 环境开关 DISABLE=1 → 透传
- 钩子异常不破坏原回复
"""

import importlib

import pytest


@pytest.fixture
def plugin():
    return importlib.import_module("plugins.completion-boundary-audit")


LONG_DONE = "功能已全部完成，三个模块都改好了，" + "细节说明" * 30
LONG_DONE_BOUNDARY = LONG_DONE + "；未验证边界：仅在本地 macOS 实测，Linux 未测。"
SHORT_DONE = "已完成"
CHAT_LONG = "今天天气不错，" + "随便聊聊" * 30


def test_needs_audit_long_completion(plugin):
    assert plugin.needs_boundary_audit(LONG_DONE) is True


def test_no_audit_when_boundary_disclosed(plugin):
    assert plugin.needs_boundary_audit(LONG_DONE_BOUNDARY) is False


def test_no_audit_short_reply(plugin):
    assert plugin.needs_boundary_audit(SHORT_DONE) is False


def test_no_audit_plain_chat(plugin):
    assert plugin.needs_boundary_audit(CHAT_LONG) is False


def test_no_audit_empty(plugin):
    assert plugin.needs_boundary_audit("") is False


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("COMPLETION_BOUNDARY_AUDIT_DISABLE", "1")

    captured = {}

    class Ctx:
        @staticmethod
        def register_hook(name, fn):
            captured[name] = fn

    plugin.register(Ctx)
    out = captured["transform_llm_output"](LONG_DONE, session_id="s1")
    assert out is None


def test_hook_appends_reminder(plugin):
    captured = {}

    class Ctx:
        @staticmethod
        def register_hook(name, fn):
            captured[name] = fn

    plugin.register(Ctx)
    out = captured["transform_llm_output"](LONG_DONE, session_id="s1")
    assert out is not None
    assert out.startswith(LONG_DONE)
    assert "反面检查" in out


def test_hook_passthrough_no_signal(plugin):
    captured = {}

    class Ctx:
        @staticmethod
        def register_hook(name, fn):
            captured[name] = fn

    plugin.register(Ctx)
    assert captured["transform_llm_output"](CHAT_LONG, session_id="s1") is None
