"""Tests for yinyang-restate-guard plugin (LLM judge 版).

行为契约：
- judge 判 True（质疑）→ 注入复述守则
- judge 判 False / None（fail-open）→ None
- 同一消息 hash 去重 → 第二次不调 judge
- judge_calls 超 30 → 不再调
- 提醒超 3 次 → None
- env 开关 → None
- 钩子内部异常 → None（fail-open）
"""
import importlib

import pytest

plugin = importlib.import_module("plugins.yinyang-restate-guard")
from plugins._shared_state import clear_session


@pytest.fixture(autouse=True)
def _reset():
    clear_session("s1")
    yield
    clear_session("s1")


def _msg(text):
    return {"session_id": "s1", "user_message": text}


def _patch_judge(monkeypatch, calls, verdict):
    """Patch the plugin's merged judge (judge_user_side) — the hook calls it
    directly since the user-side-guards consolidation; patching the legacy
    is_challenge symbol no longer intercepts anything."""
    def fake(message, timeout=8.0):
        calls.append(message)
        return {"challenge": verdict}
    monkeypatch.setattr(plugin, "judge_user_side", fake)
    monkeypatch.setattr(plugin, "_USER_SIDE_CACHE", {})


def test_judge_true_triggers(monkeypatch):
    _patch_judge(monkeypatch, [], True)
    out = plugin.on_pre_llm_call(**_msg("你说得不对"))
    assert out is not None and "复述" in out["context"]


def test_judge_false_silent(monkeypatch):
    _patch_judge(monkeypatch, [], False)
    assert plugin.on_pre_llm_call(**_msg("继续")) is None


def test_judge_none_fail_open(monkeypatch):
    _patch_judge(monkeypatch, [], None)
    assert plugin.on_pre_llm_call(**_msg("嗯")) is None


def test_dedup_same_message(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, calls, True)
    assert plugin.on_pre_llm_call(**_msg("同一句质疑")) is not None
    assert plugin.on_pre_llm_call(**_msg("同一句质疑")) is None
    assert len(calls) == 1


def test_max_judge_calls(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, calls, False)
    for i in range(35):
        plugin.on_pre_llm_call(**_msg(f"msg {i}"))
    assert len(calls) == 30


def test_max_reminders(monkeypatch):
    _patch_judge(monkeypatch, [], True)
    for i in range(5):
        out = plugin.on_pre_llm_call(**_msg(f"质疑 {i}"))
        if i < 3:
            assert out is not None
        else:
            assert out is None


def test_env_disable(monkeypatch):
    monkeypatch.setenv("YINYANG_RESTATE_GUARD_DISABLE", "1")
    _patch_judge(monkeypatch, [], True)
    assert plugin.on_pre_llm_call(**_msg("不对")) is None


def test_hook_exception_fail_open(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(plugin, "judge_user_side", boom)
    assert plugin.on_pre_llm_call(**_msg("质疑")) is None
