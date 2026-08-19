"""Tests for devil-advocate-audit plugin (LLM judge 版).

行为契约：
- judge 判 True 且未审查 → 注入红牌
- judge 判 False → None
- judge 抛异常（fail-open）→ None
- 同一消息 hash 去重 → 第二次不调 judge
- judge_calls 超 30 → 不再调 judge
- delegate_tool 调用后（reviewed=True）→ 静默
- 红牌每会话最多 2 次
- env 开关 → None
- 钩子异常 → None（fail-open）
"""

import importlib

import pytest

from plugins._shared_state import clear_session

SID = "s1"


@pytest.fixture(autouse=True)
def _clean_state():
    clear_session(SID)
    yield
    clear_session(SID)


@pytest.fixture
def plugin():
    return importlib.import_module("plugins.devil-advocate-audit")


def _msg(text):
    return {"session_id": SID, "user_message": text}


@pytest.fixture
def mock_judge(monkeypatch):
    calls = []

    def fake_call_llm(*args, **kwargs):
        calls.append(kwargs.get("messages"))
        m = type("M", (), {"content": '{"decision": true}'})()
        c = type("C", (), {"message": m})()
        return type("R", (), {"choices": [c]})()

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", fake_call_llm)
    return calls


def test_judge_true_triggers(plugin, mock_judge):
    assert mock_judge == []  # judge 尚未被调
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统迁移到新的存储架构"))
    assert out is not None and "反方审查" in out["context"]
    assert len(mock_judge) == 1


def test_judge_false_no_trigger(plugin, monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: type("R", (), {
            "choices": [type("C", (), {"message": type(
                "M", (), {"content": '{"decision": false}'})()})()]})())
    assert plugin.on_pre_llm_call(**_msg("随便聊聊今天天气")) is None


def test_judge_exception_fail_open(plugin, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no aux client")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)
    assert plugin.on_pre_llm_call(**_msg("决定采用新方案")) is None


def test_message_dedup(plugin, mock_judge):
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    assert len(mock_judge) == 1  # 第二次命中 hash 去重


def test_judge_call_cap(plugin, mock_judge):
    for i in range(35):
        plugin.on_pre_llm_call(**_msg(f"消息 {i}"))
    # 30 次上限后不再调
    assert len(mock_judge) <= 30


def test_delegate_tool_call_silences(plugin):
    plugin.on_post_tool_call(
        session_id=SID, tool_name="delegate_task",
        args={}, result={}, status="success",
    )
    assert plugin.on_pre_llm_call(**_msg("任何消息")) is None


def test_max_two_reminders(plugin, mock_judge):
    assert plugin.on_pre_llm_call(**_msg("决策一")) is not None
    assert plugin.on_pre_llm_call(**_msg("决策二")) is not None
    assert plugin.on_pre_llm_call(**_msg("决策三")) is None


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("DEVIL_ADVOCATE_AUDIT_DISABLE", "1")
    assert plugin.on_pre_llm_call(**_msg("重大决策")) is None


def test_fail_open_on_bad_state(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_count",
                        lambda sid, key="count": (_ for _ in ()).throw(
                            RuntimeError("x")))
    assert plugin.on_pre_llm_call(**_msg("方案定稿")) is None
