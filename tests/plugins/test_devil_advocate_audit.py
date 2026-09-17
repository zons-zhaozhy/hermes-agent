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
- status 词表=运行时真实值 "ok"（model_tools._tool_result_observer_fields
  派生；伪造 "success" 属历史假绿灯，已修）
- 顶层 goal= 单任务委派同样可提取判定（不再只认 tasks[].goal）
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


def _judge_false():
    return lambda *a, **k: type("R", (), {
        "choices": [type("C", (), {"message": type(
            "M", (), {"content": '{"decision": false}'})()})()]})()


def test_judge_true_triggers(plugin, mock_judge):
    assert mock_judge == []  # 期望: [] —— fixture 后未调 judge（前置计数为 0）
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统迁移到新的存储架构"))
    assert out is not None and "反方审查" in out["context"]
    assert len(mock_judge) == 2  # 期望: 2 —— 决策判定 True 后必探测豁免(不设cap)=2次调用


def test_judge_false_no_trigger(plugin, monkeypatch):
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _judge_false())
    assert plugin.on_pre_llm_call(**_msg("随便聊聊今天天气")) is None  # 期望: None —— 判 False 不触发（契约）


def test_judge_exception_fail_open(plugin, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no aux client")
    monkeypatch.setattr("agent.auxiliary_client.call_llm", boom)
    assert plugin.on_pre_llm_call(**_msg("决定采用新方案")) is None  # 期望: None —— fail-open（契约）


def test_message_dedup(plugin, mock_judge):
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    assert len(mock_judge) == 2  # 期望: 2 —— 首次判定(决策+豁免探测)；重发 hash 去重零调用


def test_judge_call_cap(plugin, mock_judge):
    for i in range(35):
        plugin.on_pre_llm_call(**_msg(f"消息 {i}"))
    assert len(mock_judge) <= 65  # 期望: ≤65 —— 决策判定cap=30；cap前armed置位后每消息1次豁免探测(35)


def test_delegate_tool_call_silences(plugin, monkeypatch):
    # status 取运行时真实词表 "ok"（_tool_result_observer_fields 派生）。
    # goal 语义判定走 _delegate_is_review → llm_judge_bool 判 review=true。
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: type("R", (), {
            "choices": [type("C", (), {"message": type(
                "M", (), {"content": '{"review": true}'})()})()]})())
    plugin.on_post_tool_call(
        session_id=SID, tool_name="delegate_task",
        args={"goal": "反方审查：挑漏洞批判审查该方案"},
        result={}, status="ok",
    )
    assert plugin.on_pre_llm_call(**_msg("任何消息")) is None  # 期望: None —— reviewed=True 已置位（契约）


def test_delegate_top_level_goal_extracted(plugin, monkeypatch):
    # 顶层 goal= 形态（单任务委派）也须被提取并判定（修 :140 只认 tasks[].goal）
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: type("R", (), {
            "choices": [type("C", (), {"message": type(
                "M", (), {"content": '{"review": true}'})()})()]})())
    plugin.on_post_tool_call(
        session_id=SID, tool_name="delegate_task",
        args={"goal": "反方审查（挑漏洞/批判）该修复方案"},
        result={}, status="ok",
    )
    from plugins._shared_state import get_session_state
    st = get_session_state(SID, "devil_advocate_audit")
    assert st.get("reviewed") is True  # 期望: True —— 顶层 goal 提取+判定通过后置位


def test_waive_not_capped(plugin, monkeypatch):
    # 豁免判定不受 judge_calls cap 限制（修：被 cap 挡=armed 死锁无解）
    from plugins._shared_state import get_session_state
    st0 = get_session_state(SID, "devil_advocate_audit")
    st0["judge_calls"] = 30
    st0["armed"] = True  # 先 armed（cap 前已置位），再打满 cap 验证豁免出口仍开
    replies = iter([
        type("R", (), {"choices": [type("C", (), {"message": type(
            "M", (), {"content": '{"waive": true}'})()})()]})(),      # cap 后豁免判定 True
    ])
    monkeypatch.setattr("agent.auxiliary_client.call_llm",
                        lambda *a, **k: next(replies))
    plugin.on_pre_llm_call(**_msg("豁免反方审查，这个方案我拍板了"))
    st = get_session_state(SID, "devil_advocate_audit")
    assert st.get("waived") is True  # 期望: True —— cap=30+armed 态豁免词仍可解除武装


def test_armed_persists_until_resolved(plugin, mock_judge):
    # 现行契约：armed 状态机替代旧 _MAX_REMINDERS 计数——armed 后持续红牌
    # 直至 reviewed(反方审查委派)/waived(豁免)解除，无次数上限。
    assert plugin.on_pre_llm_call(**_msg("决策一")) is not None  # 期望: 非None —— 置 armed+红牌
    assert plugin.on_pre_llm_call(**_msg("决策二")) is not None  # 期望: 非None —— armed 未解除持续红牌
    assert plugin.on_pre_llm_call(**_msg("决策三")) is not None  # 期望: 非None —— 无次数上限(状态机契约)


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("DEVIL_ADVOCATE_AUDIT_DISABLE", "1")
    assert plugin.on_pre_llm_call(**_msg("重大决策")) is None  # 期望: None —— env 开关整体禁用（契约）


def test_fail_open_on_bad_state(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_count",
                        lambda sid, key="count": (_ for _ in ()).throw(
                            RuntimeError("x")))
    assert plugin.on_pre_llm_call(**_msg("方案定稿")) is None  # 期望: None —— 钩子异常 fail-open（契约）
