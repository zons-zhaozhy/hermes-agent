"""聊天/问答回合不被当成「宣告了没动手」或「提案式收尾」执法。"""
from __future__ import annotations

from plugins import builder_gate
from plugins._message_intent import record_turn_intent, turn_is_engineering
from plugins.discipline import declare_act_guard

_LONG_ASSISTANT = "我先说一下我打算怎么做，后面再说具体步骤，这里只是先聊聊想法。" * 3


def _history(text: str) -> list[dict]:
    return [
        {"role": "user", "content": "在吗"},
        {"role": "assistant", "content": text},
    ]


def test_chat_turn_is_recorded_as_not_engineering():
    record_turn_intent("intent-test-chat", "你今天过得怎么样？")
    assert turn_is_engineering("intent-test-chat") is False
    record_turn_intent("intent-test-eng", "帮我修复登录接口的报错")
    assert turn_is_engineering("intent-test-eng") is True
    assert turn_is_engineering("intent-test-never-recorded") is True  # 期望: 未记录按工程话题执法


def test_declare_act_guard_skips_a_chat_turn(monkeypatch):
    sid = "intent-test-declare"
    record_turn_intent(sid, "你今天过得怎么样？")
    # 即使上一条回复像「只宣告不动手」, 聊天回合也不该被判违规
    monkeypatch.setattr(declare_act_guard, "_judge_violation", lambda text: True)
    out = declare_act_guard._on_pre_llm_call(
        session_id=sid, user_message="你今天过得怎么样？",
        conversation_history=_history(_LONG_ASSISTANT),
    )
    assert out is None  # 期望: None=聊天回合不注入「禁止裸宣告」


def test_declare_act_guard_still_polices_an_engineering_turn(monkeypatch):
    sid = "intent-test-declare-eng"
    monkeypatch.setattr(declare_act_guard, "_judge_violation", lambda text: True)
    out = declare_act_guard._on_pre_llm_call(
        session_id=sid, user_message="帮我修复登录接口的报错",
        conversation_history=_history(_LONG_ASSISTANT),
    )
    assert out is not None  # 期望: 工程回合照常注入（对照，防止放得太宽）


_PROPOSAL_TAIL = "综合以上，落地的坎坷来自四类系统性缺口。如果你想，我可以把这份复盘沉淀成一个 skill。"


def test_builder_gate_skips_a_chat_turn(monkeypatch):
    sid = "intent-test-builder"
    record_turn_intent(sid, "谢谢，今天聊得很开心")
    monkeypatch.setattr(builder_gate, "llm_judge_bool", lambda *a, **k: True)  # 模型判「是提案」
    out = builder_gate._on_pre_verify(
        attempt=0, final_response=_PROPOSAL_TAIL, changed_paths=[], session_id=sid,
    )
    assert out == {}  # 期望: 聊天回合不判提案式收尾


def test_builder_gate_still_nudges_an_engineering_turn(monkeypatch):
    sid = "intent-test-builder-eng"
    record_turn_intent(sid, "把这个 skill 文件补上")
    monkeypatch.setattr(builder_gate, "llm_judge_bool", lambda *a, **k: True)
    out = builder_gate._on_pre_verify(
        attempt=0, final_response=_PROPOSAL_TAIL, changed_paths=[], session_id=sid,
    )
    assert out.get("action") == "continue"  # 期望: 工程回合零写入 + 提案式收尾照常续跑
