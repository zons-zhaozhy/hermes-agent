"""判定预热：回复产出的那一刻就判，下一轮 pre_llm_call 只读缓存。

背景（2026-11-09 实测）：declare_act_guard 在 pre_llm_call 判**上一条**回复，那段文本
在上一轮结束时就已定型——判定却一直排在用户发下一条消息之后的关键路径上，每次 3~4s。
现在 post_llm_call 先把同一段文本判掉（判完落进 plugins._llm_judge 的内容寻址缓存），
下一轮 pre_llm_call 走同一个 (task, system, text) 直接命中。

Contract:
  1. post_llm_call 预热用**与 pre_llm_call 逐字节一致**的文本取法（同一段文本才同缓存键）；
  2. 预热只读缓存、不注入、不改回复；
  3. 预热之后，pre_llm_call 的判定不再发起真实调用（缓存已命中）；
  4. 已动手（末条 assistant 带 tool_calls）或极短文本不预热。

Preconditions: judge 与 call_llm 均被 monkeypatch，零真实调用。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def _load(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass 解析注解要靠 sys.modules 里的自己（同一文件被合成名字加载两次）
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def guard(monkeypatch):
    # 用插件真正会 import 的那个实例（plugins._llm_judge），缓存才对得上
    import plugins._llm_judge as judge

    judge.reset_verdict_cache()
    judge.reset_judge_circuit()
    calls: list[str] = []

    def counting(**kwargs):
        calls.append(str(kwargs["messages"][1]["content"]))
        return _Resp('{"violation": true, "uncertain": false, "needs_audit": false, '
                     '"done_claim": true, "has_boundary": false, "naming_drift": [], '
                     '"jargon_coined": []}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", counting)
    module = _load("declare_act_guard_under_test", "plugins/discipline/declare_act_guard.py")
    return module, calls, judge


def _history(reply: str, *, tool_calls: bool = False) -> list[dict]:
    message: dict = {"role": "assistant", "content": reply}
    if tool_calls:
        message["tool_calls"] = [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]
    return [{"role": "user", "content": "改一下这个守卫"}, message]


_REPLY = "我建议把这些判定改成异步预热的方式，这样可以省掉每轮的串行等待。"


def test_post_turn_warm_makes_the_next_turn_read_only(guard):
    """预热过的回复，下一轮 pre_llm_call 不再发起真实调用。"""
    module, calls, judge = guard

    module._on_post_llm_call(
        session_id="s1", assistant_response=_REPLY, conversation_history=_history(_REPLY),
    )
    warmed = len(calls)

    module._on_pre_llm_call(session_id="s1", conversation_history=_history(_REPLY))

    assert warmed >= 1, "post_llm_call 必须真的判一次"
    assert len(calls) == warmed, "预热之后 pre_llm_call 不得再发起调用（应命中缓存）"
    assert judge._VERDICT_CACHE, "结论要落进判定缓存（同一 module 实例）"


def test_warm_uses_the_same_text_the_next_turn_reads(guard):
    """预热与读取的文本取法必须逐字节一致，否则缓存键不同、预热白做。"""
    module, calls, _judge = guard

    module._on_post_llm_call(session_id="s2", assistant_response="被 transform 改写过",
                             conversation_history=_history(_REPLY))
    module._on_pre_llm_call(session_id="s2", conversation_history=_history(_REPLY))

    assert calls, "首轮判定发起过调用"
    target = calls[0]
    assert _REPLY[:40] in target, "预热判的应当是 conversation_history 里那段文本"


def test_warm_skips_turns_that_already_acted(guard):
    """末条 assistant 已带 tool_calls → 不预热（与 pre_llm_call 同一前置）。"""
    module, calls, _judge = guard

    module._on_post_llm_call(
        session_id="s3", assistant_response=_REPLY, conversation_history=_history(_REPLY, tool_calls=True),
    )

    assert calls == []


def test_warm_skips_short_replies(guard):
    module, calls, _judge = guard

    module._on_post_llm_call(session_id="s4", assistant_response="好", conversation_history=_history("好"))

    assert calls == []
