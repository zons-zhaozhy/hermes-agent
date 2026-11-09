"""同一 (task, system, 文本, 键集) 只判一次——判定结果在进程内按内容复用。

各插件自带的进程内缓存只覆盖自己那条通道（reply-certainty 缓存回复侧、yinyang 缓存
用户侧、read_think 缓存意图），挡不住跨插件/跨调用点的重复：同一段回复被
declare_act_guard 与 completion_boundary_audit 先后判、同一条用户消息在重发/重试时被
同一任务再判，都各付一次真实调用与一次整轮阻塞。

契约：
  1. 相同输入第二次命中缓存，**不再发起调用**，返回值等价；
  2. 文本/system 提示/键集任一变化都换判定（判定依据变了就不是同一个判定）；
  3. 全 None 判定（失败/熔断/解析不出）**不入缓存**，端点恢复后仍会重判；
  4. ``reset_verdict_cache()`` 清空。

Preconditions: call_llm 被 monkeypatch，零真实调用。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def _load_judge() -> types.ModuleType:
    if "hermes_plugins" not in sys.modules:
        namespace = types.ModuleType("hermes_plugins")
        namespace.__path__ = []
        sys.modules["hermes_plugins"] = namespace
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins._llm_judge", _REPO / "plugins" / "_llm_judge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_plugins"
    sys.modules["hermes_plugins._llm_judge"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def judge(monkeypatch):
    module = _load_judge()
    module.reset_verdict_cache()
    module.reset_judge_circuit()
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda **kwargs: _Resp('{"done": true}'),
    )
    return module


def test_same_text_is_judged_once(judge, monkeypatch):
    """第二次判同一段文本命中缓存：不发调用，结论一致。"""
    calls: list[str] = []

    def counting(**kwargs):
        calls.append(str(kwargs["messages"][1]["content"]))
        return _Resp('{"done": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", counting)

    first = judge.llm_judge_multi(task="t", system="return JSON", text="the reply", keys=["done"])
    second = judge.llm_judge_multi(task="t", system="return JSON", text="the reply", keys=["done"])

    assert first == {"done": True}
    assert second == first
    assert calls == ["the reply"], "同一文本只允许一次真实调用"


def test_changed_text_or_system_is_a_different_verdict(judge, monkeypatch):
    calls: list[str] = []

    def counting(**kwargs):
        calls.append(f"{kwargs['messages'][0]['content']}|{kwargs['messages'][1]['content']}")
        return _Resp('{"done": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", counting)

    judge.llm_judge_multi(task="t", system="return JSON", text="a", keys=["done"])
    judge.llm_judge_multi(task="t", system="return JSON", text="b", keys=["done"])
    judge.llm_judge_multi(task="t", system="another rule", text="a", keys=["done"])
    judge.llm_judge_multi(task="t", system="return JSON", text="a", keys=["done", "other"])

    assert len(calls) == 4


def test_unknown_verdict_is_never_cached(judge, monkeypatch):
    """全 None（失败/解析不出）不能进缓存，否则一次故障会被钉满 TTL。"""
    calls: list[int] = []

    def broken(**kwargs):
        calls.append(1)
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", broken)

    assert judge.llm_judge_multi(task="t", system="s", text="x", keys=["done"]) == {"done": None}
    judge.reset_judge_circuit()  # 端点恢复：下次必须真的再判一次
    assert judge.llm_judge_multi(task="t", system="s", text="x", keys=["done"]) == {"done": None}

    assert len(calls) == 2, "没有结论的判定不得被缓存"


def test_boolean_and_json_entry_points_share_the_cache(judge, monkeypatch):
    calls: list[int] = []

    def counting(**kwargs):
        calls.append(1)
        return _Resp('{"decision": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", counting)

    assert judge.llm_judge_bool(task="b", system="s", text="x", true_key="decision") is True
    assert judge.llm_judge_bool(task="b", system="s", text="x", true_key="decision") is True
    assert len(calls) == 1

    judge.reset_verdict_cache()
    assert judge.llm_judge_bool(task="b", system="s", text="x", true_key="decision") is True
    assert len(calls) == 2, "reset_verdict_cache 之后必须重新判定"


def test_concurrent_identical_judges_share_one_call(judge, monkeypatch):
    """同一判定被两个守卫同时索要：只发一次真实调用，后来者等先到者的结果。"""
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow(**kwargs):
        calls.append(1)
        started.set()
        release.wait(5.0)
        return _Resp('{"done": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", slow)
    results: list[dict] = []

    def body():
        results.append(
            judge.llm_judge_multi(task="t", system="s", text="same", keys=["done"])
        )

    leader = threading.Thread(target=body)
    leader.start()
    assert started.wait(5.0), "先到者必须真的进入调用"
    follower = threading.Thread(target=body)
    follower.start()
    release.set()
    leader.join(10)
    follower.join(10)

    assert calls == [1], "并发同键判定只允许一次真实调用"
    assert results and all(r == {"done": True} for r in results)
