"""解析失败不等于通道故障：一次答非所问不得把判定静默停摆。

2026-11-10 实测到的形态：`_record_judge_failure` 对**一次**解析失败就开 300s task 熔断。
判定件全部 fail-open，熔断期间守卫照常放行、外部没有任何信号——等于异常回答一次就把
这道闸悄悄关掉五分钟，而这正是「fail-open 最阴的形态」。

契约：
  1. 解析失败（模型答了，格式读不出）连续 < 3 次 → **不挂起**，下一次立即重判；
  2. 连续 3 次 → 才认为该 task 的 prompt/模型配错，挂起；
  3. 一次可用判定立即清零；
  4. 硬失败（超时/连不通）仍一次挂起——那是端点真死了，且挂起是端点级的；
  5. logprobs 标签读空只是**能力信号**（回退文本通道），永远不算失败、不碰熔断。

Preconditions: call_llm 被 monkeypatch，零真实调用。
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
def judge():
    module = _load_judge()
    module.reset_verdict_cache()
    module.reset_judge_circuit()
    module.reset_capability_cache()
    module.reset_judge_inflight()
    return module


_UNUSABLE = _Resp("好的，我来帮你分析一下这个回复是否符合要求：它看起来……")


def test_one_unparsable_answer_does_not_open_the_circuit(judge, monkeypatch):
    """一次答非所问之后，下一次必须真的再判一次（不能静默停摆）。"""
    calls: list[int] = []

    def unusable(**kwargs):
        calls.append(1)
        return _UNUSABLE

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unusable)

    first = judge.llm_judge_multi(task="t", system="return JSON", text="a", keys=["done"])
    second = judge.llm_judge_multi(task="t", system="return JSON", text="b", keys=["done"])

    assert first == {"done": None}
    assert second == {"done": None}
    assert len(calls) == 2, "解析失败一次不得让通道静默停摆"
    assert not judge._judge_circuit_open("t")


def test_three_consecutive_unparsable_answers_open_the_circuit(judge, monkeypatch):
    """连续 3 次读不出 → 才认定这个 task 配错并挂起（此后不再每轮空转）。"""
    calls: list[int] = []

    def unusable(**kwargs):
        calls.append(1)
        return _UNUSABLE

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unusable)

    for i in range(3):
        judge.llm_judge_multi(task="t", system="return JSON", text=f"text-{i}", keys=["done"])

    assert len(calls) == 3
    assert judge._judge_circuit_open("t"), "连续 3 次解析失败应当挂起"

    judge.llm_judge_multi(task="t", system="return JSON", text="text-later", keys=["done"])
    assert len(calls) == 3, "挂起期间不得再发调用"


def test_usable_verdict_clears_the_failure_streak(judge, monkeypatch):
    """可用判定立即清零：两次坏答案 + 一次好答案 + 两次坏答案，不得凑成挂起。"""
    calls: list[int] = []
    answers = ["不可用", "不可用", '{"done": true}', "不可用", "不可用"]

    def sequence(**kwargs):
        content = answers.pop(0)
        calls.append(1)
        return _Resp(content)

    monkeypatch.setattr("agent.auxiliary_client.call_llm", sequence)

    results = [
        judge.llm_judge_multi(task="t", system="return JSON", text=f"t{i}", keys=["done"])
        for i in range(5)
    ]

    assert results[2] == {"done": True}
    assert not judge._judge_circuit_open("t"), "好答案之后计数必须清零"
    assert len(calls) == 5


def test_hard_endpoint_failure_still_opens_at_once(judge, monkeypatch):
    """超时/连不通是端点真死：仍一次挂起，且挂起是端点级的。"""
    calls: list[int] = []

    def timeout(**kwargs):
        calls.append(1)
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", timeout)

    judge.llm_judge_multi(task="slow_guard", system="s", text="a", keys=["done"])
    judge.llm_judge_multi(task="slow_guard", system="s", text="b", keys=["done"])

    assert len(calls) == 1, "硬失败一次即挂起，后续不得再付一次超时"


def test_logprob_label_miss_never_counts_as_a_failure(judge, monkeypatch):
    """标签读空 = 能力信号（回退文本通道），不是失败，绝不能碰熔断。"""
    calls: list[str] = []

    def no_label(**kwargs):
        calls.append("call")
        extra = kwargs.get("extra_body") or {}
        if extra.get("logprobs"):
            return _Resp("好的")  # 首 token 不是任何标签 → 读出落空
        return _Resp('{"decision": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", no_label)

    decision = judge.llm_decision_typed(task="t", system="s", text="x", true_key="decision")

    assert decision.verdict is True, "读空后应回退文本通道并拿到判定"
    assert decision.source == "text"
    assert judge._capability_known("t") is False, "读空应记为「不支持 logprobs」"
    assert not judge._judge_circuit_open("t"), "读空不得打开熔断"
    assert len(calls) == 2, "一次读出尝试 + 一次文本回退"


def test_auth_and_billing_errors_are_hard_failures_not_parse_failures(judge, monkeypatch):
    """key 过期/欠费不是「模型答得不好」：一次即挂起端点，且日志说的是真因。

    2026-11-10 实测：deepseek key 过期(401)、api.z.ai 欠费(429 Insufficient balance)、
    scnet/yinfeng 全 401。当成解析失败会白跑 3 次才挂起，且误导成格式问题。
    """
    calls: list[int] = []

    def expired(**kwargs):
        calls.append(1)
        raise RuntimeError(
            "Error code: 401 - Authentication Fails, Your api key: ****ired is invalid"
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", expired)

    judge.llm_judge_multi(task="t", system="s", text="a", keys=["done"])
    judge.llm_judge_multi(task="t", system="s", text="b", keys=["done"])

    assert len(calls) == 1, "认证失败一次即挂起，不得白跑 3 次"


def test_insufficient_balance_is_a_hard_failure(judge, monkeypatch):
    calls: list[int] = []

    def broke(**kwargs):
        calls.append(1)
        raise RuntimeError("Error code: 429 - {'message': 'Insufficient balance or no resource package.'}")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", broke)

    judge.llm_judge_multi(task="t", system="s", text="a", keys=["done"])
    judge.llm_judge_multi(task="t", system="s", text="b", keys=["done"])

    assert len(calls) == 1
