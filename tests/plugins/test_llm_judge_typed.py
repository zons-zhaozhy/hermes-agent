"""tests/plugins/test_llm_judge_typed.py — 判定原语（logprobs 单 token 读出）测试。

期望值独立推导：概率 = 只在声明标签上归一化（约束 softmax），
非标签质量占比 = 非标签质量 / top-N 覆盖质量。断言处均给出推导注释。

不触发真实 LLM 调用：通道函数在测试内被 monkeypatch 替换。
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from plugins import _llm_judge as judge


def _choice(top: list[tuple[str, float]]):
    """构造与 openai SDK 同形的 choice 对象。"""
    items = [SimpleNamespace(token=token, logprob=logprob) for token, logprob in top]
    content = [SimpleNamespace(token=items[0].token, logprob=items[0].logprob, top_logprobs=items)]
    return SimpleNamespace(logprobs=SimpleNamespace(content=content))


@pytest.fixture(autouse=True)
def _clean_capability_cache():
    """每个用例前后清空能力缓存（缓存是进程级全局，防止用例互相污染）。"""
    judge.reset_capability_cache()
    yield
    judge.reset_capability_cache()


def test_readout_normalizes_over_declared_labels_only():
    """只有 true/false 参与归一化：非标签 token 不改变标签间相对比例。"""
    choice = _choice([("true", 0.0), ("false", -1.0), ("maybe", -2.0)])

    readout = judge._readout_from_logprobs(choice, judge._label_variants("decision"))

    assert readout is not None  # 期望：声明标签命中，读出成功
    label, probability, nonlabel_share = readout
    assert label == "true"  # 期望：true 质量 1.0 > false 质量 0.3679
    assert probability == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))  # 期望：1/(1+e^-1)=0.7311
    covered = math.exp(0.0) + math.exp(-1.0) + math.exp(-2.0)
    assert nonlabel_share == pytest.approx(math.exp(-2.0) / covered)  # 期望：e^-2/覆盖质量


def test_punctuation_wrapped_token_matches_label():
    """带前导标点的 token（'(false'、'.false'）归一后归属 false。"""
    choice = _choice([("true", -2.0), ("(false", -0.1), (".false", -3.0)])

    readout = judge._readout_from_logprobs(choice, judge._label_variants("decision"))

    assert readout is not None  # 期望：'(false' 归一为 false，命中标签
    label, probability, _ = readout
    false_mass = math.exp(-0.1) + math.exp(-3.0)
    true_mass = math.exp(-2.0)
    assert label == "false"  # 期望：false 质量合计 > true 质量
    assert probability == pytest.approx(false_mass / (false_mass + true_mass))  # 期望：约束 softmax


def test_readout_returns_none_without_logprobs():
    """后端不返回 logprobs 时读出为 None（由调用方回退文本通道）。"""
    choice = SimpleNamespace(logprobs=None)

    assert judge._readout_from_logprobs(choice, judge._label_variants("decision")) is None  # 期望：无 logprobs → None


def test_readout_returns_none_when_no_label_candidate():
    """候选里没有任何声明标签时读出为 None（不硬猜）。"""
    choice = _choice([("maybe", -0.1), ("unknown", -1.0)])

    assert judge._readout_from_logprobs(choice, judge._label_variants("decision")) is None  # 期望：无标签命中 → None


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.95, "high"), (0.90, "high"), (0.75, "medium"), (0.60, "medium"), (0.42, "low"), (None, "unknown")],
)
def test_confidence_tier_boundaries(probability, expected):
    """分档边界：0.90/0.60 为闭区间下界，None 归 unknown。"""
    assert judge.confidence_tier(probability) == expected  # 期望：按 HIGH/MEDIUM 阈值分档


def test_judge_action_maps_tiers_to_actions():
    """分档策略：高置信 block、中置信 warn、低置信 pass、无概率沿用布尔、失败 fail_open。"""
    high = judge.TypedDecision(verdict=True, probability=0.95, confidence="high")
    medium = judge.TypedDecision(verdict=True, probability=0.7, confidence="medium")
    low = judge.TypedDecision(verdict=True, probability=0.3, confidence="low")
    no_prob = judge.TypedDecision(verdict=True, source="text")
    negative = judge.TypedDecision(verdict=False, probability=0.99, confidence="high")
    failed = judge.TypedDecision(verdict=None, source="none")

    assert judge.judge_action(high) == "block"  # 期望：高置信且判定为真 → 拦截
    assert judge.judge_action(medium) == "warn"  # 期望：中置信 → 告警
    assert judge.judge_action(low) == "pass"  # 期望：低置信不足以动作 → 放行
    assert judge.judge_action(no_prob) == "block"  # 期望：文本通道无概率 → 沿用既有布尔语义
    assert judge.judge_action(negative) == "pass"  # 期望：判定为假 → 不动作
    assert judge.judge_action(failed) == "fail_open"  # 期望：判定失败 → fail-open


def test_capability_cache_skips_logprobs_after_unsupported(monkeypatch):
    """实证不支持后，同一 task 的后续判定不再尝试 logprobs（不浪费调用）。"""
    calls = {"logprob": 0, "text": 0}

    def _fake_logprob(task, system, text, true_key, timeout):
        calls["logprob"] += 1
        return None

    def _fake_text(task, system, text, true_key, timeout, max_tokens):
        calls["text"] += 1
        return judge.TypedDecision(verdict=True, label="true", source="text")

    monkeypatch.setattr(judge, "_logprob_verdict", _fake_logprob)
    monkeypatch.setattr(judge, "_text_verdict", _fake_text)

    first = judge.llm_decision_typed(task="demo_task", system="s", text="t")
    second = judge.llm_decision_typed(task="demo_task", system="s", text="t")

    assert first.verdict is True and second.verdict is True  # 期望：两次都落到文本通道的判定
    assert calls["logprob"] == 1  # 期望：logprobs 只探测一次，第二次被能力缓存短路
    assert calls["text"] == 2  # 期望：文本通道两次都执行


def test_logprob_exception_falls_back_and_records_unsupported(monkeypatch):
    """logprobs 通道抛错（如后端拒绝该参数）→ 回退文本且不再重试。"""
    calls = {"logprob": 0}

    def _raise(task, system, text, true_key, timeout):
        calls["logprob"] += 1
        raise RuntimeError("logprobs not supported")

    monkeypatch.setattr(judge, "_logprob_verdict", _raise)
    monkeypatch.setattr(
        judge, "_text_verdict",
        lambda task, system, text, true_key, timeout, max_tokens: judge.TypedDecision(
            verdict=False, label="false", source="text"
        ),
    )

    first = judge.llm_decision_typed(task="t1", system="s", text="t")
    second = judge.llm_decision_typed(task="t1", system="s", text="t")

    assert first.verdict is False and second.verdict is False  # 期望：两次都拿到文本通道结果
    assert calls["logprob"] == 1  # 期望：异常后能力标记为不支持，不再重试


def test_low_confidence_policy_is_opt_in(monkeypatch):
    """低置信降级默认关闭；开启后低置信判定转为 fail-open。"""
    monkeypatch.setattr(
        judge, "_logprob_verdict",
        lambda task, system, text, true_key, timeout: judge.TypedDecision(
            verdict=True, label="true", probability=0.55, confidence="low", source="logprobs"
        ),
    )

    default = judge.llm_decision_typed(task="t2", system="s", text="t")
    downgraded = judge.llm_decision_typed(task="t3", system="s", text="t", low_confidence_pass=True)

    assert default.verdict is True  # 期望：默认不改变既有布尔语义
    assert downgraded.verdict is None  # 期望：显式开启后低置信降级为 fail-open
    assert downgraded.label == "true"  # 期望：降级保留原始判定标签供诊断


def test_llm_judge_bool_uses_text_channel_only(monkeypatch):
    """既有入口保持文本单通道：不触发 logprobs 探测（调用次数与既有行为一致）。

    背景：把 llm_judge_bool 接到 logprobs 探测后，devil-advocate-audit 的
    消费者回归测试抓到判定调用次数从 2 变 3（探测期多一次 API 调用）。
    """
    logprob_calls = {"n": 0}

    def _fake_logprob(task: str, system: str, text: str, true_key: str, timeout: float) -> None:
        logprob_calls["n"] += 1
        return None

    monkeypatch.setattr(judge, "_logprob_verdict", _fake_logprob)
    monkeypatch.setattr(
        judge, "_text_verdict",
        lambda task, system, text, true_key, timeout, max_tokens: judge.TypedDecision(
            verdict=True, label="true", source="text"
        ),
    )
    assert judge.llm_judge_bool(task="t4", system="s", text="x") is True  # 期望: 透出文本通道 verdict
    assert logprob_calls["n"] == 0  # 期望: 既有入口零探测调用（文本单通道不额外请求）

    monkeypatch.setattr(
        judge, "_text_verdict",
        lambda task, system, text, true_key, timeout, max_tokens: judge.TypedDecision(
            verdict=None, source="none"
        ),
    )
    assert judge.llm_judge_bool(task="t4", system="s", text="x") is None  # 期望: 判定失败回 None(fail-open)


def test_multi_key_judgment_unchanged_by_readout_channel(monkeypatch):
    """多键判定仍走文本解析（首位置单 token 读不出多键）。"""
    from agent.auxiliary_client import call_llm as real_call_llm  # noqa: F401  # 仅确认可导入路径存在

    def _fake_call_llm(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": true, "b": false}'))]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
    result = judge.llm_judge_multi(task="t5", system="s", text="x", keys=["a", "b"])

    assert result == {"a": True, "b": False}  # 期望：逐键解析 JSON 文本，与既有行为一致
