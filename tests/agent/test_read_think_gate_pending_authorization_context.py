"""ReadThinkGate: a reply to the gate's own authorization request keeps context.

The gate may end a turn blocked and awaiting user authorization. The next user
message is then an answer to that request ("B", a rule override like "这个要求
是不恰当的"), not a fresh standalone comment. Classifying it without the pending
request as referent misreads approval as commentary and re-blocks — the exact
misfire seen twice in the 2026-11-09 session. The pending request tail must be
injected into the classification history, explicit refusals must stay read-only,
and the injection must be one-shot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.read_think_gate_host import (
    INTENT_MODIFY,
    INTENT_READ_ONLY,
    ReadThinkGate,
    ReadThinkGateConfig,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))


_PENDING = "修法已备好待放行：……请回一个「修」字授权修改 gate.py，我立即落码跑测试提交。"


def _gate(tmp_path: Path, tag: str, *, llm_classifier: bool) -> ReadThinkGate:
    return ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": llm_classifier,
                "min_read_only_calls": 2,
                "min_reasoning_chars": 200,
                "max_reasoning_rounds": 6,
            }
        ),
        session_id=f"pending-auth-{tag}-{tmp_path.name}",
    )


def _awaiting(gate: ReadThinkGate) -> None:
    gate._awaiting_user_authorization = True
    gate._pending_auth_request = _PENDING


def test_reply_to_pending_request_gets_request_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plugins.read_think_gate_host as host

    gate = _gate(tmp_path, "inject", llm_classifier=True)
    _awaiting(gate)
    captured: list[list[dict] | None] = []

    def fake_classifier(message, history=None):
        captured.append(history)
        return INTENT_MODIFY

    monkeypatch.setattr(host, "_classify_intent_via_llm", fake_classifier)
    gate.reset_for_turn("fork 禁改核心文件，这个要求是不恰当的！")

    # 期望: 分类器收到含拦截请求原文的 assistant 语境, 且判为 modify
    history = captured[0] or []
    assert any(
        isinstance(m, dict)
        and m.get("role") == "assistant"
        and "「修」" in str(m.get("content", ""))
        for m in history
    )
    assert gate.user_intent == INTENT_MODIFY


def test_explicit_refusal_stays_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plugins.read_think_gate_host as host

    gate = _gate(tmp_path, "refuse", llm_classifier=True)
    _awaiting(gate)
    called = []
    monkeypatch.setattr(
        host, "_classify_intent_via_llm", lambda m, h=None: called.append(1) or INTENT_MODIFY
    )
    gate.reset_for_turn("不要修改，只分析")

    # 期望: 显式拒绝词硬通道优先, 拒绝不可能被注入语境翻转
    assert gate.user_intent == INTENT_READ_ONLY


def test_injection_is_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plugins.read_think_gate_host as host

    gate = _gate(tmp_path, "oneshot", llm_classifier=True)
    _awaiting(gate)
    captured: list[list[dict] | None] = []
    monkeypatch.setattr(
        host,
        "_classify_intent_via_llm",
        lambda m, h=None: captured.append(h) or INTENT_MODIFY,
    )
    gate.reset_for_turn("fork 禁改核心文件，这个要求是不恰当的！")
    gate.reset_for_turn("今天天气不错")

    # 期望: 第二次分类不再携带注入语境(一次性消费, 防陈旧复用)
    second = captured[1] or []
    assert not any(
        isinstance(m, dict) and "GATE" in str(m.get("content", "")) for m in second
    )


def test_fallback_confirmation_resolves_against_pending_request(
    tmp_path: Path,
) -> None:
    gate = _gate(tmp_path, "fallback", llm_classifier=False)
    _awaiting(gate)
    gate.reset_for_turn("继续", conversation_history=[])

    # 期望: 无 LLM 时, 注入的请求原文充当 proposal 指代, 简短批准判为 modify
    assert gate.user_intent == INTENT_MODIFY
