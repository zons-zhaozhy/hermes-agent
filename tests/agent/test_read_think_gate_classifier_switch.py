"""Regression test: use_llm_classifier=False must actually bypass the LLM.

2026-08-18 audit finding: with config.yaml setting
``read_think_gate.use_llm_classifier: false``, the agent.log still showed
110 ``LLM classified complexity=...`` lines. Root cause: the else-branch of
``ReadThinkGate.reset_for_turn`` called ``detect_complexity()``, whose body
unconditionally tries ``_classify_via_llm()`` FIRST — the config switch was
decorative. Each such call is a synchronous auxiliary-LLM round-trip
(timeout=10s) on the turn-start path.

Expectations are derived from the config contract, not the implementation:
  - use_llm_classifier=False + user message present → keyword path
  - use_llm_classifier=True  + user message present → LLM path
  - empty user message → "normal" without touching either path
  - complexity_adaptive=False → no classification at all
"""

from unittest.mock import patch

from agent.read_think_gate import ReadThinkGate, ReadThinkGateConfig


def _gate(**config_overrides) -> ReadThinkGate:
    base = {
        "complexity_adaptive": True,
        "use_llm_classifier": False,
    }
    base.update(config_overrides)
    return ReadThinkGate(ReadThinkGateConfig.from_mapping(base))


def test_classifier_off_bypasses_llm():
    """Config says no LLM → _classify_via_llm must never be called."""
    gate = _gate(use_llm_classifier=False)
    with patch("agent.read_think_gate._classify_via_llm") as llm:
        with patch(
            "agent.read_think_gate._fallback_detect", return_value="complex"
        ) as fallback:
            gate.reset_for_turn(user_message="重构这个模块")
    llm.assert_not_called()
    fallback.assert_called_once()
    assert gate._active_complexity == "complex"


def test_classifier_on_uses_llm():
    """Config says LLM → LLM path is used (existing behavior preserved)."""
    gate = _gate(use_llm_classifier=True)
    with patch(
        "agent.read_think_gate._classify_via_llm", return_value="normal"
    ) as llm:
        gate.reset_for_turn(user_message="修个typo")
    llm.assert_called_once()
    assert gate._active_complexity == "normal"


def test_empty_message_defaults_normal_no_llm():
    """No user message → 'normal' without any classification machinery."""
    gate = _gate(use_llm_classifier=False)
    with patch("agent.read_think_gate._classify_via_llm") as llm:
        with patch("agent.read_think_gate._fallback_detect") as fallback:
            gate.reset_for_turn(user_message="")
    llm.assert_not_called()
    fallback.assert_not_called()
    assert gate._active_complexity == "normal"


def test_adaptive_off_skips_classification():
    """complexity_adaptive=False → no classification regardless of switch."""
    gate = _gate(complexity_adaptive=False, use_llm_classifier=True)
    with patch("agent.read_think_gate._classify_via_llm") as llm:
        gate.reset_for_turn(user_message="重构整个系统")
    llm.assert_not_called()
    assert gate._active_complexity == "normal"


def test_classifier_off_real_keyword_semantics():
    """End-to-end with the real keyword matcher (no mocks on the path):

    '重构系统' contains the 重构 keyword → complex; a chatty message with
    no keywords → normal. Proves the fallback path stays wired after the fix.
    """
    gate = _gate(use_llm_classifier=False)
    gate.reset_for_turn(user_message="重构系统")
    assert gate._active_complexity == "complex"

    gate2 = _gate(use_llm_classifier=False)
    gate2.reset_for_turn(user_message="你好呀")
    assert gate2._active_complexity == "normal"
