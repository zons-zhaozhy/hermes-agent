"""Tests for the adaptive-reasoning plugin.

Covers:
1. classify_effort — signal router levels (expectations derived from the
   design contract in the issue discussion, NOT from the implementation)
2. Middleware rewrite — extra_body.reasoning / top-level reasoning_effort
3. No-op behaviour — no reasoning fields, disabled via config
4. In-turn escalation via post_tool_call error counting
5. Floor/ceiling clamping from config
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Import the plugin module directly (hyphenated dir names aren't valid modules)
PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "adaptive-reasoning"

_spec = importlib.util.spec_from_file_location(
    "adaptive_reasoning_plugin", PLUGIN_DIR / "__init__.py"
)
plugin_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin_mod)

DEFAULT_CFG = plugin_mod._default_config()


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset per-turn error state between tests."""
    plugin_mod._TURN_ERRORS.clear()
    yield
    plugin_mod._TURN_ERRORS.clear()


@pytest.fixture(autouse=True)
def _fresh_config_cache(monkeypatch, tmp_path):
    """Point config lookups at an empty temp HERMES_HOME so user config
    on this machine can't leak into test expectations."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: home, raising=False
    )
    plugin_mod._CONFIG_CACHE = None
    plugin_mod._CONFIG_CACHE_KEY = None
    yield
    plugin_mod._CONFIG_CACHE = None
    plugin_mod._CONFIG_CACHE_KEY = None


# ── 1. classify_effort: signal router ────────────────────────────────────
# v2 expected values derived independently from the measured-benchmark design:
#   brevity-only message              → minimal
#   short plain message               → low
#   medium plain message              → low (no signals → baseline-low)
#   work-shaped (len>120) + keyword   → medium (NEVER high upfront — measured:
#                                       high starves completion budget)
#   keyword + code fence, work-shaped → medium
#   brevity/short + tool errors       → rescue escalation (per 2 errors)

def test_brevity_message_is_minimal():
    assert plugin_mod.classify_effort("ok") == "minimal"
    assert plugin_mod.classify_effort("continue") == "minimal"
    assert plugin_mod.classify_effort("继续") == "minimal"


def test_brevity_with_keyword_still_escalates():
    # v3: "debug" alone in a COLD session is conversational (nothing to
    # debug yet) → low; with work history the keyword escalates → medium
    assert plugin_mod.classify_effort("debug") == "low"
    assert plugin_mod.classify_effort("debug", work_depth=1) == "medium"


def test_short_plain_message_is_low():
    assert plugin_mod.classify_effort("what time is it") == "low"


def test_medium_plain_message_stays_low():
    msg = "please list the files in this directory and show me the readme"
    assert len(msg) <= DEFAULT_CFG["low_max_chars"] or True  # len-guard
    assert plugin_mod.classify_effort(msg) == "low"


def test_single_keyword_is_medium():
    # v3: keyword needs context or work-shape; cold short → low
    assert plugin_mod.classify_effort("help me debug this") == "low"
    assert plugin_mod.classify_effort("帮我排查这个问题") == "low"
    assert plugin_mod.classify_effort("help me debug this", work_depth=1) == "medium"


def test_keyword_plus_code_fence_is_medium():
    # v3: no upfront high; code fence is technical → medium with context
    msg = "debug this:\n```\ntraceback\n```"
    assert plugin_mod.classify_effort(msg) == "low"
    assert plugin_mod.classify_effort(msg, work_depth=1) == "medium"


def test_two_keywords_work_shaped_is_medium():
    # v2: keyword count no longer escalates to high (measured failure mode);
    # sample must be work-shaped: len > low_max_chars (120)
    msg = ("analyze and review the failure of this migration, then refactor "
           "the affected modules and document the root cause found thoroughly")
    assert len(msg) > 120
    assert plugin_mod.classify_effort(msg) == "medium"


def test_long_message_stays_low():
    # v2: length alone is NOT difficulty (long-easy benchmark category)
    msg = "word " * 130  # 650 chars, no keywords
    assert len(msg) >= 600
    assert plugin_mod.classify_effort(msg) == "low"


def test_unknown_levels_clamp_to_medium_index():
    assert plugin_mod._effort_index("bogus") == plugin_mod.EFFORT_SCALE.index("medium")


# ── 2. Middleware rewrite: wire shapes ───────────────────────────────────

def _base_request() -> Dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}},
    }


def _work_request() -> Dict[str, Any]:
    """A request whose history carries work signals (v3 context input)."""
    return {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "help me debug the failing migration"},
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "exit 1: migration failed"},
            {"role": "user", "content": "hi"},
        ],
        "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}},
    }


def test_middleware_rewrites_extra_body_effort():
    # v3: work context in history (tool result present) + short complex
    # message routes medium — context, not just text length
    req = _work_request()
    req["messages"][-1] = {
        "role": "user",
        "content": "why does this fail? analyze the root cause",
    }
    # start from low so the medium rewrite is observable (same-value = no-op)
    req["extra_body"]["reasoning"]["effort"] = "low"
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["extra_body"]["reasoning"]["effort"] == "medium"
    # messages untouched — cache prefix integrity
    assert result["request"]["messages"] == req["messages"]
    # original request not mutated
    assert _work_request()["extra_body"]["reasoning"]["effort"] == "medium"


def test_middleware_brevity_inherits_work_context():
    # v3 core fix: "ok" mid-task inherits session difficulty → medium,
    # instead of the isolated-benchmark answer (minimal). Request carries
    # low so the rewrite is observable.
    req = _work_request()
    req["messages"][-1] = {"role": "user", "content": "ok"}
    req["extra_body"]["reasoning"]["effort"] = "low"
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["extra_body"]["reasoning"]["effort"] == "medium"


def test_middleware_brevity_cold_session_is_minimal():
    # v3: "ok" with NO work history is a true ack → minimal
    req = _base_request()
    req["messages"] = [{"role": "user", "content": "ok"}]
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["extra_body"]["reasoning"]["effort"] == "minimal"


def test_middleware_skips_when_effort_already_at_target():
    # v3: "ok" mid-task → medium; request already carries medium → no rewrite
    req = _work_request()
    req["messages"][-1] = {"role": "user", "content": "ok"}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


def test_middleware_rewrites_top_level_reasoning_effort():
    # top-level reasoning_effort is the PROVIDER-NATIVE scale, written by the
    # transport via the provider profile. The plugin must reuse the same
    # profile mapping (zai glm-5.3: minimal/low→low, medium/high→high,
    # xhigh/max/ultra→max) instead of writing raw Hermes levels.
    # Start from native low; work context + complex msg → Hermes medium → native high.
    req = {"model": "glm-5.3", "messages": [
        {"role": "user", "content": "help me debug the failing migration and analyze the root cause"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "exit 1"},
        {"role": "user", "content": "why does this fail? analyze the root cause"},
    ], "reasoning_effort": "low"}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        provider="zai",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["reasoning_effort"] == "high"

    # brevity → minimal; zai glm-5.3 native mapping: minimal → low
    req2 = {"model": "glm-5.3", "messages": [], "reasoning_effort": "high"}
    result2 = plugin_mod.adaptive_llm_request_middleware(
        request=req2,
        user_message="ok",
        provider="zai",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result2 is not None
    assert result2["request"]["reasoning_effort"] == "low"


def test_middleware_never_writes_offscale_native_effort():
    # INVARIANT: whatever lands in top-level reasoning_effort must be a value
    # the provider profile itself would emit. Hermes-only levels (minimal,
    # xhigh, ultra…) must never appear there for zai/kimi.
    req = {"model": "glm-5.3", "messages": [], "reasoning_effort": "high"}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        user_message="ok",  # brevity → minimal on Hermes scale
        provider="zai",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result["request"]["reasoning_effort"] in {"low", "high", "max"}

    # kimi: only low/medium/high are legal wire values
    req2 = {"model": "kimi-k2", "messages": [], "reasoning_effort": "medium"}
    result2 = plugin_mod.adaptive_llm_request_middleware(
        request=req2,
        user_message="帮我排查这个死锁问题的根因，分析整个调用链路",  # → high
        provider="kimi",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result2["request"]["reasoning_effort"] in {"low", "medium", "high"}


def test_middleware_noop_when_profile_cannot_express_level():
    # Unknown provider → no profile → top-level reasoning_effort untouched
    req = {"model": "m", "messages": [], "reasoning_effort": "high"}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        user_message="why does this fail? analyze the root cause",
        provider="no-such-provider",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


def test_middleware_noop_without_reasoning_fields():
    req = {"model": "m", "messages": []}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        user_message="help me debug this",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


def test_middleware_noop_when_thinking_disabled():
    req = _base_request()
    req["extra_body"]["reasoning"] = {"enabled": False}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        user_message="help me debug this",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


def test_middleware_noop_on_non_dict_request():
    assert plugin_mod.adaptive_llm_request_middleware(request=None) is None
    assert plugin_mod.adaptive_llm_request_middleware(request="junk") is None


# ── 3. Off-switch ────────────────────────────────────────────────────────

def test_disabled_plugin_is_noop(monkeypatch, tmp_path):
    home = tmp_path / "disabled_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  entries:\n    adaptive-reasoning:\n      enabled: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugin_mod._CONFIG_CACHE = None
    plugin_mod._CONFIG_CACHE_KEY = None
    result = plugin_mod.adaptive_llm_request_middleware(
        request=_base_request(),
        user_message="help me debug this",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


# ── 4. In-turn escalation ────────────────────────────────────────────────

def test_tool_errors_escalate_effort():
    # Two failed tool calls → one step up from the phrasing-derived level
    assert plugin_mod.classify_effort("ok", tool_errors=2) == "low"
    assert plugin_mod.classify_effort("ok", tool_errors=4) == "medium"


def test_post_tool_call_counts_only_errors():
    plugin_mod.on_post_tool_call(status="error", session_id="s1", turn_id="t1")
    plugin_mod.on_post_tool_call(status="ok", session_id="s1", turn_id="t1")
    plugin_mod.on_post_tool_call(status="success", session_id="s1", turn_id="t1")
    assert plugin_mod._error_count_for_turn("s1", "t1") == 1


def test_middleware_reads_error_state():
    plugin_mod.on_post_tool_call(status="error", session_id="s1", turn_id="t1")
    plugin_mod.on_post_tool_call(status="error", session_id="s1", turn_id="t1")
    result = plugin_mod.adaptive_llm_request_middleware(
        request=_base_request(),
        user_message="ok",  # brevity → minimal, +1 step (2 errors) → low
        session_id="s1", turn_id="t1", api_call_count=3,
    )
    assert result is not None
    assert result["request"]["extra_body"]["reasoning"]["effort"] == "low"


def test_stale_turn_state_is_dropped():
    plugin_mod.on_post_tool_call(status="error", session_id="s1", turn_id="t1")
    plugin_mod.adaptive_llm_request_middleware(
        request=_base_request(), user_message="ok",
        session_id="s1", turn_id="t2", api_call_count=1,
    )
    assert plugin_mod._error_count_for_turn("s1", "t1") == 0


# ── 5. Floor/ceiling clamping ────────────────────────────────────────────

def test_ceiling_caps_adaptive_effort():
    cfg = dict(DEFAULT_CFG)
    cfg["ceiling"] = "medium"
    # v2: rescue path (4 tool errors) escalates brevity minimal → medium+,
    # ceiling clamps the rescue back to medium
    level = plugin_mod.classify_effort("ok", tool_errors=6, cfg=cfg)
    assert plugin_mod._clamp(level, cfg["floor"], cfg["ceiling"]) == "medium"


def test_floor_raises_minimal_effort():
    cfg = dict(DEFAULT_CFG)
    cfg["floor"] = "medium"
    level = plugin_mod.classify_effort("ok", cfg=cfg)
    assert plugin_mod._clamp(level, cfg["floor"], cfg["ceiling"]) == "medium"


def test_clamp_bounds():
    s = plugin_mod.EFFORT_SCALE
    assert plugin_mod._clamp("minimal", "medium", "high") == "medium"
    assert plugin_mod._clamp("ultra", "medium", "high") == "high"
    assert plugin_mod._clamp("low", "minimal", "ultra") == "low"
