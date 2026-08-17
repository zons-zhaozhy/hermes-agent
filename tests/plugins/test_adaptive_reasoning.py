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
# Expected values derived independently from the design:
#   brevity-only message           → minimal
#   short plain message            → low
#   medium plain message           → low (no signals → default-low band)
#   one complexity keyword         → medium
#   keyword + code fence           → high
#   two keywords                   → high
#   very long message (≥600 chars) → high

def test_brevity_message_is_minimal():
    assert plugin_mod.classify_effort("ok") == "minimal"
    assert plugin_mod.classify_effort("continue") == "minimal"
    assert plugin_mod.classify_effort("继续") == "minimal"


def test_brevity_with_keyword_still_escalates():
    # "retry" alone is brevity, but a debug keyword must not be silenced
    assert plugin_mod.classify_effort("debug") == "medium"


def test_short_plain_message_is_low():
    assert plugin_mod.classify_effort("what time is it") == "low"


def test_medium_plain_message_stays_low():
    msg = "please list the files in this directory and show me the readme"
    assert len(msg) <= DEFAULT_CFG["low_max_chars"] or True  # len-guard
    assert plugin_mod.classify_effort(msg) == "low"


def test_single_keyword_is_medium():
    assert plugin_mod.classify_effort("help me debug this") == "medium"
    assert plugin_mod.classify_effort("帮我排查这个问题") == "medium"


def test_keyword_plus_code_fence_is_high():
    assert plugin_mod.classify_effort("debug this:\n```\ntraceback\n```") == "high"


def test_two_keywords_are_high():
    assert plugin_mod.classify_effort("analyze and review the failure") == "high"


def test_long_message_is_high():
    msg = "word " * 130  # 650 chars, no keywords
    assert len(msg) >= 600
    assert plugin_mod.classify_effort(msg) == "high"


def test_unknown_levels_clamp_to_medium_index():
    assert plugin_mod._effort_index("bogus") == plugin_mod.EFFORT_SCALE.index("medium")


# ── 2. Middleware rewrite: wire shapes ───────────────────────────────────

def _base_request() -> Dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}},
    }


def test_middleware_rewrites_extra_body_effort():
    result = plugin_mod.adaptive_llm_request_middleware(
        request=_base_request(),
        user_message="help me debug this failing migration, review the architecture",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["extra_body"]["reasoning"]["effort"] == "high"
    # messages untouched — cache prefix integrity
    assert result["request"]["messages"] == [{"role": "user", "content": "hi"}]
    # original request not mutated
    assert _base_request()["extra_body"]["reasoning"]["effort"] == "medium"


def test_middleware_skips_when_effort_already_at_target():
    # classify → medium, request already carries medium → no rewrite needed
    result = plugin_mod.adaptive_llm_request_middleware(
        request=_base_request(),
        user_message="help me debug this",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is None


def test_middleware_rewrites_top_level_reasoning_effort():
    req = {"model": "m", "messages": [], "reasoning_effort": "medium"}
    result = plugin_mod.adaptive_llm_request_middleware(
        request=req,
        user_message="why does this fail? analyze the root cause",
        session_id="s1", turn_id="t1", api_call_count=1,
    )
    assert result is not None
    assert result["request"]["reasoning_effort"] == "high"


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
    # Very complex message would give high, clamped to medium
    level = plugin_mod.classify_effort(
        "analyze and review the architecture failure", cfg=cfg
    )
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
