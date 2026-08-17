"""Behavioral tests for the quality-auditor plugin.

These test the PLUGIN's contract (hook wiring, tool-stat extraction,
feedback gating), not the underlying agent/quality_auditor.py module
(that has its own suite in tests/agent/test_quality_auditor.py).

Real-import policy: the plugin module is imported for real (no mocks of
the plugin itself); the agent.quality_auditor functions it delegates to
are monkeypatched at the module boundary because they hit the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "quality-auditor"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "quality_auditor_plugin", _PLUGIN_DIR / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["quality_auditor_plugin"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def plugin():
    return _load_plugin()


# ---------------------------------------------------------------------------
# _extract_tool_stats — per-turn tool usage from the transcript
# ---------------------------------------------------------------------------


def test_tool_stats_empty_history(plugin):
    assert plugin._extract_tool_stats([]) == (0, [])
    assert plugin._extract_tool_stats(None) == (0, [])


def test_tool_stats_counts_only_current_turn(plugin):
    history = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file"}},
        ]},
        {"role": "assistant", "content": "turn 1 done"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_files"}},
            {"function": {"name": "terminal"}},
        ]},
        {"role": "assistant", "content": "turn 2 done"},
    ]
    count, names = plugin._extract_tool_stats(history)
    assert count == 2
    assert names == ["search_files", "terminal"]


def test_tool_stats_malformed_rows_are_skipped(plugin):
    history = [
        {"role": "user", "content": "q"},
        "not-a-dict",
        {"role": "assistant", "tool_calls": "not-a-list"},
        {"role": "assistant", "tool_calls": [{"no_function": 1}, {"function": {}}]},
        {"role": "assistant", "content": "done"},
    ]
    count, names = plugin._extract_tool_stats(history)
    assert count == 0
    assert names == []


# ---------------------------------------------------------------------------
# on_post_llm_call — audit firing
# ---------------------------------------------------------------------------


def test_post_llm_call_skips_trivial_responses(plugin, monkeypatch):
    fired = []

    import types
    fake_mod = types.ModuleType("agent.quality_auditor")
    fake_mod.fire_quality_audit = lambda **kw: fired.append(kw)
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)

    plugin.on_post_llm_call(assistant_response="too short", session_id="s1")
    plugin.on_post_llm_call(assistant_response="", session_id="s1")
    plugin.on_post_llm_call(assistant_response=None, session_id="s1")
    assert fired == []


def test_post_llm_call_fires_with_turn_stats(plugin, monkeypatch):
    fired = []
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")
    fake_mod.fire_quality_audit = lambda **kw: fired.append(kw)
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)

    history = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
    ]
    plugin.on_post_llm_call(
        assistant_response="A" * 80,
        user_message="do it",
        session_id="sess-1",
        model="glm-5.3",
        conversation_history=history,
    )
    assert len(fired) == 1
    kw = fired[0]
    assert kw["session_id"] == "sess-1"
    assert kw["model"] == "glm-5.3"
    assert kw["tool_call_count"] == 1
    assert kw["tool_names"] == ["read_file"]


def test_post_llm_call_never_raises(plugin, monkeypatch):
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")

    def boom(**kw):
        raise RuntimeError("aux model down")

    fake_mod.fire_quality_audit = boom
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)
    # must not propagate — a plugin must never break the host
    plugin.on_post_llm_call(assistant_response="A" * 80, session_id="s")


# ---------------------------------------------------------------------------
# on_pre_llm_call — feedback injection
# ---------------------------------------------------------------------------


def test_pre_llm_call_returns_context_when_feedback_exists(plugin, monkeypatch):
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")
    fake_mod.get_last_audit_feedback = lambda sid: "[Quality feedback] Issue: X"
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)

    out = plugin.on_pre_llm_call(session_id="sess-1")
    assert out == {"context": "[Quality feedback] Issue: X"}


def test_pre_llm_call_returns_none_when_no_feedback(plugin, monkeypatch):
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")
    fake_mod.get_last_audit_feedback = lambda sid: None
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)

    assert plugin.on_pre_llm_call(session_id="sess-1") is None


def test_pre_llm_call_requires_session_id(plugin, monkeypatch):
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")
    called = []
    fake_mod.get_last_audit_feedback = lambda sid: called.append(sid)
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)

    assert plugin.on_pre_llm_call(session_id="") is None
    assert called == []  # short-circuited before any file read


def test_pre_llm_call_never_raises(plugin, monkeypatch):
    import types
    fake_mod = types.ModuleType("agent.quality_auditor")

    def boom(sid):
        raise OSError("disk gone")

    fake_mod.get_last_audit_feedback = boom
    monkeypatch.setitem(sys.modules, "agent.quality_auditor", fake_mod)
    assert plugin.on_pre_llm_call(session_id="s") is None


# ---------------------------------------------------------------------------
# register — hook wiring
# ---------------------------------------------------------------------------


def test_register_wires_both_hooks(plugin):
    registered = {}

    class FakeCtx:
        def register_hook(self, name, cb):
            registered[name] = cb

    plugin.register(FakeCtx())
    assert set(registered) == {"post_llm_call", "pre_llm_call"}
    assert registered["post_llm_call"] is plugin.on_post_llm_call
    assert registered["pre_llm_call"] is plugin.on_pre_llm_call


# ---------------------------------------------------------------------------
# Plugin manifest sanity — the loader needs these keys
# ---------------------------------------------------------------------------


def test_manifest_declares_hooks():
    import yaml
    manifest = yaml.safe_load((_PLUGIN_DIR / "plugin.yaml").read_text())
    assert manifest["name"] == "quality-auditor"
    assert manifest.get("kind", "standalone") == "standalone"
    assert set(manifest.get("hooks", [])) == {"post_llm_call", "pre_llm_call"}
