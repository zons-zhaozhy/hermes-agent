"""Regression tests for ReadThinkGate production wiring (2026-08-16 fix).

Locks four gaps left by the partial wiring restoration (03ddfe147d):

A1. execute_tool_calls_sequential must run check_batch — single-tool calls
    (the most common write-tool shape) dispatch through this path.
A2. execute_tool_calls_segmented must preserve assistant content on the
    segment view — a bare SimpleNamespace(tool_calls=...) dropped it, so
    _scan_four_axis(None) returned before scanning and the marker was
    never written → four-axis-guard plugin blocked every write tool.
B.  check_batch must receive tool_args so terminal file-write detection
    (_terminal_writes_file) and write-target coverage work.
C.  Marker + exemption paths must resolve via get_hermes_home()
    (profile-aware), not module-level Path.home() constants (PR #3575).

Expectations derive from the 851bdcf641 design invariants:
  - gate crash never blocks execution (failsafe)
  - one check_batch per assistant_message
  - four-axis evidence accumulates within a turn
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.read_think_gate import ReadThinkGate, ReadThinkGateConfig
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_four_axis_guard():
    """The plugin dir has a hyphen — load by file path like the plugin loader."""
    import importlib.util

    plugin_path = Path(__file__).resolve().parents[2] / "plugins" / "guards" / "four_axis.py"
    assert plugin_path.exists(), f"four-axis guard module missing: {plugin_path}"
    spec = importlib.util.spec_from_file_location("four_axis_guard_under_test", plugin_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mock_tool_call(name: str, args: dict, call_id: str = "c1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(*tool_names: str, gate_enabled: bool = True) -> AIAgent:
    """Real AIAgent with the gate config pinned; relay/dispatch mocked later."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "hermes_cli.config.load_config",
            return_value={"read_think_gate": {"enabled": gate_enabled}},
        ),
        patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"read_think_gate": {"enabled": gate_enabled}},
        ),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    # Attributes assembled by the current agent_init that sequential dispatch reads.
    if not hasattr(agent, "_context_engine_tool_names"):
        agent._context_engine_tool_names = set()
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


class _GateSpy(ReadThinkGate):
    """Records check_batch invocations; verdict delegated to the real gate."""

    def __init__(self, config=None):
        super().__init__(config or ReadThinkGateConfig(enabled=True))
        self.calls: list[tuple] = []

    def check_batch(self, assistant_content, tool_names, tool_args=None):
        self.calls.append((assistant_content, list(tool_names), tool_args))
        return super().check_batch(assistant_content, tool_names, tool_args)


def _spy_on_agent(agent: AIAgent) -> _GateSpy:
    spy = _GateSpy(agent._read_think_gate.config)
    agent._read_think_gate = spy
    return spy


# ---------------------------------------------------------------------------
# C. profile-aware marker path (unit level)
# ---------------------------------------------------------------------------
class TestMarkerPathProfileAware:
    def test_marker_path_resolves_via_get_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.read_think_gate import _four_axis_marker_path

        p = _four_axis_marker_path()
        assert p == tmp_path / "cache" / "four_axis_gate.json"

    def test_plugin_marker_path_matches_gate_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.read_think_gate import _four_axis_marker_path

        four_axis_guard = _load_four_axis_guard()

        assert four_axis_guard._marker_file() == _four_axis_marker_path()

    def test_agent_owned_exemptions_follow_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        four_axis_guard = _load_four_axis_guard()

        assert four_axis_guard._is_agent_owned_path(str(tmp_path / "cron" / "output" / "r.md"))
        assert not four_axis_guard._is_agent_owned_path(
            str(Path.home() / ".hermes" / "cron" / "output" / "r.md")
        )


# ---------------------------------------------------------------------------
# A2. segmented dispatcher preserves content (executor level)
# ---------------------------------------------------------------------------
class TestSegmentedContentPreservation:
    def test_segment_view_carries_assistant_content(self, tmp_path):
        from agent.tool_executor import execute_tool_calls_segmented

        agent = _make_agent("web_search", "read_file")
        spy = _spy_on_agent(agent)
        first = _mock_tool_call("web_search", {"query": "a"}, "c1")
        second = _mock_tool_call("read_file", {"path": "/tmp/x"}, "c2")
        assistant_content = (
            "影响面: caller 已确认。原意图: git log 851bdcf641。"
            "根因: segmented 丢 content。风险: marker 永不写入。"
        )
        assistant_message = SimpleNamespace(content=assistant_content, tool_calls=[first, second])
        messages = []

        with (
            patch("agent.tool_executor.get_active_env", return_value=None),
            patch("model_tools.handle_function_call", return_value="{}"),
        ):
            execute_tool_calls_segmented(
                agent, assistant_message, messages, "task-1",
                segments=[("parallel", [first]), ("sequential", [second])],
            )

        assert len(spy.calls) >= 2, "each segment executor must invoke check_batch"
        scanned = [c[0] for c in spy.calls]
        assert all(c == assistant_content for c in scanned), (
            f"segment view dropped assistant content: {scanned!r}"
        )

    def test_segment_view_passes_tool_args(self, tmp_path):
        from agent.tool_executor import execute_tool_calls_segmented

        agent = _make_agent("web_search", "read_file")
        spy = _spy_on_agent(agent)
        first = _mock_tool_call("web_search", {"query": "a"}, "c1")
        second = _mock_tool_call("read_file", {"path": "/tmp/x"}, "c2")
        assistant_message = SimpleNamespace(content="scan me", tool_calls=[first, second])
        messages = []

        with (
            patch("agent.tool_executor.get_active_env", return_value=None),
            patch("model_tools.handle_function_call", return_value="{}"),
        ):
            execute_tool_calls_segmented(
                agent, assistant_message, messages, "task-1",
                segments=[("parallel", [first]), ("sequential", [second])],
            )

        assert spy.calls, "check_batch never invoked"
        for _content, names, args in spy.calls:
            assert args is not None, "check_batch called without tool_args (Gap B)"
            assert len(args) == len(names)


# ---------------------------------------------------------------------------
# A1. sequential path gate wiring
# ---------------------------------------------------------------------------
class TestSequentialGateWiring:
    def test_single_write_call_runs_check_batch(self, tmp_path):
        """len(tool_calls)==1 dispatches to the sequential path — the gate
        must observe it (Gap A1: the most common write shape bypassed the
        gate entirely)."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _make_agent("write_file")
        spy = _spy_on_agent(agent)
        tc = _mock_tool_call("write_file", {"path": "/tmp/a.py", "content": "x"})
        assistant_message = SimpleNamespace(content="editing after investigation", tool_calls=[tc])
        messages = []

        with (
            patch("agent.tool_executor.get_active_env", return_value=None),
            patch("model_tools.handle_function_call", return_value="{}"),
        ):
            execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

        assert len(spy.calls) == 1, (
            "sequential path must call check_batch exactly once per assistant_message"
        )
        _content, names, args = spy.calls[0]
        assert names == ["write_file"]
        assert args == [{"path": "/tmp/a.py", "content": "x"}]

    def test_gate_block_reaches_sequential_result(self, tmp_path):
        """A gate block on the sequential path must surface as the tool
        result (scope_block channel), not dispatch the tool."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _make_agent("write_file")
        # Real gate, enabled, zero investigation → must block write_file.
        agent._read_think_gate = ReadThinkGate(ReadThinkGateConfig(enabled=True))
        tc = _mock_tool_call("write_file", {"path": "/tmp/a.py", "content": "x"})
        assistant_message = SimpleNamespace(content="", tool_calls=[tc])
        messages = []

        dispatched: list = []
        with (
            patch("agent.tool_executor.get_active_env", return_value=None),
            patch(
                "model_tools.handle_function_call",
                side_effect=lambda *a, **k: dispatched.append(a) or "{}",
            ),
        ):
            execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

        assert not dispatched, "gate-blocked write must not dispatch"
        assert messages, "blocked call must still append a tool result message"
        result_text = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else str(messages[-1])
        )
        assert "ReadThink" in result_text, (
            f"block verdict must surface in the tool result: {result_text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# A2 end-to-end: four-axis-complete content writes the marker
# ---------------------------------------------------------------------------
class TestFourAxisMarkerE2E:
    """四轴 marker 现在通过 LLM judge 验证后写入（非关键词匹配）。"""

    def test_mark_four_axis_complete_writes_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.read_think_gate import _four_axis_marker_path

        gate = ReadThinkGate(ReadThinkGateConfig(enabled=True))
        gate.mark_four_axis_complete()
        marker = _four_axis_marker_path()
        assert marker.exists(), "mark_four_axis_complete must write the marker"
        data = json.loads(marker.read_text())
        assert data["verified"] is True
        assert data["source"] == "llm_judge"
        assert len(data["axes"]) == 4

    def test_reset_for_turn_clears_marker(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.read_think_gate import _four_axis_marker_path

        gate = ReadThinkGate(ReadThinkGateConfig(enabled=True))
        gate.mark_four_axis_complete()
        assert _four_axis_marker_path().exists(), "marker must exist after mark"
        gate.reset_for_turn("next turn", None)
        assert not _four_axis_marker_path().exists(), "reset_for_turn must clear the marker"


# ---------------------------------------------------------------------------
# Failsafe invariant (851bdcf641): gate crash never blocks execution
# ---------------------------------------------------------------------------
class TestGateFailsafe:
    def test_crashing_gate_does_not_block_sequential(self, tmp_path):
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _make_agent("write_file")

        class _Crasher(ReadThinkGate):
            def check_batch(self, *a, **k):
                raise RuntimeError("gate exploded")

        agent._read_think_gate = _Crasher(ReadThinkGateConfig(enabled=True))
        tc = _mock_tool_call("write_file", {"path": "/tmp/a.py", "content": "x"})
        assistant_message = SimpleNamespace(content="x", tool_calls=[tc])
        messages = []

        dispatched: list = []
        with (
            patch("agent.tool_executor.get_active_env", return_value=None),
            patch(
                "model_tools.handle_function_call",
                side_effect=lambda *a, **k: dispatched.append(a) or "{}",
            ),
        ):
            execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

        assert dispatched, "gate crash must not block execution (failsafe invariant)"
