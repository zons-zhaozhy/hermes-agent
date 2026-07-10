"""Tests for the declare-done gate in ``finalize_turn``.

The gate appends a ⚠️ footer when:
1. Core interaction path files were edited this turn
   (cli.py / run_agent.py / gateway/run.py / model_tools.py / toolsets.py)
2. AND no ``terminal`` tool call happened this turn

It is NOT triggered when:
- Only non-core files were edited
- A terminal call did happen
- The turn was interrupted
- No files were edited
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, edited_paths=None):
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._turn_file_mutation_paths = set(edited_paths or [])
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- harmless no-ops ------------------------------------------------
    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _run(agent, *, messages=None, final_response="Done.", interrupted=False):
    if messages is None:
        messages = [{"role": "user", "content": "do a thing"}]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


# ── Gate TRIGGERS ──────────────────────────────────────────────────────

def test_gate_triggers_on_cli_py_without_terminal():
    """改了 cli.py 但没有 terminal 调用 → 追加 footer。"""
    agent = _StubAgent(edited_paths=["/repo/cli.py"])
    result = _run(agent)
    assert "Declare-done gate" in result["final_response"]
    assert "cli.py" in result["final_response"]


def test_gate_triggers_on_gateway_run_py():
    """gateway/run.py 也是核心交互路径。"""
    agent = _StubAgent(edited_paths=["/repo/gateway/run.py"])
    result = _run(agent)
    assert "Declare-done gate" in result["final_response"]


def test_gate_triggers_on_run_agent_py():
    """run_agent.py 也是核心交互路径。"""
    agent = _StubAgent(edited_paths=["/repo/run_agent.py"])
    result = _run(agent)
    assert "Declare-done gate" in result["final_response"]


# ── Gate does NOT trigger ──────────────────────────────────────────────

def test_gate_no_trigger_when_terminal_called():
    """改了 cli.py 但本轮有 terminal 调用 → 不追加 footer。"""
    agent = _StubAgent(edited_paths=["/repo/cli.py"])
    messages = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "exit 0"},
    ]
    result = _run(agent, messages=messages)
    assert "Declare-done gate" not in result["final_response"]


def test_gate_no_trigger_on_non_core_file():
    """改了普通文件（非核心交互路径）→ 不追加 footer。"""
    agent = _StubAgent(edited_paths=["/repo/src/utils.py"])
    result = _run(agent)
    assert "Declare-done gate" not in result["final_response"]


def test_gate_no_trigger_when_no_files_edited():
    """没有修改任何文件 → 不追加 footer。"""
    agent = _StubAgent(edited_paths=[])
    result = _run(agent)
    assert "Declare-done gate" not in result["final_response"]


def test_gate_no_trigger_on_interrupted_turn():
    """中断的轮次 → 不追加 footer。"""
    agent = _StubAgent(edited_paths=["/repo/cli.py"])
    result = _run(agent, interrupted=True)
    assert "Declare-done gate" not in result["final_response"]


def test_gate_no_trigger_on_mixed_with_terminal():
    """同时改了核心和非核心文件，且有 terminal 调用 → 不追加 footer。"""
    agent = _StubAgent(edited_paths=["/repo/cli.py", "/repo/src/utils.py"])
    messages = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ran test"},
    ]
    result = _run(agent, messages=messages)
    assert "Declare-done gate" not in result["final_response"]
