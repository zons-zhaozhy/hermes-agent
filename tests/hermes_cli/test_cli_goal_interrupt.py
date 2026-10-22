"""Tests for CLI goal-continuation interrupt handling.

Covers:
- Ctrl+C during a /goal turn auto-pauses the goal (no more continuations).
- Empty/whitespace-only responses skip the judge (no phantom continuations).
- Clean response without interrupt still drives the judge + enqueues.

These tests exercise ``_maybe_continue_goal_after_turn`` directly on a
minimal ``HermesCLI`` stub (pattern used elsewhere in tests/cli).
"""

from __future__ import annotations

import queue
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes stay hermetic."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal module's DB cache so it re-resolves HERMES_HOME each test.
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()

def _make_cli_with_goal(session_id: str, goal_text: str = "build a thing"):
    """Build a minimal HermesCLI stub with an active goal wired in."""
    from cli import HermesCLI
    from hermes_cli.goals import GoalManager

    cli = HermesCLI.__new__(HermesCLI)
    # State the hook + helpers touch directly.
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli.conversation_history = []
    # `_get_goal_manager()` reads `self.session_id` directly, not
    # `self.agent.session_id`. Match the production lookup.
    cli.session_id = session_id
    cli.agent = MagicMock()
    cli.agent.session_id = session_id

    mgr = GoalManager(session_id=session_id, default_max_turns=5)
    mgr.set(goal_text)
    cli._goal_manager = mgr
    return cli, mgr

# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

class TestInterruptAutoPause:

    def test_interrupted_turn_is_resumable(self, hermes_home):
        """After auto-pause from Ctrl+C, /goal resume puts it back to active."""
        sid = f"sid-resume-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli.conversation_history = [
            {"role": "assistant", "content": "partial"},
        ]
        with patch("hermes_cli.goals.judge_goal"):
            cli._maybe_continue_goal_after_turn()
        assert mgr.state.status == "paused"

        mgr.resume()
        assert mgr.state.status == "active"

class TestHealthyTurnStillRuns:
    def test_clean_response_enqueues_continuation_when_judge_says_continue(
        self, hermes_home,
    ):
        """Sanity check: the hook still works in the happy path."""
        sid = f"sid-healthy-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "did some work, more to do"},
        ]

        # Force the judge to say "continue" without touching the network.
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "needs more steps", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        # Continuation prompt must be queued.
        assert not cli._pending_input.empty()
        queued = cli._pending_input.get_nowait()
        assert "Continuing toward your standing goal" in queued
        assert mgr.state.status == "active"

    def test_clean_response_marks_done_when_judge_says_done(self, hermes_home):
        sid = f"sid-done-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = False
        cli.conversation_history = [
            {"role": "assistant", "content": "all finished, here's the result"},
        ]

        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("done", "goal satisfied", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        assert cli._pending_input.empty()
        assert mgr.state.status == "done"


class TestInterruptSource:
    """Only a USER cancellation pauses; a Hermes-side cancellation is judged, not paused."""

    def test_background_review_supersede_is_judged_not_paused(self, hermes_home):
        """A fork-preemption exit reason must not read as the user abandoning the goal."""
        sid = f"sid-bgsupersede-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli._last_turn_interrupt_reason = (
            "interrupted_during_api_call(background_review_superseded)"
        )
        cli.conversation_history = [
            {"role": "assistant", "content": "did some work, more to do"},
        ]
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "needs more steps", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        state = mgr.state
        assert state is not None
        assert state.status == "active"  # 期望: 系统侧取消不暂停 goal
        assert state.paused_reason is None  # 期望: 不留 Ctrl+C 暂停原因

    def test_user_interrupt_still_pauses(self, hermes_home):
        """Ctrl+C keeps the historical auto-pause."""
        sid = f"sid-usercc-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli._last_turn_interrupt_reason = "interrupted_by_user"
        cli.conversation_history = [{"role": "assistant", "content": "partial"}]

        with patch("hermes_cli.goals.judge_goal"):
            cli._maybe_continue_goal_after_turn()

        state = mgr.state
        assert state is not None
        assert state.status == "paused"  # 期望: 用户中断仍暂停

    def test_user_message_handover_is_judged_not_paused(self, hermes_home):
        """A message that preempted the turn hands control over; the goal keeps going."""
        sid = f"sid-handover-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli._last_turn_interrupt_reason = "interrupted_by_user"
        cli._last_turn_interrupt_message = "客户表里机构是空？"
        cli.conversation_history = [
            {"role": "assistant", "content": "did some work, more to do"},
        ]
        with patch(
            "hermes_cli.goals.judge_goal",
            return_value=("continue", "needs more steps", False, None, False),
        ):
            cli._maybe_continue_goal_after_turn()

        state = mgr.state
        assert state is not None
        assert state.status == "active"  # 期望: 用户插话接管不暂停 goal

    def test_bare_ctrl_c_without_message_still_pauses(self, hermes_home):
        """A bare Ctrl+C (no preempting message) keeps the auto-pause."""
        sid = f"sid-barecc-{uuid.uuid4().hex}"
        cli, mgr = _make_cli_with_goal(sid)
        cli._last_turn_interrupted = True
        cli._last_turn_interrupt_reason = "interrupted_by_user"
        cli._last_turn_interrupt_message = None
        cli.conversation_history = [{"role": "assistant", "content": "partial"}]
        with patch("hermes_cli.goals.judge_goal"):
            cli._maybe_continue_goal_after_turn()

        state = mgr.state
        assert state is not None
        assert state.status == "paused"  # 期望: 裸 Ctrl+C 仍暂停
