"""Active /goal and queued /queue prompts share the classic CLI live-work dock."""
import queue
from types import SimpleNamespace

import pytest
from prompt_toolkit.utils import get_cwidth

from agent.i18n import t


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def test_dock_paints_goal_on_top_and_queue_last_and_follows_their_lifecycle(hermes_home):
    from cli import _VoiceInputMessage
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.goals import GoalManager
    from tools.process_registry_notifications import TimelineNotification

    mgr = GoalManager(session_id="sid-a", default_max_turns=20)
    mgr.set("ship the dock feature")
    pending = queue.Queue()
    for item in ("first follow-up\nwith a newline", ("look at this", ["shot.png"]),
                 _VoiceInputMessage("spoken ask"),
                 TimelineNotification("raw wall", "Process abc exited 0", "notice")):
        pending.put(item)
    cli = SimpleNamespace(agent=None, session_id="sid-a", _goal_manager=mgr, _pending_input=pending)
    dock = SubagentMonitor(cli)

    assert dock.refresh() and dock.has_rows
    lines = dock.dock_text(columns=80, rows=40).splitlines()
    assert "⊙ Goal (active, 0/20 turns): ship the dock feature" in lines[0]
    queue_at = next(i for i, line in enumerate(lines) if "Queue · 4 queued" in line)
    assert [line.strip() for line in lines[queue_at + 1:]] == [
        "1. first follow-up with a newline", "2. look at this", "3. spoken ask", "+1 more"]
    assert all(get_cwidth(line) <= 80 for line in lines)
    dock.collapsed = True
    collapsed = dock.dock_text(columns=80, rows=40)
    counts = f"{t('cli.subagents.goal_active')} · {t('cli.subagents.count_queued', count=4)}"
    assert "\n" not in collapsed
    assert f"{t('cli.subagents.heading_session_restore', count=counts)} · ⊙ Goal" in collapsed
    assert "Ctrl+T" not in collapsed  # the monitor only lists subagents/processes
    assert get_cwidth(collapsed) <= 80

    # A paused goal stays visible; draining the queue and clearing the goal empties the dock.
    dock.collapsed = False
    mgr.pause("user-paused")
    pending.queue.clear()
    assert dock.refresh() and dock.dock_text(columns=80, rows=40).startswith(" ⏸ Goal (paused")
    mgr.clear()
    assert dock.refresh() and not dock.has_rows and dock.dock_text(columns=80, rows=40) == ""


def test_dock_ignores_a_goal_manager_left_over_from_another_session(hermes_home):
    """After /new the cached manager still names the old session until it is rebuilt; the dock
    must not paint the previous session's goal (and never opens state.db itself)."""
    from hermes_cli.cli_session_dock import goal_line
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id="old-sid")
    mgr.set("previous session objective")
    assert goal_line(SimpleNamespace(session_id="old-sid", _goal_manager=mgr)).startswith("⊙ Goal")
    assert goal_line(SimpleNamespace(session_id="new-sid", _goal_manager=mgr)) == ""


def _remote_goal_manager(state):
    """A GatewayGoalManager with a stubbed transport: no gateway needed, refresh counted."""
    from hermes_cli.gateway_automation import GatewayGoalManager

    mgr = object.__new__(GatewayGoalManager)
    mgr.session_id = "sid-k"
    mgr._state = state
    calls = []
    mgr.refresh = lambda: calls.append(1) or None
    return mgr, calls


def test_sync_automation_keepalive_refreshes_active_remote_managers(hermes_home):
    """A long CLI turn sends no session.automation traffic, so the gateway's 30-second viewer
    liveness stamp would expire mid-turn and it would take the goal loop over at the turn's end
    (the window's own doc sizes it for the judge's seconds of latency, not a whole turn). The
    dock's poll must carry a throttled keepalive: one state read per active remote manager per
    window; a dead transport fails silently and the stamp still ages out."""
    from hermes_cli.cli_session_dock import KEEPALIVE_SECONDS, sync_automation
    from hermes_cli.goals import GoalManager

    local = GoalManager(session_id="sid-k")
    local.set("keep the loop visible")
    mgr, calls = _remote_goal_manager(local.state)
    cli = SimpleNamespace(session_id="sid-k", _goal_manager=mgr)

    sync_automation(cli)
    assert calls == [1], "an active remote goal is kept alive on the dock poll"
    sync_automation(cli)
    assert calls == [1], "throttled: the next poll inside the window refreshes nothing"

    cli._automation_keepalive_at -= KEEPALIVE_SECONDS + 1.0
    sync_automation(cli)
    assert calls == [1, 1], "past the window the next poll refreshes again"

    mgr._state = None  # goal cleared: nothing to keep alive
    cli._automation_keepalive_at = 0.0
    sync_automation(cli)
    assert calls == [1, 1]


def test_sync_automation_stale_path_unchanged_local_managers_still_refresh(hermes_home):
    """The pre-existing behaviour: a mid-turn state change flagged by the gateway refreshes the
    mirrored managers (local ones included) once per flag, and local managers never get the
    keepalive (their judge already runs in this process)."""
    from hermes_cli.cli_session_dock import sync_automation
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id="sid-s")
    mgr.set("local objective")
    calls = []
    mgr.refresh = lambda: calls.append(1) or None
    cli = SimpleNamespace(session_id="sid-s", _goal_manager=mgr, _automation_stale=True)

    sync_automation(cli)
    assert calls == [1] and cli._automation_stale is False
    sync_automation(cli)
    assert calls == [1], "the cleared flag refreshes nothing further"


def test_dock_drops_the_prompt_a_turn_takes_from_the_queue_before_the_turn_runs(hermes_home):
    """A seeded ``-q`` prompt (or any queued input) leaves ``_pending_input`` when its turn starts;
    the dock must stop listing it then, not up to a second later on the spinner thread's poll,
    or the dock is still painted when the reply prints and shrinks under it afterwards."""
    from unittest.mock import Mock

    from cli import HermesCLI, _SeededQueryMessage
    from hermes_cli.cli_subagent_monitor import SubagentMonitor

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id, cli.agent = "sid-a", None
    cli._pending_input = queue.Queue()
    cli._pending_input.put(_SeededQueryMessage("hello"))
    monitor = cli._subagent_monitor = SubagentMonitor(cli)
    monitor.refresh()
    monitor._last_poll = float("inf")  # the 1 s poll would not fire again during this turn
    assert monitor.queued == ["hello"]

    seen = []
    for attr, value in {"_pending_resume_sessions": [], "_typed_voice_stop": lambda t: False,
                        "handle_bang_shell": lambda t: False, "_turn_summary_begin": lambda: None,
                        "_tui_after_turn": lambda: None, "_app": SimpleNamespace(invalidate=lambda: None),
                        "chat": Mock(side_effect=lambda *a, **k: seen.append(list(monitor.queued)))}.items():
        setattr(cli, attr, value)
    cli._tui_process_one_input(cli._pending_input.get_nowait())

    assert cli.chat.call_args.args[0] == "hello"
    assert seen == [[]]
    assert not monitor.has_rows


def test_dock_loop_row_counts_down_to_the_next_wakeup(hermes_home):
    """A standing /loop gets a dock row whose countdown moves with the clock (it used to be a
    one-shot ``next in 45m`` transcript line that never changed)."""
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.loops import LoopManager

    mgr = LoopManager(session_id="sid-a")
    mgr.set("check the deploy", interval_seconds=45 * 60)
    mgr.fire_tick()
    mgr.complete_tick("deploy still running")
    due = mgr.state.next_due_at
    dock = SubagentMonitor(SimpleNamespace(agent=None, session_id="sid-a", _loop_manager=mgr,
                                           _pending_input=queue.Queue()))

    assert dock.refresh(now=due - 45 * 60) and dock.has_rows
    assert dock.dock_text(columns=100, rows=40).strip() == "↻ loop · every 45m · 1 run · next in 45m · check the deploy"
    assert dock.refresh(now=due - 30 * 60 + 1) and "next in 30m" in dock.loop  # rounds up, never early
    assert dock.refresh(now=due - 42) and "next in 42s" in dock.loop
    assert not dock.refresh(now=due - 42)  # unchanged text → no repaint
    dock.collapsed = True
    assert dock.dock_text(columns=100, rows=40).strip() == "↻ loop next in 42s · Ctrl+R restore · check the deploy"

    mgr.pause("user-paused")
    assert dock.refresh(now=due) and dock.loop == "⏸ loop paused · user-paused · 1 run · check the deploy"
    mgr.clear()
    assert dock.refresh(now=due) and not dock.has_rows


def test_dock_repaints_a_goal_a_tool_changed_mid_turn(hermes_home):
    """A plugin tool adds criteria to the goal mid-turn: the gateway's ``session.control.update`` frame
    marks the CLI's mirrored manager stale and the dock's next poll re-reads it (once), not at turn end."""
    from hermes_cli.cli_loops_mixin import CLILoopsMixin
    from hermes_cli.cli_subagent_monitor import SubagentMonitor
    from hermes_cli.gateway_agent import _HANDLERS
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id="sid-a")
    mgr.set("ship the dock feature")
    cli = SimpleNamespace(agent=None, session_id="sid-a", _goal_manager=mgr, _pending_input=queue.Queue())
    dock = SubagentMonitor(cli)
    dock.refresh()
    GoalManager(session_id="sid-a").add_subgoal("tests pass")  # the tool's own writer
    dock.refresh()
    assert "subgoal" not in dock.dock_text(columns=80, rows=40)

    agent = SimpleNamespace(control_update_callback=lambda: CLILoopsMixin._mark_automation_stale(cli))
    _HANDLERS["session.control.update"](agent, {"control": {}}, None, {})
    assert dock.refresh() and "1 subgoal" in dock.dock_text(columns=80, rows=40).splitlines()[0]
    assert cli._automation_stale is False
