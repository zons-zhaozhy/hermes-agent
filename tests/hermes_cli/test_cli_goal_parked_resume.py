"""A parked /goal resumes from the idle hook once its barrier lifts, without waiting for another turn."""
import queue
import time
from unittest.mock import patch

import pytest

from hermes_cli import goals
from hermes_cli.cli_loops_mixin import CLILoopsMixin


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path
    home = tmp_path / ".hermes"; home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class _Cli(CLILoopsMixin):
    def __init__(self, mgr):
        self._pending_input = queue.Queue()
        self._mgr = mgr

    def _get_goal_manager(self):
        return self._mgr


def test_idle_hook_queues_the_continuation_when_a_timed_barrier_has_elapsed(hermes_home):
    mgr = goals.GoalManager(session_id="resume-idle")
    mgr.set("finish the thing")
    # A barrier far longer than any first-import cost: patch("cli._cprint") below imports
    # the cli module the first time (12+ s on a cold testenv), which must never be what
    # lifts a "still parked" barrier built to outlast it.
    mgr.wait_for_seconds(3600, reason="cooldown")
    cli = _Cli(mgr)
    with patch("cli._cprint"), patch("cli._DIM", ""), patch("cli._RST", ""):
        cli._maybe_resume_parked_goal()
        assert cli._pending_input.empty()            # still parked
        mgr.state.waiting_until = time.time() - 1
        mgr._save()
        cli._last_goal_barrier_check = 0.0
        cli._maybe_resume_parked_goal()
    assert not cli._pending_input.empty()            # continuation queued
    assert "finish the thing" in cli._pending_input.get()
    assert mgr.state.waiting_until == 0.0            # barrier cleared


def test_idle_hook_is_a_no_op_for_an_unparked_or_inactive_goal(hermes_home):
    mgr = goals.GoalManager(session_id="resume-noop")
    mgr.set("g")
    cli = _Cli(mgr)
    cli._maybe_resume_parked_goal()
    assert cli._pending_input.empty()


def test_idle_hook_refreshes_a_goal_parked_by_the_gateway(tmp_path, monkeypatch):
    """The gateway's judge parks a goal after the turn's frames stop arriving: the CLI's mirror
    never sees the post-turn ``session.control.update``, so the idle hook must re-read the
    manager before deciding the goal is not parked (and before it can resume the barrier)."""
    from types import SimpleNamespace
    from gateway.session_viewer_automation import invoke
    from hermes_cli import goals
    from hermes_cli.cli_loops_mixin import CLILoopsMixin
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / 'owner.db')
    monkeypatch.setattr(goals, '_get_session_db', lambda: db)  # the "gateway's" store

    class _Conn:
        def call(self, method, **params):
            return invoke(SimpleNamespace(), params['kind'], params['session_id'], params['call'],
                          params['args'], params['kwargs'])

    store = GatewaySessionStore.__new__(GatewaySessionStore)
    store.connection = _Conn()
    cli = CLILoopsMixin.__new__(CLILoopsMixin)
    cli._session_db, cli._gateway_session_open, cli.session_id = store, True, 'sid-remote'
    cli._pending_input = queue.Queue()

    mgr = cli._get_goal_manager()
    mgr.set('ship it')
    # The gateway parks the goal after this mirror's last read.
    store.connection.call('session.automation', kind='goal', session_id='sid-remote',
                          call='wait_for_seconds', args=[3600], kwargs={'reason': 'cooldown'})
    assert mgr.state.waiting_until == 0.0  # the mirror is stale, as after a real turn

    with patch('cli._cprint'), patch('cli._DIM', ''), patch('cli._RST', ''):
        cli._maybe_resume_parked_goal()
    assert mgr.state.waiting_until > 0  # mirror re-read: the parked barrier is now visible
    assert cli._pending_input.empty()   # and still parked

    import json
    row = json.loads(db.get_meta('goal:sid-remote'))
    row['waiting_until'] = time.time() - 1  # the timer elapsed on the owner side
    db.set_meta('goal:sid-remote', json.dumps(row))
    cli._last_goal_barrier_check = 0.0
    with patch('cli._cprint'), patch('cli._DIM', ''), patch('cli._RST', ''):
        cli._maybe_resume_parked_goal()
    assert not cli._pending_input.empty()  # barrier lifted remotely: continuation queued
    assert 'ship it' in cli._pending_input.get()
    mgr.clear()
    cli._last_goal_barrier_check = 0.0
    cli._maybe_resume_parked_goal()
    assert cli._pending_input.empty()
