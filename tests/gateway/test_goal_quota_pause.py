"""A quota-window turn failure (zai 1308「已达到 5 小时的使用上限…限额将在 T 重置」) must park an
ACTIVE goal with a visible reason naming T and schedule one resume probe at T, instead of the goal
stalling silently while the credential pool is locked (2026-11-11 00:40–02:40: four goal sessions
looked hung in the CLI and kept nudging a locked pool)."""
import time
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_goals import GatewayGoalsMixin, QUOTA_WINDOW_PAUSE_PREFIX


class _StubRunner(GatewayGoalsMixin):
    def __init__(self):
        self.notices = []
        self.probes = []
        self.admitted = []

    async def _defer_goal_status_notice_after_delivery(self, source, message):
        self.notices.append(message)

    def _schedule_goal_quota_resume_probe(self, source, session_id, resets_at):
        self.probes.append((session_id, resets_at))

    async def _admit_goal_continuation(self, source, prompt):
        self.admitted.append(prompt)


def _set_active_goal(sid, goal="ship it"):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id=sid)
    return mgr.set(goal, max_turns=10)


def _state(sid):
    from hermes_cli.goals import GoalManager

    return GoalManager(session_id=sid).state


@pytest.mark.asyncio
async def test_quota_failure_pauses_goal_and_names_reset_time(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    state = _set_active_goal("s")

    runner = _StubRunner()
    resets_at = time.time() + 3600
    result = {"failed": True, "failure_reason": "billing", "failure_resets_at": resets_at}
    await runner._pause_goal_for_quota_window(SimpleNamespace(chat_id="s"), result)

    state = _state("s")
    assert state is not None and state.status == "paused"
    assert state.paused_reason and state.paused_reason.startswith(QUOTA_WINDOW_PAUSE_PREFIX)
    assert str(int(resets_at)) not in state.paused_reason  # human time, not a raw epoch
    assert runner.notices and runner.notices[0].startswith("⏸")
    assert runner.probes == [("s", resets_at)]


@pytest.mark.asyncio
async def test_short_throttle_and_non_quota_reasons_leave_goal_active(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    _set_active_goal("s")
    source = SimpleNamespace(chat_id="s")

    runner = _StubRunner()
    # A 60 s throttle: the ordinary retry/backoff ladder owns it, not a goal pause.
    await runner._pause_goal_for_quota_window(
        source, {"failed": True, "failure_reason": "rate_limit", "failure_resets_at": time.time() + 60})
    # No reset time (plain insufficient balance): nothing to wait for.
    await runner._pause_goal_for_quota_window(
        source, {"failed": True, "failure_reason": "billing"})
    # Non-quota verdict.
    await runner._pause_goal_for_quota_window(
        source, {"failed": True, "failure_reason": "server_error", "failure_resets_at": time.time() + 3600})

    state = _state("s")
    assert state is not None and state.status == "active"
    assert runner.notices == [] and runner.probes == []


@pytest.mark.asyncio
async def test_probe_resumes_only_its_own_quota_pause(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    from hermes_cli.goals import GoalManager
    from hermes_cli.goals import save_goal

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    state = _set_active_goal("s")
    state.turns_used = 2
    save_goal("s", state)

    runner = _StubRunner()
    await runner._pause_goal_for_quota_window(
        SimpleNamespace(chat_id="s"),
        {"failed": True, "failure_reason": "billing", "failure_resets_at": time.time() + 3600})

    # A user-paused goal (a different paused_reason) must stay parked at the probe.
    GoalManager(session_id="s").pause(reason="user-paused")
    resumed = await runner._goal_quota_resume_probe(SimpleNamespace(chat_id="s"), "s", wait_s=0.0)
    assert resumed is False
    assert _state("s").status == "paused"

    # Auto-resume is a scheduler continuation, not a fresh user decision: preserve
    # the existing turn budget.
    state = _state("s")
    assert state is not None and state.turns_used == 2

    # The quota pause itself resumes and admits exactly one continuation probe.
    GoalManager(session_id="s").pause(reason=QUOTA_WINDOW_PAUSE_PREFIX + " 23:59")
    resumed = await runner._goal_quota_resume_probe(SimpleNamespace(chat_id="s"), "s", wait_s=0.0)
    assert resumed is True
    resumed_state = _state("s")
    assert resumed_state is not None and resumed_state.status == "active"
    assert resumed_state.turns_used == 2
    assert len(runner.admitted) == 1


@pytest.mark.asyncio
async def test_post_turn_failure_wires_quota_pause_before_loop_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.session_results import execution_result
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    state = _set_active_goal("s")
    state.turns_used = 3
    from hermes_cli.goals import save_goal
    save_goal("s", state)

    runner = _StubRunner()
    runner.async_session_store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="s"))
    )
    runner._post_turn_loop_completion = AsyncMock()
    runner._publish_local_control = AsyncMock()
    resets_at = time.time() + 3600
    result = {
        "final_response": "quota exhausted",
        "failed": True,
        "failure_reason": "billing",
        "failure_resets_at": resets_at,
    }
    token = execution_result.set({"terminal_result": result})
    try:
        await runner._run_post_turn_hooks(
            agent_result=result,
            source=SimpleNamespace(chat_id="s"),
            is_internal=True,
        )
    finally:
        execution_result.reset(token)

    assert _state("s").status == "paused"
    assert runner.probes == [("s", resets_at)]
    runner._post_turn_loop_completion.assert_awaited_once()


@pytest.mark.asyncio
async def test_viewer_driven_goal_is_parked_before_cli_deferral(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.session_results import execution_result
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("viewer-s", source="test")
    _set_active_goal("viewer-s")

    runner = _StubRunner()
    runner.session_authority = SimpleNamespace(
        viewer_automation={
            "viewer-s": SimpleNamespace(subscriptions={"viewer-s": True})
        },
        db=SimpleNamespace(get_meta=lambda _key: None),
    )
    runner.async_session_store = SimpleNamespace(get_or_create_session=AsyncMock())
    resets_at = time.time() + 3600
    result = {
        "final_response": "quota exhausted",
        "failed": True,
        "failure_reason": "billing",
        "failure_resets_at": resets_at,
    }
    token = execution_result.set({"terminal_result": result})
    try:
        await runner._run_post_turn_hooks(
            agent_result=result,
            source=SimpleNamespace(chat_id="viewer-s"),
            is_internal=True,
        )
    finally:
        execution_result.reset(token)

    state = _state("viewer-s")
    assert state is not None and state.status == "paused"
    assert runner.probes == [("viewer-s", resets_at)]
    runner.async_session_store.get_or_create_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_goal_uses_session_id_not_transport_chat_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from gateway.session_results import execution_result
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("chat-c", source="test")
    db.create_session("session-s", source="test")
    _set_active_goal("chat-c")
    _set_active_goal("session-s")

    runner = _StubRunner()
    runner.async_session_store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="session-s"))
    )
    runner._post_turn_loop_completion = AsyncMock()
    runner._publish_local_control = AsyncMock()
    result = {
        "final_response": "quota exhausted",
        "failed": True,
        "failure_reason": "billing",
        "failure_resets_at": time.time() + 3600,
    }
    token = execution_result.set({"terminal_result": result})
    try:
        await runner._run_post_turn_hooks(
            agent_result=result,
            source=SimpleNamespace(chat_id="chat-c"),
            is_internal=True,
        )
    finally:
        execution_result.reset(token)

    chat_state = _state("chat-c")
    session_state = _state("session-s")
    assert chat_state is not None and chat_state.status == "active"
    assert session_state is not None and session_state.status == "paused"
    assert runner.probes == [("session-s", result["failure_resets_at"])]


@pytest.mark.asyncio
async def test_resume_probe_task_is_retained_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = _StubRunner()
    runner.retained_tasks = []
    runner._retain_background_task = lambda task: (runner.retained_tasks.append(task), task)[1]
    calls = []

    async def probe(source, session_id, wait_s):
        calls.append((session_id, wait_s))

    runner._goal_quota_resume_probe = probe
    source = SimpleNamespace(chat_id="s")
    GatewayGoalsMixin._schedule_goal_quota_resume_probe(
        runner, source, "s", time.time() + 0.01
    )

    assert len(runner.retained_tasks) == 1
    await asyncio.wait_for(runner.retained_tasks[0], timeout=1)
    assert calls == [("s", 0.0)]
