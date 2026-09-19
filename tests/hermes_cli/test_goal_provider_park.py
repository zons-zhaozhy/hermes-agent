"""Tests for the provider-failure goal park: an active goal must retry automatically after
a recoverable provider failure (429/overload/5xx) instead of stalling until the next user
message. Expectations derived from the design, not from the implementation:
- rate_limit / overloaded / server_error / upstream_rate_limit park the goal on a timer;
- billing does NOT park (needs a top-up, not a retry);
- no active goal → no park, no raise;
- parked state reports waiting and auto-clears once the timer elapses.
"""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _active_manager(hermes_home):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id="park-sid", default_max_turns=5)
    mgr.set("long-running migration goal")
    return mgr


class TestParkOnProviderFailure:

    @pytest.mark.parametrize("reason", ["rate_limit", "upstream_rate_limit", "overloaded", "server_error"])
    def test_recoverable_reasons_park(self, hermes_home, reason):
        mgr = _active_manager(hermes_home)
        state = mgr.park_on_provider_failure(reason, wait_seconds=60)
        # 期望: 可恢复原因必须成功停车并进入等待态
        assert state is not None
        assert state.waiting_until > time.time()
        assert mgr.is_waiting()

    def test_billing_does_not_park(self, hermes_home):
        mgr = _active_manager(hermes_home)
        state = mgr.park_on_provider_failure("billing", wait_seconds=60)
        # 期望: 余额耗尽需要用户充值，不许自动重试
        assert state is None
        assert not mgr.is_waiting()

    def test_unknown_reason_does_not_park(self, hermes_home):
        mgr = _active_manager(hermes_home)
        assert mgr.park_on_provider_failure("context_overflow", wait_seconds=60) is None

    def test_no_active_goal_returns_none(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="park-sid-empty", default_max_turns=5)
        # 期望: 无活动目标时静默返回 None，不抛异常
        assert mgr.park_on_provider_failure("rate_limit", wait_seconds=60) is None

    def test_parked_timer_auto_clears(self, hermes_home):
        mgr = _active_manager(hermes_home)
        state = mgr.park_on_provider_failure("rate_limit", wait_seconds=1)
        assert state is not None
        # 期望: 到期后 is_waiting 惰性清障，循环恢复判定
        deadline = time.time() + 5
        while mgr.is_waiting() and time.time() < deadline:
            time.sleep(0.1)
        assert not mgr.is_waiting()

    def test_park_does_not_burn_turn(self, hermes_home):
        mgr = _active_manager(hermes_home)
        before = mgr.state.turns_used if mgr.state else 0  # set() guarantees an active state
        mgr.park_on_provider_failure("rate_limit", wait_seconds=60)
        # 期望: 停车不消耗轮次预算
        assert (mgr.state.turns_used if mgr.state else -1) == before
