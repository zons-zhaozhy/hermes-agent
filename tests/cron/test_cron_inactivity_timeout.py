"""Tests for the cron inactivity watchdog loop (runs on its own daemon thread)."""

import threading
import time
from types import SimpleNamespace

import pytest


class TestInactivityWatchdogLoop:
    """The daemon-thread inactivity helper must not depend on the caller thread."""

    def test_fires_when_idle_crosses_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 0.0}
        results: list = []

        def _watch():
            results.append(
                _inactivity_watchdog_loop(
                    get_idle_seconds=lambda: idle["s"],
                    limit_s=0.2,
                    poll_s=0.05,
                    stop=stop,
                    future_done=lambda: False,
                )
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        time.sleep(0.12)
        idle["s"] = 1.0
        watcher.join(timeout=2.0)
        stop.set()
        # Float contract + meter clock correction: the returned awake-idle is the
        # judged sample minus microsecond-scale sleep-drift, so approx, not ==.
        import pytest as _pytest
        assert len(results) == 1
        assert results[0] == _pytest.approx(1.0, abs=1e-3)
        assert not isinstance(results[0], bool)
        assert not watcher.is_alive()

    def test_stops_when_future_completes_before_idle_limit(self):
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        fired = _inactivity_watchdog_loop(
            get_idle_seconds=lambda: 0.0,
            limit_s=10.0,
            poll_s=0.05,
            stop=stop,
            future_done=lambda: True,
        )
        assert fired is False

    def test_fires_while_caller_thread_is_blocked(self):
        """#94285: a blocked run_job thread must not disable the watchdog."""
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 1.0}
        result = {"fired": None}

        def _watch():
            result["fired"] = _inactivity_watchdog_loop(
                get_idle_seconds=lambda: idle["s"],
                limit_s=0.15,
                poll_s=0.05,
                stop=stop,
                future_done=lambda: False,
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        # Simulate the family-A stall: this thread cannot poll.
        time.sleep(0.4)
        watcher.join(timeout=2.0)
        stop.set()
        # 期望: 返回判定时刻观测到的 idle 值本身（>= limit），且不再是裸 True——
        # raiser 依赖它原样上报，杜绝 "idle for 0s (limit 600s)" 假数
        assert result["fired"] is not False
        assert result["fired"] >= 0.99  # meter 微秒级漂移校正后 ≈1.0，不卡严格下界
        assert not isinstance(result["fired"], bool)
        assert not watcher.is_alive()

    def test_returns_observed_idle_value_not_bool(self):
        """The loop must hand back the idle value it judged on, so the raiser
        never has to resample cross-thread (the 'idle for 0s (limit 600s)'
        impossible-number incidents)."""
        from cron.scheduler import _inactivity_watchdog_loop

        stop = threading.Event()
        idle = {"s": 0.0}
        results: list = []

        def _watch():
            results.append(
                _inactivity_watchdog_loop(
                    get_idle_seconds=lambda: idle["s"],
                    limit_s=0.2,
                    poll_s=0.05,
                    stop=stop,
                    future_done=lambda: False,
                )
            )

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        time.sleep(0.12)
        idle["s"] = 1.0
        watcher.join(timeout=2.0)
        stop.set()
        # 期望: 返回值恒等于触发时刻的观测 idle（≈1.0，容许 meter 微秒级睡眠漂移校正），
        # 供 _raise_inactivity_timeout 原样上报
        import pytest as _pytest
        assert len(results) == 1
        assert results[0] == _pytest.approx(1.0, abs=1e-3)
        assert not isinstance(results[0], bool)
        assert not watcher.is_alive()


class TestRaiseInactivityTimeoutObservedIdle:
    """The raiser must report the watchdog's judged idle value, not a fresh
    cross-thread resample that can read 0s right after the limit fired."""

    def _agent_double(self, seconds_since_activity: float):
        class _Agent:
            interrupted = None

            def get_activity_summary(self):
                return {
                    "last_activity_desc": "tool: terminal",
                    "seconds_since_activity": seconds_since_activity,
                    "api_call_count": 3,
                    "max_iterations": 40,
                    "current_tool": "terminal",
                }

            def interrupt(self, message=None):
                _Agent.interrupted = message

        return _Agent()

    def test_reports_observed_idle_not_resampled_zero(self):
        # 期望: 判定值 3621s 必须进异常文本；raise 时重采样到 0s（竞态/恢复瞬间）不得覆盖
        import pytest
        from cron.scheduler import _raise_inactivity_timeout

        agent = self._agent_double(seconds_since_activity=0.0)
        with pytest.raises(TimeoutError) as ei:
            _raise_inactivity_timeout(agent, "job-x", 600.0, observed_idle_s=3621.0)
        assert "idle for 3621s" in str(ei.value)
        assert agent.interrupted  # hard-interrupt 不变量保持

    def test_falls_back_to_resample_when_no_observed_value(self):
        # 期望: 无观测值（直接调用）时保留旧采样语义
        import pytest
        from cron.scheduler import _raise_inactivity_timeout

        agent = self._agent_double(seconds_since_activity=730.0)
        with pytest.raises(TimeoutError) as ei:
            _raise_inactivity_timeout(agent, "job-y", 600.0)
        assert "idle for 730s" in str(ei.value)


class TestHostSleep:
    """A sleeping host freezes the whole job, so the nap is not job inactivity."""

    @pytest.mark.parametrize("during_sample", [False, True])
    def test_sleep_is_not_idle_time_but_awake_idle_still_fires(self, monkeypatch, during_sample):
        from agent import session_activity
        from cron.scheduler import _inactivity_watchdog_loop

        # Wall time keeps running while the host sleeps; monotonic time pauses (macOS, Linux).
        clock = SimpleNamespace(wall=1_000_000.0, mono=50.0)
        monkeypatch.setattr(
            session_activity, "time", SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono))
        last_activity_wall, start_mono = clock.wall, clock.mono
        polls = []

        class _Stop:
            def wait(self, timeout):
                polls.append(timeout)
                clock.wall += timeout
                clock.mono += timeout
                if len(polls) == 3 and not during_sample:
                    clock.wall += 900.0  # the laptop sleeps for 15 minutes mid-job
                return len(polls) > 1000

        def read_idle():
            idle = clock.wall - last_activity_wall
            if during_sample and len(polls) == 3:
                clock.wall += 900.0  # suspend after the activity read, before meter clocks
            return idle

        fired = _inactivity_watchdog_loop(
            get_idle_seconds=read_idle,
            limit_s=600.0,
            poll_s=5.0,
            stop=_Stop(),
            future_done=lambda: False,
        )

        # Float contract (fork): the loop returns the awake-idle value it judged
        # on, not a bool — sleep-corrected by the meter, so still 600..605 here.
        assert fired is not False and not isinstance(fired, bool) and 600.0 <= fired < 605.0
        awake_idle = clock.mono - start_mono
        assert 600.0 <= awake_idle < 605.0


def test_sleep_before_watchdog_thread_starts(monkeypatch):
    """The worker can record activity before the watchdog gets its first timeslice."""
    from agent import activity_tracking, session_activity
    from cron import scheduler

    clock = SimpleNamespace(wall=1000.0, mono=1000.0)
    timer = SimpleNamespace(time=lambda: clock.wall, monotonic=lambda: clock.mono)
    started, finish = threading.Event(), threading.Event()
    interrupted_at = []

    class Agent(activity_tracking.ActivityTrackingMixin):
        def run_conversation(self, *args, **kwargs):
            self._touch_activity("starting new turn")
            started.set()
            finish.wait(20)
            return {"final_response": "finished"}

        def get_activity_summary(self):
            return session_activity.build_activity_snapshot(
                last_activity_at=self._last_activity_ts,
                last_activity_description=self._last_activity_desc)

        def interrupt(self, message, **kwargs):
            interrupted_at.append(clock.mono - 1000.0)
            finish.set()

    original_start, original_wait = threading.Thread.start, threading.Event.wait

    def start(thread):
        if thread.name.startswith("cron-inactivity-"):
            assert started.wait(5), "worker did not start"
            clock.wall += 900
        return original_start(thread)

    def wait(event, timeout=None):
        if threading.current_thread().name.startswith("cron-inactivity-") and timeout == 5.0:
            clock.wall += timeout
            clock.mono += timeout
            return event.is_set()
        return original_wait(event, timeout)

    for module in (activity_tracking, session_activity, scheduler):
        monkeypatch.setattr(module, "time", timer)
    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(threading.Event, "wait", wait)
    monkeypatch.setattr(scheduler, "_cron_inactivity_seconds", lambda: 600.0)
    try:
        with pytest.raises(TimeoutError):
            scheduler._run_agent_with_watchdog(
                Agent(), "probe", {"schedule": {"kind": "every"}},
                "probe", "probe", "probe", None)
    finally:
        finish.set()
    assert interrupted_at and min(interrupted_at) >= 600.0
