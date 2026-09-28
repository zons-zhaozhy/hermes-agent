"""Tests for the cron inactivity watchdog loop (runs on its own daemon thread)."""

import threading
import time


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
        assert results == [True]
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
        assert result["fired"] >= 1.0
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
        # 期望: 返回值恒等于触发时刻的观测 idle（1.0），供 _raise_inactivity_timeout 原样上报
        assert results == [1.0]
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

