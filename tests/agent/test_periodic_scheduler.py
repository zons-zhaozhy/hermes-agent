"""agent/periodic_scheduler: one timer thread dispatches isolated callbacks."""

import threading
import time
from typing import Any

from agent import periodic_scheduler
from agent.periodic_scheduler import PeriodicScheduler, schedule


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_two_intervals_fire_proportionally_and_cancel_stops_one():
    sched = PeriodicScheduler()
    fast, slow = [], []
    h_fast = sched.schedule(lambda: fast.append(time.monotonic()), 0.01)
    h_slow = sched.schedule(lambda: slow.append(time.monotonic()), 0.05)

    assert _wait_until(lambda: len(slow) >= 3)
    assert len(fast) > len(slow)  # 5x interval ratio -> clearly more fast ticks
    assert sched._thread is not None and sched._thread.is_alive()

    h_fast.cancel(wait=1.0)
    n_fast = len(fast)
    time.sleep(0.1)
    assert len(fast) == n_fast, "cancelled callback kept firing"
    assert len(slow) > 3, "sibling callback stopped when another was cancelled"
    h_slow.cancel(wait=1.0)
    # With every handle quiesced, scheduling + cancelling adds no persistent thread.
    before = threading.active_count()
    sched.schedule(lambda: None, 0.01).cancel(wait=1.0)
    assert threading.active_count() == before


def test_raising_callback_is_rescheduled_and_does_not_kill_sibling():
    sched = PeriodicScheduler()
    boom, ok = [], []

    def raises():
        boom.append(1)
        raise RuntimeError("bad callback")

    h1 = sched.schedule(raises, 0.01)
    h2 = sched.schedule(lambda: ok.append(1), 0.01)
    assert _wait_until(lambda: len(boom) >= 3 and len(ok) >= 3)
    h1.cancel()
    h2.cancel()


def test_returning_false_stops_callback_and_cancel_wait_joins_inflight():
    sched = PeriodicScheduler()
    calls = []
    sched.schedule(lambda: (calls.append(1), False)[1], 0.01)
    assert _wait_until(lambda: len(calls) == 1)
    time.sleep(0.05)
    assert calls == [1]

    entered = threading.Event()
    release = threading.Event()

    def blocking():
        entered.set()
        release.wait(2.0)

    h = sched.schedule(blocking, 0.01)
    assert entered.wait(2.0)
    threading.Timer(0.05, release.set).start()
    t0 = time.monotonic()
    h.cancel(wait=2.0)  # returns once the in-flight run finished
    assert release.is_set()
    assert time.monotonic() - t0 < 1.5


def test_module_level_schedule_uses_shared_default():
    hits = []
    h = schedule(lambda: hits.append(1), 0.01)
    assert _wait_until(lambda: hits)
    h.cancel(wait=1.0)
    thread = periodic_scheduler._DEFAULT._thread
    assert thread is not None and thread.name == "hermes-periodic-scheduler"
    # Scheduling more timers on the shared default adds no persistent OS threads.
    before = threading.active_count()
    handles = [schedule(lambda: None, 0.01) for _ in range(20)]
    for handle in handles:
        handle.cancel(wait=1.0)
    assert threading.active_count() == before


def test_blocked_callback_does_not_stall_due_sibling(monkeypatch):
    scheduler = PeriodicScheduler()
    monkeypatch.setattr(periodic_scheduler, "_DEFAULT", scheduler)
    blocker_entered = threading.Event()
    release_blocker = threading.Event()
    sibling_ran = threading.Event()

    def blocker():
        blocker_entered.set()
        release_blocker.wait(5.0)
        return False

    def sibling():
        sibling_ran.set()
        return False

    blocker_handle = schedule(blocker, 0.01)
    assert blocker_entered.wait(2.0)
    sibling_handle = schedule(sibling, 0.01)
    try:
        # Ordering, not a wall-clock bound: the sibling must fire WHILE the blocker still holds
        # its worker. On main the sibling only runs after the blocker's 5 s wait expires.
        assert sibling_ran.wait(2.0) and not release_blocker.is_set(), (
            "a blocked periodic callback stalled an unrelated due callback"
        )
    finally:
        release_blocker.set()
        blocker_handle.cancel(wait=1.0)
        sibling_handle.cancel(wait=1.0)


def test_worker_start_failure_keeps_timer(monkeypatch):
    sched = PeriodicScheduler()
    fired: list = []
    real_thread = threading.Thread
    attempts = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        # Only this scheduler's own callback worker fails, once; a leaked handle on the shared
        # _DEFAULT scheduler must not be the one that consumes the single Boom.
        # Bound methods are fresh objects per access: compare with ==, never `is`.
        if kwargs.get("target") == sched._run_callback and attempts["n"] == 0:
            attempts["n"] += 1

            class Boom:
                def start(self) -> None:
                    raise RuntimeError("no threads")

            return Boom()
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(periodic_scheduler.threading, "Thread", flaky)
    handle = sched.schedule(lambda: fired.append(1), 0.01)
    try:
        assert _wait_until(lambda: bool(fired), timeout=3.0), (
            "worker-start failure silently retired the timer"
        )
        # 期望: 恰拦截一次——首次失败由 Boom 消化，requeue 后第二次走真线程
        assert attempts["n"] == 1, "the fake never intercepted the callback worker"
        # 期望: start 失败是暂态，handle 仍被保留（模块不变量⑤：start 失败不退役）
        assert not handle.cancelled
    finally:
        handle.cancel(wait=1.0)


def test_cancel_is_not_blocked_by_a_stalled_dispatch(monkeypatch):
    """A callback worker whose Thread.start() stalls must not wedge cancel():
    cancel has to acquire the scheduler condition lock first, so holding that
    lock across start() makes the wait=timeout parameter unreachable."""
    sched = PeriodicScheduler()
    real_thread = threading.Thread
    dispatch_entered = threading.Event()
    release_dispatch = threading.Event()

    class StalledStart:
        def start(self) -> None:
            dispatch_entered.set()
            release_dispatch.wait(5.0)
            raise RuntimeError("simulated stalled start")

        def join(self, *args: object, **kwargs: object) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("target") == sched._run_callback:
            return StalledStart()
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(periodic_scheduler.threading, "Thread", flaky)
    handle = sched.schedule(lambda: None, 0.01)
    try:
        assert dispatch_entered.wait(2.0), "callback dispatch never entered start()"
        t0 = time.monotonic()
        handle.cancel(wait=0.5)
        elapsed = time.monotonic() - t0
        # 期望: cancel 界内返回（wait=0.5s 上限+调度余量）；若 _cancel 被 dispatch
        # 持有的 _cond 挡住则须等满 5s 卡段，elapsed 必然 >5s，红灯即根因复现
        assert elapsed < 2.0, f"cancel blocked {elapsed:.2f}s behind a stalled dispatch"
    finally:
        release_dispatch.set()
        handle.cancel(wait=1.0)


def test_cancel_during_dispatch_gap_skips_body(monkeypatch):
    """Cancel landing between "runner assigned" and "runner started" must skip
    the callback body: the handle was already retired by its owner."""
    sched = PeriodicScheduler()
    real_thread = threading.Thread
    body_ran = threading.Event()
    start_gate = threading.Event()

    class GatedThread:
        def start(self) -> None:
            start_gate.wait(5.0)

        def join(self, *args: object, **kwargs: object) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("target") == sched._run_callback:
            return GatedThread()
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(periodic_scheduler.threading, "Thread", flaky)
    handle = sched.schedule(body_ran.set, 0.01)
    # Wait until dispatch assigned the runner (i.e. we are inside the start gap).
    assert _wait_until(lambda: handle._runner is not None), "runner never assigned"
    handle.cancel(wait=0.5)
    start_gate.set()
    time.sleep(0.3)
    # 期望: body 永不执行——cancel 在 start 完成前置 _cancelled，所有者已退役该 handle
    assert not body_ran.is_set(), "cancelled handle still ran its body once"


def test_cancel_before_dispatch_never_runs_body():
    """Cancel immediately after schedule (runner never assigned) retires the
    timer without the body ever running or a worker thread ever starting."""
    sched = PeriodicScheduler()
    fired: list = []
    handle = sched.schedule(lambda: fired.append(1), 30.0)
    handle.cancel(wait=1.0)
    time.sleep(0.1)
    # 期望: interval=30s 远大于测试窗，任何 body 执行都是缺陷
    assert fired == []
    # 期望: cancel 先于到期，runner 从未被指派
    assert handle._runner is None
