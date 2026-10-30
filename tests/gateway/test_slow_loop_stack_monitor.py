"""Same-stack slow-loop diagnostics."""

from __future__ import annotations

import logging
import threading
import time

from gateway.slow_loop_monitor import SlowLoopStackMonitor


def _blocked_for_test() -> None:
    # health: allow HX013 -- the monitor is being tested against a real synchronous block.
    time.sleep(0.25)


def test_monitor_logs_the_blocked_loop_python_stack(caplog):
    monitor = SlowLoopStackMonitor.start(
        thread_id=threading.get_ident(), threshold_seconds=0.08
    )
    try:
        with caplog.at_level(logging.WARNING, logger="gateway.slow_loop_monitor"):
            _blocked_for_test()
    finally:
        assert monitor.stop()

    messages = [record.getMessage() for record in caplog.records]
    assert any("_blocked_for_test" in message for message in messages)
    assert any("Gateway loop callback occupied the event loop" in message for message in messages)


def test_monitor_stops_cleanly():
    monitor = SlowLoopStackMonitor.start(
        thread_id=threading.get_ident(), threshold_seconds=0.5
    )
    assert monitor.stop()
