"""Sample the gateway loop thread's Python stack while it may be blocked."""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_INTERVAL_S = 0.10
_DEFAULT_REPORT_INTERVAL_S = 60.0


@dataclass(frozen=True)
class _StackSignature:
    """A stable Python call-site signature; moving line numbers keep this identity."""

    frames: tuple[str, ...]


def _signature(frame) -> _StackSignature:
    """Return function/file identity for *frame* without retaining the frame."""

    places: list[str] = []
    while frame is not None and len(places) < 64:
        code = frame.f_code
        places.append(f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}")
        frame = frame.f_back
    return _StackSignature(tuple(places))


class SlowLoopStackMonitor:
    """Log the event-loop stack when one callback remains at the same call site."""

    def __init__(
        self,
        *,
        thread_id: int,
        threshold_seconds: float = 1.0,
        sample_interval_seconds: float = _DEFAULT_SAMPLE_INTERVAL_S,
        report_interval_seconds: float = _DEFAULT_REPORT_INTERVAL_S,
    ) -> None:
        self.thread_id = int(thread_id)
        self.threshold_seconds = max(0.05, float(threshold_seconds))
        self.sample_interval_seconds = min(
            max(0.01, float(sample_interval_seconds)), self.threshold_seconds / 2.0
        )
        self.report_interval_seconds = max(self.threshold_seconds, float(report_interval_seconds))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @classmethod
    def start(cls, *, thread_id: int, threshold_seconds: float = 1.0) -> SlowLoopStackMonitor:
        """Create and start a daemon monitor for *thread_id*."""

        monitor = cls(thread_id=thread_id, threshold_seconds=threshold_seconds)
        from agent.memory_provider import spawn_context_thread
        monitor._thread = spawn_context_thread(
            monitor._run, name="gateway-slow-loop-stack-monitor", daemon=True
        )
        monitor._thread.start()
        return monitor

    def stop(self) -> bool:
        """Request shutdown and wait briefly; True when the sampling thread ended."""

        self._stop.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(2.0)
        return not thread.is_alive()

    def _run(self) -> None:
        current: Optional[tuple[_StackSignature, float]] = None
        last_report: dict[_StackSignature, float] = {}
        while not self._stop.wait(self.sample_interval_seconds):
            frame = sys._current_frames().get(self.thread_id)
            if frame is None:
                current = None
                continue
            signature = _signature(frame)
            now = time.monotonic()
            if current is None or current[0] != signature:
                current = (signature, now)
            if now - current[1] < self.threshold_seconds:
                continue
            reported_at = last_report.get(signature, -self.report_interval_seconds)
            if now - reported_at < self.report_interval_seconds:
                continue
            last_report[signature] = now
            logger.warning(
                "Gateway loop callback occupied the event loop for %.3fs (threshold %.3fs); Python stack:\n%s",
                now - current[1],
                self.threshold_seconds,
                "".join(traceback.format_stack(frame)).rstrip(),
            )


__all__ = ["SlowLoopStackMonitor"]
