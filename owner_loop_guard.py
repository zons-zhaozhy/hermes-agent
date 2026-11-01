"""Runtime tripwire for dangerous SessionDB calls on a registered owner event loop.

Static checks keep new direct SQLite calls out of gateway coroutines. This guard catches paths that
arrive dynamically (plugins, mixins, refactors): when the registered gateway loop thread performs a
SessionDB read/write, it records a rate-limited stack in ``warn`` mode and can fail fast in ``raise``
mode. Unregistered processes (CLI helpers, one-shot DB scripts) are not affected.
"""
from __future__ import annotations

import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)

_MODES = ("off", "warn", "raise")
_REPORT_INTERVAL_S = 60.0

_mode = "warn"
_owner_thread_id: int | None = None
_last_report: dict[tuple[str, str], float] = {}


class OwnerLoopIOError(RuntimeError):
    """A protected SessionDB operation ran on a registered owner event-loop thread."""


def register_owner_loop(*, mode: str = "warn") -> None:
    """Register the current thread as the process's gateway owner loop."""
    global _mode, _owner_thread_id
    normalized = str(mode or "warn").strip().lower()
    if normalized not in _MODES:
        raise ValueError(f"owner_loop_db_guard must be one of {_MODES}, got {mode!r}")
    _mode, _owner_thread_id = normalized, threading.get_ident()


def unregister_owner_loop() -> None:
    global _owner_thread_id
    _owner_thread_id = None


def owner_loop_db_guard(label: str) -> None:
    """Call at the entry to a protected SessionDB read/write primitive."""
    if _mode == "off" or _owner_thread_id is None or threading.get_ident() != _owner_thread_id:
        return
    frame = sys._getframe(1)
    site = f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}"
    if _mode == "raise":
        raise OwnerLoopIOError(f"{label} ran on the gateway owner loop at {site}")
    now = time.monotonic()
    key = (label, site)
    if now - _last_report.get(key, -_REPORT_INTERVAL_S) < _REPORT_INTERVAL_S:
        return
    _last_report[key] = now
    logger.warning(
        "owner-loop guard: %s ran directly on the gateway event loop at %s; move it behind "
        "asyncio.to_thread (rate-limited stack follows)\n%s",
        label, site, "".join(_format_stack(frame)),
    )


def _format_stack(frame) -> list[str]:
    import traceback
    return traceback.format_stack(frame)
