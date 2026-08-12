"""Shared session-keyed state store for discipline plugins.

Replaces per-plugin ``threading.local()`` with a ``session_id``-keyed
global dictionary. This provides correct isolation between parent agent
and subagents (which share the same thread pool).

Each plugin maintains its own namespace within the store via a prefix,
so no cross-plugin state leakage occurs.

Usage::

    from plugins._shared_state import get_session_state

    # post_tool_call: record that a file was read
    state = get_session_state(session_id)
    state.setdefault("read_files", set()).add(path)

    # pre_tool_call: check if a file was read
    state = get_session_state(session_id)
    if path not in state.get("read_files", set()):
        return {"action": "block", ...}

State is automatically pruned after MAX_SESSION_IDLE turns to prevent
unbounded memory growth.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

# session_id -> {namespace -> value}
_sessions: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()

# Maximum idle time (seconds) before a session's state is pruned.
_MAX_SESSION_IDLE = 3600  # 1 hour

# Global counter: incremented every time any state is accessed, used for
# idle detection.
_last_access: Dict[str, float] = {}


def get_session_state(session_id: str, namespace: str = "") -> Dict[str, Any]:
    """Get or create the state dict for a session+namespace.

    Args:
        session_id: The Hermes session ID (from pre_tool_call kwargs).
        namespace: Optional namespace prefix to isolate plugin state.
                   If empty, returns the full session dict.

    Returns:
        Mutable dict for this session+namespace.
    """
    if not session_id:
        session_id = "__default__"
    with _lock:
        now = time.monotonic()
        _last_access[session_id] = now
        if session_id not in _sessions:
            _sessions[session_id] = {}
        _prune_idle(now)
        if namespace:
            bucket = _sessions[session_id]
            if namespace not in bucket:
                bucket[namespace] = {}
            return bucket[namespace]
        return _sessions[session_id]


def clear_session(session_id: str) -> None:
    """Remove all state for a session. Called at session end."""
    with _lock:
        _sessions.pop(session_id, None)
        _last_access.pop(session_id, None)


def _prune_idle(now: float) -> None:
    """Remove sessions idle longer than _MAX_SESSION_IDLE."""
    if len(_sessions) < 100:  # skip unless large
        return
    expired = [
        sid for sid, ts in _last_access.items()
        if now - ts > _MAX_SESSION_IDLE
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _last_access.pop(sid, None)
    if expired:
        logger.debug("_shared_state: pruned %d idle sessions", len(expired))
