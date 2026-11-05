"""Shared session-platform policy for guard state."""

from __future__ import annotations

from threading import RLock
from typing import Any


_MAX_SESSIONS = 512
_SESSION_PLATFORMS: dict[str, str] = {}
_LOCK = RLock()


def record_session_platform(
    session_id: str = "", platform: str = "", **_kwargs: Any
) -> None:
    """Record the platform field the cron exemptions key on.

    Contract:
      Postconditions: empty session IDs record nothing; the map is bounded and
      eviction only removes the oldest half when over capacity.
    """
    sid = str(session_id or "")
    if not sid:
        return
    with _LOCK:
        _SESSION_PLATFORMS[sid] = str(platform or "")
        if len(_SESSION_PLATFORMS) > _MAX_SESSIONS:
            for stale in list(_SESSION_PLATFORMS)[: len(_SESSION_PLATFORMS) // 2]:
                del _SESSION_PLATFORMS[stale]


def forget_session(session_id: str = "", **_kwargs: Any) -> None:
    """Drop one session's platform record on reset or end."""
    with _LOCK:
        _SESSION_PLATFORMS.pop(str(session_id or ""), None)


def is_cron_session(session_id: str = "") -> bool:
    """Return whether a session is cron-run, including the legacy prefix."""
    sid = str(session_id or "")
    with _LOCK:
        return sid.startswith("cron_") or _SESSION_PLATFORMS.get(sid) == "cron"


def register(ctx: Any) -> None:
    """Register the policy hooks once for the guards suite."""
    ctx.register_hook("pre_llm_call", record_session_platform)
    ctx.register_hook("on_session_reset", forget_session)
    ctx.register_hook("on_session_end", forget_session)
