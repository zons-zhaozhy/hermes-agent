"""Consumption governors and parking decisions for persistent goals."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_cli.goals import GoalState

logger = logging.getLogger(__name__)

DEFAULT_MAX_API_CALLS_PER_GOAL = 120
DEFAULT_MAX_CONCURRENT_GOALS = 2
# A goal counts toward the concurrency cap only while it actually drove turns recently: dormant
# active rows (set long ago, never evaluated) must not hold slots forever.
_CONCURRENCY_ACTIVE_WINDOW_S = 3600.0


def _goals_int_setting(key: str, default: int) -> int:
    """Resolve top-level ``goals.<key>``; 0 disables the governor, garbage falls back to
    ``default`` rather than crashing the loop. ``load_config()`` is cached on (mtime, size)."""
    try:
        from hermes_cli.config import load_config

        value = int(((load_config() or {}).get("goals") or {}).get(key, default))
        if value >= 0:
            return value
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("invalid goals.%s value; using default %s: %s", key, default, exc)
    return default


def _session_api_call_count(session_id: str) -> int:
    """The session's cumulative API-call counter (exact: get_session drains queued deltas)."""
    from hermes_cli.goals import _get_session_db

    db = _get_session_db()
    if db is None:
        return 0
    try:
        row = db.get_session(session_id)
        return int((row or {}).get("api_call_count") or 0)
    except Exception as exc:
        logger.warning(
            "goal api-call counter read failed for %s: %s", session_id, exc, exc_info=True
        )
        return 0


def _active_goal_entries() -> list[tuple[float, float, str]]:
    """``(created_at, last_turn_at, session_id)`` for ACTIVE goals that drove a turn inside the
    concurrency window — dormant active rows never hold a slot. Deterministic order keeps the
    keep-set stable: (created_at, session_id)."""
    from hermes_cli.goals import _get_session_db

    db = _get_session_db()
    if db is None:
        return []
    try:
        rows = db.list_meta_prefix("goal:")
    except Exception as exc:
        logger.warning("goal scan failed: %s", exc, exc_info=True)
        return []
    cutoff = time.time() - _CONCURRENCY_ACTIVE_WINDOW_S
    entries: list[tuple[float, float, str]] = []
    for key, raw in rows:
        try:
            data = json.loads(raw)
            created_at = float(data.get("created_at") or 0.0)
            last_turn_at = float(data.get("last_turn_at") or 0.0)
            if data.get("status") != "active" or max(created_at, last_turn_at) < cutoff:
                continue
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        entries.append((created_at, last_turn_at, key[len("goal:"):]))
    return entries


class GoalParkingMixin:
    """Pause-state helpers shared by every GoalManager continuation decision."""

    def _pause_state(self, reason: str) -> None:
        self._state.status = "paused"  # type: ignore[attr-defined]
        self._state.paused_reason = reason  # type: ignore[attr-defined]
        self._save()  # type: ignore[attr-defined]

    def _pause_decision(
        self, paused_reason: str, verdict: str, reason: str, message: str
    ) -> dict[str, Any]:
        from hermes_cli.goals import _decision

        self._pause_state(paused_reason)
        return _decision("paused", False, None, verdict, reason, message)

    def _budget_pause(
        self, state: GoalState, verdict: str, reason: str, note: str = ""
    ) -> dict[str, Any]:
        return self._pause_decision(
            f"turn budget exhausted ({state.turns_used}/{state.max_turns})",
            verdict,
            reason,
            f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used{note}. "
            "Use /goal resume to keep going, or /goal clear to stop.",
        )

    def _maybe_park_for_consumption(
        self, state: GoalState, reason: str
    ) -> Optional[dict[str, Any]]:
        """Park instead of continuing when a consumption governor trips.

        Both governors gate CONTINUATION only — a done/blocked verdict is decided before this
        runs, so a completed goal is never re-parked. The baseline anchors lazily (persisted even
        when no governor trips) so goals persisted before the governor exists are not billed for
        history, and every later evaluation measures real usage since then.

        Contract:
          Preconditions: the goal would otherwise continue (judge or a quality gate said so).
          Postconditions: never raises; returns a pause decision or None; usage reads that fail
          degrade to "no budget evidence" (fail-open) and are logged.
        """
        if self._max_api_calls > 0:  # type: ignore[attr-defined]
            current = _session_api_call_count(self.session_id)  # type: ignore[attr-defined]
            if state.api_calls_baseline is None:
                state.api_calls_baseline = current
                self._save()  # type: ignore[attr-defined]
            used = max(0, current - state.api_calls_baseline)
            if used >= self._max_api_calls:  # type: ignore[attr-defined]
                return self._pause_decision(
                    f"goal API-call budget exhausted ({used}/{self._max_api_calls} calls since the goal was set)",
                    "continue",
                    reason,
                    f"⏸ Goal paused — API-call budget used ({used}/{self._max_api_calls}). "
                    "Use /goal resume to reset it and keep going, or raise goals.max_api_calls "
                    "in config.yaml (0 disables it).",
                )
        if self._max_concurrent > 0:  # type: ignore[attr-defined]
            actives = _active_goal_entries()
            if len(actives) > self._max_concurrent:  # type: ignore[attr-defined]
                # Oldest goals keep their slots; a deterministic (created_at, session id) order
                # means simultaneous evaluations reach the same keep-set, not all-parked.
                ranked = sorted((created, sid) for created, _last, sid in actives)
                keep = {sid for _created, sid in ranked[: self._max_concurrent]}  # type: ignore[attr-defined]
                if self.session_id not in keep:  # type: ignore[attr-defined]
                    return self._pause_decision(
                        f"goal concurrency cap ({len(actives)} active goals, goals.max_concurrent={self._max_concurrent})",
                        "continue",
                        reason,
                        f"⏸ Goal paused — {len(actives)} goals are active (cap "
                        f"{self._max_concurrent}). Use /goal resume when a slot frees, or raise "
                        "goals.max_concurrent in config.yaml (0 disables it).",
                    )
        return None
