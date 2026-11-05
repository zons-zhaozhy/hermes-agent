"""Profile-scoped evidence and health reporting for Hermes policy guards."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import Any, Callable, Iterable, Mapping


GUARD_TAG = "guard"
GATE_HOOKS = frozenset({
    "pre_llm_call",
    "pre_tool_batch",
    "pre_tool_call",
    "pre_verify",
    "transform_llm_output",
})
GATE_SUITES: dict[str, tuple[str, ...]] = {
    "intent": ("pre_llm_call",),
    "execution": ("pre_tool_batch", "pre_tool_call"),
    "output": ("transform_llm_output",),
    "completion": ("pre_verify",),
    "integrity": ("on_session_start", "on_session_reset", "on_session_end"),
}
GATE_SUITE_NAMES = frozenset(GATE_SUITES)
GATE_POLICIES = frozenset({"required", "advisory"})


class GateStatus(str, Enum):
    """The observable outcome of one guard callback invocation."""

    PASS = "pass"
    BLOCK = "block"
    MODIFY = "modify"
    UNKNOWN = "unknown"


def callback_identity(callback: Callable) -> str:
    """Return a stable, payload-free callback identity for diagnostics."""
    module = getattr(callback, "__module__", "") or "unknown"
    qualname = getattr(callback, "__qualname__", None) or getattr(callback, "__name__", repr(callback))
    return f"{module}:{qualname}"


def result_status(result: Any) -> GateStatus:
    """Classify a hook result without retaining its payload."""
    if isinstance(result, Mapping):
        action = str(result.get("action", "")).lower()
        if action == "block":
            return GateStatus.BLOCK
        if action == "modify":
            return GateStatus.MODIFY
        if action in {"", "pass", "allow", "context"}:
            return GateStatus.PASS
        return GateStatus.UNKNOWN
    if result is None or isinstance(result, (str, bool, int, float, list, tuple, dict)):
        return GateStatus.PASS
    return GateStatus.UNKNOWN


@dataclass(frozen=True)
class GateEvent:
    """A safe diagnostic record for one policy callback execution."""

    hook: str
    callback: str
    session_id: str = ""
    call_id: str = ""
    status: GateStatus = GateStatus.PASS
    detail: str = ""
    duration_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "callback": self.callback,
            "session_id": self.session_id,
            "call_id": self.call_id,
            "status": self.status.value,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
        }


class GatePlane:
    """Bounded, thread-safe callback evidence owned by one PluginManager."""

    def __init__(self, *, max_events: int = 256) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._events: deque[GateEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def record(
        self,
        hook: str,
        callback: Callable,
        *,
        session_id: str = "",
        call_id: str = "",
        status: GateStatus = GateStatus.PASS,
        detail: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        event = GateEvent(
            hook=hook,
            callback=callback_identity(callback),
            session_id=str(session_id or ""),
            call_id=str(call_id or ""),
            status=status,
            detail=str(detail)[:200],
            duration_ms=max(0.0, float(duration_ms)),
        )
        with self._lock:
            self._events.append(event)

    def events(self, session_id: str | None = None) -> tuple[GateEvent, ...]:
        with self._lock:
            events = tuple(self._events)
        if session_id is None:
            return events
        return tuple(event for event in events if event.session_id == session_id)


def _guard_rows(plugins: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plugin in plugins:
        manifest = getattr(plugin, "manifest", None)
        tags = set(getattr(manifest, "tags", None) or ())
        if GUARD_TAG not in tags:
            continue
        declared = set(getattr(manifest, "provides_hooks", None) or ())
        gate_hooks = sorted(declared & GATE_HOOKS)
        registered = set(getattr(plugin, "hooks_registered", None) or ())
        missing_hooks = [hook for hook in gate_hooks if hook not in registered]
        error = getattr(plugin, "error", None)
        enabled = bool(getattr(plugin, "enabled", False))
        if not enabled:
            reason = str(error or "not enabled")
        elif error:
            reason = str(error)
        elif missing_hooks:
            reason = "declared hooks not registered: " + ", ".join(missing_hooks)
        else:
            reason = ""
        name = getattr(manifest, "name", None) or getattr(manifest, "key", "") or "unknown"
        policy = str(getattr(manifest, "gate_policy", None) or "required")
        if policy not in GATE_POLICIES:
            policy = "required"
        rows.append({
            "name": str(name),
            "hooks": gate_hooks,
            "active": not reason,
            "inactive_advisory": policy == "advisory" and not enabled,
            "reason": reason,
            "policy": policy,
        })
    return rows


def _gate_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "reason": row["reason"],
        "hooks": row["hooks"],
        "policy": row["policy"],
    }


def build_gate_report(
    plugins: Iterable[Any], plane: GatePlane, *, session_id: str | None = None,
) -> dict[str, Any]:
    """Combine declarations, registrations, and runtime evidence into one report."""
    rows = _guard_rows(plugins)
    active_rows = [row for row in rows if row["active"]]
    events = plane.events(session_id)
    event_dicts = [event.as_dict() for event in events]

    suites: list[dict[str, Any]] = []
    for suite_name, hooks in GATE_SUITES.items():
        hook_set = set(hooks)
        if suite_name == "integrity":
            guards = [row["name"] for row in active_rows]
            missing = [
                _gate_item(row)
                for row in rows
                if not row["active"] and not row["inactive_advisory"] and not row["hooks"]
            ]
            inactive_advisory = [
                _gate_item(row)
                for row in rows
                if row["inactive_advisory"] and not row["hooks"]
            ]
        else:
            guards = [
                row["name"] for row in active_rows
                if hook_set.intersection(row["hooks"])
            ]
            missing = [
                _gate_item(row)
                for row in rows
                if not row["active"] and not row["inactive_advisory"]
                and hook_set.intersection(row["hooks"])
            ]
            inactive_advisory = [
                _gate_item(row)
                for row in rows
                if row["inactive_advisory"] and hook_set.intersection(row["hooks"])
            ]
        runtime_unknown = sum(
            1 for event in events
            if event.hook in hook_set and event.status is GateStatus.UNKNOWN
        )
        status = "healthy" if guards and not missing and runtime_unknown == 0 else "unknown"
        suites.append({
            "name": suite_name,
            "hooks": list(hooks),
            "status": status,
            "guards": guards,
            "missing": missing,
            "inactive_advisory": inactive_advisory,
            "runtime_unknown": runtime_unknown,
        })

    unavailable_count = sum(
        1 for row in rows if not row["active"] and not row["inactive_advisory"]
    )
    inactive_advisory_count = sum(1 for row in rows if row["inactive_advisory"])

    return {
        "overall_status": (
            "healthy" if rows and unavailable_count == 0
            and not any(suite["missing"] or suite["runtime_unknown"] for suite in suites)
            else "unknown"
        ),
        "guard_count": len(rows),
        "active_guard_count": len(active_rows),
        "inactive_advisory_count": inactive_advisory_count,
        "suites": suites,
        "events": event_dicts,
    }
