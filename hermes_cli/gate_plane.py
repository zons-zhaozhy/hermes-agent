"""Profile-scoped evidence and health reporting for Hermes policy guards."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
import time
from typing import Any, Callable, Iterable, Mapping

from hermes_cli.gate_plane_manual import (
    MANUAL_GATE_RECEIPT_VERSION,
    MANUAL_GATE_STATUSES,
)


GUARD_TAG = "guard"
GATE_ENFORCEMENT_HOOKS = frozenset({
    "pre_llm_call",
    "pre_tool_batch",
    "pre_tool_call",
    "pre_verify",
    "transform_llm_output",
})
GATE_OBSERVER_HOOKS = frozenset({
    "post_llm_call",
    "post_tool_call",
    "on_session_start",
    "on_session_reset",
    "on_session_end",
})
GATE_EVIDENCE_HOOKS = GATE_ENFORCEMENT_HOOKS | GATE_OBSERVER_HOOKS
GATE_SUITES: dict[str, tuple[str, ...]] = {
    "intent": ("pre_llm_call",),
    "execution": ("pre_tool_batch", "pre_tool_call"),
    "output": ("transform_llm_output",),
    "completion": ("pre_verify",),
    "integrity": (
        "post_llm_call", "post_tool_call",
        "on_session_start", "on_session_reset", "on_session_end",
    ),
}
GATE_SUITE_NAMES = frozenset(GATE_SUITES)
GATE_POLICIES = frozenset({"required", "advisory"})
GATE_KINDS = frozenset({"enforcement", "observer", "manual"})


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
        runtime_hooks = sorted(declared & GATE_EVIDENCE_HOOKS)
        registered = set(getattr(plugin, "hooks_registered", None) or ())
        missing_hooks = [hook for hook in runtime_hooks if hook not in registered]
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
        key = str(getattr(manifest, "key", None) or name)
        policy = str(getattr(manifest, "gate_policy", None) or "required")
        if policy not in GATE_POLICIES:
            policy = "required"
        kind = str(getattr(manifest, "gate_kind", None) or "")
        if kind not in GATE_KINDS:
            declared_kinds = set(runtime_hooks)
            kind = (
                "enforcement" if declared_kinds & GATE_ENFORCEMENT_HOOKS
                else "observer" if declared_kinds & GATE_OBSERVER_HOOKS
                else "manual"
            )
        if kind == "manual":
            active = False
        else:
            active = not reason
        rows.append({
            "name": str(name),
            "key": key,
            "hooks": runtime_hooks,
            "kind": kind,
            "active": active,
            "inactive_advisory": kind != "manual" and policy == "advisory" and not enabled,
            "reason": reason,
            "policy": policy,
            "gate_audit": dict(getattr(manifest, "gate_audit", None) or {}),
        })
    return rows


def _gate_item(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": row["name"],
        "reason": row["reason"],
        "hooks": row["hooks"],
        "policy": row["policy"],
    }


def _receipt_timestamp(value: Any) -> float | None:
    """Parse an ISO-8601 timestamp and require an explicit timezone."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.timestamp()


def _manual_item(
    row: Mapping[str, Any],
    evidence: Any = None,
    current_fingerprint: str | None = None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": row["name"],
        "policy": row["policy"],
        "reason": "no fresh audit evidence",
        "status": "unknown",
    }
    if row["reason"]:
        item["reason"] = row["reason"]
        return item
    declaration = row.get("gate_audit") or {}
    if not declaration:
        return item
    if evidence is None:
        return item
    if isinstance(evidence, Mapping) and evidence.get("_read_error"):
        item["reason"] = f"audit evidence unreadable: {evidence['_read_error']}"
        return item
    if not isinstance(evidence, Mapping):
        item["reason"] = "audit evidence malformed"
        return item

    metadata = {
        "subject": str(evidence.get("subject", "")),
        "checked_at": str(evidence.get("checked_at", "")),
        "fingerprint": str(evidence.get("fingerprint", "")),
    }
    item.update(metadata)
    if evidence.get("version") != MANUAL_GATE_RECEIPT_VERSION:
        item["reason"] = "audit evidence malformed"
        return item
    status = str(evidence.get("status", ""))
    if status not in MANUAL_GATE_STATUSES:
        item["reason"] = "audit evidence malformed"
        return item
    if metadata["subject"] != declaration.get("subject"):
        item["reason"] = "audit subject mismatch"
        return item
    checked_at = _receipt_timestamp(metadata["checked_at"])
    current_time = time.time() if now is None else float(now)
    if checked_at is None:
        item["reason"] = "audit evidence malformed"
        return item
    if current_time < checked_at:
        item["reason"] = "audit timestamp is in the future"
        return item
    ttl_seconds = int(declaration.get("ttl_seconds", 0))
    if current_time >= checked_at + ttl_seconds:
        item["reason"] = "audit evidence stale"
        return item
    if declaration.get("fingerprint") == "repo":
        if not current_fingerprint:
            item["reason"] = "current repo fingerprint unavailable"
            return item
        if not metadata["fingerprint"]:
            item["reason"] = "audit fingerprint missing"
            return item
        if metadata["fingerprint"] != current_fingerprint:
            item["reason"] = "audit fingerprint mismatch"
            return item
    if status == "pass":
        item["status"] = "pass"
        item["reason"] = ""
        return item
    item["status"] = "fail"
    item["reason"] = "latest audit failed" if status == "fail" else "latest audit errored"
    return item


def build_gate_report(
    plugins: Iterable[Any], plane: GatePlane, *, session_id: str | None = None,
    manual_evidence: Mapping[str, Any] | None = None, now: float | None = None,
    manual_fingerprints: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Combine declarations, registrations, and runtime evidence into one report."""
    rows = _guard_rows(plugins)
    runtime_rows = [row for row in rows if row["kind"] != "manual"]
    manual_rows = [row for row in rows if row["kind"] == "manual"]
    active_rows = [row for row in runtime_rows if row["active"]]
    events = plane.events(session_id)
    event_dicts = [event.as_dict() for event in events]

    suites: list[dict[str, Any]] = []
    for suite_name, hooks in GATE_SUITES.items():
        hook_set = set(hooks)
        if suite_name == "integrity":
            relevant_rows = [
                row for row in runtime_rows
                if not row["hooks"] or hook_set.intersection(row["hooks"])
            ]
            relevant_active = {
                row["name"] for row in relevant_rows if row["active"]
            }
            guards = [
                row["name"] for row in active_rows if row["name"] in relevant_active
            ]
            missing = [
                _gate_item(row)
                for row in relevant_rows
                if not row["active"] and not row["inactive_advisory"]
            ]
            inactive_advisory = [
                _gate_item(row)
                for row in relevant_rows
                if row["inactive_advisory"]
            ]
        else:
            relevant_rows = [
                row for row in runtime_rows if hook_set.intersection(row["hooks"])
            ]
            guards = [row["name"] for row in relevant_rows if row["active"]]
            missing = [
                _gate_item(row)
                for row in relevant_rows
                if not row["active"] and not row["inactive_advisory"]
            ]
            inactive_advisory = [
                _gate_item(row)
                for row in relevant_rows if row["inactive_advisory"]
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
        1 for row in runtime_rows if not row["active"] and not row["inactive_advisory"]
    )
    inactive_advisory_count = sum(1 for row in rows if row["inactive_advisory"])
    evidence_by_key = manual_evidence or {}
    fingerprints_by_key = manual_fingerprints or {}
    manual_guards = [
        _manual_item(
            row,
            evidence_by_key.get(row["key"]),
            fingerprints_by_key.get(row["key"]),
            now=now,
        )
        for row in manual_rows
    ]
    required_manual_unhealthy = any(
        item["policy"] == "required" and item["status"] != "pass"
        for item in manual_guards
    )

    return {
        "overall_status": (
            "healthy" if rows and unavailable_count == 0 and not required_manual_unhealthy
            and all(suite["status"] == "healthy" for suite in suites)
            else "unknown"
        ),
        "guard_count": len(rows),
        "active_guard_count": len(active_rows),
        "inactive_advisory_count": inactive_advisory_count,
        "manual_guards": manual_guards,
        "suites": suites,
        "events": event_dicts,
    }
