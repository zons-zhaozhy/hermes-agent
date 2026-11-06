"""Durable receipt helpers for manual guards.

Runtime guard health is owned by :mod:`hermes_cli.gate_plane`. This module owns only the
profile-scoped audit receipt written by a manual guard's CLI and read by Gate Plane.
"""

from __future__ import annotations

from datetime import datetime, UTC
import logging
from typing import Any, Iterable


logger = logging.getLogger(__name__)

MANUAL_GATE_STATE_KEY = "gate_audit"
MANUAL_GATE_STATUSES = frozenset({"pass", "fail", "error"})
MANUAL_GATE_RECEIPT_VERSION = 1


class PluginManualGateCtxMixin:
    """Manual-guard receipt API mixed into the native plugin context."""

    manifest: Any

    @property
    def state(self) -> Any:
        """Implemented by :class:`hermes_cli.plugins.PluginContext`."""
        raise NotImplementedError

    def record_manual_gate_audit(
        self,
        *,
        status: str,
        subject: str,
        fingerprint: str = "",
        detail: str = "",
    ) -> dict[str, Any]:
        """Persist one standardized manual-guard audit receipt in this profile."""
        if self.manifest.gate_kind != "manual" or not self.manifest.gate_audit:
            raise ValueError(
                "manual audit receipts require a manual guard with a gate_audit declaration"
            )
        receipt = manual_gate_receipt(
            status=status,
            subject=subject,
            fingerprint=fingerprint,
            detail=detail,
        )
        self.state.set(MANUAL_GATE_STATE_KEY, receipt)
        return receipt


def manual_gate_receipt(
    status: str,
    subject: str,
    fingerprint: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Build one safe, JSON-serializable manual-audit receipt."""
    clean_status = str(status or "").strip().lower()
    if clean_status not in MANUAL_GATE_STATUSES:
        raise ValueError(
            "manual audit status must be one of "
            f"{', '.join(sorted(MANUAL_GATE_STATUSES))}"
        )
    clean_subject = str(subject or "").strip()
    if not clean_subject:
        raise ValueError("manual audit subject must be a non-empty string")
    return {
        "version": MANUAL_GATE_RECEIPT_VERSION,
        "status": clean_status,
        "subject": clean_subject,
        "checked_at": datetime.now(UTC).isoformat(),
        "fingerprint": str(fingerprint or "")[:200],
        "detail": str(detail or "")[:200],
    }


def load_manual_gate_evidence(plugins: Iterable[Any]) -> dict[str, Any]:
    """Read declared manual-guard receipts from their profile-scoped plugin state.

    A read failure is returned as ``{"_read_error": ...}`` so report construction can degrade
    visibly instead of crashing or inferring success.
    """
    evidence: dict[str, Any] = {}
    for plugin in plugins:
        manifest = getattr(plugin, "manifest", None)
        if getattr(manifest, "gate_kind", None) != "manual":
            continue
        if not getattr(manifest, "gate_audit", None):
            continue
        key = str(getattr(manifest, "key", None) or getattr(manifest, "name", "") or "unknown")
        try:
            from hermes_cli.plugins_state import PluginState

            state = PluginState(key, str(getattr(manifest, "data_namespace", "") or ""))
            evidence[key] = state.get(MANUAL_GATE_STATE_KEY)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Cannot read manual gate evidence for plugin %s: %s", key, exc,
            )
            evidence[key] = {"_read_error": str(exc)}
    return evidence
