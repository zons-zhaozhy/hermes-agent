"""Durable receipt helpers for manual guards.

Runtime guard health is owned by :mod:`hermes_cli.gate_plane`. This module owns only the
profile-scoped audit receipt written by a manual guard's CLI and read by Gate Plane.
"""

from __future__ import annotations

from datetime import datetime, UTC
import hashlib
import logging
import subprocess
from pathlib import Path
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


def manual_repo_fingerprint(subject: str) -> str | None:
    """Return the payload-free git fingerprint for a ``repo:<path>`` subject."""
    if not subject.startswith("repo:"):
        return None
    repo = Path(subject[len("repo:"):].strip()).expanduser()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git failed")
        return result.stdout

    try:
        head = git("rev-parse", "HEAD").strip()
        worktree = git("status", "--porcelain").splitlines()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.warning("Manual gate repo fingerprint unavailable for %s: %s", repo, exc)
        return None
    digest = hashlib.sha256()
    digest.update(head.encode("utf-8"))
    for line in sorted(worktree):
        digest.update(line.encode("utf-8"))
    return f"git:{head}:worktree:{digest.hexdigest()[:16]}"


def load_manual_gate_fingerprints(plugins: Iterable[Any]) -> dict[str, str | None]:
    """Read current fingerprints for manual guards that declare repo binding."""
    fingerprints: dict[str, str | None] = {}
    seen_subjects: dict[str, str | None] = {}
    for plugin in plugins:
        manifest = getattr(plugin, "manifest", None)
        declaration = getattr(manifest, "gate_audit", None) or {}
        if getattr(manifest, "gate_kind", None) != "manual" or not declaration:
            continue
        if declaration.get("fingerprint") != "repo":
            continue
        key = str(getattr(manifest, "key", None) or getattr(manifest, "name", "") or "unknown")
        subject = str(declaration.get("subject", ""))
        if subject not in seen_subjects:
            seen_subjects[subject] = manual_repo_fingerprint(subject)
        fingerprints[key] = seen_subjects[subject]
    return fingerprints
