"""Content fingerprints for read-before-write freshness checks.

Hashline-style protocol (adapted from oh-my-pi's hashline, Apache-2.0):
``read_file`` stamps every successful read with a short content hash; the
model echoes it back on ``patch``/``write_file`` as ``expected_fingerprint``;
the writer rejects the edit when the file on disk no longer matches —
catching "edited from a stale mental copy" BEFORE the fuzzy matcher can
graft a patch onto the wrong content.

A per-task registry additionally records the last-seen fingerprint per
resolved path so the writer can WARN on staleness even when the model
doesn't pass the credential explicitly (transition-period enforcement:
warn, don't block).

Contract:
  Preconditions:  path exists and is readable text (callers check first).
  Postconditions: fingerprints are stable 12-hex strings — same bytes
                  always hash identically; line-ending/BOM normalization
                  keeps editor re-serialization from false-rejecting.
"""
from __future__ import annotations

import hashlib
import threading

_FP_LEN = 12
_REGISTRY_CAP = 200  # per-task entries; oldest evicted first (insertion order)

_registries: dict[str, dict[str, str]] = {}
_registry_lock = threading.Lock()


def content_fingerprint(text: str) -> str:
    """Stable short hash of normalized file content.

    Normalization: strip a leading UTF-8 BOM, unify CRLF/CR to LF — the
    same normalization patch_replace applies before matching, so the
    fingerprint tracks the bytes the writer actually compares against.
    """
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:_FP_LEN]


def fingerprint_matches(text: str, expected: str) -> bool:
    """True when ``expected`` is the fingerprint of ``text``'s current content."""
    return content_fingerprint(text) == expected


def record_read(task_id: str, resolved_path: str, fingerprint: str) -> None:
    """Remember the fingerprint the task last saw for ``resolved_path``."""
    with _registry_lock:
        reg = _registries.setdefault(task_id, {})
        reg.pop(resolved_path, None)  # move to end = most recent
        reg[resolved_path] = fingerprint
        while len(reg) > _REGISTRY_CAP:
            reg.pop(next(iter(reg)))


def last_seen(task_id: str, resolved_path: str) -> str | None:
    """Fingerprint recorded at this task's last full read, if any."""
    with _registry_lock:
        return _registries.get(task_id, {}).get(resolved_path)


def record_write(task_id: str, resolved_path: str, new_fingerprint: str) -> None:
    """After a successful write, refresh the registry so the NEXT edit of the
    post-write content doesn't false-warn (the writer itself has fresh state)."""
    record_read(task_id, resolved_path, new_fingerprint)


def drop_task(task_id: str) -> None:
    """Forget a task's registry entirely (session teardown)."""
    with _registry_lock:
        _registries.pop(task_id, None)
