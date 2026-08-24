"""Memory write ledger — append-only history + rollback for MEMORY.md/USER.md.

Mirrors the proven pattern of ``tools/skill_ledger.py`` (in production since
the curator ledger, 460+ entries) for the built-in memory stores:

- Every mutation of ``~/.hermes/memories/MEMORY.md`` / ``USER.md`` records a
  ``before``/``after`` content snapshot (content-addressed blobs, deduped)
  into ``~/.hermes/memories/.memory_ledger.jsonl``.
- ``rollback_entry(entry_id)`` restores the ``before`` state and appends its
  own safety entry, so rollbacks are themselves undoable.
- NEVER raises into the mutation path: recording failures are logged and
  swallowed (``record_mutation`` returns ``None``). The one deliberate
  exception is ``rollback_entry``, which fails closed on validation errors —
  an explicit user-requested rollback must not silently no-op.

Contract:
    Preconditions:
        - store_blob/append happen under the caller's memory file lock where
          available (MemoryStore.save_to_disk holds the target file lock).
    Postconditions:
        - append_entry returns an entry id or None (failure logged, never raised).
        - rollback_entry restores before-files, removes files created by the
          entry, appends a safety entry, and returns (ok, message).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def ledger_path() -> Path:
    return _hermes_home() / "memories" / ".memory_ledger.jsonl"


def blobs_dir() -> Path:
    return _hermes_home() / ".memory_backups" / "blobs"


def _store_blob(data: bytes) -> Optional[str]:
    """Write *data* to the blob store keyed by its sha256 (deduped)."""
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    path = blobs_dir() / digest
    if path.exists():
        return digest
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return digest
    except OSError as e:
        logger.warning("memory_ledger: blob write failed (%s)", e)
        return None


def read_blob(sha256: str) -> Optional[bytes]:
    path = blobs_dir() / sha256
    try:
        return path.read_bytes()
    except OSError:
        return None


def _memory_file(target: str) -> Path:
    name = "USER.md" if target == "user" else "MEMORY.md"
    return _hermes_home() / "memories" / name


def _entry_id() -> str:
    return os.urandom(6).hex()


def append_entry(
    action: str,
    target: str,
    before: Optional[str],
    after: Optional[str],
    actor: Optional[str] = None,
) -> Optional[str]:
    """Append one ledger entry; returns its id or None on failure."""
    entry: Dict[str, Any] = {
        "id": _entry_id(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "target": target,
        "actor": actor or "agent",
    }
    for key, content in (("before", before), ("after", after)):
        if content is None:
            entry[key + "_missing"] = True
        else:
            digest = _store_blob(content.encode("utf-8"))
            if digest is None:
                return None
            entry[key] = digest
    try:
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry["id"]
    except OSError as e:
        logger.warning("memory_ledger: failed to append entry (%s)", e)
        return None


def record_mutation(
    action: str,
    target: str,
    before: Optional[str],
    after: Optional[str],
    actor: Optional[str] = None,
) -> Optional[str]:
    """One-stop hook for mutation call sites. NEVER raises, never blocks."""
    try:
        return append_entry(action, target, before, after, actor)
    except Exception as e:  # defensive: callers are in the hot write path
        logger.warning("memory_ledger: record_mutation failed (%s)", e)
        return None


def read_target(target: str) -> Optional[str]:
    """Current on-disk content of a memory target, or None if absent."""
    path = _memory_file(target)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning("memory_ledger: read %s failed (%s)", path, e)
        return None


def list_entries(
    limit: Optional[int] = None, target: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Newest-first ledger entries; malformed lines are skipped.

    Enriches each row with ``size`` (after-bytes) and ``delta`` (after minus
    before bytes) for display; both are None when a side is missing.
    """
    try:
        raw = ledger_path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and "id" in parsed:
                if target and parsed.get("target") != target:
                    continue
                rows.append(_with_sizes(parsed))
        except json.JSONDecodeError:
            continue  # a single bad append must not break history/rollback
    rows.reverse()
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def _with_sizes(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Augment an entry with after-bytes size and before/after byte delta."""
    sizes = {}
    for side in ("before", "after"):
        digest = entry.get(side)
        if digest and not entry.get(side + "_missing"):
            blob = read_blob(digest)
            sizes[side] = len(blob) if blob is not None else None
        else:
            sizes[side] = None
    entry["size"] = sizes["after"]
    entry["delta"] = (
        sizes["after"] - sizes["before"]
        if sizes["after"] is not None and sizes["before"] is not None
        else None
    )
    return entry


def get_entry(entry_id: str) -> Optional[Dict[str, Any]]:
    if not entry_id:
        return None
    for row in list_entries():
        if row.get("id") == entry_id:
            return row
    return None


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".ledger-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def rollback_entry(entry_id: str) -> Tuple[bool, str]:
    """Restore the ``before`` state of *entry_id*; fails closed on bad input.

    Appends a safety entry capturing the pre-rollback state first, so a
    rollback is itself undoable (same guarantee as skill_ledger).
    """
    entry = get_entry(entry_id)
    if entry is None:
        return False, f"no ledger entry with id {entry_id!r}"

    target = entry.get("target")
    if target not in {"memory", "user"}:
        return False, f"entry {entry_id} has invalid target {target!r}"

    before_digest = entry.get("before")
    if before_digest:
        data = read_blob(before_digest)
        if data is None:
            return False, f"before blob {before_digest} missing from blob store"
        restore = data.decode("utf-8")
    elif entry.get("before_missing"):
        # Rollback of a create: restore the pre-create state (file absent → empty).
        restore = ""
    else:
        return False, f"entry {entry_id} has no before snapshot to restore"

    # Safety snapshot of the current (pre-rollback) state.
    current = read_target(target)
    safety_id = append_entry("pre-rollback-safety", target, current, current)

    _atomic_write(_memory_file(target), restore)

    append_entry(
        "rollback",
        target,
        before=current,
        after=restore,
        actor="rollback",
    )
    msg = (
        f"rolled back entry {entry_id} ({entry.get('action')} on "
        f"{_memory_file(target).name}): content restored "
        f"({len(restore):,} bytes). Safety entry {safety_id} captured the "
        f"pre-rollback state."
    )
    return True, msg
