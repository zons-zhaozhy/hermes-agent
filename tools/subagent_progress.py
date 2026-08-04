"""
Subagent Progress Ledger — durable, compaction-resistant task tracking.

Inspired by Superpowers' progress.md pattern: conversation memory does not
survive context compression, but a file on disk does.  When a delegation batch
completes tasks, each result is appended here.  On resume after compaction,
the parent reads the ledger to skip already-finished tasks instead of
re-dispatching them — the single most expensive failure observed in
long-running delegation sessions (Superpowers SDD SKILL.md, "Durable Progress").

The ledger lives at ``<HERMES_HOME>/delegation_ledger.jsonl`` — one JSON object
per line, each representing a completed task.  This is a scratch file
(git-ignored, user-invisible) and is NOT a session store.

Design decisions:
- JSONL (not JSON array) so appends are O(1) — no need to read/parse/rewrite.
- Per-session keying: each delegation batch gets a unique ``ledger_session_id``
  so concurrent batches don't collide.
- Idempotent: appending the same task twice is harmless — consumers dedupe
  by (ledger_session_id, task_index).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_ledger_path() -> Path:
    """Return the absolute path to the delegation progress ledger."""
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    return hermes_home / "delegation_ledger.jsonl"


def append_task_completion(
    ledger_session_id: str,
    task_index: int,
    goal: str,
    status: str,
    summary: str,
    model: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    files_written: Optional[List[str]] = None,
) -> None:
    """Append a single task completion record to the ledger.

    Called after each child agent finishes (success or failure).  Failures
    are logged too — on resume we want to know the task was *attempted*, not
    re-dispatch it obliviously.

    Parameters
    ----------
    ledger_session_id : str
        Unique ID for this delegation batch.  Generated in ``delegate_task``
        and shared across all tasks in the batch.
    task_index : int
        Zero-based task index within the batch.
    goal : str
        The task goal (truncated for storage).
    status : str
        ``"completed"``, ``"failed"``, ``"interrupted"``, ``"timeout"``, or
        ``"error"``.
    summary : str
        The child agent's summary (truncated to 500 chars for the ledger).
    """
    try:
        record = {
            "ledger_session_id": ledger_session_id,
            "task_index": task_index,
            "goal": (goal or "")[:200],
            "status": status,
            "summary": (summary or "")[:500],
            "model": model,
            "duration_seconds": duration_seconds,
            "files_written": (files_written or [])[:20],
            "timestamp": time.time(),
        }
        ledger_path = _get_ledger_path()
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning(
            "Failed to append progress ledger entry for task %d",
            task_index,
            exc_info=True,
        )


def read_completed_tasks(ledger_session_id: str) -> Dict[int, Dict[str, Any]]:
    """Read completed tasks for a given ledger session.

    Returns a dict mapping ``task_index`` → record.  If the ledger file
    doesn't exist or is unreadable, returns an empty dict (safe degradation —
    the caller just re-dispatches all tasks, which is the pre-ledger behavior).

    Only tasks with status ``"completed"`` are returned.  Failed/errored
    tasks are NOT skipped — they should be retried on resume.
    """
    try:
        ledger_path = _get_ledger_path()
        if not ledger_path.exists():
            return {}
        completed: Dict[int, Dict[str, Any]] = {}
        seen: Set[int] = set()
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("subagent_progress: malformed JSONL line in ledger: %s", line[:100])
                    continue
                if record.get("ledger_session_id") != ledger_session_id:
                    continue
                idx = record.get("task_index")
                if not isinstance(idx, int):
                    continue
                # Last-write-wins: a later record for the same task_index
                # (e.g. first "failed", then "completed" on retry) overwrites.
                if record.get("status") == "completed":
                    completed[idx] = record
                elif idx in completed:
                    # A later non-completed record means the task was retried
                    # but hasn't succeeded yet — remove from completed.
                    del completed[idx]
        return completed
    except Exception:
        logger.warning("Failed to read progress ledger", exc_info=True)
        return {}


def clear_session(ledger_session_id: str) -> None:
    """Remove all records for a given ledger session.

    Called when a delegation batch finishes successfully and the caller
    wants to clean up.  Optional — the ledger is append-only and old
    sessions are harmless (they just take up space).  In practice this is
    rarely needed; the file is a scratch log.
    """
    try:
        ledger_path = _get_ledger_path()
        if not ledger_path.exists():
            return
        lines_to_keep: List[str] = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("subagent_progress: malformed JSONL line during clear: %s", line[:100])
                    continue
                if record.get("ledger_session_id") != ledger_session_id:
                    lines_to_keep.append(line)
        with open(ledger_path, "w", encoding="utf-8") as f:
            for line in lines_to_keep:
                f.write(line + "\n")
    except Exception:
        logger.warning(
            "Failed to clear progress ledger for session %s",
            ledger_session_id,
            exc_info=True,
        )
