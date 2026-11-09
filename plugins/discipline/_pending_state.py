"""Session-scoped pending state shared by discipline tool-event reminders."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hermes_constants import get_scratch_dir

logger = logging.getLogger(__name__)
_PENDING_TTL_SECONDS = 60


def owner_id(kwargs: Mapping[str, Any]) -> str:
    """Return the hook's stable session identity, or ``""`` when absent."""
    return str(kwargs.get("session_id") or kwargs.get("task_id") or "")


def state_path(prefix: str, session_id: str) -> Path:
    """Return a per-session scratch path under the active Hermes home."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:20]
    return get_scratch_dir(prune=False) / f"{prefix}_{digest}.json"


def write_pending(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one session's pending payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(payload, handle)
        os.replace(tmp_name, path)
    except OSError:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def take_pending(path: Path, session_id: str) -> dict[str, Any] | None:
    """Load and remove pending state only when it belongs to this session."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        _remove(path)
        logger.warning("discipline pending state is unreadable (%s): %s", path.name, exc)
        return None

    if not isinstance(payload, dict) or payload.get("session_id") != session_id:
        _remove(path)
        logger.warning("discipline pending state has a foreign owner: %s", path.name)
        return None
    try:
        age = time.time() - float(payload.get("timestamp") or 0)
    except (TypeError, ValueError):
        _remove(path)
        logger.warning("discipline pending state has an invalid timestamp: %s", path.name)
        return None
    if age > _PENDING_TTL_SECONDS:
        _remove(path)
        return None

    _remove(path)
    return payload


def _remove(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("discipline pending state cleanup failed (%s): %s", path.name, exc)
