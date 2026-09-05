"""guards.pre_write — enforce pre-investigation gate.

Blocks write_file/patch/execute_code(write) if the target file has not been
read (read_file/search_files) earlier in the same turn.

ACTIVATION: ON by default. Set PRE_WRITE_GUARD_DISABLE=1 to turn off.

STATE: Per-session keyed set of "files that have been read" and
"SQL schemas that have been confirmed". Scoped by session_id via
plugins._shared_state.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Set

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "pre_write_guard"


def _get_read_files(sid: str) -> Set[str]:
    return get_session_state(sid, _NAMESPACE).setdefault("read_files", set())


def _get_schema_confirmed(sid: str) -> Set[str]:
    """Set of connection identifiers whose schema has been confirmed."""
    return get_session_state(sid, _NAMESPACE).setdefault("schema_confirmed", set())


# ── Configuration ────────────────────────────────────────────────────

_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})
_READ_TOOLS = frozenset({"read_file", "search_files"})


def _plugin_disabled() -> bool:
    return os.environ.get("PRE_WRITE_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _contains_write(code: str) -> bool:
    """Check if execute_code code contains file-write operations."""
    write_indicators = (
        "write_file(", "patch(", ".write(", "open(",
        "pathlib.Path(", ".write_text(", ".write_bytes(",
    )
    lower = code.lower()
    return any(ind in lower for ind in write_indicators)


def _extract_paths_from_code(code: str) -> list:
    """Extract string literals that look like file paths from Python code."""
    import re  # noqa: R1 — import validation requires regex pattern matching
    # Match string literals (single or double quoted)
    patterns = re.findall(r'["\']([^"\']+)["\']', code)
    return [p for p in patterns if "/" in p or p.endswith(".py") or p.endswith(".md")]


def _paths_overlap(target: str, read_set: Set[str]) -> bool:
    """Check if target path was read or any parent/child was read."""
    target_norm = os.path.normpath(target)
    for read_path in read_set:
        read_norm = os.path.normpath(read_path)
        if target_norm == read_norm:
            return True
        if target_norm.startswith(read_norm + "/"):
            return True
        if read_norm.startswith(target_norm + "/"):
            return True
    return False


# ── post_tool_call hook ──────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Track which files have been read in this turn."""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")

    if tool_name in _READ_TOOLS:
        args = kwargs.get("args") or {}
        if tool_name == "read_file":
            path = args.get("path", "")
            if path:
                _get_read_files(sid).add(path)
        elif tool_name == "search_files":
            # Any search counts as investigation for the searched path
            path = args.get("path", ".")
            if path:
                _get_read_files(sid).add(path)

    # Also track schema confirmations
    if tool_name == "terminal":
        args = kwargs.get("args") or {}
        cmd = str(args.get("command", "")).lower()
        schema_cmds = ("\\dt", "\\d ", "information_schema", "show tables",
                       "describe ", "\\dt ", "\\d\\s")
        if any(sc in cmd for sc in schema_cmds):
            _get_schema_confirmed(sid).add("confirmed")


# ── pre_tool_call hook ────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Block write operations if the target hasn't been read first."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}

    if tool_name not in _WRITE_TOOLS:
        return None

    # Determine the target path(s)
    target_path = ""

    if tool_name == "write_file":
        target_path = args.get("path", "")
    elif tool_name == "patch":
        target_path = args.get("path", "")
    elif tool_name == "execute_code":
        code = args.get("code", "")
        if not _contains_write(code):
            return None  # read-only code, don't block
        paths = _extract_paths_from_code(code)
        if not paths:
            return None  # can't determine targets
        target_path = paths[0]  # check the first extracted path

    if not target_path:
        return None  # can't determine target, allow

    # Creating a brand-new file is always valid — only block modifications
    # to existing files that haven't been read yet.
    # Try both absolute and relative-to-cwd paths.
    import os as _os
    target_norm = _os.path.normpath(target_path)
    if not _os.path.exists(target_norm):
        # Try relative to cwd
        try:
            cwd = _os.getcwd()
            if _os.path.exists(_os.path.join(cwd, target_norm)):
                pass  # exists relative to cwd
            else:
                return None  # truly new file
        except Exception:  # noqa: D5 — file existence check, non-critical
            return None

    read_files = _get_read_files(sid)

    if _paths_overlap(target_path, read_files):
        return None  # file has been read, allow

    return {
        "action": "block",
        "message": (
            f"[PreWriteGuard] 预调查门禁拦截：write_file/patch/execute_code 前"
            f"必须先 read_file 或 search_files 读取目标文件。\n"
            f"  目标: {target_path}\n"
            f"  已读: {sorted(read_files) if read_files else '（无）'}\n"
            f"  修复: 先执行 read_file('{target_path}') 确认磁盘内容。"
        ),
    }


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("guards.pre_write registered")
