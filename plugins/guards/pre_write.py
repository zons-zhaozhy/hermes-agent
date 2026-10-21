"""guards.pre_write — enforce pre-investigation gate.

Blocks write_file/patch/execute_code(write) if the target file has not been
read (read_file/search_files) earlier in the same session (the read-set is
session-persistent, not per-turn).

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


def _get_read_files(sid: str) -> set[str]:
    return get_session_state(sid, _NAMESPACE).setdefault("read_files", set())


def _get_schema_confirmed(sid: str) -> set[str]:
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
    import re  # re-ok: 字符串字面量提取,引号嵌套非str方法可解析 # noqa: R1
    # Match string literals (single or double quoted)
    patterns = re.findall(r'["\']([^"\']+)["\']', code)  # re-ok: 字符串字面量提取
    return [p for p in patterns if "/" in p or p.endswith(".py") or p.endswith(".md")]


def _norm(path: str) -> str:
    """Normalize a path to absolute form for cross-form comparison.

    read_file/search_files record paths exactly as passed (sometimes relative),
    while the pre_tool_call layer may hand us an already-resolved absolute
    target. Comparing raw strings makes the same file mismatch itself.
    """
    return os.path.normpath(os.path.abspath(path))


def _paths_overlap(target: str, read_set: set[str]) -> bool:
    """Check if target path was read or any parent/child was read."""
    target_norm = _norm(target)
    for read_path in read_set:
        read_norm = _norm(read_path)
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
                _get_read_files(sid).add(_norm(path))
        elif tool_name == "search_files":
            # Any search counts as investigation for the searched path
            path = args.get("path", ".")
            if path:
                _get_read_files(sid).add(_norm(path))

    # Also track schema confirmations
    if tool_name == "terminal":
        args = kwargs.get("args") or {}
        cmd = str(args.get("command", "")).lower()
        schema_cmds = ("\\dt", "\\d ", "information_schema", "show tables",
                       "describe ", "\\dt ", "\\d\\s")
        if any(sc in cmd for sc in schema_cmds):
            _get_schema_confirmed(sid).add("confirmed")


# ── pre_tool_call hook ────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """Block write operations if the target hasn't been read first."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}

    if tool_name not in _WRITE_TOOLS:
        return None

    # Determine the target path(s)
    targets: list[str] = []

    if tool_name == "write_file" or tool_name == "patch":
        _single = args.get("path", "")
        if _single:
            targets.append(_single)
    elif tool_name == "execute_code":
        code = args.get("code", "")
        if not _contains_write(code):
            return None  # read-only code, don't block
        # check ALL extracted paths — checking only paths[0] left writes to
        # paths[1:] unguarded (2026-10-21 audit)
        targets = _extract_paths_from_code(code)

    if not targets:
        return None  # can't determine target, allow

    # Creating a brand-new file is always valid — only block modifications
    # to existing files that haven't been read yet.
    # Try both absolute and relative-to-cwd paths.
    import os as _os
    read_files = _get_read_files(sid)

    unread_existing: list[str] = []
    for target_path in targets:
        target_norm = _os.path.normpath(target_path)
        if not _os.path.exists(target_norm):
            # Try relative to cwd
            try:
                cwd = _os.getcwd()
                if not _os.path.exists(_os.path.join(cwd, target_norm)):
                    continue  # truly new file
            except Exception:  # noqa: D5 — 探测失败按不存在处理,但留痕观测
                logger.warning("guards.pre_write: target existence check failed for %s", target_norm, exc_info=True)
                continue
        if not _paths_overlap(target_path, read_files):
            unread_existing.append(target_path)

    if not unread_existing:
        return None

    return {
        "action": "block",
        "message": (
            f"[PreWriteGuard] 预调查门禁拦截：write_file/patch/execute_code 前"
            f"必须先 read_file 或 search_files 读取目标文件。\n"
            f"  未读已存在目标: {unread_existing}\n"
            f"  已读: {sorted(read_files) if read_files else '（无）'}\n"
            f"  修复: 先执行 read_file('上述目标') 确认磁盘内容。"
        ),
    }


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("guards.pre_write registered")
