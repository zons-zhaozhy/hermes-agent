"""skill_manage result diff echo (v2: unified diff from disk before/after).

Hooks:
  pre_tool_call          snapshot target file contents on disk BEFORE skill_manage runs
  transform_tool_result  on success, append a standard unified diff (difflib, @@ hunks)
                         of what actually changed on disk

v1 behavior (old/new string echo rebuilt from call args) survives as the
fallback when no snapshot is available (hook miss, unresolvable skill path).
Fail-open: any internal error leaves the original result untouched.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("plugins.skill_diff_echo")

_MAX_CHARS_PER_OP = 2000
_WRITE_ACTIONS = {"patch", "edit", "create", "write_file", "remove_file", "delete"}
_OP_KEYS = ("action", "name", "content", "file_path", "file_content",
            "old_string", "new_string", "replace_all")

# (op_index, resolved_target_path) -> disk content before the call ("" = file
# absent). Populated by pre_tool_call, consumed once by transform_tool_result.
_snapshots: dict[tuple[int, str], str] = {}


def on_pre_tool_call(**kwargs: Any) -> None:
    """Snapshot target files on disk before skill_manage mutates them.

    Contract:
        Preconditions — kwargs carry tool_name/args; only skill_manage write actions snapshot.
        Postconditions — _snapshots holds pre-call disk text (or "" for absent files);
        resolution failure stores nothing (transform falls back to v1 echo).
    """
    try:
        if kwargs.get("tool_name") != "skill_manage":
            return
        ops = _collect_ops(kwargs.get("args") or {})
        for idx, op in enumerate(ops):
            if op.get("action") not in _WRITE_ACTIONS:
                continue
            target = _target_path(op)
            if target is None:
                continue
            _snapshots[(idx, str(target))] = (target.read_text(encoding="utf-8-sig")
                                              if target.exists() else "")
    except Exception:
        logger.warning("skill_diff_echo pre-snapshot failed", exc_info=True)


def transform_tool_result(tool_name: str, args: dict[str, Any], result: Any, **_: Any) -> str | None:
    """Append a unified diff of actual disk changes to a successful skill_manage result.

    Contract:
        Postconditions — success result keeps its prefix and gains one diff section
        per write op; ops without a snapshot fall back to the v1 args echo; any
        failure returns None (original result untouched).
    """
    try:
        if tool_name != "skill_manage" or not isinstance(args, dict) or not isinstance(result, str):
            return None
        parsed = json.loads(result)
        if not (isinstance(parsed, dict) and parsed.get("success")):
            return None
        sections = [s for s in (_diff_for_op(op, idx) for idx, op in enumerate(_collect_ops(args))) if s]
        if not sections:
            return None
        return result + "\n\n[skill_diff] " + str(len(sections)) + " op(s):\n" + "\n".join(sections)
    except Exception:
        logger.warning("skill_diff_echo failed, returning original result", exc_info=True)
        return None


def _collect_ops(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize flat single-op args and the atomic batch shape into one list."""
    if isinstance(args.get("operations"), list) and args["operations"]:
        return [op for op in args["operations"] if isinstance(op, dict)]
    return [{k: args.get(k) for k in _OP_KEYS if k in args}]


def _target_path(op: dict[str, Any],
                 resolver: Callable[[str], dict[str, Any] | None] | None = None) -> Path | None:
    """Resolve an op's on-disk target: skill_dir/SKILL.md or the supporting file.

    Contract:
        Postconditions — returns the absolute target Path, or None when the skill
        name does not resolve (injectable resolver keeps this unit-testable).
    """
    name = op.get("name") or ""
    if not name:
        return None
    if resolver is None:
        from tools.skill_manager_tool import _find_skill
        resolver = _find_skill
    found = resolver(name)
    if not found or not found.get("path"):
        return None
    skill_dir = Path(found["path"])
    rel = op.get("file_path")
    if rel and op.get("action") != "create":
        return skill_dir / rel
    return skill_dir / "SKILL.md"


def _unified(snapshot: str, current: str, label: str) -> str:
    """Render difflib.unified_diff without the ---/+++ header lines (label shown separately)."""
    lines = list(difflib.unified_diff(
        snapshot.splitlines(keepends=False),
        current.splitlines(keepends=False),
        fromfile=label + " (before)",
        tofile=label + " (after)",
        n=2))
    if len(lines) <= 2:
        return "(no textual change on disk)\n"
    return "".join(ln + "\n" for ln in lines[2:])


def _clip(text: str, limit: int = _MAX_CHARS_PER_OP) -> str:
    """Clip a text block to limit chars, appending a marker when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit} chars)"


def _diff_for_op(op: dict[str, Any], idx: int) -> str:
    """One op -> unified diff from its pre-call snapshot, or the v1 args echo."""
    action = op.get("action") or "?"
    label = op.get("name") or op.get("file_path") or "?"
    if action in _WRITE_ACTIONS:
        target = _target_path(op)
        snap = None
        if target is not None:
            snap = _snapshots.pop((idx, str(target)), None)
            if snap is not None:
                current = (target.read_text(encoding="utf-8-sig") if target.exists() else "")
                body = _unified(snap, current, str(target))
                return f"--- {label} ({action}, unified diff disk before/after)\n{_clip(body)}"
    return _fallback_echo(op, action, label)


def _echo_full_file(text: str) -> str:
    """Full-file echo body: head+tail+count when long, every line when short."""
    lines = text.splitlines()
    if len(lines) > 10:
        return ("".join(f"+{ln}\n" for ln in lines[:5]) + f"...({len(lines)} lines total)\n" +
                "".join(f"+{ln}\n" for ln in lines[-5:]))
    return "".join(f"+{ln}\n" for ln in lines)


def _fallback_echo(op: dict[str, Any], action: str, label: str) -> str:
    """v1 behavior: echo the call args when no disk snapshot is available."""
    old, new = op.get("old_string"), op.get("new_string")
    if isinstance(old, str) and isinstance(new, str):
        body = "".join(f"-{ln}\n" for ln in old.splitlines()) + \
               "".join(f"+{ln}\n" for ln in new.splitlines())
        return f"--- {label} ({action}, fallback args echo)\n{_clip(body)}"
    for key in ("content", "file_content"):
        if isinstance(op.get(key), str):
            n = len(op[key].splitlines())
            return f"--- {label} ({action}, {n} lines, fallback full-file echo)\n{_clip(_echo_full_file(op[key]))}"
    if action in ("remove_file", "delete"):
        return f"--- {label} ({action}, file removed, no snapshot)"
    return ""


def register(ctx: Any) -> None:
    """Plugin entry: register the pre-call snapshot and tool-result transform hooks."""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("transform_tool_result", transform_tool_result)
    logger.info("skill_diff_echo v2 registered: skill_manage results carry unified diffs")
