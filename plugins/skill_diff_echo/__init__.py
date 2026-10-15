"""skill_manage result diff echo.

transform_tool_result hook: when skill_manage succeeds, append a plain-text
diff of every applied operation (old_string -> new_string for patches, head/tail
for full-file writes) to the tool result so the change is visible in-band
without a follow-up read. Fail-open: any internal error leaves the original
result untouched.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("plugins.skill_diff_echo")

_MAX_CHARS_PER_OP = 1200
_OP_KEYS = ("action", "name", "content", "file_path", "file_content",
            "old_string", "new_string", "replace_all")


def transform_tool_result(tool_name: str, args: dict[str, Any], result: Any, **_: Any) -> str | None:
    """Append an operation diff to a successful skill_manage result string."""
    try:
        if tool_name != "skill_manage" or not isinstance(args, dict) or not isinstance(result, str):
            return None
        parsed = json.loads(result)
        if not (isinstance(parsed, dict) and parsed.get("success")):
            return None
        sections = [s for s in (_diff_for_op(op) for op in _collect_ops(args)) if s]
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


def _clip(text: str, limit: int = _MAX_CHARS_PER_OP) -> str:
    """Clip a text block to limit chars, appending a marker when truncated."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit} chars)"


def _patch_body(old: str, new: str) -> str:
    """Render a patch operation as -old/+new lines."""
    return "".join(f"-{line}\n" for line in old.splitlines()) + \
           "".join(f"+{line}\n" for line in new.splitlines())


def _full_file_body(val: str) -> str:
    """Render a full-file write: all lines when short, head+tail+count when long."""
    lines = val.splitlines()
    if len(lines) > 10:
        return "".join(f"+{line}\n" for line in lines[:5]) + \
               f"...({len(lines)} lines total)\n" + \
               "".join(f"+{line}\n" for line in lines[-5:])
    return "".join(f"+{line}\n" for line in lines)


def _diff_for_op(op: dict[str, Any]) -> str:
    """Render one operation as a label plus its diff body."""
    action = op.get("action") or "?"
    label = op.get("name") or op.get("file_path") or "?"
    old, new = op.get("old_string"), op.get("new_string")
    if isinstance(old, str) and isinstance(new, str):
        return f"--- {label} ({action})\n{_clip(_patch_body(old, new))}"
    for key in ("content", "file_content"):
        if isinstance(op.get(key), str):
            body = _full_file_body(op[key])
            return f"--- {label} ({action}, {len(op[key].splitlines())} lines, new/overwritten)\n{_clip(body)}"
    if action in ("remove_file", "delete"):
        return f"--- {label} ({action}, file removed)"
    return ""


def register(ctx):
    """Plugin entry: register the tool-result transform hook."""
    ctx.register_hook("transform_tool_result", transform_tool_result)
    logger.info("skill_diff_echo v1 registered: skill_manage results carry inline diffs")
