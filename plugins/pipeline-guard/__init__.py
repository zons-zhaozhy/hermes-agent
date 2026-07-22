"""pipeline-guard plugin — code-level stage gate for feature-dev-pipeline.

ACTIVATION MODEL
================
This plugin is OFF by default. It only activates when a pipeline session is
explicitly started:

1. Agent loads feature-dev-pipeline skill and calls the first phase update
2. Phase file appears at ~/.hermes/cache/pipeline_phase_<session>.json
3. From that point, write gate + done gate are active for that session
4. When session ends (phase 7 reached or file deleted), guard deactivates

No phase file = no guard = normal agent behavior. This prevents the guard
from blocking simple tasks (typo fixes, quick edits) that don't need the
pipeline.

PHASE STATE
===========
~/.hermes/cache/pipeline_phase_<session_id>.json:
    {"phase": 4, "active": true}

- phase: 1-7 current pipeline stage
- active: if missing/false, guard is inactive even if file exists
"""

from __future__ import annotations

import json
import logging
import os
import re  # noqa: session-id sanitization — regex is essential
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_WRITE_TOOLS = frozenset({"write_file", "patch"})

_ALWAYS_ALLOWED = frozenset({
    "read_file", "search_files", "web_search", "web_extract",
    "browser_navigate", "browser_snapshot", "browser_console",
    "browser_vision", "browser_get_images", "browser_scroll",
    "skill_view", "skills_list", "session_search", "todo",
    "vision_analyze", "clarify", "delegate_task",
    "mcp__codegraph__codegraph_search", "mcp__codegraph__codegraph_node",
    "mcp__codegraph__codegraph_context", "mcp__codegraph__codegraph_explore",
    "mcp__codegraph__codegraph_files", "mcp__codegraph__codegraph_status",
    "mcp__codegraph__codegraph_callees", "mcp__codegraph__codegraph_callers",
    "mcp__codegraph__codegraph_trace", "mcp__codegraph__codegraph_impact",
    "mcp__gitnexus__query", "mcp__gitnexus__context",
    "mcp__dbhub__execute_sql_aml_v7", "mcp__dbhub__execute_sql_aml_v8",
    "mcp__dbhub__execute_sql_b2b_trade", "mcp__dbhub__execute_sql_dbchat",
    "mcp__dbhub__execute_sql_default",
    "mcp__dbhub__search_objects_aml_v7", "mcp__dbhub__search_objects_aml_v8",
    "mcp__dbhub__search_objects_b2b_trade", "mcp__dbhub__search_objects_dbchat",
    "mcp__dbhub__search_objects_default",
})

_READONLY_CMD_PREFIXES = (
    "ls ", "cat ", "grep ", "rg ", "find ", "wc ",
    "git status", "git log", "git diff", "git show", "git branch",
    "python ", "python3 ",
    "echo ", "head ", "tail ", "file ",
    "npm run typecheck", "npm run lint",
)


def _cache_dir() -> Path:
    home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
    cache = home / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _phase_file(session_id: str) -> Path:
    sid = session_id or "default"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", sid)
    return _cache_dir() / f"pipeline_phase_{safe}.json"


def _read_phase_state(session_id: str) -> Optional[dict]:
    """Read phase state. Returns None if not in pipeline mode."""
    try:
        f = _phase_file(session_id)
        if f.exists():
            return json.loads(f.read_text())
    except Exception:
        logger.warning("[PipelineGuard] failed to read phase file", exc_info=True)
    return None


def _is_pipeline_active(session_id: str) -> bool:
    """Check if pipeline mode is active for this session."""
    state = _read_phase_state(session_id)
    if state is None:
        return False
    return bool(state.get("active", False))


def _is_readonly_terminal(args: Optional[Dict[str, Any]]) -> bool:
    if not args:
        return False
    cmd = str(args.get("command", "")).strip().lower()
    if not cmd:
        return False
    if "scripts/run_tests.sh" in cmd:
        return True
    return any(cmd.startswith(p) for p in _READONLY_CMD_PREFIXES)


def _unlock_hint(session_id: str) -> str:
    f = _phase_file(session_id)
    return (
        f"完成阶段 1-3 后，执行以下命令解锁写工具:\n"
        f"  python3 -c \"import json,pathlib; "
        f"p=pathlib.Path('{f}'); "
        f"p.parent.mkdir(parents=True,exist_ok=True); "
        f"p.write_text(json.dumps({{'phase':4,'active':True}}))\""
    )


# ── pre_tool_call hook ─────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Block write tools in pipeline stages 1-3. Only when pipeline is active."""
    tool_name = kwargs.get("tool_name", "")
    if tool_name in _ALWAYS_ALLOWED:
        return None

    session_id = kwargs.get("session_id", "") or kwargs.get("task_id", "")

    # KEY: if no phase file exists, pipeline mode is OFF → allow everything
    if not _is_pipeline_active(session_id):
        return None

    state = _read_phase_state(session_id)
    phase = int(state.get("phase", 1)) if state else 1

    # Terminal: allow read-only commands even in early stages
    if tool_name == "terminal":
        if _is_readonly_terminal(kwargs.get("args")):
            return None
        if phase < 4:
            return {
                "action": "block",
                "message": f"[Pipeline] 阶段 {phase}/3 禁止写操作。\n{_unlock_hint(session_id)}",
            }
        return None

    # Write tools in stages 1-3
    if tool_name in _WRITE_TOOLS and phase < 4:
        return {
            "action": "block",
            "message": f"[Pipeline] 阶段 {phase}/3 禁止写操作。\n{_unlock_hint(session_id)}",
        }

    return None


# ── pre_verify hook ────────────────────────────────────────────────────

def on_pre_verify(**kwargs) -> Optional[Dict[str, Any]]:
    """Block premature 'done' — force verification. Only when pipeline active."""
    session_id = kwargs.get("session_id", "")

    if not _is_pipeline_active(session_id):
        return None

    attempt = kwargs.get("attempt", 0)
    if attempt >= 3:
        return None

    coding = kwargs.get("coding", False)
    if not coding:
        return None

    state = _read_phase_state(session_id)
    phase = int(state.get("phase", 1)) if state else 1

    if phase < 5:
        return {
            "action": "continue",
            "message": (
                f"[Pipeline] 阶段 {phase}/7 — 还没到验证阶段。"
                f"声明阶段 5 并跑测试后才能完成。"
            ),
        }

    return None


# ── Registration ───────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("pre_verify", on_pre_verify)
    logger.info("pipeline-guard registered: inactive until pipeline session started")
