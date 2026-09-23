"""negative-conclusion-guard plugin — block "not found / doesn't exist" claims based on a single directory probe.

Enforces the rule: before concluding "X 不存在/不在线/未找到", the agent must
have consulted an authoritative entry point (hermes plugins list / gitnexus /
codegraph / get_bundled_plugins_dir()), not just a bare `ls`/`find` on one
directory.

Background (2026-08-12 血训): an agent diagnosed coding-standards-guard as
"not online" because it only ran `ls ~/.hermes/plugins/coding-standards-guard/`
and `find ~/.hermes -name coding-standards-guard`, never realizing the plugin
is a *bundled* plugin living in <hermes-repo>/plugins/ (a separate scan source).
The lesson generalizes: Hermes plugins are scanned from THREE independent
sources (bundled / user / project); a miss in one directory proves nothing.

ACTIVATION: ON by default. Set NEGATIVE_CONCLUSION_GUARD_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Set

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "negative_conclusion_guard"

# Regexes are FINITE sets of tool commands — same rationale as db-safety's
# SQL keyword whitelist. No free-form signal-word enumeration.

# Directory-probe commands that are NOT authoritative for existence checks:
# ls on a specific dir, find scoped to a dir, `test -e`, `stat`.
_PROBE_COMMANDS = re.compile(
    r"(^|\s)(ls(\s+-[a-zA-Z]+)*\s|find\s|stat\s|test\s+-[efd]\s|"
    r"\[?\s*-[efd]\s)",
    re.IGNORECASE,
)

# Authoritative entry points that CAN settle an existence question.
_AUTHORITATIVE_COMMANDS = re.compile(
    r"(hermes\s+plugins\s+(list|show|info)|"
    r"get_bundled_plugins_dir|"
    r"gitnexus|"
    r"codegraph|"
    r"hermes\s+config\s+get\s+plugins|"
    r"pip\s+(show|list)|"
    r"which\s+)",
    re.IGNORECASE,
)

# Tools that return structured "no matches" signals we can key on.
_ZERO_HIT_TOOLS = {"search_files"}

_MAX_PROBE_HITS_BEFORE_NUDGE = 2


def _plugin_disabled() -> bool:
    return os.environ.get("NEGATIVE_CONCLUSION_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _get_probe_count(sid: str) -> int:
    return int(get_session_state(sid, _NAMESPACE).get("probe_count", 0))


def _get_authority_seen(sid: str) -> bool:
    return bool(get_session_state(sid, _NAMESPACE).get("authority_seen", False))


def _get_last_probe_tool(sid: str) -> str:
    return str(get_session_state(sid, _NAMESPACE).get("last_probe_tool", ""))


def _mark_probe(sid: str, tool: str) -> None:
    st = get_session_state(sid, _NAMESPACE)
    st["probe_count"] = _get_probe_count(sid) + 1
    st["last_probe_tool"] = tool


def _mark_authority(sid: str) -> None:
    get_session_state(sid, _NAMESPACE)["authority_seen"] = True


# ── post_tool_call hook ────────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Record directory probes, authority use, and zero-hit searches."""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}
    result = kwargs.get("result") or {}
    status = kwargs.get("status") or ""

    # Authoritative command run → existence question can be settled.
    if tool_name == "terminal":
        cmd = str(args.get("command", ""))
        if _AUTHORITATIVE_COMMANDS.search(cmd):
            _mark_authority(sid)

    # Directory probe on terminal (ls/find/stat/test -e).
    if tool_name == "terminal":
        cmd = str(args.get("command", ""))
        if _PROBE_COMMANDS.search(cmd):
            _mark_probe(sid, "terminal")

    # search_files returning zero matches on a content/files search.
    if tool_name in _ZERO_HIT_TOOLS:
        total = result.get("total_count") if isinstance(result, dict) else None
        if total == 0:
            _mark_probe(sid, tool_name)
        elif total is None and status == "error":
            _mark_probe(sid, tool_name)


# ── pre_llm_call hook ──────────────────────────────────────────────────

_REMINDER = (
    "[NegativeConclusionGuard] 注意：本会话已出现 {n} 次\"目录探测/零命中搜索\"，"
    "且尚未调用任何权威验证入口。\n"
    "  断言\"X 不存在/不在线/未找到\"之前，必须先用权威入口验证：\n"
    "    • 插件是否在线？→ hermes plugins list | grep <name>（真实加载管线）\n"
    "    • bundled 插件实体？→ python3 -c \"import hermes_cli.plugins as p; "
    "print(p.get_bundled_plugins_dir())\"\n"
    "    • 代码/符号是否存在？→ gitnexus query / codegraph_context\n"
    "    • 只搜一个目录就断定\"不存在\"= 必错：bundled/user/project 三源独立扫描，"
    "一个目录 miss 证明不了全局。\n"
    "  若已用权威入口验证过，忽略本条。"
)


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Inject a reminder when probes accumulated without authority verification."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if _get_authority_seen(sid):
        return None
    n = _get_probe_count(sid)
    if n < _MAX_PROBE_HITS_BEFORE_NUDGE:
        return None

    return {"context": _REMINDER.format(n=n)}


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("negative-conclusion-guard registered")
