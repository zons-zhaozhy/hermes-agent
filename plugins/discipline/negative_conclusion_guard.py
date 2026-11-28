"""discipline.negative_conclusion_guard — block "not found / doesn't exist" claims based on a single directory probe.

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
from typing import Any, Optional

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
# 提醒一次即清零探测计数, 所以计数只代表「上一轮之后」的探测; 一场封顶防反复打扰
_MAX_INJECTS = 3


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
        if total == 0 or total is None and status == "error":
            _mark_probe(sid, tool_name)


# ── pre_llm_call hook ──────────────────────────────────────────────────

_REMINDER = (
    "[否定结论先核实] 上一轮找过 {n} 次东西都没找到。要下「某某不存在/不在线」的结论之前，"
    "先用权威入口确认：\n"
    "  • 插件在不在 → hermes plugins list\n"
    "  • 代码里有没有 → gitnexus / codegraph\n"
    "  • 自带插件在哪 → get_bundled_plugins_dir()\n"
    "  只搜了一个目录不能下结论（插件分自带、用户、项目三处存放）。\n"
    "  这一轮没下这种结论，或已经确认过，不用管这条。"
)


def on_pre_llm_call(**kwargs) -> Optional[dict[str, Any]]:
    """Inject a reminder when probes accumulated without authority verification.

    Contract:
      Postconditions: 探测满两次且未核实 → 提醒一次并把探测计数清零; 一场最多
        _MAX_INJECTS 次; 已核实/未满两次/超封顶返回 None。
    """
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if _get_authority_seen(sid):
        return None
    st = get_session_state(sid, _NAMESPACE)
    n = int(st.get("probe_count", 0))
    if n < _MAX_PROBE_HITS_BEFORE_NUDGE:
        return None
    if int(st.get("inject_count", 0)) >= _MAX_INJECTS:
        return None

    st["probe_count"] = 0
    st["inject_count"] = int(st.get("inject_count", 0)) + 1
    return {"context": _REMINDER.format(n=n)}


# ── Registration ──────────────────────────────────────────────────────

def register(ctx: Any) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("discipline.negative_conclusion_guard registered")
