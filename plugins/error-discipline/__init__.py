"""error-discipline plugin — enforce log-first diagnosis.

Blocks repeated trial-and-error tool calls when the agent is firing similar
commands without having read/analyzed the error message from the previous failure.

Detects:
- Consecutive terminal calls to different URLs (URL enumeration)
- Consecutive config file modifications (parameter enumeration)
- Consecutive similar commands with only minor parameter changes

ACTIVATION: ON by default. Set ERROR_DISCIPLINE_DISABLE=1 to turn off.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "error_discipline"

_MAX_RECENT = 10  # max recent tool calls to track per session
_MAX_CONSECUTIVE_ERRORS = 2  # block after N consecutive errors without diagnosis


def _get_recent_calls(sid: str) -> List[Dict[str, Any]]:
    return get_session_state(sid, _NAMESPACE).setdefault("recent_calls", [])


def _get_last_error_read(sid: str) -> bool:
    """Whether the agent has read/analyzed an error since the last error occurred."""
    return bool(get_session_state(sid, _NAMESPACE).get("last_error_read", False))


def _set_last_error_read(sid: str, val: bool):
    get_session_state(sid, _NAMESPACE)["last_error_read"] = val


def _get_error_count(sid: str) -> int:
    return get_session_state(sid, _NAMESPACE).get("consecutive_error_count", 0)


def _set_error_count(sid: str, val: int):
    get_session_state(sid, _NAMESPACE)["consecutive_error_count"] = val


# ── Configuration ────────────────────────────────────────────────────

_DIAGNOSIS_TOOLS = frozenset({
    "read_file", "search_files", "browser_console",
    "terminal",  # when used for reading logs (grep, cat, tail)
})

_CURL_PATTERN = "curl "

_READONLY_PREFIXES = (
    "cat ", "grep ", "rg ", "head ", "tail ", "less ", "more ",
    "docker logs", "journalctl", "find ", "ls ", "file ",
)

# Exit codes that are NOT code bugs — network/environment/grep-no-match
_NON_ERROR_EXIT_CODES = frozenset({124, 127})

# Commands where exit_code=1 means "no match" (normal, not an error)
_NO_MATCH_EXIT1_PREFIXES = ("grep ", "rg ", "fgrep ", "egrep ")

_NETWORK_ERROR_PHRASES = (
    "connection refused", "no such host", "could not resolve",
    "timed out", "name or service not known",
)


def _plugin_disabled() -> bool:
    return os.environ.get("ERROR_DISCIPLINE_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _extract_exit_code(text: str) -> int | None:
    """Extract exit_code from terminal result.

    Contract:
      Preconditions: text is a string (may be empty)
      Postconditions: returns int or None (None if not found)

    Handles JSON ({"exit_code": 1}), Python repr ({'exit_code': 1}),
    and simple format (exit_code=1). Returns None if not found.
    """
    if not text:
        return None
    # Try JSON parse first (no exception — validate before parse)
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "exit_code" in obj:
                return int(obj["exit_code"])
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("exit_code extraction: malformed JSON in result: %s", exc)
    # Try repr format: {'exit_code': 1}
    if "'exit_code':" in text:
        for part in text.split("'exit_code':"):
            rest = part.lstrip()
            if rest and rest[0].isdigit():
                digits = []
                for ch in rest:
                    if ch.isdigit():
                        digits.append(ch)
                    else:
                        break
                if digits:
                    return int("".join(digits))
    # Simple format: exit_code=1
    if "exit_code=" in text:
        idx = text.index("exit_code=") + len("exit_code=")
        digits = []
        for ch in text[idx:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return int("".join(digits))
    return None


def _is_error_result(result: Any, status: str = "", error_type: str = "", tool_name: str = "") -> bool:
    """Check if a tool result indicates an error.

    Only trust structured signals (status/error_type), NOT content keywords.
    Content-based detection has high false-positive rate.

    Excludes non-code-error exit codes:
    - 124: SSH/process timeout (network issue, not code bug)
    - 127: command not found (environment issue)

    Note: grep/rg exit_code=1 (no match) is handled in on_post_tool_call
    via command-level context, since this function lacks the command string.
    """
    if status in ("error", "blocked"):
        # 2026-08-15 修复：plugin_block 是防线插件的行动指引（如四轴闸门要求
        # 输出四轴分析），不是待诊断的技术异常。把它计为 consecutive error 会
        # 与拦截方形成互锁死锁：拦截 -> 计数 -> terminal 封锁 -> 无法继续。
        # approval deny 仍按错误计。
        if error_type == "plugin_block":
            return False
        return True
    if error_type:
        return True
    result_str = str(result) if result else ""
    # Check for non-zero exit code (exclude timeout/not-found)
    code = _extract_exit_code(result_str)
    if code is not None and code != 0 and code not in _NON_ERROR_EXIT_CODES:
        return True
    if result_str.strip().startswith("Traceback "):
        return True
    # Network failures in terminal output (only if not a known non-error exit code)
    if tool_name == "terminal" and code not in _NON_ERROR_EXIT_CODES:
        if any(phrase in result_str.lower() for phrase in _NETWORK_ERROR_PHRASES):
            return True
    return False


def _is_readonly_terminal(cmd: str) -> bool:
    """Check if a terminal command is read-only (logs, grep, etc.).

    Also recognizes SSH-wrapped read-only commands — extracts the last
    quoted string as the remote command and checks its prefix.
    Uses rfind instead of regex to avoid escaping hell.
    """
    lower = cmd.strip().lower()
    # Direct readonly commands
    if any(lower.startswith(p) for p in _READONLY_PREFIXES):
        return True
    # SSH-wrapped: find last quoted string as remote command
    # Try single quotes first, then double quotes
    for quote in ("'", '"'):
        end = lower.rfind(quote)
        if end > 0:
            start = lower.rfind(quote, 0, end)
            if start >= 0:
                remote_cmd = lower[start + 1 : end].lstrip()
                if any(remote_cmd.startswith(p) for p in _READONLY_PREFIXES):
                    return True
                break
    return False


def _is_url_probe(cmd: str) -> bool:
    """Check if command is a URL probe (curl/wget)."""
    return cmd.strip().lower().startswith(_CURL_PATTERN)


def _is_similar_command(a: str, b: str, threshold: float = 0.7) -> bool:
    """Check if two commands are similar (only minor differences)."""
    if not a or not b:
        return False
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


def _commands_are_url_enumeration(recent: List[Dict]) -> bool:
    """Detect pattern: multiple curl calls to different URLs.

    Threshold: 2+ curl calls with different URLs AND at least one failed.
    Catches curl-fail → curl-different-url enumeration while allowing
    2 attempts at the same URL (normal retry).
    """
    curl_calls = [
        r
        for r in recent[-3:]
        if r["tool"] == "terminal" and _is_url_probe(r["args"].get("command", ""))
    ]
    if len(curl_calls) >= 2:
        urls = [c["args"].get("command", "").split()[-1] for c in curl_calls if c["args"].get("command", "")]
        has_error = any(c.get("is_error") for c in curl_calls)
        return len(set(urls)) >= 2 and has_error
    return False


# ── post_tool_call hook ──────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Track tool call results to detect error patterns."""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}
    result = kwargs.get("result", "")
    status = kwargs.get("status", "")
    error_type = kwargs.get("error_type", "")

    recent = _get_recent_calls(sid)
    recent.append({
        "tool": tool_name,
        "args": args,
        "result": str(result)[:500] if result else "",
        "status": status,
        "is_error": _is_error_result(result, status, error_type, tool_name),
    })
    # Trim to max size
    if len(recent) > _MAX_RECENT:
        recent[:] = recent[-_MAX_RECENT:]

    # Determine if this is an error, considering command context for grep
    is_err = _is_error_result(result, status, error_type, tool_name)
    # Override: grep/rg with exit_code=1 = no match, not an error
    if is_err and tool_name == "terminal":
        cmd = str(args.get("command", "")).lower()
        code = _extract_exit_code(str(result))
        if code == 1 and (cmd.startswith(_NO_MATCH_EXIT1_PREFIXES) or "| grep" in cmd or "| rg " in cmd):
            is_err = False

    # Track consecutive errors
    if is_err:
        _set_error_count(sid, _get_error_count(sid) + 1)
        _set_last_error_read(sid, False)  # error occurred, need new diagnosis
    else:
        _set_error_count(sid, 0)

    # Detect diagnosis (reading errors after an error)
    if _get_error_count(sid) > 0 and not _get_last_error_read(sid):
        if tool_name in _DIAGNOSIS_TOOLS:
            if tool_name == "terminal":
                cmd = str(args.get("command", ""))
                if _is_readonly_terminal(cmd):
                    _set_last_error_read(sid, True)
            else:
                # read_file, search_files, etc. = diagnosis
                _set_last_error_read(sid, True)


# ── pre_tool_call hook ────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Block trial-and-error patterns."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}

    if tool_name != "terminal":
        return None

    cmd = str(args.get("command", ""))
    recent = _get_recent_calls(sid)

    # Check 1: URL enumeration (multiple curl to different URLs)
    if _is_url_probe(cmd) and _commands_are_url_enumeration(recent):
        return {
            "action": "block",
            "message": (
                "[ErrorDiscipline] 检测到 URL 枚举试探模式。"
                "错误信息本身往往是答案（no such host=DNS失败，"
                "ReadTimeout=对端慢，401=认证缺失）。"
                "停止逐个 curl，先读错误信息提取文件名:行号定位根因。"
            ),
        }

    # Check 2: Consecutive errors without diagnosis
    if (
        _get_error_count(sid) >= _MAX_CONSECUTIVE_ERRORS
        and not _get_last_error_read(sid)
        and not _is_readonly_terminal(cmd)
    ):
        last_error = ""
        for r in reversed(recent):
            if r.get("is_error"):
                last_error = r["result"][:200]
                break

        return {
            "action": "block",
            "message": (
                f"[ErrorDiscipline] 连续 {_get_error_count(sid)} 次错误未诊断。"
                f"异常诊断铁律：①读错误信息提取文件名:行号 ②打开代码读逻辑"
                f" ③沿调用链追踪根因 ④一句话说清根因。\n"
                f"  最近错误: {last_error}\n"
                f"  修复: 先 read_file / grep / docker logs 读错误信息。"
            ),
        }

    return None


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("error-discipline registered")
