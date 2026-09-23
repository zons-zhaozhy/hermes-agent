"""tool-safety plugin — guard against tool misuse patterns.

Detects and blocks:
1. execute_code with batch str.replace across multiple files (silent failure risk)
2. read_file retry loops (same file 3+ times returning empty/error)
3. patch falsification (same file patched 3+ times without verification)
4. execute_code file writes without verification step

ACTIVATION: ON by default. Set TOOL_SAFETY_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re  # noqa: R1 — dangerous command detection requires regex
from typing import Any, Dict, List, Optional, Set

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "tool_safety"


def _get_read_attempts(sid: str) -> Dict[str, int]:
    """Track read_file attempts per path."""
    return get_session_state(sid, _NAMESPACE).setdefault("read_attempts", {})


def _get_patch_attempts(sid: str) -> Dict[str, int]:
    """Track patch attempts per path."""
    return get_session_state(sid, _NAMESPACE).setdefault("patch_attempts", {})


def _get_patch_verified(sid: str) -> Set[str]:
    """Track paths that have been verified after patch."""
    return get_session_state(sid, _NAMESPACE).setdefault("patch_verified", set())


def _get_recent_tool_calls(sid: str) -> List[Dict[str, Any]]:
    """Recent tool calls in this session."""
    return get_session_state(sid, _NAMESPACE).setdefault("recent_calls", [])


# ── Configuration ────────────────────────────────────────────────────

_MAX_READ_RETRIES = 3
_MAX_PATCH_ATTEMPTS = 3


def _plugin_disabled() -> bool:
    return os.environ.get("TOOL_SAFETY_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _is_batch_replace(code: str) -> bool:
    """Detect execute_code with batch str.replace pattern."""
    indicators = (
        r"\.replace\(", r"\.replace\s*\(",
        "for.*in.*files", "for.*in.*paths",
        "read_file.*write_file", "open.*\.write",
    )
    lower = code.lower()
    count = sum(1 for ind in indicators if re.search(ind, lower))
    return count >= 3  # at least 3 indicators to reduce false positives


def _is_empty_or_error_result(result: Any, status: str = "") -> bool:
    """Check if read_file returned empty content or an error."""
    result_str = str(result).lower() if result else ""
    if status == "error":
        return True
    # read_file returns empty for binary files or non-existent files
    if "is_binary" in result_str or "does not exist" in result_str:
        return True
    # Empty content
    if len(result_str.strip()) < 20:
        return True
    return False


def _is_verification_tool_call(tool_name: str, args: Dict) -> bool:
    """Check if tool call is verifying a previous write."""
    if tool_name == "search_files":
        return True  # grep after write = verification
    if tool_name == "read_file":
        return True  # re-read after patch = verification
    if tool_name == "terminal":
        cmd = str(args.get("command", "")).lower()
        verify_prefixes = ("grep ", "cat ", "head ", "diff ", "git diff")
        return any(cmd.startswith(p) for p in verify_prefixes)
    return False


# ── post_tool_call hook ──────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Track tool usage patterns for safety detection."""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}
    result = kwargs.get("result", "")
    status = kwargs.get("status", "")

    recent = _get_recent_tool_calls(sid)
    recent.append({
        "tool": tool_name,
        "args": args,
        "status": status,
    })
    if len(recent) > 15:
        recent[:] = recent[-15:]

    # Track read_file attempts
    if tool_name == "read_file":
        path = args.get("path", "")
        if path:
            attempts = _get_read_attempts(sid)
            attempts[path] = attempts.get(path, 0) + 1

            # Reset counter on successful read
            if not _is_empty_or_error_result(result, status):
                attempts[path] = 0

    # Track patch attempts
    if tool_name == "patch":
        path = args.get("path", "")
        if path:
            attempts = _get_patch_attempts(sid)
            attempts[path] = attempts.get(path, 0) + 1

            # Reset counter on successful patch — the guard targets
            # repeated FAILURES (stale old_string), not repeated successes.
            # Without this, 3 successful patches on the same file block
            # the 4th even though nothing went wrong.
            if status != "error" and "success" in str(result).lower():
                attempts[path] = 0

    # Track verification after patch
    if _is_verification_tool_call(tool_name, args):
        # Mark recent patched files as verified
        patch_attempts = _get_patch_attempts(sid)
        for path in list(patch_attempts.keys()):
            _get_patch_verified(sid).add(path)


# ── pre_tool_call hook ────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Block unsafe tool usage patterns."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}

    # ── Check 1: execute_code batch str.replace ───────────────────
    if tool_name == "execute_code":
        code = args.get("code", "")
        if _is_batch_replace(code):
            return {
                "action": "block",
                "message": (
                    "[ToolSafety] execute_code 批量 str.replace 检测到。"
                    "str.replace 匹配失败静默跳过，无法确认写入结果。\n"
                    "  修复: 使用 patch 工具逐文件修改（显示 diff），"
                    "或拆分为独立的 patch + search_files 验证。"
                ),
            }

    # ── Check 2: read_file retry loop ─────────────────────────────
    if tool_name == "read_file":
        path = args.get("path", "")
        if path:
            attempts = _get_read_attempts(sid)
            count = attempts.get(path, 0)
            if count >= _MAX_READ_RETRIES:
                return {
                    "action": "block",
                    "message": (
                        f"[ToolSafety] read_file 对 {path} 已重试 {count} 次。"
                        f"可能文件是 binary、不存在或 path 有误。\n"
                        f"  修复: 切换到 execute_code Python open() 读取，"
                        f"或先用 terminal ls/检查文件是否存在。"
                    ),
                }

    # ── Check 3: patch falsification (3+ attempts, unverified) ──
    if tool_name == "patch":
        path = args.get("path", "")
        if path:
            attempts = _get_patch_attempts(sid)
            count = attempts.get(path, 0)
            if count >= _MAX_PATCH_ATTEMPTS and path not in _get_patch_verified(sid):
                return {
                    "action": "block",
                    "message": (
                        f"[ToolSafety] 对 {path} 已执行 {count} 次 patch。"
                        f"连续 patch 失败可能因 old_string 不匹配（HMR 修改了磁盘文件）。\n"
                        f"  修复: 先 read_file 重读文件获取当前内容，"
                        f"然后切换到 write_file 全量重写。"
                    ),
                }

    # ── Check 4: catastrophic delete commands ─────────────────────
    # 2026-08-28 审计修复：rm -rf 指向根/家目录此前无任何插件拦截
    if tool_name == "terminal":
        cmd = str(args.get("command", "")).strip()
        for rm_token in cmd.split():
            if not rm_token.startswith(("rm", "-")):
                continue
            if rm_token in ("rm", "rm -r", "rm -rf", "rm -fr", "rm -f") or (
                rm_token.startswith("-") and "r" in rm_token and "f" in rm_token
            ):
                # 找到 rm/-rf 后的第一个非选项参数即目标
                idx = cmd.split().index(rm_token) if rm_token in cmd.split() else -1
                toks = cmd.split()
                target = next((t for t in toks[idx + 1:] if not t.startswith("-")), "")
                norm = os.path.expanduser(target).rstrip("/")
                if norm in ("/", "", ".") or norm == os.path.expanduser("~").rstrip("/"):
                    return {
                        "action": "block",
                        "message": (
                            "[ToolSafety] 灾难性删除命令被拦截：rm -rf 指向根目录或家目录。\n"
                            "  修复: 明确列出要删除的具体子路径；确需清空请逐项确认后操作。"
                        ),
                    }

    return None

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("tool-safety registered")
