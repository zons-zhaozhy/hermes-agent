"""guards.pre_write — enforce user intent and pre-investigation gates.

Blocks write_file/patch/execute_code(write) if the target file has not been
read (read_file/search_files) earlier in the same session (the read-set is
session-persistent, not per-turn).

When ``read_think_gate`` has published a verdict for this turn, an explicit
``read_only`` verdict refuses the write; a ``modify`` or ``ambiguous`` verdict
requires a completed investigation. This is a second line of defense against
direct dispatcher paths that skip ``pre_tool_batch``. An absent published
verdict preserves the standalone read-before-write guard; an active verdict
without investigation evidence fails closed on the investigation contract.

ACTIVATION: ON by default. Set PRE_WRITE_GUARD_DISABLE=1 to turn off.

STATE: Per-session keyed set of "files that have been read" and
"SQL schemas that have been confirmed". Scoped by session_id via
plugins._shared_state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "pre_write_guard"


def _get_read_files(sid: str) -> set[str]:
    return get_session_state(sid, _NAMESPACE).setdefault("read_files", set())


_PW_PROBE_TTL = 86400  # 24h：marker 过期即失效，防残留导致永久免检
# 每个 session 一个 marker 文件；过期后既失效又没人删，长期运行的 home 会攒出几千个
# 死文件（实测 2233 个），拖慢目录扫描。写入时顺带清扫，按进程节流到每小时一次。
_PW_PRUNE_INTERVAL = 3600
_pw_last_prune = 0.0


def _prune_stale_markers(marker: Path) -> None:
    """删掉缓存目录里已过 TTL 的 marker——与 ``_disk_read_paths`` 的失效判定同一条线。

    Contract:
      Preconditions: marker 的父目录存在。
      Postconditions: mtime 早于 ``now - _PW_PROBE_TTL`` 的 ``pw_read_probe_*.json``
        被删除；仍有效的 marker（含本次刚写的）保留；任何 OSError 都只下调为
        debug 日志，绝不影响钩子主路径。
    """
    global _pw_last_prune
    now = time.time()
    if now - _pw_last_prune < _PW_PRUNE_INTERVAL:
        return
    _pw_last_prune = now
    cutoff = now - _PW_PROBE_TTL
    try:
        for stale in marker.parent.glob("pw_read_probe_*.json"):
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
            except OSError:
                logger.debug("pre_write: stale marker not removable: %s", stale, exc_info=True)
    except OSError:
        logger.debug("pre_write: marker prune scan failed", exc_info=True)


def _read_marker(sid: str) -> Path:
    """已读证据的磁盘 marker 路径——pre/post 钩子分属不同进程时的共享载体。

    Contract:
      Postconditions: 恒返回 <PW_READ_PROBE_CACHE 或 ~/.hermes/cache>/
        pw_read_probe_<sha256(sid)[:16]>.json，不触碰其它路径。
    """
    digest = hashlib.sha256(sid.encode("utf-8")).hexdigest()[:16]
    cache = os.environ.get("PW_READ_PROBE_CACHE") or os.path.expanduser("~/.hermes/cache")
    return Path(cache) / f"pw_read_probe_{digest}.json"


def _write_read_marker(sid: str, paths: set[str]) -> None:
    """原子写 marker——内存态之外的双保险；失败仅告警不抛（记证据是尽力路径）。

    Contract:
      Postconditions: marker 内容为 {"sid":…,"ts":…,"paths":[…]}；OSError 时日志
        告警且不改既有 marker；写成功后按节流顺手清掉过期 marker。
    """
    try:
        marker = _read_marker(sid)
        marker.parent.mkdir(parents=True, exist_ok=True)
        tmp = marker.with_suffix(".tmp")
        payload = json.dumps({"sid": sid, "ts": time.time(), "paths": sorted(paths)})
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(marker)
        _prune_stale_markers(marker)
    except OSError:
        logger.warning("pre_write: read marker write failed", exc_info=True)


def _disk_read_paths(sid: str) -> set[str]:
    """从 marker 恢复已读集合；不可读/过期/格式错一律空集（fail-closed）。

    Contract:
      Postconditions: 返回值 ⊆ marker 中 paths；marker 缺失、sid 不符或超 TTL
        返回空集。
    """
    marker = _read_marker(sid)
    try:
        data = json.loads(marker.read_text(encoding="utf-8-sig"))
        if not data.get("sid") or time.time() - float(data.get("ts", 0)) >= _PW_PROBE_TTL:
            return set()
        return {str(p) for p in data.get("paths", []) if p}
    except (OSError, ValueError, TypeError):
        logger.debug("pre_write: read marker unreadable: %s", marker, exc_info=True)
        return set()


def _get_schema_confirmed(sid: str) -> set[str]:
    """Set of connection identifiers whose schema has been confirmed."""
    return get_session_state(sid, _NAMESPACE).setdefault("schema_confirmed", set())


# ── Configuration ────────────────────────────────────────────────────

from plugins.guards.tool_classification import STANDARD_WRITE_TOOLS as _WRITE_TOOLS
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


def _read_think_block(sid: str) -> Optional[dict[str, Any]]:
    """Return a block verdict from the current read-think turn, if active."""
    state = get_session_state(sid, "read_think_gate")
    if not state.get("active"):
        return None
    intent = str(state.get("intent") or "ambiguous")
    if intent == "read_only":
        return {
            "action": "block",
            "message": (
                "[PreWriteGuard — 用户意图门禁] 写入被阻止。\n"
                "  用户请求是只读的（询问、分析、审查或诊断），没有授权修改。\n"
                "  只有用户消息及其明确确认可以授权修改；继续回答或调查即可。"
            ),
        }
    # An ambiguous verdict is NOT a refusal: ReadThinkGate runs the
    # investigation contract for it (see _fallback_detect_intent and the intent
    # corpus: "both let the write proceed through the investigation contract").
    # Refusing here turned an unavailable classifier into a user-authorization
    # wait, which deadlocked goal/loop turns until the human retyped.
    if not state.get("investigation_complete"):
        return {
            "action": "block",
            "message": (
                "[PreWriteGuard — 调查门禁] 本 turn 的全面调查尚未完成。\n"
                "  必须先检查目标现状、调用方/相关实现、测试与配置、"
                "必要的历史意图，以及验收和验证方案。\n"
                "  新建文件也不能跳过这次调查。"
            ),
        }
    return None


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
                files = _get_read_files(sid)
                files.add(_norm(path))
                _write_read_marker(sid, files)
        elif tool_name == "search_files":
            # Any search counts as investigation for the searched path
            path = args.get("path", ".")
            if path:
                files = _get_read_files(sid)
                files.add(_norm(path))
                _write_read_marker(sid, files)

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

    gate_block = _read_think_block(sid)
    if gate_block is not None:
        return gate_block

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
    read_files = _get_read_files(sid) | _disk_read_paths(sid)

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
