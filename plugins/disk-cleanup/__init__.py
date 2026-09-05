"""disk-cleanup plugin — auto-cleanup of ephemeral Hermes session files.

``post_tool_call`` silently tracks test/temp paths created by write_file/patch/terminal;
``on_session_end`` runs :func:`disk_cleanup.quick` when any test file was tracked this turn;
``/disk-cleanup`` exposes status / dry-run / quick / deep / track / forget.
"""

from __future__ import annotations

import contextlib
import logging
import json
import re
import shlex
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from . import disk_cleanup as dg

logger = logging.getLogger(__name__)


# Test files newly tracked this turn, keyed by task_id (or session_id) so on_session_end can
# decide whether to run cleanup. Locked: post_tool_call fires concurrently on parallel calls.
_recent_test_tracks: Dict[str, Set[str]] = {}
_lock = threading.Lock()

_TERMINAL_PATH_REGEX = re.compile(r"(?:^|\s)(/[^\s'\"`]+|\~/[^\s'\"`]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracker_key(task_id: str, session_id: str) -> str:
    return task_id or session_id or "default"


def _record_track(task_id: str, session_id: str, path: Path, category: str) -> None:
    """Record that we tracked *path* as *category* during this turn."""
    if category != "test":
        return
    key = _tracker_key(task_id, session_id)
    with _lock:
        _recent_test_tracks.setdefault(key, set()).add(str(path))


def _drain(task_id: str, session_id: str) -> Set[str]:
    """Pop the set of test paths tracked during this turn."""
    key = _tracker_key(task_id, session_id)
    with _lock:
        return _recent_test_tracks.pop(key, set())


def _attempt_track(path_str: str, task_id: str, session_id: str) -> None:
    """Best-effort auto-track. Never raises.

    Contract:
      Preconditions: path_str 是任意字符串（可能超长/含换行/非法字节）。
      Postconditions: 本函数绝不向上抛异常——所有 OSError（含超长路径的
      [Errno 63] ENAMETOOLONG）在 exists() 探测处就地消化，只记 debug 日志。
    """
    try:
        p = Path(path_str).expanduser()
        if not p.exists():
            return
    except OSError:
        # 超长/非法路径触发 ENAMETOOLONG 等——best-effort 语义下静默放弃，
        # 不让钩子分发器把 WARNING 刷进 errors.log（历史缺陷：156 条 Errno 63）。
        logger.debug("disk-cleanup: skip unstat-able path (%d chars)", len(path_str))
        return
    category = dg.guess_category(p)
    if category is None:
        return
    newly = dg.track(str(p), category, silent=True)
    if newly:
        _record_track(task_id, session_id, p, category)


def _extract_paths_from_write_file(args: Dict[str, Any]) -> Set[str]:
    path = args.get(_WRITE_FILE_PATH_KEY)
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_patch(args: Dict[str, Any]) -> Set[str]:
    # The patch tool creates new files via the `mode="patch"` path too, but
    # most of its use is editing existing files — we only care about new
    # ephemeral creations, so treat patch conservatively and only pick up
    # the single-file `path` arg.  Track-then-cleanup is idempotent, so
    # re-tracking an already-tracked file is a no-op (dedup in track()).
    path = args.get("path")
    return {path} if isinstance(path, str) and path else set()


def _extract_paths_from_terminal(args: Dict[str, Any], result: str) -> Set[str]:
    """Candidate paths from a terminal command + output; guess_category/is_safe_path filter later."""
    paths: Set[str] = set()
    cmd = args.get("command") or ""
    if isinstance(cmd, str) and cmd:
        with contextlib.suppress(ValueError):  # tokenise — catches `touch /tmp/hermes-x/test_foo.py`
            paths.update(tok for tok in shlex.split(cmd, posix=True) if tok.startswith(("/", "~")))
    # Only scan the result text if it's a reasonable size (avoid 50KB dumps).
    # result 可能是 JSON 字符串（terminal 工具的钩子负载形态）：JSON 内换行是
    # 字面 \n 两字符，正则的 [^\s]+ 会把它当普通字符吞掉，跨"行"匹配出
    # /tmp/foo.md\n---\ndeleg_* 超长伪路径。先解码 JSON 取 output 字段再扫描。
    if isinstance(result, str) and len(result) < 4096:
        scan_text = result
        if result.lstrip().startswith("{"):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and isinstance(parsed.get("output"), str):
                    scan_text = parsed["output"]
            except (json.JSONDecodeError, ValueError):
                pass
        for match in _TERMINAL_PATH_REGEX.findall(scan_text):
            paths.add(match)
    return paths


_PATH_EXTRACTORS: Dict[str, Callable[[Dict[str, Any], str], Set[str]]] = {
    "write_file": _extract_path_arg,
    "patch": _extract_path_arg,
    "terminal": _extract_paths_from_terminal}


def _on_post_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, result: Any = None,
                       task_id: str = "", session_id: str = "", tool_call_id: str = "", **_: Any) -> None:
    """Auto-track ephemeral files created by recent tool calls. Best-effort, never raises."""
    extractor = _PATH_EXTRACTORS.get(tool_name)
    if not isinstance(args, dict) or extractor is None:
        return
    for path_str in extractor(args, result if isinstance(result, str) else ""):
        try:
            p = Path(path_str).expanduser()
        except Exception:
            continue
        category = dg.guess_category(p) if p.exists() else None
        if category is not None and dg.track(str(p), category, silent=True) and category == "test":
            with _lock:
                _recent_test_tracks.setdefault(task_id or session_id or "default", set()).add(str(p))


def _on_session_end(
    session_id: str = "", completed: bool = True, interrupted: bool = False, **_: Any) -> None:
    """Run quick cleanup if any test files were tracked during this turn."""
    # Drain the session bucket plus every task-scoped bucket (subagents record into their own).
    with _lock:
        had_tracks = bool(_recent_test_tracks.pop(session_id or "default", None) or _recent_test_tracks)
        _recent_test_tracks.clear()
    if not had_tracks:
        return
    try:
        summary = dg.quick()
    except Exception as exc:
        logger.debug("disk-cleanup quick cleanup failed: %s", exc)
        return
    if summary["deleted"] or summary["empty_dirs"]:
        dg._log(f"AUTO_QUICK (session_end): deleted={summary['deleted']} "
                f"dirs={summary['empty_dirs']} freed={dg.fmt_size(summary['freed'])}")


_HELP_TEXT = """\
/disk-cleanup — ephemeral-file cleanup

Subcommands:
  status                     Per-category breakdown + top-10 largest
  dry-run                    Preview what quick/deep would delete
  quick                      Run safe cleanup now (no prompts)
  deep                       Run quick, then list items that need prompts
  track <path> <category>    Manually add a path to tracking
  forget <path>              Stop tracking a path (does not delete)

Categories: temp | test | research | download | chrome-profile | cron-output | other

All operations are scoped to HERMES_HOME and /tmp/hermes-*.
Test files are auto-tracked on write_file / terminal and auto-cleaned at session end.
"""


def _fmt_summary(summary: Dict[str, Any]) -> str:
    base = (f"[disk-cleanup] Cleaned {summary['deleted']} files + "
            f"{summary['empty_dirs']} empty dirs, freed {dg.fmt_size(summary['freed'])}.")
    if summary.get("errors"):
        base += f"\n  {len(summary['errors'])} error(s); see cleanup.log."
    return base


def _item_block(header: str, items: List[Dict], indent: str) -> List[str]:
    """``header`` formatted with ``{n}`` / ``{size}``, followed by one line per item."""
    size = dg.fmt_size(sum(i["size"] for i in items))
    return [header.format(n=len(items), size=size)] + [f"{indent}[{item['category']}] {item['path']}" for item in items]


def _cmd_dry_run(argv: List[str]) -> str:
    auto, prompt = dg.dry_run()
    lines = ["Dry-run preview (nothing deleted):"]
    lines += _item_block("  Auto-delete : {n} files ({size})", auto, "    ")
    lines += _item_block("  Needs prompt: {n} files ({size})", prompt, "    ")
    lines.append(f"\n  Total potential: {dg.fmt_size(sum(i['size'] for i in auto + prompt))}")
    return "\n".join(lines)


def _cmd_deep(argv: List[str]) -> str:
    # In-session deep can't prompt — show what quick cleaned plus items needing confirmation.
    quick_summary = dg.quick()
    _auto, prompt_items = dg.dry_run()
    lines = [_fmt_summary(quick_summary)]
    if prompt_items:
        lines += _item_block("\n{n} item(s) need confirmation ({size}):", prompt_items, "  ")
        lines.append("\nRun `/disk-cleanup forget <path>` to skip, or delete manually via terminal.")
    return "\n".join(lines)


def _cmd_track(argv: List[str]) -> str:
    if len(argv) < 3:
        return "Usage: /disk-cleanup track <path> <category>"
    path_arg, category = argv[1], argv[2]
    if category not in dg.ALLOWED_CATEGORIES:
        return f"Unknown category '{category}'. Allowed: {sorted(dg.ALLOWED_CATEGORIES)}"
    if dg.track(path_arg, category, silent=True):
        return f"Tracked {path_arg} as '{category}'."
    return f"Not tracked (already present, missing, or outside HERMES_HOME): {path_arg}"


def _cmd_forget(argv: List[str]) -> str:
    if len(argv) < 2:
        return "Usage: /disk-cleanup forget <path>"
    n = dg.forget(argv[1])
    return (f"Removed {n} tracking entr{'y' if n == 1 else 'ies'} for {argv[1]}." if n
            else f"Not found in tracking: {argv[1]}")


_SUBCOMMANDS: Dict[str, Callable[[List[str]], str]] = {
    "status": lambda argv: dg.format_status(dg.status()),
    "dry-run": _cmd_dry_run,
    "quick": lambda argv: _fmt_summary(dg.quick()),
    "deep": _cmd_deep,
    "track": _cmd_track,
    "forget": _cmd_forget}


def _handle_slash(raw_args: str) -> Optional[str]:
    argv = raw_args.strip().split()
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _HELP_TEXT
    handler = _SUBCOMMANDS.get(argv[0])
    if handler is None:
        return f"Unknown subcommand: {argv[0]}\n\n{_HELP_TEXT}"
    return handler(argv)


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_command("disk-cleanup", handler=_handle_slash,
                         description="Track and clean up ephemeral Hermes session files.")
