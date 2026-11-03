"""guards.slow_search — auto-harden doomed recursive grep in terminal commands.

Motivation: subagents (and the main agent) run bare recursive grep over large
monorepos. Measured on delegation transcripts (2026-10-24): 4 of 25 adversarial
subagent tasks each burned the full 420 s default tool timeout
(``_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S``, agent/tool_executor.py) on a single
``grep -rn ... .`` that scans node_modules/dist, produced nothing, then got
retried with a different keyword — worst case 1025 s of a 1472 s task spent on
doomed/slow greps.

2026-11-03 rework (block → auto-rewrite): the block form cost one wasted API
round per hit — measured 1,385 blocks in 7 days (18% of all blocks), the single
largest interceptor. The correction is fully mechanical, so this guard now
REWRITES the command instead of blocking it: each doomed grep segment gets
``timeout 120`` wrapping and ``--exclude-dir`` guards appended, then proceeds.
Protection strength is unchanged (worst case still bounded by 120 s); zero
rounds are wasted. Waiver ``# slowgrep-ok: <reason>`` skips the rewrite.

Detection shape (ALL must hold for a shell segment):
  1. segment head is ``grep`` (direct command; ``git grep``/``rg`` pass)
  2. a recursive flag (``-r``-prefixed short option) is present
  3. segment is NOT wrapped in ``timeout``
  4. segment has NO ``--exclude-dir`` guard
  5. positional path targets are not all temp/cache paths
  6. no inline waiver comment ``# slowgrep-ok: <reason>``

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args.
  Postconditions: returns None (allow unchanged) or
    {"action": "modify", "args": {"command": hardened}}; never raises; pure
    string analysis, no subprocess; blocks are never returned.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from plugins.guards.source_write import _is_safe_path, _split_compound

logger = logging.getLogger(__name__)

_WAIVER_MARK = "# slowgrep-ok"


def _segment_tokens(seg: str) -> list[str]:
    """段 token 化（shlex 引号感知；不闭合退化空格切分）。

    Contract:
      Postconditions: 恒返回列表（可能为空表），不抛异常。
    """
    try:
        import shlex

        return shlex.split(seg.splitlines()[0], posix=True)
    except (ValueError, IndexError):
        return seg.split()


def _head_name(tokens: list[str]) -> str:
    """段头命令名（basename，剥路径前缀）。"""
    return tokens[0].rsplit("/", 1)[-1] if tokens else ""


def _has_recursive_flag(tokens: list[str]) -> bool:
    """grep 是否带递归短选项（-r/-rn/-rE…；-- 长选项不算）。"""
    return any(t.startswith("-r") and not t.startswith("--") for t in tokens[1:])


def _path_targets(tokens: list[str]) -> list[str]:
    """grep 位置参数中的路径目标（首个位置参数是 PATTERN，其余视为路径）。"""
    positional = [t for t in tokens[1:] if not t.startswith("-")]
    return positional[1:] if len(positional) > 1 else []


def _segment_is_doomed(seg: str, tokens: list[str]) -> bool:
    """单段是否为无保护的递归 grep（判定顺序：命令头→递归标志→豁免）。

    Contract:
      Preconditions: tokens 为 seg 首行 token 化结果（可为空表）。
      Postconditions: 返回 bool；纯判定，无副作用。
    """
    if not tokens or _head_name(tokens) != "grep":
        return False
    if not _has_recursive_flag(tokens):
        return False
    if "--exclude-dir" in seg:
        return False
    targets = _path_targets(tokens)
    if targets and all(_is_safe_path(os.path.expanduser(t)) for t in targets):
        return False
    return True


_HARDEN_EXCLUDES = (
    "--exclude-dir=node_modules",
    "--exclude-dir=.venv",
    "--exclude-dir=.git",
    "--exclude-dir=dist",
)

_HARDEN_TIMEOUT_S = 120


def _harden_segment(seg: str) -> str:
    """改写单个 doomed grep 段：timeout 包裹 + 追加 --exclude-dir 守卫。

    flags 插在 grep 段头之后（管道符前），保证 ``timeout 120 grep ... | head``
    语义为 timeout 只作用于 grep。段内含管道时 shlex 重排仍保持参数次序语义。

    Contract:
      Preconditions: seg 为命中判定（doomed）的 shell 段文本。
      Postconditions: 返回加固定化后的段文本；不抛异常。
    """
    try:
        import shlex

        tokens = shlex.split(seg.splitlines()[0], posix=True)
    except (ValueError, IndexError):
        tokens = seg.split()
    if not tokens:
        return seg
    hardened = ["timeout", str(_HARDEN_TIMEOUT_S), tokens[0], *_HARDEN_EXCLUDES, *tokens[1:]]
    try:
        import shlex

        return shlex.join(hardened)
    except ValueError:
        return " ".join(hardened)


def detect_doomed_grep(command: str) -> Optional[str]:
    """检测无保护的递归 grep，命中返回该段文本，未命中返回 None。

    Contract:
      Preconditions: command 为非空字符串。
      Postconditions: 返回 None 或命中的 shell 段原文；纯字符串分析，
        不执行命令、不抛异常。
    """
    if _WAIVER_MARK in command:
        return None
    for seg in _split_compound(command):
        if _segment_is_doomed(seg, _segment_tokens(seg)):
            return seg
    return None


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook: auto-harden unprotected recursive grep (never block).

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict with command key).
      Postconditions: returns None (allow unchanged) or
        {"action": "modify", "args": {"command": <hardened>}}; never blocks.
    """
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "terminal":
        return None

    args = kwargs.get("args") or {}
    command = str(args.get("command", ""))
    if not command:
        return None

    hit = detect_doomed_grep(command)
    if hit is None:
        return None

    hardened = command.replace(hit, _harden_segment(hit), 1)
    logger.warning(
        "guards.slow_search: auto-hardened doomed recursive grep (timeout=%ss, excludes=%s)",
        _HARDEN_TIMEOUT_S,
        ",".join(_HARDEN_EXCLUDES),
    )
    return {"action": "modify", "args": {"command": hardened}}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.slow_search registered")
