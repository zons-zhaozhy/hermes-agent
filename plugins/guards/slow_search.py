"""guards.slow_search — intercept doomed recursive grep in terminal commands.

Motivation: subagents (and the main agent) run bare recursive grep over large
monorepos. Measured on delegation transcripts (2026-10-24): 4 of 25 adversarial
subagent tasks each burned the full 420 s default tool timeout
(``_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S``, agent/tool_executor.py) on a single
``grep -rn ... .`` that scans node_modules/dist, produced nothing, then got
retried with a different keyword — worst case 1025 s of a 1472 s task spent on
doomed/slow greps.

Blocked shape (ALL must hold for a shell segment):
  1. segment head is ``grep`` (direct command; ``git grep``/``rg`` pass)
  2. a recursive flag (``-r``-prefixed short option) is present
  3. segment is NOT wrapped in ``timeout``
  4. segment has NO ``--exclude-dir`` guard
  5. positional path targets are not all temp/cache paths
  6. no inline waiver comment ``# slowgrep-ok: <reason>``

Escape hatches: add ``--exclude-dir`` or wrap in ``timeout``, or annotate the
command with ``# slowgrep-ok: <reason>``.

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args.
  Postconditions: returns None (allow) or {"action": "block", "message": str};
    never raises; pure string analysis, no subprocess.
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


_BLOCK_MESSAGE = (
    "[SlowSearchGuard] terminal 中检测到无保护的递归 grep（无 timeout 包裹、无 --exclude-dir），已拦截。\n"
    "实测此类命令在大仓上单条可跑满 420 秒工具超时（扫描 node_modules/dist），白等零产出。\n"
    "替代方案（按优先级）：\n"
    "  1. search_files 工具（rg 内核，自动跳过 node_modules/.git，毫秒级）\n"
    "  2. git grep -n \"<pattern>\"（只搜 tracked 文件，秒级）\n"
    "  3. 确需 grep：timeout 60 grep -rn --exclude-dir=node_modules --exclude-dir=.venv \"<pattern>\" <具体子目录>\n"
    "豁免通道：任选其一即放行——加 --exclude-dir、用 timeout 包裹、或行内注释 # slowgrep-ok: <理由>"
)


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook: block unprotected recursive grep.

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict with command key).
      Postconditions: returns None (allow) or {"action":"block","message":str}.
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

    logger.warning(
        "guards.slow_search: blocked doomed recursive grep (command length=%d, hit=%s)",
        len(command),
        hit,
    )
    return {"action": "block", "message": _BLOCK_MESSAGE}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.slow_search registered")
