"""guards.watch_pattern — block bare-noise notify patterns on background terminal.

Motivation (2026-11-09, measured in session): a dev-server background task
carried notify=["error", ...]. webpack progress lines print the full path of
every module being processed, and dozens of filenames naturally contain
"error" (errorCode.js, errors.js, errorReceiptTrans.vue ...). Every match is
delivered as a session message that each consumes a turn; 10 queued messages
piled up while the session was blocked. The skill layer records the lesson,
but skills are advisory — this guard makes it mechanical.

Detection (ALL must hold):
  1. tool_name == "terminal"
  2. args.notify is a list
  3. some entry is an exact (case-insensitive) match of a bare noise word:
     error / fail / failed / failure / exception / traceback / warn / warning
  4. no inline waiver ``# watch-ok: <reason>`` in the command

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args.
  Postconditions: returns {} (allow) or {"action": "block", "message": str};
    never raises; pure argument analysis, no subprocess.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_WAIVER_MARK = "# watch-ok"

# 裸噪声词: 作为整词出现时与海量文件名/日志常规词撞车, 只许精确短语替代。
_BARE_NOISE_WORDS = frozenset(
    {"error", "fail", "failed", "failure", "exception", "traceback", "warn", "warning"}
)

_BLOCK_MSG = (
    "[WATCH-PATTERN BLOCK] terminal 后台任务的 notify pattern 含裸噪声词"
    "（error/fail/failed/failure/exception/traceback/warn/warning 整词）。"
    "编译进度行会打印文件全路径，文件名含这些词即误命中——每次命中都投递"
    "一条消息各占一个回合，实测堆出 10 条排队消息。\n"
    "正解：pattern 用只出现在真错误的精确短语（webpack 用 \"ERROR in\"，"
    "pytest 用 \"FAILED\"）；或去掉 pattern 改用完成时通知（notify=true）。"
    "确属有意监听该裸词，命令内加 `# watch-ok: <理由>` 豁免。"
)


def _bare_noise_hits(patterns: list) -> list[str]:
    """返回 patterns 中命中的裸噪声词列表。

    Contract:
      Preconditions: patterns 为 list（元素任意类型）。
      Postconditions: 返回命中词列表（可能为空表）；不抛异常。
    """
    hits = []
    for p in patterns:
        if isinstance(p, str) and p.strip().lower() in _BARE_NOISE_WORDS:
            hits.append(p.strip())
    return hits


def on_pre_tool_call(**kwargs) -> dict[str, Any]:
    """pre_tool_call hook: block bare-noise notify patterns (waiver: # watch-ok).

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict or None).
      Postconditions: returns {} or {"action": "block", "message": str};
        never raises (fail-open on probe errors).
    """
    try:
        tool_name = kwargs.get("tool_name", "") or ""
        if tool_name != "terminal":
            return {}
        args = kwargs.get("args", {}) or {}
        command = str(args.get("command", "") or "")
        if _WAIVER_MARK in command:
            return {}
        patterns = args.get("notify")
        if not isinstance(patterns, list) or not patterns:
            return {}
        hits = _bare_noise_hits(patterns)
        if not hits:
            return {}
        logger.warning("guards.watch_pattern: bare noise pattern blocked: %s", hits)
        return {"action": "block", "message": _BLOCK_MSG}
    except Exception:  # fail-open: 守卫自身异常不拦业务调用
        logger.warning("guards.watch_pattern: probe error, fail-open", exc_info=True)
        return {}


def register(ctx: Any) -> None:
    """注册进 guards 套件（与 slow_search 同层）。

    Contract:
      Preconditions: ctx 提供 register_hook。
      Postconditions: pre_tool_call 钩子注册一次。
    """
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.watch_pattern registered")
