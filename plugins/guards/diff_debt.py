"""guards.diff_debt — diff 欠账闸门（三钩子协作防线）。

规则来源：.hermes-rules.md 编辑纪律——四写通道改文件当轮必须贴写前 diff。
口头宣称已刻/已改而未出示 diff=违规。本模块把该纪律机器化：

  post_tool_call   写工具执行后记欠账（路径+时间戳，进程内 state）
  pre_tool_batch   扫 assistant_content 中的 diff 证据，清欠账
  pre_tool_call    下一次写操作前查欠账未清 → block

证据形态（与规则文本逐字对齐）：
  1. ```diff 代码块
  2. "diff --git" / "@@ -N" unified diff 痕迹 / "N file(s) changed"
  3. skill_manage 无 diff 回显的合规代偿：old_string/new_string 对照，
     或「前后对照/改动前/改动后/写前 diff」关键词

Contract:
  Preconditions: plugin 系统提供 post_tool_call / pre_tool_batch /
    pre_tool_call 钩子；三钩子同进程派发（进程内 state 即会话隔离）。
  Postconditions: 欠账未清时写工具被 block；证据出现即清账；
    cron 会话（session_id 前缀 cron_）全链路豁免；state 读取异常
    按无欠账处理（fail-open——纪律增强层，误放行优于误死锁）。
  Invariants: 同一写操作只记一笔欠账；清欠账只认 assistant 正文证据；
    永不阻断读工具；pre_tool_batch 层永不阻断（执法只在 pre_tool_call）。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 写工具集合（.hermes-rules.md「四写通道」中的结构化工具）。
# patch 工具自带 diff 回显——但义务是「贴进回复正文」，回显≠履行，
# 统一记账统一判证据。execute_code/terminal 写入由 source_write 按
# 命令特征另管，本防线不按工具名拦只读分析。
_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "patch",
    "skill_manage",
})

# 欠账账本：进程内单例。post 与 pre 同进程（工具执行器同进程派发钩子），
# 无需四轴的跨进程 marker 文件。
_DebtEntry = Dict[str, Any]  # {"tool": str, "path": str, "ts": float}
_DIFF_DEBT: List[_DebtEntry] = []
_DEBT_MAX_AGE_SECONDS = 1800  # 过期防误拦隔轮写；过期时 warning 留痕（遗忘即违规记录）

# 证据判定：行首锚定/有限结构，无灾难回溯（历史 re.search CPU 100% 教训）
_DIFF_FENCE_RE = re.compile(r"```diff\b", re.IGNORECASE)  # re-ok: fenced 标记,无回溯
_UNIFIED_RE = re.compile(r"^diff --git ", re.MULTILINE)    # re-ok: 行首锚定字面量
_HUNK_RE = re.compile(r"^@@ -\d+", re.MULTILINE)           # re-ok: 行首锚定固定结构
_COMPENSATION_KEYWORDS: tuple[str, ...] = (
    "old_string",
    "new_string",
    "前后对照",
    "改动前",
    "改动后",
    "写前 diff",
    "写前diff",
)


def _has_diff_evidence(content: str) -> bool:
    """判定 assistant 正文是否出示了 diff 或合规代偿。

    Contract:
      Preconditions: content 为本批 assistant 消息正文（可能为空串）
      Postconditions: 返回 True 当且仅当命中任一证据形态
    """
    if not content:
        return False
    if _DIFF_FENCE_RE.search(content) or _UNIFIED_RE.search(content) or _HUNK_RE.search(content):
        return True
    if "1 file changed" in content or "files changed" in content:
        return True
    return any(kw in content for kw in _COMPENSATION_KEYWORDS)


def _is_cron_session(kwargs: Dict[str, Any]) -> bool:
    """cron 会话豁免——与 four_axis.on_pre_tool_call 同判定依据。"""
    return str(kwargs.get("session_id") or "").startswith("cron_")


def on_post_tool_call(**kwargs: Any) -> None:
    """post_tool_call：写工具执行后记欠账。observer-only，永不阻断。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name not in _WRITE_TOOLS:
        return
    if _is_cron_session(kwargs):
        return
    args = kwargs.get("args") or {}
    path = str(args.get("path") or args.get("file_path") or args.get("name") or "")
    _DIFF_DEBT.append({"tool": tool_name, "path": path, "ts": time.time()})
    logger.info("diff-debt guard: debt recorded (tool=%s path=%s)", tool_name, path)


def on_pre_tool_batch(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """pre_tool_batch：扫 assistant 正文 diff 证据清欠账。

    永不返回 block——执法在 pre_tool_call 层，batch 层拦截会连坐
    同批读工具。
    """
    if not _DIFF_DEBT:
        return None
    if _is_cron_session(kwargs):
        _DIFF_DEBT.clear()
        return None
    content = str(kwargs.get("assistant_content") or "")
    if _has_diff_evidence(content):
        _DIFF_DEBT.clear()
        logger.info("diff-debt guard: debts cleared — diff evidence present in assistant content")
    return None


def on_pre_tool_call(**kwargs: Any) -> Optional[Dict[str, Any]]:
    """pre_tool_call：下一次写操作前查欠账未清即 block。执法层。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name not in _WRITE_TOOLS:
        return None
    if _is_cron_session(kwargs):
        return None
    now = time.time()
    stale = [d for d in _DIFF_DEBT if now - float(d.get("ts", 0)) > _DEBT_MAX_AGE_SECONDS]
    for d in stale:
        _DIFF_DEBT.remove(d)
        logger.warning(
            "diff-debt guard: debt expired without diff shown (tool=%s path=%s)",
            d.get("tool"), d.get("path"),
        )
    if not _DIFF_DEBT:
        return None
    debts = "; ".join(f"{d.get('tool')}→{d.get('path') or '?'}" for d in _DIFF_DEBT[:3])
    return {
        "action": "block",
        "message": (
            f"[diff 欠账闸门] 工具 '{tool_name}' 被阻断。\n\n"
            f"存在未出示 diff 的写操作欠账：{debts}\n\n"
            "规则（.hermes-rules.md 编辑纪律）：四写通道（write_file/patch/execute_code 写入/"
            "terminal 写入/skill_manage）改动文件，当轮必须在回复正文贴对写前状态 diff。\n"
            "补救：在回复中贴出欠账操作的改动前后对照（```diff 块或 old_string/new_string 对照），"
            "再重试本写操作。"
        ),
    }


def register(ctx: Any) -> None:
    """注册三钩子（post 记账 / batch 清账 / pre 执法）。"""
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_tool_batch", on_pre_tool_batch)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
