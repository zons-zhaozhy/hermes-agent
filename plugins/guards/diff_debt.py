"""guards.diff_debt — diff 欠账闸门（三钩子协作防线）。

规则来源：.hermes-rules.md 编辑纪律——四写通道改文件当轮必须贴写前 diff。
口头宣称已刻/已改而未出示 diff=违规。本模块把该纪律机器化：

  post_tool_call   写工具执行后记欠账（路径+时间戳，进程内 state，按 session 分桶）
  pre_tool_batch   登记本会话本批正文证据；有证据即清本会话欠账
  pre_tool_call    本会话欠账未清且本批无证据时 block（执法只在写工具）

证据形态（与规则文本逐字对齐）：
  1. ```diff 代码块
  2. "diff --git" / "@@ -N" unified diff 痕迹 / "N file(s) changed"
  3. skill_manage 无 diff 回显的合规代偿：old_string/new_string 对照，
     或「前后对照/改动前/改动后/写前 diff」关键词

Contract:
  Preconditions: plugin 系统提供 post_tool_call / pre_tool_batch /
    pre_tool_call 钩子；三钩子同进程派发，kwargs 携带 session_id。
  Postconditions: 本会话欠账未清且本会话本批正文无证据时写工具被 block；
    证据出现即清本会话欠账；cron 会话（session_id 前缀 cron_）全链路豁免；
    state 读取异常按无欠账处理（fail-open——纪律增强层，误放行优于误死锁）。
  Invariants: 欠账与证据均按 session_id 分桶——网关单进程服务多会话
    （同进程可挂多个 bot profile），进程级共享 state 会让 A 会话的欠账
    拦下 B 会话的写、A 会话的证据给 B 会话放行，故不得用全局单例账本；
    同一写操作只记一笔欠账；清欠账只认 assistant 正文证据；
    当批证据当批生效——规则要求「贴对写前状态 diff」，证据与写操作同批
    即已履行义务（缺此，同批第一笔记账会拦下同批后续写，把合规的写前贴
    误判成欠账）；跨批次正文无证据仍拦；
    永不阻断读工具；pre_tool_batch 层永不阻断（执法只在 pre_tool_call）。
"""

from __future__ import annotations

import logging
import re  # re-ok: 三条证据正则均行首锚定/字面无回溯（见使用处各条 re-ok 说明）
import threading
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

# 欠账账本与会话证据位：按 session_id 分桶。state 只在进程内（三钩子同进程
# 派发，无需四轴的跨进程 marker 文件），但键必须带会话——网关单进程服务多
# 会话，进程级单例会让并发会话互相拦写、互相放行。
_DebtEntry = dict[str, Any]  # {"tool": str, "path": str, "ts": float}
_DIFF_DEBT: dict[str, list[_DebtEntry]] = {}   # session_id → 欠账列表
_BATCH_EVIDENCE: dict[str, bool] = {}          # session_id → 本批正文是否出示证据
_LEDGER_LOCK = threading.Lock()                # 多会话钩子可能并发派发
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


def _session_key(kwargs: dict[str, Any]) -> str:
    """会话键——进程内 state 的隔离维度。

    Contract:
      Preconditions: kwargs 为钩子参数（session_id 可能缺失或为 None）
      Postconditions: 返回非 None 的 str 键；同一会话恒等（缺失时归入空键桶）
    """
    return str(kwargs.get("session_id") or "")


def _is_cron_session(kwargs: dict[str, Any]) -> bool:
    """cron 会话豁免——与 four_axis.on_pre_tool_call 同判定依据。"""
    return _session_key(kwargs).startswith("cron_")


def _record_debt(sid: str, tool_name: str, path: str) -> None:
    """记一笔欠账到本会话桶。

    Contract:
      Preconditions: sid 为非 None 会话键；tool_name 已在 _WRITE_TOOLS 内
      Postconditions: 本会话欠账列表尾部新增一笔（持锁），返回 None
    """
    entry: _DebtEntry = {"tool": tool_name, "path": path, "ts": time.time()}
    with _LEDGER_LOCK:
        _DIFF_DEBT.setdefault(sid, []).append(entry)


def _prune_stale(sid: str, debts: list[_DebtEntry], now: float) -> None:
    """剔除过期欠账（隔轮遗忘留痕）。调用方须持 _LEDGER_LOCK。

    Contract:
      Preconditions: debts 为 _DIFF_DEBT[sid] 的引用；now 为当前 epoch 秒
      Postconditions: 超 _DEBT_MAX_AGE_SECONDS 的条目被就地移除并 warning 留痕
    """
    stale = [d for d in debts if now - float(d.get("ts", 0)) > _DEBT_MAX_AGE_SECONDS]
    for d in stale:
        debts.remove(d)
        logger.warning(
            "diff-debt guard: debt expired without diff shown (tool=%s path=%s session=%s)",
            d.get("tool"), d.get("path"), sid,
        )


def _block_verdict(tool_name: str, sid: str, debts: list[_DebtEntry]) -> dict[str, Any]:
    """构造 block 判决（本会话欠账清单 + 规则出处 + 补救指引）。

    Contract:
      Preconditions: debts 非空列表
      Postconditions: 返回 {"action": "block", "message": str}，message 含 diff 字样
    """
    debts_desc = "; ".join(f"{d.get('tool')}→{d.get('path') or '?'}" for d in debts[:3])
    return {
        "action": "block",
        "message": (
            f"[diff 欠账闸门] 工具 '{tool_name}' 被阻断。\n\n"
            f"本会话（{sid or '未命名'}）存在未出示 diff 的写操作欠账：{debts_desc}\n\n"
            "规则（.hermes-rules.md 编辑纪律）：四写通道（write_file/patch/execute_code 写入/"
            "terminal 写入/skill_manage）改动文件，当轮必须在回复正文贴对写前状态 diff。\n"
            "补救：在回复中贴出欠账操作的改动前后对照（```diff 块或 old_string/new_string 对照），"
            "再重试本写操作。"
        ),
    }


def on_post_tool_call(**kwargs: Any) -> None:
    """post_tool_call：写工具执行后记欠账。observer-only，永不阻断。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name not in _WRITE_TOOLS or _is_cron_session(kwargs):
        return
    # 失败/被拦/取消的写未改动文件——无 diff 义务。若记账，被拦的写会
    # 自我繁殖欠账，与 block_escalation 叠成死亡螺旋（10-06 会话 85 连拦实录）。
    if str(kwargs.get("status") or "ok") != "ok":
        return
    args = kwargs.get("args") or {}
    path = str(args.get("path") or args.get("file_path") or args.get("name") or "")
    sid = _session_key(kwargs)
    _record_debt(sid, tool_name, path)
    logger.info("diff-debt guard: debt recorded (tool=%s path=%s session=%s)", tool_name, path, sid)


def on_pre_tool_batch(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_tool_batch：登记本会话本批正文证据，有证据即清本会话欠账。

    永不返回 block——执法在 pre_tool_call 层，batch 层拦截会连坐
    同批读工具。
    """
    sid = _session_key(kwargs)
    if _is_cron_session(kwargs):
        with _LEDGER_LOCK:
            _DIFF_DEBT.pop(sid, None)
            _BATCH_EVIDENCE.pop(sid, None)
        return None
    evidence = _has_diff_evidence(str(kwargs.get("assistant_content") or ""))
    with _LEDGER_LOCK:
        _BATCH_EVIDENCE[sid] = evidence
        cleared = len(_DIFF_DEBT.pop(sid, [])) if evidence else 0
    if cleared:
        logger.info(
            "diff-debt guard: %d debt(s) cleared — diff evidence present in assistant content (session=%s)",
            cleared, sid,
        )
    return None


def on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_tool_call：本会话欠账未清且本会话本批无证据即 block。执法层。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name not in _WRITE_TOOLS or _is_cron_session(kwargs):
        return None
    sid = _session_key(kwargs)
    with _LEDGER_LOCK:
        debts = _DIFF_DEBT.get(sid)
        # 本会话本批正文已出示证据（规则要的「写前贴」）→ 本批写操作全部合规；
        # 他会话的欠账/证据在各自桶里，互不影响。
        if not debts or _BATCH_EVIDENCE.get(sid):
            return None
        _prune_stale(sid, debts, time.time())
        if not debts:
            _DIFF_DEBT.pop(sid, None)
            return None
        return _block_verdict(tool_name, sid, debts)


def register(ctx: Any) -> None:
    """注册三钩子（post 记账 / batch 清账 / pre 执法）。"""
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_tool_batch", on_pre_tool_batch)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
