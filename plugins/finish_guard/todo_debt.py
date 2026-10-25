"""todo_debt — 跨会话任务债务账本（挂账闭环的消费端）。

根因（2026-10-25 调查，state.db 113,164 条回复实测）：挂账介质的生命周期
短于任务的生命周期——TodoStore 是会话内存态（tools/todo_tool.py，one per
session），会话结束即蒸发；新会话启动不注入任何未清账，登记了的事项永远
没有触发点（24 个会话的末条消息停在挂账措辞上，无一下文）。本模块补上
闭环的消费端：

  1. on_todo_write（post_tool_call）：todo_list 每次写入后，把
     pending/in_progress 项按会话槽位持久化到 <HERMES_HOME>/cache/
     todo_debt.json（原子写）；快照中 completed/cancelled 的 id 从其他
     会话槽位剔除（跨槽位销账）。
  2. on_session_open（pre_llm_call）：会话首轮注入其他会话的未清账清单，
     要求 merge 进本会话 todo_list（保留原 id）逐项消化，或显式 cancelled
     并说明理由。

Contract:
  Preconditions: post_tool_call 载荷含 tool_name/args/result/session_id
    （model_tools._emit_post_tool_call_hook 的键）；pre_llm_call 载荷含
    conversation_history/session_id/task_id/turn_id。
  Postconditions: 回调永不 raise（异常吞入 logger.exception）；账本文件
    损坏/不可解析时视为空账本；删除账本文件即完全清零（可恢复）。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes.plugins.finish_guard.todo_debt")

_NAMESPACE = "todo_debt"
_MAX_LISTED_ITEMS = 10
_MAX_CONTENT_CHARS = 200
_ACTIVE_STATUSES = ("pending", "in_progress")
_SETTLED_STATUSES = ("completed", "cancelled")


def _debt_path() -> Path:
    """账本路径（HERMES_HOME 作用域，profile 隔离）。

    Contract: Postconditions: 返回 <hermes_home>/cache/todo_debt.json；不缓存
    路径（HERMES_HOME 可能在进程生命周期内被 profile 作用域切换）。
    """
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "cache" / "todo_debt.json"


def _load_debt() -> dict[str, Any]:
    """读账本；文件缺失或 JSON 损坏时返回空结构。

    Contract: Postconditions: 返回含 "sessions" 键的 dict；损坏文件不 raise。
    """
    path = _debt_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"sessions": {}}
    if not isinstance(data, dict) or not isinstance(data.get("sessions"), dict):
        return {"sessions": {}}
    return data


def _save_debt(data: dict[str, Any]) -> None:
    """原子落盘（pid 后缀临时文件 + replace，末写者胜）。

    Contract: Preconditions: data 为可 JSON 序列化 dict；
    Postconditions: 读者永远看到完整文件，不存在半写状态；IO 失败 raise
    OSError 由调用方的 try 块记录。
    """
    path = _debt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def _snapshot_items(result: Any) -> list[Any] | None:
    """从 todo_list 工具结果提取全量 items；非 todo 快照返回 None。

    Contract: Postconditions: 结果为 JSON 字符串或 dict 且含 todos 列表时返回
    该列表（可为空列表），否则 None（读调用/异常形态，不触发账本更新）。
    """
    payload: Any = None
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except ValueError:
            return None
    elif isinstance(result, dict):
        payload = result
    if not isinstance(payload, dict):
        return None
    items = payload.get("todos")
    return items if isinstance(items, list) else None


def _apply_own_slot(sessions: dict[str, Any], sid: str, active: list[dict[str, Any]]) -> None:
    """把本会话槽位写为 active 项快照；无 active 项则删槽。

    Contract: Preconditions: active 为本快照中 pending/in_progress 的项；
    Postconditions: sessions[sid] 恰好反映当前未清项（空则不存在）。
    """
    if not active:
        sessions.pop(sid, None)
        return
    sessions[sid] = {
        "updated_at": int(time.time()),
        "items": [
            {
                "id": str(i.get("id")),
                "content": str(i.get("content") or "")[:_MAX_CONTENT_CHARS],
                "status": str(i.get("status")),
            }
            for i in active
        ],
    }


def _settle_foreign_slots(sessions: dict[str, Any], sid: str, settled_ids: set[str]) -> None:
    """用本快照已结算的 id 清其他会话槽位（跨槽位销账）。

    Contract: Preconditions: settled_ids 为本快照中 completed/cancelled 的 id 集；
    Postconditions: 其他槽位不再含 settled_ids 中的项；清空的槽位被删除。
    """
    for other_sid in [s for s in sessions if s != sid]:
        kept = [
            it for it in sessions[other_sid].get("items", [])
            if str(it.get("id")) not in settled_ids
        ]
        if kept:
            sessions[other_sid]["items"] = kept
        else:
            del sessions[other_sid]


def _update_ledger(sid: str, items: list[Any]) -> None:
    """按 todo 全量快照更新账本：本槽位 + 跨槽位销账 + 落盘。

    Contract: Preconditions: items 为 todo_list 返回的全量 items；
    Postconditions: 全部槽位空时删除账本文件（零残留），否则原子落盘。
    """
    data = _load_debt()
    sessions: dict[str, Any] = data["sessions"]
    settled_ids = {
        str(i.get("id"))
        for i in items
        if isinstance(i, dict) and str(i.get("status") or "") in _SETTLED_STATUSES
    }
    active = [
        i for i in items
        if isinstance(i, dict) and str(i.get("status") or "") in _ACTIVE_STATUSES
    ]
    _apply_own_slot(sessions, sid, active)
    _settle_foreign_slots(sessions, sid, settled_ids)
    if not sessions:
        _debt_path().unlink(missing_ok=True)
        return
    _save_debt(data)


def _foreign_debts(sid: str) -> list[dict[str, Any]]:
    """收集非本会话槽位的未清项，最老优先。

    Contract: Postconditions: 返回 [{slot, updated_at, item}]，按 updated_at
    升序；本会话槽位（本轮刚登记的正常在途任务）不列入。
    """
    rows: list[dict[str, Any]] = []
    for slot, block in _load_debt()["sessions"].items():
        if slot == sid:
            continue
        for item in block.get("items", []):
            rows.append({
                "slot": slot,
                "updated_at": int(block.get("updated_at") or 0),
                "item": item,
            })
    rows.sort(key=lambda r: r["updated_at"])
    return rows


def _format_age(updated_at: int) -> str:
    """账龄人话化。Contract: Postconditions: 返回如「3.5 小时」「2 天」。"""
    seconds = max(0, int(time.time()) - updated_at)
    if seconds < 3600:
        return f"{seconds // 60} 分钟"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds // 86400} 天"


def _render_debt_context(rows: list[dict[str, Any]]) -> str:
    """把未清账行渲染为注入文本。

    Contract: Preconditions: rows 非空（调用方已判）；
    Postconditions: 返回含清单与处置纪律的完整提醒文本。
    """
    lines = [
        f"- [账龄 {_format_age(r['updated_at'])}] {r['item'].get('content') or '(无内容)'}"
        f"（来源会话 {r['slot'][:12]}，id={r['item'].get('id')}）"
        for r in rows[:_MAX_LISTED_ITEMS]
    ]
    extra = f"（另有 {len(rows) - _MAX_LISTED_ITEMS} 项未列出）" if len(rows) > _MAX_LISTED_ITEMS else ""
    return (
        f"【跨会话未清账 todo_debt】检测到 {len(rows)} 项已登记但从未消化的事项{extra}。\n"
        "这些事项登记进了会话内存态 TODO，会话结束即蒸发——本提醒是它们唯一的消费触发点：\n"
        + "\n".join(lines)
        + "\n处置纪律：开工前把上述事项 merge 进本会话 todo_list（保留原 id），"
        "逐项做完并验证，或显式标记 cancelled 并在回复中说明理由——"
        "禁止再次登记后停轮（登记不等于完成）。完成或取消后账本自动销账。"
        "若本轮是定时任务且与该账无关，忽略本提醒，勿在定时任务会话中处理。"
    )


def on_todo_write(**kwargs: Any) -> None:
    """post_tool_call 观察者：todo_list 写入后同步债务账本。

    Contract: Preconditions: kwargs 含 tool_name/result/session_id/task_id；
    Postconditions: todo_list 快照触发 _update_ledger；其他工具直接返回；
    任何异常吞入 logger.exception（观察者绝不影响工具链）。
    """
    try:
        if kwargs.get("tool_name") != "todo_list":
            return
        items = _snapshot_items(kwargs.get("result"))
        if items is None:
            return
        sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "__default__")
        _update_ledger(sid, items)
    except Exception:
        logger.exception("todo_debt on_todo_write failed")


def on_session_open(**kwargs: Any) -> dict[str, Any] | None:
    """pre_llm_call 注入器：会话进程内首见时注入跨会话未清账（一次/进程会话）。

    首轮判定=debt_injected 标记缺失（进程内该会话首次 pre_llm_call），不按
    历史推断——续开会话（恢复的 conversation_history 已含 assistant 消息）
    同样命中注入，修复「按历史推断把续开首轮误判非首轮、永不注入」的缺口；
    进程重启后同会话重注入一次属幂等无害（账单文本只读提示）。

    Contract: Preconditions: kwargs 含 session_id/task_id（is_first_turn/
    conversation_history 不再参与判定）；
    Postconditions: 首见且存在他账时返回 {"context": 提醒文本} 并置 injected
    标记；否则 None；永不 raise。
    """
    try:
        from plugins._shared_state import get_session_state

        sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "__default__")
        state = get_session_state(sid, _NAMESPACE)
        if state.get("debt_injected"):
            return None
        rows = _foreign_debts(sid)
        if not rows:
            return None
        state["debt_injected"] = True
        logger.info("todo_debt: 会话首轮注入 %d 项跨会话未清账", len(rows))
        return {"context": _render_debt_context(rows)}
    except Exception:
        logger.exception("todo_debt on_session_open failed")
        return None
