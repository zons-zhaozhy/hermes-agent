"""finish_guard todo_debt 跨会话任务债务账本回归测试。

事故形态（2026-10-25 用户批评原话）：「hermes 总是喜欢先挂账、先登记、
后续做，却又停着不做」——登记介质是会话内存态 TodoStore（one per
session），会话结束即蒸发；新会话启动不注入任何未清账，「后续做」永远
没有触发点（state.db 实测 24 个会话的末条消息停在挂账措辞上）。

修法（闭环消费端）：todo_list 写入后 pending 项按会话槽位持久化到
<HERMES_HOME>/cache/todo_debt.json；新会话首轮注入他账清单；同 id 完成
跨槽位自动销账。措辞面同步补 _DEFER_RE 实测逃逸词（待办/先记录/后续做
/后续会话/稍后做）与完成态否定豁免（无待办等）。

本文件锁五件事：T1 逃逸词命中、T2 完成态否定不误伤、T3 账本持久化与
跨槽位销账、T4 首轮注入一次/会话且排除本会话槽位、T5 非 todo 工具零副作用。
"""
from __future__ import annotations

import json

from plugins import finish_guard as fg
from plugins.finish_guard import todo_debt


def _write_snapshot(sid: str, items: list[dict]) -> None:
    """以 post_tool_call 载荷形态喂一次 todo_list 快照。"""
    todo_debt.on_todo_write(
        tool_name="todo_list",
        result=json.dumps({"todos": items, "summary": {}}, ensure_ascii=False),
        session_id=sid,
    )


def test_defer_escape_words_now_caught():
    """T1：2026-10-25 实测逃逸措辞全部命中（待办 491 次/先记录 9 次的来源形态）。"""
    for phrase in ("先记录这条，后续做", "登记为待办，稍后做", "此事后续会话落盘", "列为待办"):
        assert fg._is_defer_pushout(phrase) is True  # 期望: 逃逸词命中拦截


def test_done_negation_not_flagged():
    """T2：完成态否定（无待办/待办清零）是完成声明，不误伤。"""
    for phrase in ("无待办，全部缺口已落地并验证。", "待办清零，无未清事项。", "没有待办，工作状态不变。"):
        assert fg._is_defer_pushout(phrase) is False  # 期望: 否定形态放行
    out = fg._transform_llm_output(response_text="无待办，全部完成。")
    assert "[finish_guard]" not in out  # 期望: 完成态不打标


def test_ledger_persists_only_active_items():
    """T3：todo 快照只落 pending/in_progress 项，completed 不落。"""
    _write_snapshot("s_ledger", [
        {"id": "1", "content": "修 A", "status": "pending"},
        {"id": "2", "content": "修 B", "status": "in_progress"},
        {"id": "3", "content": "修 C", "status": "completed"},
    ])
    sessions = todo_debt._load_debt()["sessions"]
    ids = [it["id"] for it in sessions["s_ledger"]["items"]]
    assert ids == ["1", "2"]  # 期望: 仅 active 项落盘


def test_cross_slot_settlement_clears_foreign():
    """T3：同 id 在他账槽位被 completed/cancelled 后跨槽位销账；清空即删文件。"""
    _write_snapshot("s_old", [{"id": "9", "content": "旧账", "status": "pending"}])
    _write_snapshot("s_new", [{"id": "9", "content": "旧账", "status": "cancelled"}])
    sessions = todo_debt._load_debt()["sessions"]
    assert "s_old" not in sessions  # 期望: 旧槽位被销账删除
    assert "s_new" not in sessions  # 期望: cancelled 项不落本槽（全账清空删文件）
    assert not todo_debt._debt_path().exists()  # 期望: 零残留


def test_session_open_injects_foreign_debt_once():
    """T4：进程内首见注入他账一次；同会话不重复；续开会话（历史含 assistant）同样注入；本会话槽位排除。"""
    _write_snapshot("s_owner", [{"id": "5", "content": "跨会话未清事项X", "status": "pending"}])
    first = todo_debt.on_session_open(conversation_history=[{"role": "user", "content": "hi"}],
                                      session_id="s_fresh")
    assert first is not None and "todo_debt" in first["context"]  # 期望: 进程内首见注入
    assert "跨会话未清事项X" in first["context"]  # 期望: 清单含他账内容
    again = todo_debt.on_session_open(conversation_history=[{"role": "user", "content": "hi"}],
                                      session_id="s_fresh")
    assert again is None  # 期望: 同会话只注入一次（debt_injected 标记）
    own = todo_debt.on_session_open(conversation_history=[], session_id="s_owner")
    assert own is None  # 期望: 本会话自己的在途任务不算他账
    resumed = todo_debt.on_session_open(
        conversation_history=[{"role": "user", "content": "早"},
                              {"role": "assistant", "content": "已开工"}],
        session_id="s_resume")
    assert resumed is not None and "跨会话未清事项X" in resumed["context"]  # 期望: 续开会话进程内首见同样注入


def test_non_todo_tool_no_side_effect():
    """T5：非 todo_list 工具不触发账本写入。"""
    before = todo_debt._debt_path().exists()
    todo_debt.on_todo_write(tool_name="read_file", result="{}", session_id="s_x")
    assert todo_debt._debt_path().exists() == before  # 期望: 账本无变化
