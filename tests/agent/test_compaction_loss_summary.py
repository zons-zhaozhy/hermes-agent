"""_compaction_loss_summary 行为契约（期望值独立推导自丢弃告知设计）。

契约：
1. 零丢弃（无标记消息）→ 空串（裸 COMPACTION_DONE_STATUS 照发）
2. 技能剪枝+工具降级并存 → 一行内含两类计数与 session_search 恢复指引
3. None / 非 dict 混合输入 → 不抛异常
"""

from __future__ import annotations

from agent.conversation_compression import _compaction_loss_summary


def test_zero_drops_returns_empty() -> None:
    msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert _compaction_loss_summary(msgs) == ""  # 期望: 无标记即零丢弃


def test_counts_pruned_skills_and_demoted_outputs() -> None:
    msgs = [
        {"role": "user", "content": "[SKILL_PRUNED: content lost in compression]"},
        {"role": "user", "content": "another [SKILL_PRUNED: gone] marker"},
        {"role": "tool", "content": "[terminal output demoted at compaction — 1,234 chars preserved]"},
    ]
    out = _compaction_loss_summary(msgs)
    assert "2 skill body(ies) pruned" in out  # 期望: 两个剪枝标记计数
    assert "1 tool output(s) demoted" in out  # 期望: 一个降级指针计数
    assert "session_search" in out  # 期望: 给出找回路径


def test_none_and_mixed_input_never_raises() -> None:
    # skill-pruned-ok: new contract test, unrelated to any skill body
    assert _compaction_loss_summary(None) == ""  # 期望: None 视为零丢弃
    assert _compaction_loss_summary(["bare string", 42, None]) == ""  # 期望: 非 dict 混合不抛
