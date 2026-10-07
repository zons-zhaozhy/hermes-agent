"""block_escalation 升级标记会话隔离防线。

病案（10-07 哨兵 19:07 班，执行记录实测）：三 cron 任务同秒死于拦截升级，
真凶=插件 `_escalated: set` 进程级全局标记无会话归属——A 会话（如 cron 会话）
连续 2 次被拦触发升级后未消费标记，同进程内恰好收尾的 B 会话（夜间三任务）
的 transform_llm_output 被追加 ESCALATION_MARKER，scheduler 按失败记账。

修前行为：_escalated 全局单集合，B 会话输出被 A 会话的升级标记污染。
修后行为：_escalated 键控 session_id，A 的标记只污染 A 自己的输出。
"""

from __future__ import annotations

from typing import Iterator

import pytest

from plugins import block_escalation


@pytest.fixture(autouse=True)
def _clean_state(
    tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    # 期望: 每用例独立 db + 干净升级标记，测试间零串扰
    monkeypatch.setattr(block_escalation, "_db_path", tmp_path / "be.db")
    block_escalation._reset_for_test()
    yield
    block_escalation._reset_for_test()


def _arm_escalation(session_id: str) -> None:
    # 期望: 同 session 两次 blocked 写意图 → 升级标记落本 session 且未消费
    # （复现病案时序：A 置标后没有再输出 turn，标记悬在进程里）
    args = {"path": "/tmp/victim_%s.py" % session_id, "content": "x"}
    for _ in range(2):
        block_escalation._on_post_tool_call(
            tool_name="write_file", status="blocked",
            args=args, session_id=session_id)


def test_escalated_marker_scoped_to_own_session() -> None:
    # 期望: A 置标未消费时，B 会话收尾输出原样返回（病案核心）——
    # 依据: 修前全局 set 使 B 的 transform 命中悬置标记被追加 marker
    # （探针实测 marker in B: True）；修后键控 session_id，sess_B 桶空零污染
    _arm_escalation("sess_A")
    b = block_escalation._on_transform_llm_output(
        response_text="B 任务正常收尾", session_id="sess_B")
    assert block_escalation.ESCALATION_MARKER not in b  # 期望: B 零污染
    assert b == "B 任务正常收尾"  # 期望: 原文逐字返回
    # A 自己的消费不受影响
    a = block_escalation._on_transform_llm_output(
        response_text="A 会话收尾", session_id="sess_A")
    assert block_escalation.ESCALATION_MARKER in a  # 期望: A 自身仍被终止


def test_unknown_session_defaults_isolated() -> None:
    # 期望: kwargs 未带 session_id 的调用（空串桶）不继承他处标记——
    # 依据: sess_A 标记只在 sess_A 桶，与 "" 桶无交集
    _arm_escalation("sess_A")
    plain = block_escalation._on_transform_llm_output(
        response_text="无会话身份的输出")
    assert block_escalation.ESCALATION_MARKER not in plain  # 期望: 空桶零污染


def test_marker_single_consumption_per_session() -> None:
    # 期望: 同一 session 标记单次消费（第二次输出原样返回，禁重复堆叠）——
    # 依据: 单次消费不变量收窄作用域后逐 session 保持
    _arm_escalation("sess_A")
    first = block_escalation._on_transform_llm_output(
        response_text="A 会话第一个 turn 的输出", session_id="sess_A")
    assert block_escalation.ESCALATION_MARKER in first  # 期望: 首次消费生效
    second = block_escalation._on_transform_llm_output(
        response_text="A 会话第二个 turn 的输出", session_id="sess_A")
    assert block_escalation.ESCALATION_MARKER not in second  # 期望: 单次消费
