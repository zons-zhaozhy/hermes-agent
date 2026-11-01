"""零写入回合的 pre_verify 通道可达性回归。

背景（2026-11-01 用户主诉）：方案输出后以「是否按X开工？如无异议我按序施工」
停轮，本回合零文件写入。契约（v2，2026-11-01 回归修正）：上游默认契约=零写入
回合不咨询 pre_verify（test_hold_stream_exits 锁定）；需要守卫零写入停轮的
回调（finish_guard 的 todo 债执法）以 ``pre_verify_zero_edit = True`` 函数
属性声明式 opt-in（hermes_cli.plugins.pre_verify_zero_edit_requested 聚合），
机制与 ``hold_stream`` 同型。
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent.turn_stop_gates import _pre_verify_nudge


def _agent(**over: Any) -> SimpleNamespace:
    base = dict(
        _turn_file_mutation_paths=set(),
        session_id="s0",
        platform="cli",
        model="m",
        _resolved_is_coding=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_zero_edit_turn_consults_declared_hook() -> None:
    """零写入回合：仅当注册回调声明 pre_verify_zero_edit 时咨询（changed_paths=[] 如实透传）。"""
    agent = _agent()
    with (
        patch("agent.verify_hooks.max_verify_nudges", return_value=3),
        patch("hermes_cli.lifecycle.has_hook", side_effect=lambda n: n == "pre_verify"),
        patch(
            "hermes_cli.plugins.pre_verify_zero_edit_requested",
            return_value=True,
        ),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="continue working",
        ) as consult,
    ):
        out = _pre_verify_nudge(agent, "方案已列好，是否开工？", 0)
    assert out == "continue working"  # 期望: 声明回调的零写入咨询消息透传为续跑 nudge
    assert consult.call_args.kwargs["changed_paths"] == []  # 期望: 空改动集如实透传


def test_zero_edit_turn_skips_undeclared_hook() -> None:
    """零写入回合：未声明 pre_verify_zero_edit 的回调不被咨询（上游默认契约，
    test_hold_stream_exits::test_answer_kept_as_the_budget_fallback 锁定）。"""
    agent = _agent()
    with (
        patch("agent.verify_hooks.max_verify_nudges", return_value=3),
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.plugins.pre_verify_zero_edit_requested",
            return_value=False,
        ),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="should not be consulted",
        ) as consult,
    ):
        out = _pre_verify_nudge(agent, "Done, the deploy key is in place.", 0)
    assert out is None  # 期望: 未声明回调零写入不咨询、回合正常结束
    consult.assert_not_called()  # 期望: 分发器完全未被触发


def test_no_hook_registered_zero_edit_still_noop() -> None:
    """无 pre_verify 钩子注册时零写入回合零开销放行（无默认续跑成本语义保留）。"""
    agent = _agent()
    with (
        patch("agent.verify_hooks.max_verify_nudges", return_value=3),
        patch("hermes_cli.lifecycle.has_hook", return_value=False),
    ):
        out = _pre_verify_nudge(agent, "plain answer", 0)
    assert out is None  # 期望: 未注册钩子不产生续跑


def test_edited_turn_consult_unchanged() -> None:
    """有写入回合行为不变：钩子咨询与 changed_paths 透传照旧（opt-in 声明不参与）。"""
    agent = _agent(_turn_file_mutation_paths={"src/a.py"})
    with (
        patch("agent.verify_hooks.max_verify_nudges", return_value=3),
        patch("hermes_cli.lifecycle.has_hook", return_value=True),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="run tests",
        ) as consult,
    ):
        out = _pre_verify_nudge(agent, "edited", 0)
    assert out == "run tests"  # 期望: 有写入回合照常续跑
    assert consult.call_args.kwargs["changed_paths"] == ["src/a.py"]  # 期望: 路径排序透传
