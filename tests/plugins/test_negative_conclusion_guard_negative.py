"""negative_conclusion_guard 负例契约测试——零命中探测积累必须触发权威入口提醒（违规必红）。

Contract:
    Preconditions: plugins/discipline/negative_conclusion_guard.py 存在
    Postconditions: 探测计数达阈值且无权威验证时 pre_llm_call 注入提醒
"""
from typing import Callable

from plugins import discipline
from plugins.discipline import negative_conclusion_guard as guard


class _OwnerCtx:
    def __init__(self) -> None:
        self.hooks: dict[str, list[Callable[..., object]]] = {}

    def register_hook(self, name: str, callback: Callable[..., object]) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_system_prompt_section(self, section_id: str, renderer: Callable[..., str]) -> None:
        return None


def test_negative_probe_accumulation_triggers_reminder():
    sid = "neg-test-s1"
    for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
        guard.on_post_tool_call(
            tool_name="search_files", args={"pattern": "zzz_absent"},
            result={"total_count": 0}, session_id=sid,
        )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is not None and "权威" in out.get("context", "")  # 期望: 注入权威入口提醒（拦截裸断言）


def test_negative_authority_check_settles_reminder():
    sid = "neg-test-s2"
    for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
        guard.on_post_tool_call(
            tool_name="search_files", args={"pattern": "zzz_absent"},
            result={"total_count": 0}, session_id=sid,
        )
    guard.on_post_tool_call(
        tool_name="terminal", args={"command": "hermes plugins list"},
        result={}, session_id=sid,
    )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is None  # 期望: None=权威验证后不再提醒


def test_negative_below_threshold_no_reminder():
    sid = "neg-test-s3"
    guard.on_post_tool_call(
        tool_name="search_files", args={"pattern": "zzz"},
        result={"total_count": 0}, session_id=sid,
    )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is None  # 期望: None=未达阈值不提醒


def test_discipline_owner_registers_negative_conclusion_guard_hooks() -> None:
    ctx = _OwnerCtx()

    discipline.register(ctx)

    assert guard.on_post_tool_call in ctx.hooks["post_tool_call"]
    assert guard.on_pre_llm_call in ctx.hooks["pre_llm_call"]
