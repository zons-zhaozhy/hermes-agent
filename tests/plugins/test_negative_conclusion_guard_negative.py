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


def test_reminder_resets_probe_count_so_it_does_not_repeat_every_turn():
    """The reproduced false positive: two probes, then the reminder, then idle turns."""
    sid = "neg-test-s4-reset"
    for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
        guard.on_post_tool_call(
            tool_name="search_files", args={"pattern": "zzz_absent"},
            result={"total_count": 0}, session_id=sid,
        )
    assert guard.on_pre_llm_call(session_id=sid) is not None
    assert guard.on_pre_llm_call(session_id=sid) is None  # 期望: None=已提醒过, 不再逐轮重复
    guard.on_post_tool_call(
        tool_name="terminal", args={"command": "ls somewhere"}, result={}, session_id=sid,
    )
    assert guard.on_pre_llm_call(session_id=sid) is None  # 期望: None=清零后一次探测不满阈值


def test_reminder_is_capped_per_conversation():
    sid = "neg-test-s5-cap"
    injected = 0
    for _ in range(guard._MAX_INJECTS + 2):
        for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
            guard.on_post_tool_call(
                tool_name="search_files", args={"pattern": "zzz_absent"},
                result={"total_count": 0}, session_id=sid,
            )
        injected += guard.on_pre_llm_call(session_id=sid) is not None
    assert injected == guard._MAX_INJECTS  # 期望: 封顶, 不无限打扰


def test_discipline_owner_registers_negative_conclusion_guard_hooks() -> None:
    ctx = _OwnerCtx()

    discipline.register(ctx)

    assert guard.on_post_tool_call in ctx.hooks["post_tool_call"]
    assert guard.on_pre_llm_call in ctx.hooks["pre_llm_call"]
