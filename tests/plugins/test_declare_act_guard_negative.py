"""declare_act_guard 负例契约测试——宣告-动手配对守卫的注入闸必须真拦（违规必红）。

Contract:
    Preconditions: plugins/discipline/declare_act_guard.py 存在
    Postconditions: 超封顶/同文本重放被注入闸拒绝；违规注入文本含配对纪律要求
"""
from typing import Callable

from plugins import discipline
from plugins.discipline import declare_act_guard as guard


class _OwnerCtx:
    def __init__(self) -> None:
        self.hooks: dict[str, list[Callable[..., object]]] = {}

    def register_hook(self, name: str, callback: Callable[..., object]) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_system_prompt_section(self, section_id: str, renderer: Callable[..., str]) -> None:
        return None


def test_negative_gate_blocks_over_cap():
    state = {"injected_count": guard._MAX_INJECTS, "last_text": ""}
    assert guard._gate_ok(state, "some text") is False  # 期望: False=超封顶注入闸拦截


def test_negative_gate_blocks_same_text_replay():
    state = {"injected_count": 0, "last_text": "repeat me"}
    assert guard._gate_ok(state, "repeat me") is False  # 期望: False=同文本重放拦截


def test_negative_injection_carries_pairing_rule():
    out = guard._build_injection()
    assert "禁止裸宣告" in out.get("context", "")  # 期望: 注入文本含配对纪律（拦截语义）
    assert "必须先确认" in out.get("context", "")  # 期望: 注入文本给出须拍板出路


def test_negative_gate_allows_fresh_text_under_cap():
    state = {"injected_count": 0, "last_text": "old"}
    assert guard._gate_ok(state, "brand new text") is True  # 期望: True=未超限新文本放行


def test_discipline_owner_registers_declare_act_guard_hooks() -> None:
    ctx = _OwnerCtx()

    discipline.register(ctx)

    assert guard._on_pre_llm_call in ctx.hooks["pre_llm_call"]
    assert guard._on_post_llm_call in ctx.hooks["post_llm_call"]
