"""failure_preflight 负例契约测试——动手类任务必须被注入失败预演要求（违规必红）。

Contract:
    Preconditions: plugins/discipline/failure_preflight.py 存在
    Postconditions: 动词开头任务消息触发 context 注入；讨论句式不触发
"""
from typing import Callable

from plugins import discipline
from plugins.discipline import failure_preflight as guard


class _OwnerCtx:
    def __init__(self) -> None:
        self.hooks: dict[str, list[Callable[..., object]]] = {}

    def register_hook(self, name: str, callback: Callable[..., object]) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_system_prompt_section(self, section_id: str, renderer: Callable[..., str]) -> None:
        return None


def test_negative_action_task_gets_preflight_injection():
    out = guard._on_pre_llm_call(user_message="部署这个服务到云端")
    assert "context" in out  # 期望: 动手任务被注入失败预演要求（拦截裸动手）


def test_negative_action_detection():
    assert guard._is_action_task("写一个脚本处理数据") is True  # 期望: True=动词开头识别
    assert guard._is_action_task("为什么这个设计是这样") is False  # 期望: False=讨论句式豁免
    assert guard._is_action_task("") is False  # 期望: False=空消息豁免


def test_negative_discussion_not_injected():
    out = guard._on_pre_llm_call(user_message="能不能实现这个功能")
    assert out == {}  # 期望: {}=疑问句式不注入


def test_discipline_owner_registers_failure_preflight() -> None:
    ctx = _OwnerCtx()

    discipline.register(ctx)

    contexts = [
        result.get("context", "")
        for callback in ctx.hooks["pre_llm_call"]
        if (result := callback(user_message="部署这个服务到云端", session_id="preflight-owner"))
    ]
    assert any("失败说明书" in context for context in contexts)
