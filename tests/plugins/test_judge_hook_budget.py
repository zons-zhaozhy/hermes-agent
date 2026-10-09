"""judge 预算治理契约：插件不写死秒数，预算落到 auxiliary.<task>.timeout。

背景：pre_llm_call / post_tool_call / transform_llm_output 都在
hermes_cli/plugins_dispatch.py 的 _HOOK_TIMEOUT_BOUNDED_HOOKS 内，预算
_HOOK_CALLBACK_TIMEOUT_SECS=30s：超时即 abandon，且调用路径先白等满预算。
judge 侧一旦写死秒数（形参默认或实参），auxiliary.<task>.timeout 配置面即被
永久屏蔽——config 写 10s 挡不住写死的 20s，钩子被丢弃、状态与注入静默丢失。
契约：① 三个公开入口的 timeout 默认一律 None（预算交配置面解析）
      ② 默认调用确实把 None 透传给 call_llm（配置面保持开放、可调）

Contract:
  Preconditions: call_llm 全部打桩（零真实网络调用）；三个入口名存在
  Postconditions: 默认值 None 且透传 None，解析权留在 _effective_aux_timeout
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import pytest

from plugins import _llm_judge

_ENTRIES = ("llm_judge_bool", "llm_judge_multi", "llm_decision_typed")

# (入口名, 调用时的必需 kwargs)
_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("llm_judge_bool", {"true_key": "decision"}),
    ("llm_judge_multi", {"keys": ["a"]}),
    ("llm_decision_typed", {"true_key": "decision"}),
)


@pytest.mark.parametrize("entry", _ENTRIES)
def test_timeout_default_defers_to_config(entry: str) -> None:
    fn = getattr(_llm_judge, entry)
    # 期望: None —— 预算唯一治理面是 auxiliary.<task>.timeout（config.yaml）
    assert inspect.signature(fn).parameters["timeout"].default is None  # 期望: 默认 None


@pytest.mark.parametrize(("entry", "extra"), _CALLS)
def test_default_call_forwards_none_to_call_llm(
    entry: str, extra: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    def _fake_call_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs.get("timeout"))
        return {}

    # call_llm 是函数内延迟导入 → 打桩点必须是源模块属性
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
    fn: Callable[..., Any] = getattr(_llm_judge, entry)
    fn(task="budget_probe", system="s", text="话题", **extra)
    assert seen, f"{entry} 未调用 call_llm（测试与实现漂移）"  # 期望: 至少有调用发生
    assert all(t is None for t in seen), seen  # 期望: 全部透传 None（写死 20.0 会在此现形）
