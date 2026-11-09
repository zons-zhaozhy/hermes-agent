"""Shared semantic-judge failures open a bounded circuit — task-local for parse failures,
endpoint-scoped for hard (timeout/transport) failures."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_REPO = Path(__file__).resolve().parents[2]


class _Resp:
    """Minimal chat-completion stub for the "unusable answer" failure mode."""

    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))]


def _load_judge() -> types.ModuleType:
    if "hermes_plugins" not in sys.modules:
        namespace = types.ModuleType("hermes_plugins")
        namespace.__path__ = []
        sys.modules["hermes_plugins"] = namespace
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins._llm_judge", _REPO / "plugins" / "_llm_judge.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "hermes_plugins"
    sys.modules["hermes_plugins._llm_judge"] = module
    spec.loader.exec_module(module)
    return module


def test_endpoint_timeout_suppresses_every_task_on_that_endpoint(monkeypatch):
    """A dead judge endpoint costs ONE timeout per turn, not one per registered judge task.

    Regression for the 2026-11-06 gateway stall: six guard tasks each paid a 5-10 s timeout
    for the same unreachable endpoint (≈48 s per turn), which exhausted the ``pre_llm_call``
    hook budget and ended a turn with ``blocked_by_plugin_pre_llm_call`` and ``api_calls=0``.
    """
    module = _load_judge()
    calls: list[str] = []

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    first = module.llm_judge_multi(
        task="reply_side_guards", system="return JSON", text="finished", keys=["done"],
    )
    second = module.llm_judge_multi(
        task="reply_side_guards", system="return JSON", text="finished again", keys=["done"],
    )
    other_task = module.llm_judge_multi(
        task="user_side_guards", system="return JSON", text="continue", keys=["done"],
    )
    third_task = module.llm_judge_multi(
        task="scope_gate", system="return JSON", text="continue", keys=["done"],
    )

    assert first == {"done": None}
    assert second == {"done": None}
    assert other_task == {"done": None}
    assert third_task == {"done": None}
    # Only the first call actually reached the endpoint; the rest short-circuited.
    assert calls == ["reply_side_guards"]


def test_endpoint_timeout_leaves_a_task_on_another_endpoint_alone(monkeypatch):
    """Tasks may be configured onto different auxiliary models: a dead endpoint must not
    silence a task whose own endpoint is healthy."""
    module = _load_judge()
    calls: list[str] = []

    def resolve(task, provider=None, model=None, base_url=None, api_key=None):
        if task == "slow_guard":
            return ("local_judge", "qwen3.5:4b-mlx-judge", "http://127.0.0.1:11434/v1", None, None)
        return ("zai", "glm-5.3-flash", "https://open.bigmodel.cn/api/coding/paas/v4", None, None)

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model", resolve)

    assert module.llm_judge_multi(
        task="slow_guard", system="return JSON", text="finished", keys=["done"],
    ) == {"done": None}
    assert module.llm_judge_multi(
        task="fast_guard", system="return JSON", text="finished", keys=["done"],
    ) == {"done": None}

    assert calls == ["slow_guard", "fast_guard"]


def test_unconfigured_task_shares_the_circuit_of_a_task_pinned_to_the_same_endpoint(monkeypatch):
    """An ``auxiliary.<task>`` with no provider/model/base_url calls the MAIN provider.

    Keying its circuit on the "auto" sentinel put it on a different endpoint identity than the
    pinned tasks that really call the same host, so a dead endpoint made it pay a fresh timeout
    every turn while its peers were already short-circuited. Regression for the 2026-11-09 stall:
    ``read_think_intent`` (12 s × main + 3 global fallbacks) against the 30 s ``pre_llm_call``
    budget, twice, with ``api_calls=0``.
    """
    module = _load_judge()
    calls: list[str] = []
    endpoint = "https://open.bigmodel.cn/api/coding/paas/v4"

    def resolve(task, provider=None, model=None, base_url=None, api_key=None):
        if task == "read_think_intent":
            return ("auto", None, None, None, None)
        return ("zai", "glm-5.3-flash", endpoint, None, None)

    def main_route(_runtime, _task):
        return ("zai", "glm-5.3", endpoint, None, None)

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model", resolve)
    monkeypatch.setattr("agent.auxiliary_client._main_route_target", main_route)
    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)

    assert module._judge_endpoint_key("read_think_intent") == module._judge_endpoint_key("scope_gate")
    assert module.llm_judge_multi(
        task="scope_gate", system="return JSON", text="continue", keys=["done"],
    ) == {"done": None}
    assert module.llm_judge_multi(
        task="read_think_intent", system="return JSON", text="continue", keys=["done"],
    ) == {"done": None}

    assert calls == ["scope_gate"], "the dead endpoint costs one timeout per turn, not one per task"


def test_unusable_answer_stays_task_scoped(monkeypatch):
    """A parse failure is model misbehaviour on ONE prompt, not a dead endpoint: a sibling
    task on the same endpoint must still be judged."""
    module = _load_judge()
    calls: list[str] = []

    def garbage(**kwargs):
        calls.append(str(kwargs["task"]))
        return _Resp("这不是判定结果，只是一段散文。")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", garbage)

    assert module.llm_judge_multi(
        task="reply_side_guards", system="return JSON", text="finished", keys=["done"],
    ) == {"done": None}
    assert module.llm_judge_multi(
        task="user_side_guards", system="return JSON", text="continue", keys=["done"],
    ) == {"done": None}

    assert calls == ["reply_side_guards", "user_side_guards"]


def test_bool_failure_suppresses_repeat_calls_for_same_task(monkeypatch):
    module = _load_judge()
    calls: list[str] = []

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    first = module.llm_judge_bool(
        task="completion_boundary_audit", system="return JSON", text="finished",
    )
    second = module.llm_judge_bool(
        task="completion_boundary_audit", system="return JSON", text="finished again",
    )

    assert first is None
    assert second is None
    assert calls == ["completion_boundary_audit"]


def test_typed_failure_suppresses_repeat_calls_for_same_task(monkeypatch):
    module = _load_judge()
    calls: list[str] = []

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    first = module.llm_decision_typed(
        task="typed_guard", system="return JSON", text="finished", use_logprobs=False,
    )
    second = module.llm_decision_typed(
        task="typed_guard", system="return JSON", text="finished again", use_logprobs=False,
    )

    assert first.verdict is None
    assert second.verdict is None
    assert calls == ["typed_guard"]


def test_json_failure_suppresses_repeat_calls_for_same_task(monkeypatch):
    module = _load_judge()
    calls: list[str] = []

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    first = module.llm_judge_json(
        task="terminology_ledger", system="return JSON", text="finished", keys=["terms"],
    )
    second = module.llm_judge_json(
        task="terminology_ledger", system="return JSON", text="finished again", keys=["terms"],
    )

    assert first == {"terms": []}
    assert second == {"terms": []}
    assert calls == ["terminology_ledger"]


def test_repeated_failures_extend_the_cooldown(monkeypatch):
    module = _load_judge()
    clock = [1000.0]
    calls: list[float] = []

    def unavailable(**kwargs):
        calls.append(clock[0])
        raise TimeoutError("judge timed out")

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)

    def call():
        return module.llm_judge_multi(
            task="degraded_guard", system="return JSON", text="finished", keys=["done"],
        )

    assert call() == {"done": None}       # failure #1: cooldown until 1300
    clock[0] = 1301.0
    assert call() == {"done": None}       # failure #2: cooldown until 1901
    clock[0] = 1900.0
    assert call() == {"done": None}       # still open
    clock[0] = 1901.0
    assert call() == {"done": None}       # first retry after the second cooldown

    assert calls == [1000.0, 1301.0, 1901.0]


def test_circuit_is_partitioned_by_profile(monkeypatch, tmp_path):
    module = _load_judge()
    calls: list[str] = []

    def unavailable(**kwargs):
        calls.append(str(kwargs["task"]))
        raise TimeoutError("judge timed out")

    monkeypatch.setattr("agent.auxiliary_client.call_llm", unavailable)
    profile_a = set_hermes_home_override(tmp_path / "profiles" / "a")
    try:
        assert module.llm_judge_bool(
            task="shared_guard", system="return JSON", text="finished",
        ) is None
        assert module.llm_judge_bool(
            task="shared_guard", system="return JSON", text="finished again",
        ) is None
    finally:
        reset_hermes_home_override(profile_a)

    profile_b = set_hermes_home_override(tmp_path / "profiles" / "b")
    try:
        assert module.llm_judge_bool(
            task="shared_guard", system="return JSON", text="finished",
        ) is None
    finally:
        reset_hermes_home_override(profile_b)

    assert calls == ["shared_guard", "shared_guard"]
