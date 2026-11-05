"""Shared semantic-judge failures open a bounded, task-local circuit."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_REPO = Path(__file__).resolve().parents[2]


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


def test_multi_failure_suppresses_repeat_calls_for_same_task(monkeypatch):
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

    assert first == {"done": None}
    assert second == {"done": None}
    assert other_task == {"done": None}
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
