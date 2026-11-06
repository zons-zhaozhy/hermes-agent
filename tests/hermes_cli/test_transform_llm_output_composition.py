"""Reply transforms compose in registration order and failures stay visible.

Several policy plugins transform the same final reply (evidence, boundary and
SOP markers). Running every callback against the original and taking the first
string silently hid every later gate; running them sequentially keeps each
gate's contribution and lets the next gate inspect the previous rewrite.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from agent.turn_finalizer import apply_llm_output_transform
from hermes_cli.plugins import PluginManager


def test_output_transforms_compose_and_see_prior_rewrite() -> None:
    manager = PluginManager()
    seen: list[str] = []

    def first(response_text: str, **_kwargs) -> str:
        seen.append(response_text)
        return response_text + "\n[first gate]"

    def second(response_text: str, **_kwargs) -> str:
        seen.append(response_text)
        return response_text + "\n[second gate]"

    manager._hooks["transform_llm_output"] = [first, second]
    results = manager.invoke_hook(
        "transform_llm_output", response_text="answer", session_id="compose", turn_id="t1",
    )

    assert seen == ["answer", "answer\n[first gate]"]
    assert results == ["answer\n[first gate]\n[second gate]"]


def test_output_transform_failure_is_visible_unknown() -> None:
    manager = PluginManager()

    def broken(response_text: str, **_kwargs) -> str:
        raise RuntimeError("judge backend unavailable")

    manager._hooks["transform_llm_output"] = [broken]
    results = manager.invoke_hook(
        "transform_llm_output", response_text="answer", session_id="broken", turn_id="t1",
    )

    assert len(results) == 1
    assert results[0].startswith("answer")
    assert "门禁不可用 / UNKNOWN" in results[0]
    assert "broken" in results[0]


def test_held_output_transforms_also_compose() -> None:
    manager = PluginManager()

    def first(response_text: str, **_kwargs) -> str:
        return response_text + "[first]"

    def second(response_text: str, **_kwargs) -> str:
        return response_text + "[second]"

    rewrite, failed = manager.run_held_transforms(
        (first, second),
        response_text="answer",
        session_id="held",
        turn_id="t1",
    )

    assert failed is False
    assert rewrite == "answer[first][second]"


def test_async_output_transforms_compose_and_see_prior_rewrite() -> None:
    manager = PluginManager()
    seen: list[str] = []

    async def first(response_text: str, **_kwargs) -> str:
        seen.append(response_text)
        return response_text + "\n[first gate]"

    async def second(response_text: str, **_kwargs) -> str:
        seen.append(response_text)
        return response_text + "\n[second gate]"

    manager._hooks["transform_llm_output"] = [first, second]
    results = asyncio.run(manager.ainvoke_hook(
        "transform_llm_output", response_text="answer", session_id="async-compose", turn_id="t1",
    ))

    assert seen == ["answer", "answer\n[first gate]"]
    assert results == ["answer\n[first gate]\n[second gate]"]


def test_async_output_transform_failure_is_visible_unknown() -> None:
    manager = PluginManager()

    async def broken(response_text: str, **_kwargs) -> str:
        raise RuntimeError("judge backend unavailable")

    manager._hooks["transform_llm_output"] = [broken]
    results = asyncio.run(manager.ainvoke_hook(
        "transform_llm_output", response_text="answer", session_id="async-broken", turn_id="t1",
    ))

    assert len(results) == 1
    assert results[0].startswith("answer")
    assert "门禁不可用 / UNKNOWN" in results[0]
    assert "broken" in results[0]


def test_transform_dispatch_failure_is_visible_unknown() -> None:
    agent = SimpleNamespace(
        session_id="dispatch-broken",
        session_cwd="/workspace/project",
        model="m",
        platform="cli",
        _llm_output_transform=None,
    )
    with patch(
        "hermes_cli.lifecycle.invoke_hook",
        side_effect=RuntimeError("plugin manager unavailable"),
    ):
        final, transformed, _ = apply_llm_output_transform(
            agent, "answer", turn_id="t1",
        )

    assert transformed is True
    assert final.startswith("answer")
    assert "门禁不可用 / UNKNOWN" in final
