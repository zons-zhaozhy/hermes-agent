"""reply-certainty-checker direct enforcement contract."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import pytest

from plugins._shared_state import clear_session

reply_certainty_checker = importlib.import_module("plugins.reply-certainty-checker")


class _Ctx:
    def __init__(self) -> None:
        self.hooks: dict[str, list[Callable[..., Any]]] = {}

    def register_hook(self, name: str, callback: Callable[..., Any]) -> None:
        self.hooks.setdefault(name, []).append(callback)


def test_uncertain_reply_injects_one_reminder_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "reply-certainty-direct"
    ctx = _Ctx()
    reply_certainty_checker.register(ctx)
    monkeypatch.setattr(
        reply_certainty_checker,
        "judge_reply_side",
        lambda _text: {"uncertain": True},
    )

    try:
        visible_reply = ctx.hooks["transform_llm_output"][0](
            "这个配置应该已经生效，但没有标注验证来源。",
            session_id=session_id,
        )
        first_reminder = ctx.hooks["pre_llm_call"][0](session_id=session_id)
        second_reminder = ctx.hooks["pre_llm_call"][0](session_id=session_id)

        assert visible_reply is None
        assert first_reminder is not None
        assert "确定性检查" in first_reminder["context"]
        assert "标注[实测]" in first_reminder["context"]
        assert second_reminder is None
    finally:
        clear_session(session_id)
