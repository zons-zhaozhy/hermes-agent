"""diff_debt shares guard session policy and releases ended-session state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Iterator

import pytest

from plugins.guards import diff_debt, four_axis, session_policy
from plugins.guards import register as register_guards


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    diff_debt._DIFF_DEBT.clear()
    diff_debt._BATCH_EVIDENCE.clear()
    four_axis._absence_warned.clear()
    yield
    diff_debt._DIFF_DEBT.clear()
    diff_debt._BATCH_EVIDENCE.clear()
    four_axis._absence_warned.clear()


def _capture_registration() -> dict[str, Callable[..., Any]]:
    callbacks: dict[str, Callable[..., Any]] = {}
    diff_debt.register(SimpleNamespace(
        register_hook=lambda name, callback: callbacks.setdefault(name, callback),
        register_system_prompt_section=lambda *_args: None,
    ))
    return callbacks


def test_guards_registration_installs_shared_session_policy() -> None:
    callbacks: dict[str, list[Callable[..., Any]]] = {}
    register_guards(SimpleNamespace(
        register_hook=lambda name, callback: callbacks.setdefault(name, []).append(callback),
        register_system_prompt_section=lambda *_args: None,
    ))

    assert session_policy.record_session_platform in callbacks["pre_llm_call"]
    assert session_policy.forget_session in callbacks["on_session_reset"]
    assert session_policy.forget_session in callbacks["on_session_end"]


def test_unified_cron_platform_exempts_diff_debt() -> None:
    four_axis.on_pre_llm_call(session_id="local-cron", platform="cron")

    diff_debt.on_post_tool_call(
        tool_name="write_file", args={"path": "/tmp/a.py"},
        session_id="local-cron",
    )

    assert "local-cron" not in diff_debt._DIFF_DEBT
    assert diff_debt.on_pre_tool_call(
        tool_name="write_file", args={"path": "/tmp/a.py"},
        session_id="local-cron",
    ) is None


def test_session_end_clears_diff_debt_and_batch_flag() -> None:
    callbacks = _capture_registration()
    assert "on_session_end" in callbacks
    diff_debt.on_post_tool_call(
        tool_name="write_file", args={"path": "/tmp/a.py"}, session_id="s-end",
    )
    diff_debt.on_pre_tool_batch(assistant_content="done", session_id="s-end")

    callbacks["on_session_end"](session_id="s-end")

    assert "s-end" not in diff_debt._DIFF_DEBT
    assert "s-end" not in diff_debt._BATCH_EVIDENCE


def test_session_reset_keeps_other_sessions_isolated() -> None:
    callbacks = _capture_registration()
    for sid in ("s-reset", "s-other"):
        diff_debt.on_post_tool_call(
            tool_name="write_file", args={"path": f"/tmp/{sid}.py"}, session_id=sid,
        )
        diff_debt.on_pre_tool_batch(assistant_content="done", session_id=sid)

    callbacks["on_session_reset"](session_id="s-reset")

    assert "s-reset" not in diff_debt._DIFF_DEBT
    assert "s-reset" not in diff_debt._BATCH_EVIDENCE
    assert len(diff_debt._DIFF_DEBT["s-other"]) == 1
    assert diff_debt._BATCH_EVIDENCE["s-other"] is False
