"""The discipline suite routes engineering work to bundled workflow skills."""

from __future__ import annotations

import pytest


def test_first_engineering_turn_receives_one_routing_hint() -> None:
    from plugins.discipline import workflow_router

    session_id = "workflow-router-first"
    workflow_router._forget(session_id)
    try:
        first = workflow_router.on_pre_llm_call(session_id=session_id, platform="cli")
        second = workflow_router.on_pre_llm_call(session_id=session_id, platform="cli")
    finally:
        workflow_router._forget(session_id)

    assert set(first) == {"context"}
    assert "brainstorming" in first["context"]
    assert "systematic-debugging" in first["context"]
    assert "test-driven-development" in first["context"]
    assert "writing-plans" in first["context"]
    assert "executing-plans" in first["context"]
    assert "receiving-code-review" in first["context"]
    assert "codebase-inspection" in first["context"]
    assert "verification-before-completion" in first["context"]
    assert "skill_view" in first["context"]
    assert second == {}


@pytest.mark.parametrize("platform", ("cron", "subagent", "batch"))
def test_automation_surfaces_do_not_receive_the_hint(platform: str) -> None:
    from plugins.discipline import workflow_router

    session_id = f"workflow-router-{platform}"
    workflow_router._forget(session_id)
    try:
        assert workflow_router.on_pre_llm_call(
            session_id=session_id, platform=platform
        ) == {}
    finally:
        workflow_router._forget(session_id)


def test_session_end_forgets_state() -> None:
    from plugins.discipline import workflow_router

    session_id = "workflow-router-reset"
    workflow_router._forget(session_id)
    try:
        workflow_router.on_pre_llm_call(session_id=session_id, platform="cli")
        workflow_router.on_session_end(session_id=session_id)
        assert workflow_router.on_pre_llm_call(
            session_id=session_id, platform="cli"
        ).get("context")
    finally:
        workflow_router._forget(session_id)
