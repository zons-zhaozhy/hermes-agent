"""The discipline suite routes engineering work to bundled workflow skills."""

from __future__ import annotations

import pytest


def test_first_engineering_turn_receives_one_routing_hint() -> None:
    from plugins.discipline import workflow_router

    session_id = "workflow-router-first"
    workflow_router._forget(session_id)
    try:
        first = workflow_router.on_pre_llm_call(
            session_id=session_id, platform="cli", user_message="帮我修复这个报错"
        )
        second = workflow_router.on_pre_llm_call(
            session_id=session_id, platform="cli", user_message="继续修复测试"
        )
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


def test_workflow_routes_are_registered_as_data() -> None:
    from plugins.discipline import workflow_router

    assert dict(workflow_router.WORKFLOW_ROUTES) == {
        "新功能/重构/接口迁移": "brainstorming",
        "工程调查/结构/调用链": "repository-investigation",
        "故障/报错/性能异常": "systematic-debugging",
        "行为变更/缺陷修复": "test-driven-development",
        "等待/轮询/异步就绪": "condition-based-waiting",
        "隔离改造/脏工作区": "using-git-worktrees",
        "已批准的多步架构方案": "writing-plans",
        "按已写计划执行": "executing-plans",
        "收到评审反馈": "receiving-code-review",
        "实现后/提交前评审": "requesting-code-review",
        "完成/集成/清理分支": "finishing-a-development-branch",
        "代码规模/语言构成": "codebase-inspection",
        "宣称完成/修复/通过前": "verification-before-completion",
    }
    assert workflow_router.WORKFLOW_SKILLS == frozenset(
        skill for _situation, skill in workflow_router.WORKFLOW_ROUTES
    )


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


def test_ordinary_conversation_does_not_receive_routing_hint() -> None:
    from plugins.discipline import workflow_router

    session_id = "workflow-router-ordinary"
    workflow_router._forget(session_id)
    try:
        assert workflow_router.on_pre_llm_call(
            session_id=session_id,
            platform="cli",
            user_message="你还在工作吗",
            conversation_history=[],
        ) == {}
        assert workflow_router.on_pre_llm_call(
            session_id=session_id,
            platform="cli",
            user_message="我今天有点累",
            conversation_history=[],
        ) == {}
        assert workflow_router.on_pre_llm_call(
            session_id=session_id,
            platform="cli",
            user_message="帮我看这个报错",
            conversation_history=[],
        ).get("context")
    finally:
        workflow_router._forget(session_id)


def test_session_end_forgets_state() -> None:
    from plugins.discipline import workflow_router

    session_id = "workflow-router-reset"
    workflow_router._forget(session_id)
    try:
        workflow_router.on_pre_llm_call(
            session_id=session_id, platform="cli", user_message="重构这个接口"
        )
        workflow_router.on_session_end(session_id=session_id)
        assert workflow_router.on_pre_llm_call(
            session_id=session_id, platform="cli", user_message="继续重构"
        ).get("context")
    finally:
        workflow_router._forget(session_id)
