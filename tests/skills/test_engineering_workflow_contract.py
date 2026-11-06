"""Engineering workflow routes must resolve through the real skill loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import skills_tool


REPO = Path(__file__).resolve().parents[2]
EXPECTED_SKILLS = (
    "brainstorming",
    "repository-investigation",
    "systematic-debugging",
    "test-driven-development",
    "condition-based-waiting",
    "using-git-worktrees",
    "writing-plans",
    "executing-plans",
    "receiving-code-review",
    "requesting-code-review",
    "finishing-a-development-branch",
    "codebase-inspection",
    "verification-before-completion",
)


@pytest.fixture(autouse=True)
def _bundled_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", REPO / "skills")
    skills_tool._SKILLS_CACHE.clear()


@pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
def test_workflow_target_loads_through_real_loader(skill_name: str) -> None:
    payload = json.loads(skills_tool.skill_view(skill_name))

    assert payload["success"], payload
    assert payload["name"] == skill_name
    assert payload["content"].strip()


def test_repository_investigation_is_discoverable() -> None:
    listing = json.loads(
        skills_tool.skills_list(query="repository callers history invariants")
    )

    assert any(skill["name"] == "repository-investigation" for skill in listing["skills"])


def test_condition_based_waiting_is_discoverable() -> None:
    listing = json.loads(skills_tool.skills_list(query="waiting condition deadline event"))

    assert any(skill["name"] == "condition-based-waiting" for skill in listing["skills"])


def test_using_git_worktrees_is_discoverable() -> None:
    listing = json.loads(skills_tool.skills_list(query="git worktree isolation baseline"))

    assert any(skill["name"] == "using-git-worktrees" for skill in listing["skills"])


def test_finishing_development_branch_is_discoverable() -> None:
    listing = json.loads(skills_tool.skills_list(query="branch integration verification cleanup"))

    assert any(
        skill["name"] == "finishing-a-development-branch" for skill in listing["skills"]
    )


def test_branch_completion_preserves_refused_worktree_cleanup() -> None:
    payload = json.loads(skills_tool.skill_view("finishing-a-development-branch"))

    assert "force-remove it" in payload["content"]
    assert "uncommitted or untracked files" in payload["content"]


def test_tdd_skill_uses_repository_test_runner() -> None:
    payload = json.loads(skills_tool.skill_view("test-driven-development"))
    content = payload["content"]

    assert "scripts/run_tests.sh" in content
    assert "pytest tests/" not in content


def test_debugging_skill_uses_repository_test_runner() -> None:
    payload = json.loads(skills_tool.skill_view("systematic-debugging"))
    content = payload["content"]

    assert "scripts/run_tests.sh" in content
    assert "pytest tests/" not in content


def test_review_skill_does_not_auto_commit_or_delegate() -> None:
    payload = json.loads(skills_tool.skill_view("requesting-code-review"))
    content = payload["content"]

    assert "does not authorize" in content
    assert "git add -A" not in content
    assert "git commit" not in content
    assert "delegate_task" not in content


def test_workflow_chain_explicitly_hands_off_to_new_skills() -> None:
    brainstorming = json.loads(skills_tool.skill_view("brainstorming"))["content"]
    writing_plans = json.loads(skills_tool.skill_view("writing-plans"))["content"]
    executing_plans = json.loads(skills_tool.skill_view("executing-plans"))["content"]

    assert "repository-investigation" in brainstorming
    assert "using-git-worktrees" in writing_plans
    assert "using-git-worktrees" in executing_plans
    assert "finishing-a-development-branch" in executing_plans
