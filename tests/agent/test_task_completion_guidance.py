"""Autonomy and approval boundaries in the universal task-completion guidance."""

from agent.prompt_builder import TASK_COMPLETION_GUIDANCE


def test_autonomy_has_cost_and_external_effect_boundaries():
    assert "local, reversible, in-scope work" in TASK_COMPLETION_GUIDANCE
    assert "without making the user approve routine steps" in TASK_COMPLETION_GUIDANCE
    assert "Before spending money" in TASK_COMPLETION_GUIDANCE
    assert "external side effect" in TASK_COMPLETION_GUIDANCE
    assert "explicit user authorization" in TASK_COMPLETION_GUIDANCE
    assert "external read-only lookup" in TASK_COMPLETION_GUIDANCE
