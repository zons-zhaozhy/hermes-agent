"""``display.busy_interrupt_overrides_subagents`` — opt-in for interrupt over delegates.

By default a typed follow-up while the agent drives ``delegate_task`` children is demoted
from interrupt to queue (#30170: an interrupt cascades through ``_active_children`` and
aborts in-flight subagent work). Some users would rather have the interrupt fire and let the
goal judge redo the delegated work next turn; that is a cost trade, so it stays opt-in and
defaults OFF. Context compression in flight keeps its demotion unconditionally (#56391:
interrupting mid-rotation forks orphaned siblings).
"""

from __future__ import annotations

import pytest

from gateway.run_busy_interrupt_policy import (
    compression_demotion_reason,
    subagent_demotion_reason,
)


class _Runner:
    """Minimal stand-in carrying only the config flags the policy reads."""

    def __init__(self, *, overrides: bool = False):
        self._busy_interrupt_overrides_subagents = overrides


class TestSubagentDemotionReason:
    def test_default_off_keeps_demotion(self):
        runner = _Runner(overrides=False)
        reason = subagent_demotion_reason(runner, has_active_subagents=True)
        assert reason and "#30170" in reason  # 期望: 默认仍为子代理保护降级

    def test_opt_in_drops_subagent_demotion(self):
        runner = _Runner(overrides=True)
        assert subagent_demotion_reason(runner, has_active_subagents=True) is None  # 期望: 开启后不降级

    def test_no_subagents_never_demotes(self):
        for overrides in (False, True):
            runner = _Runner(overrides=overrides)
            assert subagent_demotion_reason(runner, has_active_subagents=False) is None


class TestCompressionDemotionReason:
    def test_compression_always_demotes(self):
        """#56391 is not configurable: rotating the session id under an interrupt forks siblings."""
        reason = compression_demotion_reason(compression_in_flight=True)
        assert reason and "#56391" in reason  # 期望: 压缩在途始终降级

    def test_no_compression_never_demotes(self):
        assert compression_demotion_reason(compression_in_flight=False) is None


class TestFlagParsing:
    def test_absent_config_defaults_false(self):
        from gateway.run_busy_interrupt_policy import parse_busy_interrupt_overrides_subagents

        assert parse_busy_interrupt_overrides_subagents(None) is False
        assert parse_busy_interrupt_overrides_subagents("") is False

    @pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
    def test_truthy_strings_enable(self, raw):
        from gateway.run_busy_interrupt_policy import parse_busy_interrupt_overrides_subagents

        assert parse_busy_interrupt_overrides_subagents(raw) is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off", "garbage"])
    def test_falsy_and_garbage_disable(self, raw):
        from gateway.run_busy_interrupt_policy import parse_busy_interrupt_overrides_subagents

        assert parse_busy_interrupt_overrides_subagents(raw) is False
