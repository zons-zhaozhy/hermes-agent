"""Tests for agent.deliberation_gate — two-phase structural deliberation gate.

Covers: reasoning-phase gating, investigation tracking, unlock conditions,
round-based anti-loop, config parsing, tool classification.
"""

import pytest

from agent.read_think_gate import (
    ReadThinkGate,
    ReadThinkGateConfig,
    GATED_TOOL_NAMES,
)


# ── Config ──────────────────────────────────────────────────────────


class TestReadThinkGateConfig:
    def test_default_config(self):
        c = ReadThinkGateConfig()
        assert c.enabled is True
        assert c.max_reasoning_rounds == 5
        assert c.min_reasoning_chars == 80
        assert c.min_reflection_chars == 20

    def test_from_mapping_empty(self):
        c = ReadThinkGateConfig.from_mapping(None)
        assert c.enabled is True

    def test_from_mapping_partial(self):
        c = ReadThinkGateConfig.from_mapping(
            {"enabled": False, "max_reasoning_rounds": 3}
        )
        assert c.enabled is False
        assert c.max_reasoning_rounds == 3
        assert c.min_reasoning_chars == 80  # default

    def test_from_mapping_invalid(self):
        c = ReadThinkGateConfig.from_mapping(
            {"max_reasoning_rounds": -1, "min_reasoning_chars": 0}
        )
        assert c.max_reasoning_rounds == 5  # clamped
        assert c.min_reasoning_chars == 80  # clamped


# ── Phase state ─────────────────────────────────────────────────────


class TestPhaseState:
    """Gate starts in reasoning phase, transitions to execution phase."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gate = ReadThinkGate()

    def test_starts_in_reasoning_phase(self):
        assert self.gate.phase == "reasoning"
        assert self.gate.is_satisfied is False
        assert self.gate._reasoning_rounds == 0
        assert self.gate._investigation_done is False

    def test_reset_for_turn(self):
        # Get to execution phase
        self.gate.check_batch("x" * 80, ["terminal"])
        assert self.gate.phase == "execution"

        # Reset
        self.gate.reset_for_turn()
        assert self.gate.phase == "reasoning"
        assert self.gate._reasoning_rounds == 0
        assert self.gate._investigation_done is False


# ── Unlock conditions ───────────────────────────────────────────────


class TestUnlockConditions:
    """Three ways to unlock: direct reasoning, investigation+reflection, timeout."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gate = ReadThinkGate()

    # ── Direct reasoning ─────────────────────────────────────────────

    def test_direct_reasoning_unlocks(self):
        """Sufficient analysis text → immediate unlock."""
        result = self.gate.check_batch("x" * 80, ["terminal"])
        assert result is None
        assert self.gate.phase == "execution"
        assert self.gate._reasoning_rounds == 0  # direct reasoning doesn't consume a round

    def test_direct_reasoning_allows_tools(self):
        self.gate.check_batch("x" * 80, ["terminal"])
        # Once unlocked, even empty content passes
        result = self.gate.check_batch("", ["write_file", "patch"])
        assert result is None

    def test_below_threshold_stays_locked(self):
        """79 chars is not direct reasoning yet."""
        result = self.gate.check_batch("x" * 79, ["terminal"])
        assert result is not None
        assert self.gate.phase == "reasoning"

    # ── Investigation + reflection ───────────────────────────────────

    def test_investigation_then_reflection_unlocks(self):
        # Round 1: read-only tools (investigation)
        r1 = self.gate.check_batch("", ["read_file", "search_files"])
        assert r1 is None  # pass through
        assert self.gate._investigation_done is True
        assert self.gate.phase == "execution"  # unlocked — investigation done

        # Round 2: mutating tool with reflection → unlocks
        r2 = self.gate.check_batch("x" * 25, ["terminal"])
        assert r2 is None  # investigation done + 25 >= 20 → unlock
        assert self.gate.phase == "execution"

    def test_investigation_without_reflection_unlocks(self):
        """调查后无条件解锁——unlock_after_investigation=True (default)."""
        self.gate.check_batch("", ["read_file"])  # investigate
        # Try to mutate with NO reflection
        result = self.gate.check_batch("", ["terminal"])
        assert result is None  # unlocked — investigation done is sufficient
        assert self.gate.phase == "execution"

    def test_investigation_unlock_can_be_disabled(self):
        """unlock_after_investigation=False → 调查后仍需反射文本."""
        gate = ReadThinkGate(ReadThinkGateConfig(unlock_after_investigation=False))
        gate.check_batch("", ["read_file"])  # investigate
        result = gate.check_batch("", ["terminal"])  # no reflection
        assert result is not None  # blocked — need reflection
        assert gate.phase == "reasoning"

    def test_investigation_with_insufficient_reflection_unlocks(self):
        """调查后无条件解锁——即使 content < min_reflection_chars 也放行."""
        self.gate.check_batch("", ["read_file"])  # investigate
        result = self.gate.check_batch("OK", ["terminal"])  # 2 chars < 20
        assert result is None  # unlocked — investigation done is sufficient
        assert self.gate.phase == "execution"

    # ── Anti-loop: timeout ──────────────────────────────────────────

    def test_max_rounds_auto_unlock(self):
        """After max_reasoning_rounds blocked calls, the next call auto-unlocks."""
        # 先消耗 max_reasoning_rounds 次被拦截的调用，第 N+1 次才触发自动解锁
        for i in range(self.gate.config.max_reasoning_rounds):
            result = self.gate.check_batch("", ["terminal"])
            assert result is not None  # blocked
            assert self.gate.phase == "reasoning"
        # Next call: auto-unlock (counter >= max)
        result = self.gate.check_batch("", ["terminal"])
        assert result is None
        assert self.gate.phase == "execution"


# ── Block message quality ──────────────────────────────────────────


class TestBlockMessage:
    """Block messages should guide, not punish."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gate = ReadThinkGate()

    def test_no_investigation_message_guides_to_search(self):
        result = self.gate.check_batch("", ["terminal"])
        assert result is not None
        assert "search_files" in result
        assert "read_file" in result
        assert "web_search" in result

    def test_investigation_done_message_asks_for_reflection(self):
        """With unlock_after_investigation=True (default), investigation unlocks immediately.
        To test the 'needs reflection' message, disable unconditional unlock."""
        gate = ReadThinkGate(ReadThinkGateConfig(unlock_after_investigation=False))
        gate.check_batch("", ["read_file"])
        result = gate.check_batch("", ["terminal"])
        assert result is not None
        assert "现状全貌" in result


# ── Disabled gate ───────────────────────────────────────────────────


class TestDisabled:
    def test_disabled_passes_mutating_without_content(self):
        gate = ReadThinkGate(ReadThinkGateConfig(enabled=False))
        result = gate.check_batch("", ["terminal", "write_file"])
        assert result is None


# ── Tool classification ────────────────────────────────────────────


class TestGatedToolNames:
    def test_execution_tools_are_gated(self):
        for t in ["terminal", "write_file", "patch", "execute_code", "delegate_task"]:
            assert t in GATED_TOOL_NAMES

    def test_read_only_tools_not_gated(self):
        for t in ["read_file", "search_files", "web_search", "skill_view", "memory"]:
            assert t not in GATED_TOOL_NAMES

    def test_browser_tools_are_gated(self):
        for t in ["browser_navigate", "browser_click", "browser_type", "browser_dialog"]:
            assert t in GATED_TOOL_NAMES


# ── Mixed batches ───────────────────────────────────────────────────


class TestMixedBatches:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gate = ReadThinkGate()

    def test_mixed_batch_marks_investigation(self):
        """Batch has both read and write tools → marks investigation.
        With unlock_after_investigation=True (default), investigation unlocks immediately.
        To test blocking with mixed batch, disable unconditional unlock."""
        gate = ReadThinkGate(ReadThinkGateConfig(unlock_after_investigation=False))
        result = gate.check_batch("", ["read_file", "terminal"])
        assert result is not None  # terminal blocked
        assert gate._investigation_done is True  # but read_file counted

    def test_pure_read_batch_does_not_block(self):
        result = self.gate.check_batch(None, ["read_file", "search_files"])
        assert result is None
        assert self.gate._investigation_done is True
        assert self.gate.phase == "execution"  # investigation done → unlocked


# ── Turn lifecycle simulation (mirrors tool_executor path) ───────────


class TestTurnLifecycle:
    """Simulate a full turn: how gate behaves across multiple LLM rounds."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gate = ReadThinkGate()

    def test_full_reasoning_to_execution_cycle(self):
        """Round 1: investigate → Round 2: unlocked (investigation done)."""
        assert self.gate.phase == "reasoning"
        assert self.gate._investigation_done is False

        # Round 1: LLM searches (content="", read-only tools)
        r1 = self.gate.check_batch("", ["read_file", "search_files"])
        assert r1 is None  # allowed
        assert self.gate._investigation_done is True
        assert self.gate.phase == "execution"  # investigation done → unlocked

        # Round 2: investigation done → unlocked unconditionally
        r2 = self.gate.check_batch("", ["write_file"])
        assert r2 is None  # unlocked — investigation done
        assert self.gate.phase == "execution"

    def test_full_cycle_strict_mode(self):
        """With unlock_after_investigation=False: investigate → blocked → reflect → unlocked."""
        gate = ReadThinkGate(ReadThinkGateConfig(unlock_after_investigation=False))
        gate.check_batch("", ["read_file", "search_files"])
        r2 = gate.check_batch("", ["write_file"])
        assert r2 is not None  # blocked — need reflection
        assert "调查已完成" in r2
        r3 = gate.check_batch("Found auth bug in login handler.", ["write_file"])
        assert r3 is None  # unlocked — investigation + reflection
        assert gate.phase == "execution"

    def test_direct_reasoning_skips_investigation(self):
        """If LLM provides analysis upfront, skip directly to execution."""
        r1 = self.gate.check_batch(
            "The bug is in auth.py line 42. The token validation is missing entirely. Need to add JWT verification.",
            ["write_file"],
        )
        assert r1 is None
        assert self.gate.phase == "execution"

    def test_block_message_json_friendly(self):
        """Block message survives JSON wrapping without double-encoding."""
        import json

        msg = self.gate.check_batch("", ["terminal"])
        assert msg is not None

        # This is what line 1220 in tool_executor.py does:
        wrapped = json.dumps({"error": msg}, ensure_ascii=False)
        parsed = json.loads(wrapped)

        # Should be: {"error": "[ReadThink Gate — 推理阶段] ..."}
        assert "ReadThink Gate" in parsed["error"]
        # NOT: {"error": "{\"read_think_gate\": ...}"}
        assert "deliberation_gate" not in str(parsed)

    def test_concurrent_path_block_message_works(self):
        """Gate returns correct format for concurrent path block injection.
        With default config, investigation unlocks immediately, so use strict mode."""
        gate = ReadThinkGate(ReadThinkGateConfig(unlock_after_investigation=False))
        msg = gate.check_batch("", ["read_file", "terminal"])

        # Used as: elif _gate_block and func in GATED_TOOL_NAMES: block_result = _gate_block
        assert msg is not None
        assert "terminal" in msg
        # Gate should still track investigation (read_file was in the batch)
        assert gate._investigation_done is True


# ── min_read_only_calls config ──────────────────────────────────────


class TestMinReadOnlyCalls:
    """Verify that min_read_only_calls controls how many reads are required."""

    def test_default_one_call_suffices(self):
        """Default min_read_only_calls=1: one read unlocks."""
        gate = ReadThinkGate()
        gate.check_batch("", ["read_file"])
        assert gate._investigation_done is True

    def test_two_calls_required(self):
        """min_read_only_calls=2: one read is not enough."""
        gate = ReadThinkGate(ReadThinkGateConfig(min_read_only_calls=2))
        gate.check_batch("", ["read_file"])
        assert gate._investigation_done is False
        gate.check_batch("", ["search_files"])
        assert gate._investigation_done is True

    def test_three_calls_required(self):
        """min_read_only_calls=3."""
        gate = ReadThinkGate(ReadThinkGateConfig(min_read_only_calls=3))
        gate.check_batch("", ["read_file"])
        gate.check_batch("", ["search_files"])
        assert gate._investigation_done is False
        gate.check_batch("", ["web_search"])
        assert gate._investigation_done is True

    def test_two_calls_blocks_then_unlocks(self):
        """With min_read_only_calls=2, terminal is blocked after 1 read, unlocked after 2."""
        gate = ReadThinkGate(ReadThinkGateConfig(min_read_only_calls=2))
        # First read
        gate.check_batch("", ["read_file"])
        # terminal still blocked
        result = gate.check_batch("", ["terminal"])
        assert result is not None
        assert gate.phase == "reasoning"
        # Second read
        gate.check_batch("", ["search_files"])
        # Now terminal unlocked
        result = gate.check_batch("", ["terminal"])
        assert result is None
        assert gate.phase == "execution"

    def test_block_message_shows_remaining(self):
        """Block message tells agent how many more reads are needed."""
        gate = ReadThinkGate(ReadThinkGateConfig(min_read_only_calls=3))
        gate.check_batch("", ["read_file"])  # 1/3
        gate.check_batch("", ["search_files"])  # 2/3
        result = gate.check_batch("", ["terminal"])
        assert result is not None
        assert "1" in result  # "还需要 1 次"
