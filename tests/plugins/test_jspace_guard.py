"""Tests for jspace-guard plugin — invariant 6 (coverage) and invariant 8 (goal-rebase)."""

import importlib.util
import os
from pathlib import Path

import pytest

# Load the plugin module directly (plugins/ is not a Python package)
_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent / "plugins" / "jspace-guard"
_INIT_PATH = _PLUGIN_DIR / "__init__.py"


@pytest.fixture(autouse=True)
def mod():
    """Load and reset the jspace-guard module for each test."""
    spec = importlib.util.spec_from_file_location("jspace_guard", str(_INIT_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # Reset per-session state
    m._state = m._SessionState()
    yield m


# ── Activation threshold ──────────────────────────────────────


class TestActivationThreshold:
    """Below threshold, no warnings should fire."""

    def test_below_threshold_no_warning(self, mod):
        for _ in range(3):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="已验证修复成功。"
        )
        assert result == ""

    def test_exactly_threshold_fires(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="已验证修复成功。"
        )
        assert "[J-SPACE-GUARD" in result


# ── Invariant 6: coverage-gate ──────────────────────────────────


class TestCoverageGate:
    """Verification claims without coverage description should trigger warning."""

    def test_verify_no_coverage_triggers(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="修复已验证，测试全绿。"
        )
        assert "不变量6" in result
        assert "覆盖" in result

    def test_verify_with_coverage_passes(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="修复已验证，覆盖了正反例共9个测试用例。"
        )
        assert result == ""

    def test_verify_with_pytest_passes(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="pytest 64/64 passed, 0 failed."
        )
        assert result == ""

    def test_verify_with_includes_passes(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="验证通过，covering all attack and teaching cases."
        )
        assert result == ""

    def test_no_verify_claim_no_trigger(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="我建议下一步查看日志文件。"
        )
        assert result == ""

    def test_english_verified_no_coverage(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="The fix has been verified and all tests pass."
        )
        assert "不变量6" in result

    def test_green_emoji_no_coverage(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(
            response_text="✅ 修复完成。"
        )
        assert "不变量6" in result

    def test_empty_text_no_trigger(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

        result = mod._on_transform_llm_output(response_text="")
        assert result == ""


# ── Invariant 8: goal-rebase-gate ─────────────────────────────


class TestGoalRebaseGate:
    """Completion claims without goal re-read should trigger warning."""

    def test_done_no_rebase_triggers(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="任务已完成。"
        )
        assert "不变量8" in result
        assert "回读" in result

    def test_done_with_read_tool_passes(self, mod):
        mod._on_post_tool_call(tool_name="patch")
        mod._on_post_tool_call(tool_name="patch")
        mod._on_post_tool_call(tool_name="read_file")  # rebase tool
        for _ in range(5):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="任务已完成。"
        )
        assert result == ""

    def test_done_with_todo_tool_passes(self, mod):
        for _ in range(7):
            mod._on_post_tool_call(tool_name="patch")
        mod._on_post_tool_call(tool_name="todo")  # rebase tool

        result = mod._on_transform_llm_output(
            response_text="任务已完成。"
        )
        assert result == ""

    def test_done_with_rebase_words_passes(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="对照原始需求，任务已完成。"
        )
        assert result == ""

    def test_no_done_claim_no_trigger(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="代码已修改，待测试。"
        )
        assert result == ""

    def test_english_done_no_rebase(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="All done, the fix is complete."
        )
        assert "不变量8" in result

    def test_both_invariants_trigger(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="patch")

        result = mod._on_transform_llm_output(
            response_text="✅ 任务已完成，测试全绿。"
        )
        assert "不变量6" in result
        assert "不变量8" in result


# ── Helper function tests ──────────────────────────────────────


class TestHelperFunctions:
    """Verify _contains_any and _find_assertion_word behave correctly."""

    def test_contains_any_found(self, mod):
        assert mod._contains_any("hello world", frozenset({"world", "foo"}))

    def test_contains_any_miss(self, mod):
        assert not mod._contains_any("hello world", frozenset({"bar", "foo"}))

    def test_find_assertion_longest_match(self, mod):
        result = mod._find_assertion_word(
            "测试全绿", frozenset({"全绿", "全部通过"})
        )
        # "全部通过" (4 chars) checked first (sorted by len desc), not in text
        # "全绿" (2 chars) checked second, found in text
        assert result == "全绿"

    def test_find_assertion_not_found(self, mod):
        assert mod._find_assertion_word(
            "no assertion here", frozenset({"验证", "完成"})
        ) == ""

    def test_find_assertion_longest_wins(self, mod):
        result = mod._find_assertion_word(
            "测试全部通过", frozenset({"全绿", "全部通过"})
        )
        # "全部通过" (4 chars) checked first, found in text
        assert result == "全部通过"


# ── Invariant 9: tri-state annotation ──────────────────────────


class TestTriStateGate:
    """Invariant 9: factual assertions must carry source annotation."""

    def _activate(self, mod):
        for _ in range(8):
            mod._on_post_tool_call(tool_name="read_file")

    def test_assertion_without_tag_triggers(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="修复了 XXX 漏洞，根因是 YYY 模块。"
        )
        assert result != ""
        assert "[实测]" not in "修复了 XXX 漏洞"  # original has no tag
        assert "三态标注" in result

    def test_assertion_with_verified_tag_passes(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="修复了 XXX 漏洞 [实测]，根因是 YYY [推断]。"
        )
        assert result == ""

    def test_assertion_with_documented_tag_passes(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="根据文档，原因是 ZZZ [文档]。"
        )
        assert result == ""

    def test_assertion_with_unverified_tag_passes(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="推测根因是 AAA [未查证]。"
        )
        assert result == ""

    def test_no_assertion_no_trigger(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="接下来需要做 XYZ 和 ABC。"
        )
        assert result == ""

    def test_git_log_in_context_passes(self, mod):
        self._activate(mod)
        result = mod._on_transform_llm_output(
            response_text="修复了 bug，git log 显示是 commit abc 引入的。"
        )
        assert result == ""

    def test_below_threshold_no_trigger(self, mod):
        result = mod._on_transform_llm_output(
            response_text="修复了 XXX 漏洞，根因是 YYY。"
        )
        assert result == ""

