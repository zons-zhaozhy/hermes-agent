"""Tests for agent/analysis_stop_guard.py.

Behavioral tests — assert the guard's decision logic, not its implementation.
No live network calls; LLM judge is mocked.
"""

import pytest
from unittest.mock import patch, MagicMock
from agent.analysis_stop_guard import (
    check_analysis_stop,
    _has_unfinished_heuristics,
    _prev_turn_has_tool_results,
    _count_structure_markers,
    _build_nudge,
    _MAX_NUDGES,
)


# ── Heuristic tests ──────────────────────────────────────────────


class TestUnfinishedHeuristics:
    def test_chinese_next_step_pattern(self):
        assert _has_unfinished_heuristics("现在可以继续下一步。")

    def test_chinese_let_me_pattern(self):
        assert _has_unfinished_heuristics("让我检查一下这个文件。")

    def test_chinese_i_will_pattern(self):
        assert _has_unfinished_heuristics("我将运行测试。")

    def test_chinese_suggestion_pattern(self):
        """建议 without action = analysis."""
        assert _has_unfinished_heuristics("建议修复这个 bug。")

    def test_english_i_will_pattern(self):
        assert _has_unfinished_heuristics("I will run the tests now.")

    def test_english_let_me_pattern(self):
        assert _has_unfinished_heuristics("Let me check the file.")

    def test_english_should_fix_pattern(self):
        assert _has_unfinished_heuristics("We should fix this by updating the config.")

    def test_finished_done(self):
        assert not _has_unfinished_heuristics("done.")

    def test_finished_fixed(self):
        assert not _has_unfinished_heuristics("已修复。")

    def test_finished_tests_passed(self):
        assert not _has_unfinished_heuristics("验证通过。")

    def test_empty_string(self):
        assert not _has_unfinished_heuristics("")

    def test_plain_factual_no_pattern(self):
        """A factual statement with no unfinished markers."""
        assert not _has_unfinished_heuristics("The function returns 42.")

    def test_short_factual_with_keyword_but_done(self):
        """Final-response pattern overrides unfinished keywords."""
        # Even if it contains "让我", a final pattern should NOT trigger.
        # (This is an edge case — final patterns are checked first.)
        assert not _has_unfinished_heuristics("已完成。")

    # ── Structural analysis detection (Layer 3) ──

    def test_long_structured_analysis_triggers(self):
        """Long text with 3+ structure markers → unfinished."""
        text = "\n".join([
            "## 根因分析",
            "",
            "问题出在以下环节：",
            "",
            "1. 第一层：文件读取",
            "2. 第二层：数据处理",
            "3. 第三层：结果输出",
            "",
            "- 根因：缺少空值检查",
            "- 影响：所有用户",
            "",
            "```python",
            "x = data[key]  # 这里会 KeyError",
            "```",
            "",
            "需要进一步确认具体调用路径。",
        ])
        assert _has_unfinished_heuristics(text)

    def test_long_text_without_structure_does_not_trigger(self):
        """Long text with no markdown structure → not flagged by Layer 3."""
        text = "这是一个关于天气的描述。" * 50  # long, no structure
        assert not _has_unfinished_heuristics(text)

    def test_short_structured_does_not_trigger(self):
        """Short text with structure → Layer 3 needs long text too."""
        text = "## Title\n- item 1\n- item 2"
        assert not _has_unfinished_heuristics(text)

    def test_english_structured_analysis_triggers(self):
        """English structured report → unfinished."""
        text = "\n".join([
            "## Root Cause Analysis",
            "",
            "1. First issue: missing validation",
            "2. Second issue: race condition",
            "",
            "- Impact: all endpoints",
            "- Severity: high",
            "",
            "```",
            "result = process(data)  # no null check",
            "```",
        ])
        assert _has_unfinished_heuristics(text)


# ── Structure marker counting ────────────────────────────────────


class TestStructureCount:
    def test_headers_counted(self):
        assert _count_structure_markers("## Title\n### Sub") == 2

    def test_numbered_list_counted(self):
        assert _count_structure_markers("1. first\n2. second\n3. third") == 3

    def test_code_block_counted(self):
        text = "```\ncode here\n```"
        assert _count_structure_markers(text) == 2

    def test_mixed_markers(self):
        text = "## Title\n1. item\n- bullet\n> quote"
        assert _count_structure_markers(text) == 4

    def test_no_markers(self):
        assert _count_structure_markers("just plain text") == 0


# ── Tool-results context tests ───────────────────────────────────


class TestPrevTurnToolResults:
    def test_has_tool_before_assistant(self):
        msgs = [
            {"role": "user", "content": "check this"},
            {"role": "assistant", "content": "let me read", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "file contents"},
            {"role": "assistant", "content": "here is the analysis"},
        ]
        assert _prev_turn_has_tool_results(msgs)

    def test_no_tool_results(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        assert not _prev_turn_has_tool_results(msgs)

    def test_tool_far_back_not_counted(self):
        """Tool results more than 5 messages back don't count."""
        msgs = [
            {"role": "tool", "content": "old result"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
            {"role": "assistant", "content": "final analysis"},
        ]
        assert not _prev_turn_has_tool_results(msgs)

    def test_empty_messages(self):
        assert not _prev_turn_has_tool_results([])

    def test_single_message(self):
        assert not _prev_turn_has_tool_results([{"role": "user", "content": "hi"}])


# ── Progressive nudge tests ──────────────────────────────────────


class TestProgressiveNudge:
    def test_first_nudge_is_standard(self):
        msg = _build_nudge(0)
        assert "tool call" in msg.lower()
        assert "FINAL" not in msg

    def test_second_nudge_escalates(self):
        msg = _build_nudge(1)
        assert "FINAL" in msg
        assert "task failure" in msg.lower()

    def test_nudge_index_clamped(self):
        """Beyond the last nudge index, clamp to the last message."""
        msg = _build_nudge(99)
        assert "FINAL" in msg

    def test_nudge_count_not_exceeded(self):
        """After max nudges, check_analysis_stop returns None."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=[
                    {"role": "tool", "content": "data"},
                    {"role": "assistant", "content": "下一步分析"},
                ],
                assistant_content="下一步分析",
                finish_reason="stop",
                user_message="task",
                nudge_count=_MAX_NUDGES,
            )
        assert result is None


# ── Integration: check_analysis_stop ─────────────────────────────


class TestCheckAnalysisStop:
    def _msgs_with_tool_context(self, assistant_text="现在可以继续下一步。"):
        return [
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "reading", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "file content here"},
            {"role": "assistant", "content": assistant_text},
        ]

    def test_returns_none_for_finish_reason_tool_calls(self):
        """If finish_reason is tool_calls, the agent IS executing — no nudge."""
        msgs = self._msgs_with_tool_context()
        result = check_analysis_stop(
            messages=msgs,
            assistant_content="some text",
            finish_reason="tool_calls",
            user_message="fix the bug",
        )
        assert result is None

    def test_returns_none_for_empty_content(self):
        result = check_analysis_stop(
            messages=self._msgs_with_tool_context(),
            assistant_content="",
            finish_reason="stop",
            user_message="fix the bug",
        )
        assert result is None

    def test_returns_none_when_nudge_count_exceeded(self):
        result = check_analysis_stop(
            messages=self._msgs_with_tool_context(),
            assistant_content="现在可以继续下一步。",
            finish_reason="stop",
            user_message="fix the bug",
            nudge_count=_MAX_NUDGES,
        )
        assert result is None

    def test_returns_none_when_no_tool_results_in_context(self):
        """Normal conversation without prior tool calls — not analysis-then-stop."""
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "让我检查一下"},
        ]
        result = check_analysis_stop(
            messages=msgs,
            assistant_content="让我检查一下",
            finish_reason="stop",
            user_message="hello",
        )
        assert result is None

    def test_returns_none_when_no_unfinished_heuristics(self):
        """Genuine final answer — no heuristic match, no nudge."""
        result = check_analysis_stop(
            messages=self._msgs_with_tool_context("done."),
            assistant_content="done.",
            finish_reason="stop",
            user_message="fix the bug",
        )
        assert result is None

    def test_nudge_returned_when_judge_says_unfinished(self):
        """Heuristic matches + judge says UNFINISHED → nudge."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(),
                assistant_content="现在可以继续下一步。",
                finish_reason="stop",
                user_message="fix the bug",
            )
        assert result is not None
        assert "tool call" in result.lower()

    def test_no_nudge_when_judge_says_finished(self):
        """Heuristic matches but judge overrides → no nudge."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=False):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(),
                assistant_content="现在可以继续下一步。",
                finish_reason="stop",
                user_message="fix the bug",
            )
        assert result is None

    def test_nudge_includes_system_prefix(self):
        """Nudge should look like a system instruction."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(),
                assistant_content="让我检查一下这个文件。",
                finish_reason="stop",
                user_message="do the task",
            )
        assert result is not None
        assert result.startswith("[System:")

    def test_structured_analysis_after_tools_triggers(self):
        """Long structured analysis after tool results → nudge."""
        long_analysis = "\n".join([
            "## 根因分析",
            "",
            "问题出在以下环节：",
            "",
            "1. 第一层：文件读取阶段没有校验输入",
            "2. 第二层：数据处理阶段缺少空值检查",
            "3. 第三层：结果输出阶段格式不正确",
            "",
            "- 根因：缺少空值检查导致 KeyError",
            "- 影响：所有调用该接口的用户",
            "",
            "```python",
            "x = data[key]  # 这里会 KeyError",
            "```",
            "",
            "需要进一步确认具体调用路径和修复方案。",
        ])
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(long_analysis),
                assistant_content=long_analysis,
                finish_reason="stop",
                user_message="analyze",
            )
        assert result is not None

    def test_first_nudge_uses_standard_message(self):
        """First nudge (count=0) uses standard, not FINAL."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(),
                assistant_content="下一步",
                finish_reason="stop",
                user_message="task",
                nudge_count=0,
            )
        assert result is not None
        assert "FINAL" not in result

    def test_second_nudge_uses_escalated_message(self):
        """Second nudge (count=1) uses FINAL escalation."""
        with patch("agent.analysis_stop_guard._llm_judge", return_value=True):
            result = check_analysis_stop(
                messages=self._msgs_with_tool_context(),
                assistant_content="下一步",
                finish_reason="stop",
                user_message="task",
                nudge_count=1,
            )
        assert result is not None
        assert "FINAL" in result


# ── LLM judge fallback tests ─────────────────────────────────────


class TestLLMJudgeFallback:
    def test_judge_failopen_on_exception(self):
        """If auxiliary client raises, fall back to heuristic verdict (True)."""
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            side_effect=Exception("network error"),
        ):
            from agent.analysis_stop_guard import _llm_judge
            result = _llm_judge("task", "response")
        assert result is True  # trust heuristics

    def test_judge_failopen_on_no_client(self):
        """If no auxiliary client available, trust heuristics."""
        mock_none = MagicMock(return_value=(None, ""))
        with patch("agent.auxiliary_client.get_text_auxiliary_client", mock_none):
            from agent.analysis_stop_guard import _llm_judge
            result = _llm_judge("task", "response")
        assert result is True
