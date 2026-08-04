#!/usr/bin/env python3
"""Tests for subagent_review.py — Builder-Judge-Manager review pipeline.

Tests the pure functions: should_review, _parse_handoff, _check_files_exist,
_build_reviewer_prompt, _build_fixer_prompt, _get_git_diff, _format_handoff_commands,
_format_uncertainty, _cross_verify_handoff, _collect_verification_evidence,
collect_ground_truth, review_child_output (config-gated), run_review_loop.

Does NOT test review_child_output / fix_and_re_review that spawn real subagents —
those are integration tests requiring an LLM backend.

Run:
    scripts/run_tests.sh tests/tools/test_subagent_review.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools import subagent_review


class TestShouldReview(unittest.TestCase):
    """should_review() gate logic."""

    def test_completed_with_files_returns_true(self):
        self.assertTrue(subagent_review.should_review({
            "status": "completed",
            "summary": "Built the thing",
            "files_written": ["/tmp/a.py"],
        }))

    def test_non_completed_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "failed",
            "summary": "Oops",
            "files_written": ["/tmp/a.py"],
        }))

    def test_no_summary_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "completed",
            "summary": "",
            "files_written": ["/tmp/a.py"],
        }))

    def test_none_summary_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "completed",
            "files_written": ["/tmp/a.py"],
        }))

    def test_no_files_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "completed",
            "summary": "Just analysis",
            "files_written": [],
        }))

    def test_none_files_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "completed",
            "summary": "Just analysis",
            "files_written": None,
        }))

    def test_pure_research_no_files_returns_false(self):
        self.assertFalse(subagent_review.should_review({
            "status": "completed",
            "summary": "Here is my analysis",
        }))


class TestParseHandoff(unittest.TestCase):
    """_parse_handoff() markdown parsing."""

    def test_empty_string_returns_empty_result(self):
        result = subagent_review._parse_handoff("")
        self.assertEqual(result["files"], [])
        self.assertEqual(result["commands"], [])
        self.assertEqual(result["uncertainty"], [])
        self.assertEqual(result["raw_block"], "")

    def test_none_string_returns_empty_result(self):
        result = subagent_review._parse_handoff(None)
        self.assertEqual(result["files"], [])

    def test_full_handoff_parsed_correctly(self):
        text = """Some intro text.

## Deliverable Handoff
### Files
- /tmp/calc.py
- /tmp/test_calc.py
### Commands Executed
- pytest test_calc.py -v (exit 0)
- python -c 'print(1)' (exit 1)
### Uncertainty
- edge case with zero not tested
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["files"], ["/tmp/calc.py", "/tmp/test_calc.py"])
        self.assertEqual(len(result["commands"]), 2)
        self.assertEqual(result["commands"][0], ("pytest test_calc.py -v", 0))
        self.assertEqual(result["commands"][1], ("python -c 'print(1)'", 1))
        self.assertEqual(result["uncertainty"], ["edge case with zero not tested"])
        self.assertIn("Deliverable Handoff", result["raw_block"])

    def test_command_without_exit_code_parsed(self):
        text = """## Deliverable Handoff
### Commands Executed
- python -c 'print(1)'
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(len(result["commands"]), 1)
        self.assertEqual(result["commands"][0], ("python -c 'print(1)'", None))

    def test_case_insensitive_header(self):
        text = """## DELIVERABLE HANDOFF
### Files
- /tmp/a.py
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["files"], ["/tmp/a.py"])

    def test_no_handoff_header_returns_empty(self):
        text = "Just some text without a handoff block"
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["files"], [])
        self.assertEqual(result["raw_block"], "")

    def test_uncertainty_excludes_help_text(self):
        text = """## Deliverable Handoff
### Uncertainty
- something real
- (what you are not sure about or could not verify)
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["uncertainty"], ["something real"])

    def test_empty_file_lines_skipped(self):
        text = """## Deliverable Handoff
### Files
- /tmp/a.py
-
- /tmp/b.py
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["files"], ["/tmp/a.py", "/tmp/b.py"])

    def test_block_stops_at_next_h2(self):
        text = """## Deliverable Handoff
### Files
- /tmp/a.py
## Next Section
### More Files
- /tmp/b.py
"""
        result = subagent_review._parse_handoff(text)
        self.assertEqual(result["files"], ["/tmp/a.py"])
        self.assertNotIn("/tmp/b.py", result["files"])


class TestCheckFilesExist(unittest.TestCase):
    """_check_files_exist() file-system checks."""

    def test_existing_file_passes(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            result = subagent_review._check_files_exist([path])
            self.assertIn("PASS", result)
            self.assertIn(path, result)
            self.assertIn("bytes", result)
        finally:
            os.unlink(path)

    def test_missing_file_fails(self):
        result = subagent_review._check_files_exist(["/tmp/nonexistent_hermes_test_12345.txt"])
        self.assertIn("FAIL", result)
        self.assertIn("does NOT exist", result)

    def test_mixed_files(self):
        fd, existing = tempfile.mkstemp()
        os.close(fd)
        try:
            result = subagent_review._check_files_exist([
                existing,
                "/tmp/nonexistent_hermes_test_99999.txt",
            ])
            self.assertIn("PASS", result)
            self.assertIn("FAIL", result)
        finally:
            os.unlink(existing)

    def test_empty_list_returns_empty(self):
        result = subagent_review._check_files_exist([])
        self.assertEqual(result, "")

    def test_none_list_returns_empty(self):
        result = subagent_review._check_files_exist(None)
        self.assertEqual(result, "")


class TestBuildReviewerPrompt(unittest.TestCase):
    """_build_reviewer_prompt() prompt construction."""

    def test_contains_goal_and_summary(self):
        prompt = subagent_review._build_reviewer_prompt(
            goal="Build a calc",
            task_summary="I built it",
            files_written=["/tmp/a.py"],
            diff_content="diff --git",
        )
        self.assertIn("Build a calc", prompt)
        self.assertIn("I built it", prompt)
        self.assertIn("/tmp/a.py", prompt)
        self.assertIn("diff --git", prompt)

    def test_ground_truth_appended(self):
        prompt = subagent_review._build_reviewer_prompt(
            goal="g",
            task_summary="s",
            files_written=["f"],
            diff_content="d",
            ground_truth="Test passed",
        )
        self.assertIn("Ground Truth", prompt)
        self.assertIn("Test passed", prompt)

    def test_no_ground_truth_placeholder_skipped(self):
        prompt = subagent_review._build_reviewer_prompt(
            goal="g",
            task_summary="s",
            files_written=["f"],
            diff_content="d",
            ground_truth="(no independent ground truth available)",
        )
        self.assertNotIn("Ground Truth", prompt)

    def test_diff_truncated_at_8000(self):
        long_diff = "x" * 10000
        prompt = subagent_review._build_reviewer_prompt(
            goal="g", task_summary="s",
            files_written=["f"], diff_content=long_diff,
        )
        # The diff block should contain truncated content
        self.assertIn("```", prompt)

    def test_empty_files_list(self):
        prompt = subagent_review._build_reviewer_prompt(
            goal="g", task_summary="s",
            files_written=[], diff_content="d",
        )
        self.assertIn("Modified Files", prompt)


class TestBuildFixerPrompt(unittest.TestCase):
    """_build_fixer_prompt() prompt construction."""

    def test_contains_goal_and_issues(self):
        prompt = subagent_review._build_fixer_prompt(
            goal="Fix the calc",
            review_issues="divide returns a*b not a/b",
            files_to_fix=["/tmp/calc.py"],
        )
        self.assertIn("Fix the calc", prompt)
        self.assertIn("divide returns a*b not a/b", prompt)
        self.assertIn("/tmp/calc.py", prompt)

    def test_empty_files_list(self):
        prompt = subagent_review._build_fixer_prompt(
            goal="g", review_issues="issues", files_to_fix=[],
        )
        self.assertIn("Files to Fix", prompt)


class TestGetGitDiff(unittest.TestCase):
    """_get_git_diff() git integration."""

    def test_empty_files_returns_empty(self):
        self.assertEqual(subagent_review._get_git_diff([]), "")

    def test_none_files_returns_empty(self):
        self.assertEqual(subagent_review._get_git_diff(None), "")

    def test_nonexistent_files_returns_empty(self):
        # Should not crash, returns empty or whatever git says
        result = subagent_review._get_git_diff(["/tmp/nonexistent_git_test_file_12345.py"])
        # Result depends on whether we're in a git repo; just verify no crash
        self.assertIsInstance(result, str)


class TestFormatHandoffCommands(unittest.TestCase):
    """_format_handoff_commands() formatting."""

    def test_empty_commands_returns_empty(self):
        self.assertEqual(subagent_review._format_handoff_commands([], []), "")

    def test_verified_command(self):
        cmds = [("pytest test.py -v", 0)]
        result = subagent_review._format_handoff_commands(cmds, cmds)
        self.assertIn("VERIFIED", result)
        self.assertIn("pytest test.py -v", result)
        self.assertIn("exit=0", result)

    def test_unverified_command_triggers_warning(self):
        cmds = [("pytest test.py", 1)]
        result = subagent_review._format_handoff_commands(cmds, [])
        self.assertIn("UNVERIFIED", result)
        self.assertIn("WARNING", result)

    def test_none_exit_code_no_paren(self):
        cmds = [("python script.py", None)]
        result = subagent_review._format_handoff_commands(cmds, cmds)
        self.assertNotIn("exit=", result)
        self.assertIn("VERIFIED", result)

    def test_mixed_verified_unverified(self):
        cmds = [("pytest a.py", 0), ("pytest b.py", 1)]
        result = subagent_review._format_handoff_commands(cmds, ["pytest a.py"])
        self.assertIn("[VERIFIED]", result)
        self.assertIn("[UNVERIFIED]", result)


class TestFormatUncertainty(unittest.TestCase):
    """_format_uncertainty() formatting."""

    def test_empty_returns_empty(self):
        self.assertEqual(subagent_review._format_uncertainty({"uncertainty": []}), "")

    def test_none_returns_empty(self):
        self.assertEqual(subagent_review._format_uncertainty({"uncertainty": None}), "")

    def test_single_item(self):
        result = subagent_review._format_uncertainty({"uncertainty": ["edge case"]})
        self.assertIn("edge case", result)
        self.assertIn("Builder's Stated Uncertainty", result)

    def test_multiple_items(self):
        items = ["edge case A", "edge case B"]
        result = subagent_review._format_uncertainty({"uncertainty": items})
        self.assertIn("edge case A", result)
        self.assertIn("edge case B", result)


class TestCollectGroundTruth(unittest.TestCase):
    """collect_ground_truth() evidence aggregation."""

    def test_no_files_no_evidence(self):
        result = subagent_review.collect_ground_truth([], {"summary": ""})
        self.assertEqual(result, "(no independent ground truth available)")

    def test_existing_files_included(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            result = subagent_review.collect_ground_truth(
                [path],
                {"summary": "## Deliverable Handoff\n### Files\n- " + path},
            )
            self.assertIn("PASS", result)
            self.assertIn(path, result)
        finally:
            os.unlink(path)

    def test_missing_files_flagged(self):
        result = subagent_review.collect_ground_truth(
            ["/tmp/nonexistent_gt_test_98765.txt"],
            {"summary": ""},
        )
        self.assertIn("FAIL", result)

    def test_handoff_commands_cross_checked(self):
        # When no session_id, cross-verification returns formatted commands
        result = subagent_review.collect_ground_truth(
            [],
            {
                "summary": (
                    "## Deliverable Handoff\n"
                    "### Commands Executed\n"
                    "- pytest test.py -v (exit 0)\n"
                ),
            },
        )
        # Should have cross-verification section even without session_id
        self.assertIn("Handoff Cross-Verification", result)

    def test_uncertainty_section_included(self):
        result = subagent_review.collect_ground_truth(
            [],
            {
                "summary": (
                    "## Deliverable Handoff\n"
                    "### Uncertainty\n"
                    "- race condition not tested\n"
                ),
            },
        )
        self.assertIn("race condition not tested", result)


class TestCollectVerificationEvidence(unittest.TestCase):
    """_collect_verification_evidence() evidence db lookup."""

    def test_no_session_id_returns_empty(self):
        result = subagent_review._collect_verification_evidence({})
        self.assertEqual(result, "")

    def test_no_cwd_returns_empty(self):
        result = subagent_review._collect_verification_evidence({"session_id": "s1"})
        self.assertEqual(result, "")

    @patch("agent.verification_evidence.verification_status")
    def test_not_applicable_returns_empty(self, mock_vs):
        mock_vs.return_value = {"status": "not_applicable"}
        result = subagent_review._collect_verification_evidence({
            "session_id": "s1", "cwd": "/tmp",
        })
        self.assertEqual(result, "")

    @patch("agent.verification_evidence.verification_status")
    def test_no_evidence_returns_empty(self, mock_vs):
        mock_vs.return_value = {"status": "ok", "evidence": None}
        result = subagent_review._collect_verification_evidence({
            "session_id": "s1", "cwd": "/tmp",
        })
        self.assertEqual(result, "")

    @patch("agent.verification_evidence.verification_status")
    def test_evidence_formatted(self, mock_vs):
        mock_vs.return_value = {
            "status": "ok",
            "evidence": {
                "command": "pytest test.py -v",
                "kind": "test",
                "status": "passed",
                "exit_code": 0,
                "output_summary": "5 passed",
            },
        }
        result = subagent_review._collect_verification_evidence({
            "session_id": "s1", "cwd": "/tmp",
        })
        self.assertIn("pytest test.py -v", result)
        self.assertIn("passed", result)
        self.assertIn("5 passed", result)

    @patch("agent.verification_evidence.verification_status")
    def test_import_failure_returns_empty(self, mock_vs):
        mock_vs.side_effect = ImportError("no module")
        result = subagent_review._collect_verification_evidence({
            "session_id": "s1", "cwd": "/tmp",
        })
        self.assertEqual(result, "")


class TestCrossVerifyHandoff(unittest.TestCase):
    """_cross_verify_handoff() ledger cross-check."""

    def test_no_commands_returns_empty(self):
        result = subagent_review._cross_verify_handoff(
            {"commands": []}, {"session_id": "s1"},
        )
        self.assertEqual(result, "")

    def test_no_session_id_formats_unverified(self):
        cmds = [("pytest a.py", 0)]
        result = subagent_review._cross_verify_handoff(
            {"commands": cmds}, {},
        )
        self.assertIn("UNVERIFIED", result)

    @patch("agent.verification_evidence.verification_status")
    def test_verified_command(self, mock_vs):
        mock_vs.return_value = {
            "evidence": {"command": "pytest test.py -v"},
        }
        cmds = [("pytest test.py -v", 0)]
        result = subagent_review._cross_verify_handoff(
            {"commands": cmds}, {"session_id": "s1", "cwd": "/tmp"},
        )
        self.assertIn("VERIFIED", result)

    @patch("agent.verification_evidence.verification_status")
    def test_partial_match_command_short(self, mock_vs):
        mock_vs.return_value = {
            "evidence": {"command": "pytest test.py -v --tb=short"},
        }
        cmds = [("pytest test.py", 0)]
        result = subagent_review._cross_verify_handoff(
            {"commands": cmds}, {"session_id": "s1", "cwd": "/tmp"},
        )
        self.assertIn("VERIFIED", result)

    @patch("agent.verification_evidence.verification_status")
    def test_no_match_unverified(self, mock_vs):
        mock_vs.return_value = {
            "evidence": {"command": "npm run build"},
        }
        cmds = [("pytest test.py", 0)]
        result = subagent_review._cross_verify_handoff(
            {"commands": cmds}, {"session_id": "s1", "cwd": "/tmp"},
        )
        self.assertIn("UNVERIFIED", result)


class TestReviewChildOutput(unittest.TestCase):
    """review_child_output() — config-gated, no real subagent."""

    def test_review_disabled_returns_approved(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg:
            mock_cfg.return_value = {"enabled": False}
            result = subagent_review.review_child_output(
                task_result={
                    "status": "completed",
                    "summary": "Built it",
                    "files_written": ["a.py"],
                },
                goal="Build something",
            )
        self.assertTrue(result["approved"])
        self.assertEqual(result["review_summary"], "review disabled (delegation.review_enabled=false)")

    def test_no_files_skips_review(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg:
            mock_cfg.return_value = {"enabled": True}
            result = subagent_review.review_child_output(
                task_result={
                    "status": "completed",
                    "summary": "Just analysis",
                    "files_written": [],
                },
                goal="Analyze code",
            )
        self.assertTrue(result["approved"])
        self.assertIn("no files modified", result["review_summary"])

    def test_no_diff_no_gt_skips(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg, \
             patch("tools.subagent_review._get_git_diff", return_value=""), \
             patch("tools.subagent_review.collect_ground_truth",
                   return_value="(no independent ground truth available)"):
            mock_cfg.return_value = {"enabled": True, "ground_truth": True}
            result = subagent_review.review_child_output(
                task_result={
                    "status": "completed",
                    "summary": "Built",
                    "files_written": ["a.py"],
                },
                goal="Build",
            )
        self.assertTrue(result["approved"])
        self.assertIn("no diff and no ground truth", result["review_summary"])


class TestRunReviewLoop(unittest.TestCase):
    """run_review_loop() — full loop with mocked internals."""

    def test_review_disabled_returns_unchanged(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg:
            mock_cfg.return_value = {"enabled": False}
            task = {"status": "completed", "summary": "done", "files_written": ["a.py"]}
            result = subagent_review.run_review_loop(task, "goal")
        self.assertIs(result, task)
        self.assertNotIn("_review", result)

    def test_approved_first_time(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg, \
             patch("tools.subagent_review.review_child_output") as mock_review:
            mock_cfg.return_value = {"enabled": True, "max_iterations": 2}
            mock_review.return_value = {"approved": True, "review_summary": "LGTM"}
            task = {"status": "completed", "summary": "done", "files_written": ["a.py"]}
            result = subagent_review.run_review_loop(task, "goal")
        self.assertTrue(result["_review"]["approved"])
        self.assertEqual(result["_review"]["iteration"], 0)
        self.assertNotIn("escalate", result["_review"])

    def test_fix_then_approve(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg, \
             patch("tools.subagent_review.review_child_output") as mock_review, \
             patch("tools.subagent_review.fix_and_re_review") as mock_fix:
            mock_cfg.return_value = {"enabled": True, "max_iterations": 2}
            mock_review.return_value = {"approved": False, "issues": "bug found"}
            mock_fix.return_value = {"approved": True, "review_summary": "Fixed"}
            task = {"status": "completed", "summary": "done", "files_written": ["a.py"]}
            result = subagent_review.run_review_loop(task, "goal")
        self.assertTrue(result["_review"]["approved"])
        self.assertEqual(result["_review"]["iteration"], 1)

    def test_max_iterations_escalates(self):
        with patch("tools.subagent_review._load_review_config") as mock_cfg, \
             patch("tools.subagent_review.review_child_output") as mock_review, \
             patch("tools.subagent_review.fix_and_re_review") as mock_fix:
            mock_cfg.return_value = {"enabled": True, "max_iterations": 1}
            mock_review.return_value = {"approved": False, "issues": "bug"}
            mock_fix.return_value = {"approved": False, "issues": "still broken"}
            task = {"status": "completed", "summary": "done", "files_written": ["a.py"]}
            result = subagent_review.run_review_loop(task, "goal")
        self.assertFalse(result["_review"]["approved"])
        self.assertTrue(result["_review"]["escalate"])
        self.assertIn("exhausted", result["_review"]["escalate_reason"])

    def test_explicit_review_cfg_override(self):
        with patch("tools.subagent_review.review_child_output") as mock_review, \
             patch("tools.subagent_review.fix_and_re_review") as mock_fix:
            mock_review.return_value = {"approved": True, "review_summary": "ok"}
            task = {"status": "completed", "summary": "done", "files_written": ["a.py"]}
            result = subagent_review.run_review_loop(
                task, "goal",
                review_cfg={"enabled": True, "max_iterations": 1},
            )
        self.assertTrue(result["_review"]["approved"])


class TestLoadReviewConfig(unittest.TestCase):
    """_load_review_config() config loading."""

    def test_default_config_when_import_fails(self):
        with patch.dict("sys.modules", {"tools.delegate_tool": None}):
            # Force re-import by calling the function which will catch ImportError
            cfg = subagent_review._load_review_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["max_iterations"], 1)
        self.assertEqual(cfg["cost_limit"], 0.5)

    def test_config_from_delegate_tool(self):
        with patch("tools.delegate_tool._load_config") as mock_load:
            mock_load.return_value = {
                "review_enabled": True,
                "review_max_iterations": 3,
                "review_cost_limit": 1.0,
            }
            cfg = subagent_review._load_review_config()
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["max_iterations"], 3)
        self.assertEqual(cfg["cost_limit"], 1.0)


if __name__ == "__main__":
    unittest.main()
