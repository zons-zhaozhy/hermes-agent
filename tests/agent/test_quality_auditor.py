#!/usr/bin/env python3
"""Tests for agent/quality_auditor.py — lightweight quality audit module.

Tests pure functions: _build_audit_prompt, _format_audit_feedback,
inject_audit_feedback, get_last_audit_feedback, aggregate_daily, fire_quality_audit.

Does NOT test _call_auxiliary_llm (requires HTTP backend) — mock it out.

Run:
    scripts/run_tests.sh tests/agent/test_quality_auditor.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.quality_auditor import (
    _build_audit_prompt,
    _format_audit_feedback,
    _write_audit_entry,
    aggregate_daily,
    fire_quality_audit,
    get_last_audit_feedback,
    inject_audit_feedback,
)


class TestBuildAuditPrompt(unittest.TestCase):
    """_build_audit_prompt() prompt construction."""

    def test_basic_prompt_contains_user_and_assistant(self):
        prompt = _build_audit_prompt("hello", "world")
        self.assertIn("hello", prompt)
        self.assertIn("world", prompt)

    def test_truncation_applied(self):
        long_msg = "x" * 3000
        prompt = _build_audit_prompt(long_msg, long_msg)
        self.assertIn("...(截断)", prompt)

    def test_no_truncation_when_short(self):
        prompt = _build_audit_prompt("hi", "bye")
        self.assertNotIn("...(截断)", prompt)

    def test_tool_call_count_included(self):
        prompt = _build_audit_prompt("hi", "bye", tool_call_count=5)
        self.assertIn("5", prompt)

    def test_tool_names_included(self):
        prompt = _build_audit_prompt("hi", "bye", tool_call_count=2,
                                     tool_names=["read_file", "terminal"])
        self.assertIn("read_file", prompt)
        self.assertIn("terminal", prompt)

    def test_tool_names_truncated_at_15(self):
        names = [f"tool_{i}" for i in range(20)]
        prompt = _build_audit_prompt("hi", "bye", tool_call_count=20, tool_names=names)
        self.assertIn("tool_14", prompt)
        self.assertNotIn("tool_15", prompt)

    def test_zero_tool_calls_no_tool_section(self):
        prompt = _build_audit_prompt("hi", "bye", tool_call_count=0)
        self.assertNotIn("本轮工具调用", prompt)


class TestFormatAuditFeedback(unittest.TestCase):
    """_format_audit_feedback() feedback formatting."""

    def test_non_dict_returns_none(self):
        self.assertIsNone(_format_audit_feedback("not a dict"))
        self.assertIsNone(_format_audit_feedback(42))
        self.assertIsNone(_format_audit_feedback(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(_format_audit_feedback({}))

    def test_no_issues_suggestions_fatal_returns_none(self):
        self.assertIsNone(_format_audit_feedback({
            "issues": [], "suggestions": [], "fatal_issues": [],
        }))

    def test_fatal_issues_formatted(self):
        result = _format_audit_feedback({
            "issues": [], "suggestions": [],
            "fatal_issues": ["fabricated evidence"],
        })
        self.assertIsNotNone(result)
        self.assertIn("CRITICAL", result)
        self.assertIn("fabricated evidence", result)

    def test_issues_and_suggestions_formatted(self):
        result = _format_audit_feedback({
            "issues": ["bad accuracy", "incomplete"],
            "suggestions": ["use more tools"],
            "fatal_issues": [],
        })
        self.assertIsNotNone(result)
        self.assertIn("bad accuracy", result)
        self.assertIn("incomplete", result)
        self.assertIn("use more tools", result)

    def test_top_2_issues_top_1_suggestion(self):
        result = _format_audit_feedback({
            "issues": ["i1", "i2", "i3"],
            "suggestions": ["s1", "s2"],
            "fatal_issues": [],
        })
        self.assertIn("i1", result)
        self.assertIn("i2", result)
        self.assertNotIn("i3", result)
        self.assertIn("s1", result)
        self.assertNotIn("s2", result)

    def test_score_included(self):
        result = _format_audit_feedback({
            "issues": ["bad"], "suggestions": [], "fatal_issues": [],
            "total_score": 3.5,
        })
        self.assertIn("3.5", result)

    def test_none_values_treated_as_empty(self):
        result = _format_audit_feedback({
            "issues": None, "suggestions": None, "fatal_issues": None,
        })
        self.assertIsNone(result)


class TestInjectAuditFeedback(unittest.TestCase):
    """inject_audit_feedback() message wrapping."""

    def test_returns_original_when_no_feedback(self):
        with patch("agent.quality_auditor.get_last_audit_feedback", return_value=None):
            result = inject_audit_feedback("sess1", "hello")
        self.assertEqual(result, "hello")

    def test_prepends_feedback(self):
        with patch("agent.quality_auditor.get_last_audit_feedback",
                   return_value="[Quality feedback] Score: 4/10"):
            result = inject_audit_feedback("sess1", "hello")
        self.assertIn("[Quality feedback]", result)
        self.assertIn("hello", result)
        # feedback comes before user message
        self.assertTrue(result.index("[Quality feedback]") < result.index("hello"))

    def test_empty_session_id_returns_original(self):
        result = inject_audit_feedback("", "hello")
        self.assertEqual(result, "hello")

    def test_non_string_message_returns_original(self):
        result = inject_audit_feedback("sess1", 42)
        self.assertEqual(result, 42)


class TestWriteAuditEntry(unittest.TestCase):
    """_write_audit_entry() JSONL persistence."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._hermes_home = Path(self._tmpdir) / ".hermes"
        self._state_dir = self._hermes_home / "state"
        self._state_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_writes_valid_jsonl(self):
        audit_file = self._state_dir / "quality_audit.jsonl"
        with patch("agent.quality_auditor.get_hermes_home",
                   return_value=self._hermes_home):
            _write_audit_entry({"total_score": 7.5, "session_id": "s1"})
        lines = audit_file.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["total_score"], 7.5)
        self.assertEqual(entry["session_id"], "s1")
        self.assertIn("_ts", entry)

    def test_appends_multiple_entries(self):
        audit_file = self._state_dir / "quality_audit.jsonl"
        with patch("agent.quality_auditor.get_hermes_home",
                   return_value=self._hermes_home):
            _write_audit_entry({"total_score": 5.0})
            _write_audit_entry({"total_score": 8.0})
        lines = audit_file.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)


class TestGetLastAuditFeedback(unittest.TestCase):
    """get_last_audit_feedback() freshness-windowed lookup."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_dir = Path(self._tmpdir) / "state"
        self._state_dir.mkdir(parents=True)
        self._audit_file = self._state_dir / "quality_audit.jsonl"
        self._orig_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self._tmpdir

    def tearDown(self):
        if self._orig_home is not None:
            os.environ["HERMES_HOME"] = self._orig_home
        elif "HERMES_HOME" in os.environ:
            del os.environ["HERMES_HOME"]
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_missing_file_returns_none(self):
        result = get_last_audit_feedback("sess1")
        self.assertIsNone(result)

    def test_recent_entry_returned(self):
        entry = {
            "session_id": "sess1",
            "total_score": 3.0,
            "issues": ["low accuracy"],
            "suggestions": [],
            "fatal_issues": [],
            "_ts": time.time(),
        }
        self._audit_file.write_text(json.dumps(entry) + "\n")
        result = get_last_audit_feedback("sess1")
        self.assertIsNotNone(result)
        self.assertIn("low accuracy", result)

    def test_stale_entry_returns_none(self):
        entry = {
            "session_id": "sess1",
            "total_score": 3.0,
            "issues": ["old issue"],
            "suggestions": [],
            "fatal_issues": [],
            "_ts": time.time() - 7200,
        }
        self._audit_file.write_text(json.dumps(entry) + "\n")
        result = get_last_audit_feedback("sess1")
        self.assertIsNone(result)

    def test_skips_non_json_lines(self):
        self._audit_file.write_text("garbage line\n")
        result = get_last_audit_feedback("sess1")
        self.assertIsNone(result)

    def test_session_id_filter(self):
        entry = {
            "session_id": "other_session",
            "total_score": 3.0,
            "issues": ["wrong session"],
            "suggestions": [],
            "fatal_issues": [],
            "_ts": time.time(),
        }
        self._audit_file.write_text(json.dumps(entry) + "\n")
        result = get_last_audit_feedback("sess1")
        self.assertIsNone(result)


class TestAggregateDaily(unittest.TestCase):
    """aggregate_daily() daily statistics."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._state_dir = Path(self._tmpdir) / "state"
        self._state_dir.mkdir(parents=True)
        self._audit_file = self._state_dir / "quality_audit.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _patch_home(self):
        """Patch get_hermes_home in the quality_auditor module's namespace."""
        p = patch.object(
            __import__("agent.quality_auditor", fromlist=["get_hermes_home"]),
            "get_hermes_home",
            return_value=Path(self._tmpdir),
        )
        return p

    def test_missing_file_returns_empty_report(self):
        with self._patch_home():
            result = aggregate_daily()
        self.assertEqual(result["total"], 0)

    def test_aggregates_scores(self):
        now = time.time()
        for score in [6.0, 8.0, 4.0]:
            entry = {
                "session_id": "s1",
                "total_score": score,
                "scores": {"accuracy": score},
                "issues": [f"issue_{int(score)}"],
                "suggestions": [],
                "fatal_issues": [],
                "_ts": now,
            }
            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        with self._patch_home():
            result = aggregate_daily()
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["avg_score"], 6.0)
        self.assertEqual(result["min_score"], 4.0)
        self.assertEqual(result["max_score"], 8.0)

    def test_filters_by_24h_cutoff(self):
        # Old entry (2 days ago)
        old_entry = {"total_score": 9.0, "issues": [], "suggestions": [],
                     "fatal_issues": [], "_ts": time.time() - 172800}
        # Fresh entry
        fresh_entry = {"total_score": 5.0, "issues": [], "suggestions": [],
                       "fatal_issues": [], "_ts": time.time()}
        self._audit_file.write_text(json.dumps(old_entry) + "\n")
        self._audit_file.write_text(json.dumps(fresh_entry) + "\n")
        with self._patch_home():
            result = aggregate_daily()
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["avg_score"], 5.0)

    def test_fatal_count(self):
        now = time.time()
        for has_fatal in [True, False, True]:
            entry = {
                "total_score": 3.0,
                "fatal_issues": ["fatal"] if has_fatal else [],
                "issues": [], "suggestions": [], "_ts": now,
            }
            with open(self._audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        with self._patch_home():
            result = aggregate_daily()
        self.assertEqual(result["fatal_count"], 2)

    def test_output_file_written(self):
        self._audit_file.write_text(json.dumps({
            "total_score": 7.0, "issues": [], "suggestions": [],
            "fatal_issues": [], "_ts": time.time(),
        }) + "\n")
        output_path = os.path.join(self._tmpdir, "report.json")
        with self._patch_home():
            result = aggregate_daily(output_file=output_path)
        self.assertTrue(os.path.exists(output_path))
        report = json.loads(open(output_path).read())
        self.assertEqual(report["total"], 1)


class TestFireQualityAudit(unittest.TestCase):
    """fire_quality_audit() dispatch logic."""

    def test_short_response_skipped(self):
        result = fire_quality_audit("hello", "too short")
        self.assertIsNone(result)

    def test_disabled_auditor_skipped(self):
        with patch("agent.quality_auditor._AUDIT_ENABLED", False):
            result = fire_quality_audit("hello world" * 10, "response" * 20)
        self.assertIsNone(result)

    def test_fast_timeout_returns_entry(self):
        with patch("agent.quality_auditor._AUDIT_ENABLED", True), \
             patch("agent.quality_auditor._build_audit_entry") as mock_build, \
             patch("agent.quality_auditor._write_audit_entry") as mock_write:
            mock_build.return_value = {"total_score": 7.0, "issues": [], "fatal_issues": []}
            result = fire_quality_audit(
                "message" * 20, "response" * 20,
                fast_timeout=5,
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["total_score"], 7.0)
        mock_write.assert_called_once()

    def test_fast_timeout_returns_none_on_timeout(self):
        import threading
        call_started = threading.Event()

        def slow_build(*args, **kwargs):
            call_started.set()
            time.sleep(100)  # way longer than timeout
            return {"total_score": 7.0}

        with patch("agent.quality_auditor._AUDIT_ENABLED", True), \
             patch("agent.quality_auditor._build_audit_entry", side_effect=slow_build):
            result = fire_quality_audit(
                "message" * 20, "response" * 20,
                fast_timeout=0.01,
            )
        self.assertIsNone(result)

    def test_background_mode_returns_none(self):
        with patch("agent.quality_auditor._AUDIT_ENABLED", True):
            result = fire_quality_audit("message" * 20, "response" * 20)
        self.assertIsNone(result)  # background mode returns None


if __name__ == "__main__":
    unittest.main()
