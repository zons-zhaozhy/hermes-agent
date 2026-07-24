#!/usr/bin/env python3
"""
Tests for subagent progress ledger (P0).

Tests the JSONL append/read/clear cycle, multi-session isolation,
and compaction-resume scenarios.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestProgressLedger(unittest.TestCase):
    """Test the progress ledger write/read/clear cycle."""

    def setUp(self):
        """Redirect HERMES_HOME to a temp dir for each test."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.hermes_home = self.tmpdir / ".hermes"
        self.hermes_home.mkdir(parents=True, exist_ok=True)
        # Patch get_hermes_home so the ledger writes to our temp dir
        self._patcher = patch(
            "hermes_constants.get_hermes_home",
            return_value=self.hermes_home,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_append_and_read_single_task(self):
        """A single task completion should be readable back."""
        from tools.subagent_progress import (
            append_task_completion,
            read_completed_tasks,
        )

        append_task_completion(
            ledger_session_id="test-session-1",
            task_index=0,
            goal="Write unit tests for foo()",
            status="completed",
            summary="Wrote 5 tests, all passing",
            model="test-model",
            duration_seconds=12.5,
            files_written=["tests/test_foo.py"],
        )

        completed = read_completed_tasks("test-session-1")
        self.assertIn(0, completed)
        self.assertEqual(completed[0]["status"], "completed")
        self.assertEqual(completed[0]["goal"], "Write unit tests for foo()")

    def test_multiple_tasks_same_session(self):
        """Multiple tasks in the same session should all be readable."""
        from tools.subagent_progress import (
            append_task_completion,
            read_completed_tasks,
        )

        for i in range(3):
            append_task_completion(
                ledger_session_id="test-session-2",
                task_index=i,
                goal=f"Task {i}",
                status="completed",
                summary=f"Summary {i}",
            )

        completed = read_completed_tasks("test-session-2")
        self.assertEqual(len(completed), 3)
        self.assertEqual(set(completed.keys()), {0, 1, 2})

    def test_multi_session_isolation(self):
        """Sessions should be isolated — reading one shouldn't return another's tasks."""
        from tools.subagent_progress import (
            append_task_completion,
            read_completed_tasks,
        )

        append_task_completion(
            ledger_session_id="session-A",
            task_index=0,
            goal="Task A",
            status="completed",
            summary="A",
        )
        append_task_completion(
            ledger_session_id="session-B",
            task_index=0,
            goal="Task B",
            status="completed",
            summary="B",
        )

        completed_a = read_completed_tasks("session-A")
        completed_b = read_completed_tasks("session-B")
        self.assertEqual(len(completed_a), 1)
        self.assertEqual(len(completed_b), 1)
        self.assertEqual(completed_a[0]["goal"], "Task A")
        self.assertEqual(completed_b[0]["goal"], "Task B")

    def test_failed_tasks_not_in_completed(self):
        """Failed tasks should NOT be returned by read_completed_tasks."""
        from tools.subagent_progress import (
            append_task_completion,
            read_completed_tasks,
        )

        append_task_completion(
            ledger_session_id="test-session-3",
            task_index=0,
            goal="Succeeding task",
            status="completed",
            summary="OK",
        )
        append_task_completion(
            ledger_session_id="test-session-3",
            task_index=1,
            goal="Failing task",
            status="failed",
            summary="Error",
        )

        completed = read_completed_tasks("test-session-3")
        self.assertIn(0, completed)
        self.assertNotIn(1, completed)

    def test_retry_overwrites_failure(self):
        """A task that first fails then succeeds should appear as completed."""
        from tools.subagent_progress import (
            append_task_completion,
            read_completed_tasks,
        )

        # First attempt fails
        append_task_completion(
            ledger_session_id="test-session-4",
            task_index=0,
            goal="Retryable task",
            status="failed",
            summary="First attempt failed",
        )
        # Second attempt succeeds
        append_task_completion(
            ledger_session_id="test-session-4",
            task_index=0,
            goal="Retryable task",
            status="completed",
            summary="Second attempt succeeded",
        )

        completed = read_completed_tasks("test-session-4")
        self.assertIn(0, completed)
        self.assertEqual(completed[0]["status"], "completed")

    def test_clear_session(self):
        """clear_session should remove only the specified session's records."""
        from tools.subagent_progress import (
            append_task_completion,
            clear_session,
            read_completed_tasks,
        )

        append_task_completion(
            ledger_session_id="to-clear",
            task_index=0,
            goal="Clear me",
            status="completed",
            summary="",
        )
        append_task_completion(
            ledger_session_id="keep-me",
            task_index=0,
            goal="Keep me",
            status="completed",
            summary="",
        )

        clear_session("to-clear")
        self.assertEqual(read_completed_tasks("to-clear"), {})
        self.assertEqual(len(read_completed_tasks("keep-me")), 1)

    def test_read_empty_ledger(self):
        """Reading a non-existent ledger should return {} safely."""
        from tools.subagent_progress import read_completed_tasks

        result = read_completed_tasks("nonexistent-session")
        self.assertEqual(result, {})

    def test_append_failure_is_safe(self):
        """append_task_completion should never raise — it logs and returns."""
        from tools.subagent_progress import append_task_completion

        # Even with invalid data, it should not raise
        append_task_completion(
            ledger_session_id=None,  # invalid
            task_index=0,
            goal="test",
            status="completed",
            summary="",
        )
        # No exception = pass


if __name__ == "__main__":
    unittest.main()
