#!/usr/bin/env python3
"""
Tests for pipeline-guard plugin (physical design gate) and SelfCheck R15.

Verifies:
1. pipeline-guard plugin blocks write_file/patch when phase < 4
2. pipeline-guard plugin allows write_file/patch when phase >= 4
3. pipeline-guard plugin allows everything when no phase file exists
4. pipeline-guard plugin allows everything when pipeline is inactive
5. SelfCheck R15 warns when feature-dev-pipeline loaded + phase < 4
6. SelfCheck R15 does NOT warn when pipeline not loaded
7. SelfCheck R15 does NOT warn when phase >= 4
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPipelineGuardPlugin(unittest.TestCase):
    """Test the pipeline-guard pre_tool_call hook."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.hermes_home = self.tmpdir / ".hermes"
        self.cache_dir = self.hermes_home / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._patcher = patch.dict(os.environ, {"HERMES_HOME": str(self.hermes_home)})
        self._patcher.start()
        # Load plugin module directly (hyphen in dir name prevents normal import)
        import importlib.util
        plugin_path = Path(__file__).parent.parent.parent / "plugins" / "pipeline-guard" / "__init__.py"
        spec = importlib.util.spec_from_file_location("pipeline_guard", plugin_path)
        self._pg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._pg)

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _write_phase_file(self, phase, active=True, session_id="test"):
        """Write a phase file to the cache directory."""
        phase_file = self.cache_dir / f"pipeline_phase_{session_id}.json"
        phase_file.write_text(json.dumps({"phase": phase, "active": active}))

    def test_blocks_write_file_when_phase_below_4(self):
        """write_file must be blocked when phase=2, active=true."""
        self._write_phase_file(phase=2, active=True)
        result = self._pg.on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/test.py", "content": "print('hello')"},
            session_id="test",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "block")

    def test_blocks_patch_when_phase_below_4(self):
        """patch must be blocked when phase=3, active=true."""
        self._write_phase_file(phase=3, active=True)
        result = self._pg.on_pre_tool_call(
            tool_name="patch",
            args={"path": "/test.py", "old_string": "a", "new_string": "b"},
            session_id="test",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "block")

    def test_allows_write_file_when_phase_4(self):
        """write_file must be allowed when phase=4 (implementation)."""
        self._write_phase_file(phase=4, active=True)
        result = self._pg.on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/test.py", "content": "print('hello')"},
            session_id="test",
        )
        self.assertIsNone(result)

    def test_allows_when_no_phase_file(self):
        """No phase file = pipeline not active = everything allowed."""
        result = self._pg.on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/test.py", "content": "x"},
            session_id="test",
        )
        self.assertIsNone(result)

    def test_allows_when_pipeline_inactive(self):
        """active=false = pipeline finished = everything allowed."""
        self._write_phase_file(phase=2, active=False)
        result = self._pg.on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/test.py", "content": "x"},
            session_id="test",
        )
        self.assertIsNone(result)

    def test_allows_read_only_tools(self):
        """read-only tools must never be blocked."""
        self._write_phase_file(phase=1, active=True)
        for tool in ("read_file", "search_files", "web_search"):
            result = self._pg.on_pre_tool_call(tool_name=tool, args={}, session_id="test")
            self.assertIsNone(result, f"{tool} should not be blocked")

    def test_pre_verify_blocks_done_before_phase_5(self):
        """pre_verify must block premature 'done' claims when phase < 5."""
        self._write_phase_file(phase=4, active=True)
        result = self._pg.on_pre_verify(
            coding=True,
            attempt=0,
            session_id="test",
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "continue")

    def test_pre_verify_allows_done_at_phase_5(self):
        """pre_verify must allow 'done' when phase >= 5."""
        self._write_phase_file(phase=5, active=True)
        result = self._pg.on_pre_verify(
            coding=True,
            attempt=0,
            session_id="test",
        )
        self.assertIsNone(result)


class TestSelfCheckR15(unittest.TestCase):
    """Test the SelfCheck R15 design gate warning."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self.hermes_home = self.tmpdir / ".hermes"
        self.cache_dir = self.hermes_home / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._patcher = patch(
            "hermes_constants.get_hermes_home",
            return_value=self.hermes_home,
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _make_sc(self, pipeline_loaded=False):
        """Create a SelfCheckManager with optional pipeline skill loaded."""
        from agent.self_check import SelfCheckManager

        sc = SelfCheckManager()
        sc._loaded = True
        if pipeline_loaded:
            sc._loaded_skill_dirs = {
                "/home/user/.hermes/skills/engineering-methodology/feature-dev-pipeline"
            }
        return sc

    def _write_phase_file(self, phase, active=True):
        phase_file = self.cache_dir / "pipeline_phase_test.json"
        phase_file.write_text(json.dumps({"phase": phase, "active": active}))

    def test_r15_warns_when_pipeline_loaded_and_phase_below_4(self):
        """R15 should warn: pipeline loaded + phase=2 + write_file."""
        sc = self._make_sc(pipeline_loaded=True)
        self._write_phase_file(phase=2, active=True)

        warning = sc._check_r15_design_gate("write_file", {"path": "/test.py"})
        self.assertIsNotNone(warning)
        self.assertIn("phase=2", warning)
        self.assertIn("RL-5", warning)

    def test_r15_silent_when_pipeline_not_loaded(self):
        """R15 must NOT warn when feature-dev-pipeline skill is not loaded."""
        sc = self._make_sc(pipeline_loaded=False)
        self._write_phase_file(phase=2, active=True)

        warning = sc._check_r15_design_gate("write_file", {"path": "/test.py"})
        self.assertIsNone(warning)

    def test_r15_silent_when_phase_4(self):
        """R15 must NOT warn when phase >= 4 (implementation reached)."""
        sc = self._make_sc(pipeline_loaded=True)
        self._write_phase_file(phase=4, active=True)

        warning = sc._check_r15_design_gate("write_file", {"path": "/test.py"})
        self.assertIsNone(warning)

    def test_r15_silent_when_no_phase_file(self):
        """R15 must NOT warn when no phase file exists (pipeline not active)."""
        sc = self._make_sc(pipeline_loaded=True)
        warning = sc._check_r15_design_gate("write_file", {"path": "/test.py"})
        self.assertIsNone(warning)

    def test_r15_silent_when_pipeline_inactive(self):
        """R15 must NOT warn when phase file exists but active=false."""
        sc = self._make_sc(pipeline_loaded=True)
        self._write_phase_file(phase=2, active=False)

        warning = sc._check_r15_design_gate("write_file", {"path": "/test.py"})
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
