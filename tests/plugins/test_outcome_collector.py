"""Tests for the outcome-collector plugin.

Tests three layers:
1. Plugin hooks fire and buffer outcomes
2. Buffer flushes to outcomes.db with correct schema
3. analyze.py produces correct findings from the data
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import the plugin module directly (hyphenated dir names aren't valid Python modules)
PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "outcome-collector"

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "outcome_collector_plugin", PLUGIN_DIR / "__init__.py"
)
plugin_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin_mod)

# Also load analyze.py as a module
_a_spec = importlib.util.spec_from_file_location(
    "outcome_analyzer", PLUGIN_DIR / "analyze.py"
)
analyze_mod = importlib.util.module_from_spec(_a_spec)
_a_spec.loader.exec_module(analyze_mod)
run_analysis = analyze_mod.run_analysis


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Redirect outcomes.db to a temp directory."""
    db_path = tmp_path / "outcomes.db"
    # Reset the module-level cached path
    monkeypatch.setattr(plugin_mod, "_db_path", db_path)
    monkeypatch.setattr(plugin_mod, "_schema_ready", False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield db_path


@pytest.fixture
def clean_plugin(monkeypatch, isolated_db):
    """Reset all plugin buffers."""
    monkeypatch.setattr(plugin_mod, "_buffers", {})
    monkeypatch.setattr(plugin_mod, "_schema_ready", False)
    yield


# ── Hook tests ────────────────────────────────────────────────────────

class TestPostToolCallHook:
    """Layer 0: post_tool_call hook captures outcomes."""

    def test_ok_result_buffered(self, clean_plugin, isolated_db):
        """A successful tool call should be buffered."""
        plugin_mod.on_post_tool_call(
            tool_name="read_file",
            args={"path": "/some/file.py"},
            result="file content",
            session_id="test-session-1",
            turn_id="turn-1",
            tool_call_id="tc-1",
            status="ok",
            duration_ms=42,
        )

        buf = plugin_mod._buffers.get("test-session-1", [])
        assert len(buf) == 1
        assert buf[0]["tool_name"] == "read_file"
        assert buf[0]["status"] == "ok"
        assert buf[0]["duration_ms"] == 42
        assert buf[0]["args_summary"]["path"] == "/some/file.py"

    def test_error_result_buffered(self, clean_plugin, isolated_db):
        """An error tool call should capture error fields."""
        plugin_mod.on_post_tool_call(
            tool_name="patch",
            args={"path": "/app/main.py", "old_string": "def old():", "new_string": "def new():"},
            result=None,
            session_id="test-session-2",
            status="error",
            error_type="tool_error",
            error_message="old_string not found in file",
            duration_ms=100,
        )

        buf = plugin_mod._buffers.get("test-session-2", [])
        assert len(buf) == 1
        assert buf[0]["status"] == "error"
        assert buf[0]["error_type"] == "tool_error"
        assert "not found" in buf[0]["error_message"]

    def test_missing_session_id_skipped(self, clean_plugin, isolated_db):
        """No session_id → silently skip."""
        plugin_mod.on_post_tool_call(
            tool_name="read_file",
            args={"path": "/x"},
            result="ok",
            session_id="",
            status="ok",
        )
        assert len(plugin_mod._buffers) == 0

    def test_disabled_plugin_noop(self, clean_plugin, isolated_db, monkeypatch):
        """When OUTCOME_COLLECTOR_DISABLE=1, nothing is captured."""
        monkeypatch.setenv("OUTCOME_COLLECTOR_DISABLE", "1")
        plugin_mod.on_post_tool_call(
            tool_name="read_file",
            args={"path": "/x"},
            result="ok",
            session_id="test-session-x",
            status="ok",
        )
        assert len(plugin_mod._buffers) == 0

    def test_diagnostic_args_extraction_per_tool(self, clean_plugin, isolated_db):
        """Different tools extract different diagnostic fields."""
        test_cases = [
            ("terminal", {"command": "git status"}, "cmd_prefix"),
            ("skill_view", {"name": "coding-conventions"}, "skill"),
            ("web_search", {"query": "fastapi async patterns"}, "query_prefix"),
            ("delegate_task", {"goal": "fix all linting errors"}, "goal_prefix"),
            ("browser_click", {"ref": "@e5"}, "ref"),
        ]

        for tool, args, expected_key in test_cases:
            plugin_mod.on_post_tool_call(
                tool_name=tool,
                args=args,
                result="ok",
                session_id="test-args",
                status="ok",
            )

        buf = plugin_mod._buffers.get("test-args", [])
        assert len(buf) == len(test_cases)
        for i, (_, _, key) in enumerate(test_cases):
            assert key in buf[i]["args_summary"], f"Missing {key} for {test_cases[i][0]}"

    def test_no_arg_tools_empty_summary(self, clean_plugin, isolated_db):
        """browser_snapshot, memory, todo → empty args_summary."""
        for tool in ("browser_snapshot", "memory", "todo"):
            plugin_mod.on_post_tool_call(
                tool_name=tool,
                args={"some": "stuff"},
                result="ok",
                session_id="test-noargs",
                status="ok",
            )

        buf = plugin_mod._buffers.get("test-noargs", [])
        for entry in buf:
            assert entry["args_summary"] == {}

    def test_error_message_truncated(self, clean_plugin, isolated_db):
        """Long error messages should be truncated to 300 chars."""
        long_msg = "x" * 500
        plugin_mod.on_post_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            result=None,
            session_id="test-trunc",
            status="error",
            error_message=long_msg,
        )

        buf = plugin_mod._buffers.get("test-trunc", [])
        assert len(buf[0]["error_message"]) == 300

    def test_uses_task_id_when_session_id_missing(self, clean_plugin, isolated_db):
        """task_id should be used as fallback for session identification."""
        plugin_mod.on_post_tool_call(
            tool_name="read_file",
            args={"path": "/x"},
            result="ok",
            session_id="",
            task_id="task-fallback-id",
            status="ok",
        )
        assert "task-fallback-id" in plugin_mod._buffers


# ── Flush & persistence tests ─────────────────────────────────────────

class TestBufferFlush:
    """Buffer flushing to outcomes.db."""

    def test_session_end_flushes_buffer(self, clean_plugin, isolated_db):
        """on_session_end should flush all buffered entries to DB."""
        # Buffer 3 entries
        for i in range(3):
            plugin_mod.on_post_tool_call(
                tool_name="read_file",
                args={"path": f"/file{i}.py"},
                result="content",
                session_id="flush-test",
                status="ok" if i < 2 else "error",
                error_message="fail" if i == 2 else None,
            )

        # Flush
        plugin_mod.on_session_end(session_id="flush-test")

        # Verify DB
        conn = sqlite3.connect(str(isolated_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tool_outcomes WHERE session_id = ?", ("flush-test",)
        ).fetchall()
        conn.close()

        assert len(rows) == 3
        statuses = [r["status"] for r in rows]
        assert "ok" in statuses
        assert "error" in statuses

    def test_auto_flush_at_threshold(self, clean_plugin, isolated_db, monkeypatch):
        """Buffer should auto-flush when reaching _FLUSH_THRESHOLD."""
        monkeypatch.setattr(plugin_mod, "_FLUSH_THRESHOLD", 3)

        for i in range(3):
            plugin_mod.on_post_tool_call(
                tool_name="terminal",
                args={"command": f"echo {i}"},
                result="ok",
                session_id="threshold-test",
                status="ok",
            )

        # Buffer should have been flushed
        assert len(plugin_mod._buffers.get("threshold-test", [])) == 0

        # DB should have 3 rows
        conn = sqlite3.connect(str(isolated_db))
        count = conn.execute(
            "SELECT COUNT(*) FROM tool_outcomes WHERE session_id = ?", ("threshold-test",)
        ).fetchone()[0]
        conn.close()
        assert count == 3

    def test_session_summary_computed(self, clean_plugin, isolated_db):
        """on_session_end should compute a session summary."""
        for i in range(5):
            plugin_mod.on_post_tool_call(
                tool_name="terminal" if i < 3 else "read_file",
                args={"command": "ls"} if i < 3 else {"path": "/x"},
                result="ok",
                session_id="summary-test",
                status="ok" if i < 4 else "error",
            )

        plugin_mod.on_session_end(session_id="summary-test")

        conn = sqlite3.connect(str(isolated_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM session_summaries WHERE session_id = ?", ("summary-test",)
        ).fetchone()
        conn.close()

        assert row is not None
        assert row["total_calls"] == 5
        assert row["error_count"] == 1
        breakdown = json.loads(row["tool_breakdown"])
        assert breakdown["terminal"] == 3
        assert breakdown["read_file"] == 2


# ── DB schema tests ───────────────────────────────────────────────────

class TestDatabaseSchema:
    """Schema correctness and index presence."""

    def test_schema_created_on_first_use(self, clean_plugin, isolated_db):
        """Tables and indexes should be created on first flush."""
        plugin_mod.on_post_tool_call(
            tool_name="read_file",
            args={"path": "/x"},
            result="ok",
            session_id="schema-test",
            status="ok",
        )
        plugin_mod.on_session_end(session_id="schema-test")

        conn = sqlite3.connect(str(isolated_db))
        conn.row_factory = sqlite3.Row

        # Check tables exist
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "tool_outcomes" in tables
        assert "session_summaries" in tables

        # Check indexes exist
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_to_session" in indexes
        assert "idx_to_tool" in indexes
        assert "idx_to_status" in indexes

        conn.close()


# ── Registration test ─────────────────────────────────────────────────

class TestPluginRegistration:
    """Plugin registration with PluginContext."""

    def test_registers_hooks(self):
        """register() should register three hooks: post_tool_call, pre_llm_call, on_session_end."""
        ctx = MagicMock()
        plugin_mod.register(ctx)

        registered_hooks = [call.args for call in ctx.register_hook.call_args_list]
        hook_names = {args[0] for args in registered_hooks}
        assert "post_tool_call" in hook_names
        assert "pre_llm_call" in hook_names
        assert "on_session_end" in hook_names


# ── Analyzer tests ────────────────────────────────────────────────────

class TestAnalyzer:
    """Layer 2: analyze.py pattern detection."""

    @pytest.fixture
    def populated_db(self, isolated_db, clean_plugin):
        """Populate outcomes.db with known data for analysis."""
        # Ensure schema exists before inserting
        plugin_mod._ensure_schema_once()

        # Insert 10 calls for 'patch': 6 ok, 4 error (40% error rate)
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        for i in range(10):
            status = "error" if i < 4 else "ok"
            conn.execute(
                """INSERT INTO tool_outcomes
                   (session_id, tool_name, status, error_type, error_message,
                    duration_ms, args_summary, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "analysis-session",
                    "patch",
                    status,
                    "tool_error" if status == "error" else None,
                    "old_string not found" if status == "error" else None,
                    50,
                    json.dumps({"path": "/app/main.py"}),
                    now,
                ),
            )

        # Insert 5 calls for 'read_file': all ok
        for i in range(5):
            conn.execute(
                """INSERT INTO tool_outcomes
                   (session_id, tool_name, status, duration_ms, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                ("analysis-session", "read_file", "ok", 10, now),
            )

        conn.commit()
        conn.close()
        yield isolated_db

    def test_high_error_tool_detected(self, populated_db):
        """Analyzer should flag 'patch' as a high-error tool (>30%)."""
        report = run_analysis(populated_db, days=7)

        high_error_tools = [
            f for f in report["findings"] if f["type"] == "high_error_tool"
        ]
        assert len(high_error_tools) >= 1

        patch_finding = next(
            f for f in high_error_tools if f["tool"] == "patch"
        )
        assert patch_finding["error_rate"] == 40.0
        assert patch_finding["total_calls"] == 10
        assert patch_finding["error_count"] == 4
        assert patch_finding["severity"] == "medium"

    def test_low_error_tool_not_flagged(self, populated_db):
        """read_file (0% error) should NOT appear in high_error_tool findings."""
        report = run_analysis(populated_db, days=7)

        read_file_findings = [
            f for f in report["findings"]
            if f.get("tool") == "read_file" and f["type"] == "high_error_tool"
        ]
        assert len(read_file_findings) == 0

    def test_recurring_error_detected(self, populated_db):
        """Analyzer should detect the 'old_string not found' recurring error."""
        report = run_analysis(populated_db, days=7)

        recurring = [
            f for f in report["findings"] if f["type"] == "recurring_error"
        ]
        assert len(recurring) >= 1
        assert recurring[0]["count"] == 4
        assert "not found" in recurring[0]["error_message"]

    def test_usage_breakdown_correct(self, populated_db):
        """Usage breakdown should show correct totals per tool."""
        report = run_analysis(populated_db, days=7)
        usage = report["usage_breakdown"]

        tool_map = {t["tool"]: t for t in usage["tools"]}
        assert tool_map["patch"]["total"] == 10
        assert tool_map["read_file"]["total"] == 5


# ── Layer 1 tests ─────────────────────────────────────────────────────

class TestLayer1TaskOutcome:
    """Layer 1: task-level outcome inference from tool-call sequence patterns."""

    def test_repeated_same_tool_error_is_failure(self):
        """Same tool erroring twice → failure/repeated_same_tool_error."""
        outcome, pattern = plugin_mod._classify_tool_sequence([
            {"tool_name": "patch", "status": "error"},
            {"tool_name": "patch", "status": "error"},
            {"tool_name": "patch", "status": "error"},
        ])
        assert outcome == "failure"
        assert pattern == "repeated_same_tool_error"

    def test_retry_then_success_is_success(self):
        """Tool errors then succeeds on retry → success/retry_then_success."""
        outcome, pattern = plugin_mod._classify_tool_sequence([
            {"tool_name": "patch", "status": "error"},
            {"tool_name": "patch", "status": "ok"},
            {"tool_name": "read_file", "status": "ok"},
        ])
        assert outcome == "success"
        assert pattern == "retry_then_success"

    def test_clean_completion_is_success(self):
        """All tool calls ok → success/clean_completion."""
        outcome, pattern = plugin_mod._classify_tool_sequence([
            {"tool_name": "read_file", "status": "ok"},
            {"tool_name": "search_files", "status": "ok"},
            {"tool_name": "write_file", "status": "ok"},
        ])
        assert outcome == "success"
        assert pattern == "clean_completion"

    def test_high_error_density_is_partial(self):
        """≥50% error rate with ≥5 calls but different tools → partial/high_error_density."""
        outcome, pattern = plugin_mod._classify_tool_sequence([
            {"tool_name": "patch", "status": "error"},
            {"tool_name": "search_files", "status": "error"},
            {"tool_name": "read_file", "status": "ok"},
            {"tool_name": "terminal", "status": "error"},
            {"tool_name": "web_search", "status": "error"},
        ])
        assert outcome == "partial"
        assert pattern == "high_error_density"

    def test_empty_tools_unknown(self):
        """No tool calls → unknown outcome."""
        outcome, pattern = plugin_mod._classify_tool_sequence([])
        assert outcome == "unknown"

    def test_no_keyword_dependency(self):
        """Classification should NOT depend on any user message text.

        This is the fundamental design change — outcome is derived purely
        from tool-call status sequence, never from user message keywords.
        """
        # Same tool pattern, different 'user messages' — outcome must be identical
        seq = [
            {"tool_name": "patch", "status": "ok"},
            {"tool_name": "read_file", "status": "ok"},
        ]
        outcome_a, _ = plugin_mod._classify_tool_sequence(seq)
        # _classify_tool_sequence takes NO user_message parameter at all
        assert outcome_a == "success"

    def test_pre_llm_call_classifies_and_persists(self, isolated_db, clean_plugin, monkeypatch):
        """pre_llm_call hook should classify tool sequence and persist outcome."""
        sid = "test-layer1-seq"
        plugin_mod._turn_outcomes.clear()
        plugin_mod._turn_outcomes[sid] = [{
            "turn_id": "turn-1",
            "tool_calls": [
                {"tool_name": "patch", "status": "error"},
                {"tool_name": "patch", "status": "error"},
            ],
        }]

        # pre_llm_call now classifies from sequence — no user_message needed
        plugin_mod.on_pre_llm_call(session_id=sid, is_first_turn=False)

        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(isolated_db))
        conn.row_factory = _sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM tool_outcomes WHERE tool_name = '_turn_outcome'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["status"] == "failure"
        assert rows[0]["error_type"] == "repeated_same_tool_error"


# ── Layer 3 context injection tests ───────────────────────────────────

class TestLayer3ContextInjection:
    """Test that pre_llm_call injects findings context on first turn."""

    def test_first_turn_with_findings_returns_context(self, clean_plugin, monkeypatch, tmp_path):
        """pre_llm_call on first turn should return context when findings exist."""
        findings_dir = tmp_path / "outcomes"
        findings_dir.mkdir()
        (findings_dir / "findings.md").write_text(
            "# Outcome Analysis\n\n## High-Severity Findings\n"
            "  ⚠ patch: 80.0% error rate (8/10 calls)\n"
        )

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        plugin_mod._injected_sessions.clear()
        result = plugin_mod.on_pre_llm_call(
            session_id="test-ctx-1", is_first_turn=True
        )
        assert result is not None
        assert "context" in result
        assert "patch" in result["context"]
        assert "80.0%" in result["context"]

    def test_first_turn_no_findings_returns_none(self, clean_plugin, monkeypatch, tmp_path):
        """pre_llm_call should return None when no findings file exists."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        plugin_mod._injected_sessions.clear()
        result = plugin_mod.on_pre_llm_call(
            session_id="test-ctx-2", is_first_turn=True
        )
        assert result is None

    def test_injection_once_per_session(self, clean_plugin, monkeypatch, tmp_path):
        """Findings context should only be injected once per session."""
        findings_dir = tmp_path / "outcomes"
        findings_dir.mkdir()
        (findings_dir / "findings.md").write_text(
            "# Analysis\n  ⚠ patch: 50.0% error rate\n"
        )

        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        plugin_mod._injected_sessions.clear()
        sid = "test-ctx-3"

        # First turn → context injected
        r1 = plugin_mod.on_pre_llm_call(session_id=sid, is_first_turn=True)
        assert r1 is not None

        # Second turn → no injection (already done)
        r2 = plugin_mod.on_pre_llm_call(session_id=sid, is_first_turn=False)
        assert r2 is None


# ── Findings file tests ───────────────────────────────────────────────

class TestWriteFindingsFile:
    """Test that analyze.py write_findings_file writes to findings.md."""

    def test_write_findings_creates_file(self, monkeypatch, tmp_path):
        """write_findings_file should create findings.md in the outcomes dir."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        report = {
            "generated_at": "2026-01-01 00:00:00 UTC",
            "period_days": 7,
            "high_signal_findings": [{
                "type": "high_error_tool",
                "tool": "patch",
                "error_rate": 45.0,
                "error_count": 9,
                "total_calls": 20,
            }],
            "findings": [],
            "turn_outcomes": {"total_turns": 0, "by_outcome": {}},
            "temporal_trend": {},
        }

        result = analyze_mod.write_findings_file(report)
        assert result == 1

        findings_path = tmp_path / "outcomes" / "findings.md"
        assert findings_path.exists()
        content = findings_path.read_text()
        assert "patch" in content
        assert "45.0%" in content

    def test_write_findings_no_findings(self, monkeypatch, tmp_path):
        """write_findings_file should return 0 when nothing to write."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        report = {
            "generated_at": "2026-01-01 00:00:00 UTC",
            "period_days": 7,
            "high_signal_findings": [],
            "findings": [],
            "turn_outcomes": {"total_turns": 0, "by_outcome": {}},
            "temporal_trend": {},
        }

        result = analyze_mod.write_findings_file(report)
        assert result == 0

    def test_write_findings_includes_turn_outcomes(self, monkeypatch, tmp_path):
        """findings.md should include Layer 1 turn outcome data."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )

        report = {
            "generated_at": "2026-01-01 00:00:00 UTC",
            "period_days": 7,
            "high_signal_findings": [],
            "findings": [],
            "turn_outcomes": {
                "total_turns": 10,
                "by_outcome": {"failure": 4, "success": 6},
                "by_failure_pattern": {"repeated_same_tool_error": 3, "high_error_density": 1},
                "turn_failure_rate": 40.0,
            },
            "temporal_trend": {},
        }

        analyze_mod.write_findings_file(report)
        content = (tmp_path / "outcomes" / "findings.md").read_text()
        assert "Task-Level Outcomes" in content
        assert "failure: 4" in content
        assert "repeated_same_tool_error: 3" in content


# ── Tier 2: Cross-turn pattern tests ──────────────────────────────────


class TestCrossTurnPatterns:
    """Tier 2: cross-turn retry and death loop detection."""

    def test_cross_turn_retry_detected(self, isolated_db, clean_plugin):
        """Same tool erroring in ≥2 distinct turns → cross_turn_retry finding."""
        plugin_mod._ensure_schema_once()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        # Session has 3 turns; patch errors in turn-1 and turn-3
        for turn_id, tool, status in [
            ("turn-1", "patch", "error"),
            ("turn-1", "read_file", "ok"),
            ("turn-2", "search_files", "ok"),
            ("turn-2", "read_file", "ok"),
            ("turn-3", "patch", "error"),
            ("turn-3", "write_file", "ok"),
        ]:
            conn.execute(
                """INSERT INTO tool_outcomes
                   (session_id, turn_id, tool_name, status, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                ("retry-session", turn_id, tool, status, now),
            )
        conn.commit()
        conn.close()

        report = run_analysis(isolated_db, days=7)
        cross_turn = [
            f for f in report["findings"] if f["type"] == "cross_turn_retry"
        ]
        assert len(cross_turn) >= 1
        assert cross_turn[0]["tool"] == "patch"
        assert cross_turn[0]["retry_turns"] == 2

    def test_death_loop_direct_repetition(self, isolated_db, clean_plugin):
        """Same fingerprint repeating ≥3 turns → death_loop finding."""
        plugin_mod._ensure_schema_once()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        # 4 turns, all with the same exact tool sequence: patch→read_file→patch
        for turn_id in ["t1", "t2", "t3", "t4"]:
            for tool, status in [("patch", "error"), ("read_file", "ok"), ("patch", "error")]:
                conn.execute(
                    """INSERT INTO tool_outcomes
                       (session_id, turn_id, tool_name, status, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("loop-session", turn_id, tool, status, now),
                )
        conn.commit()
        conn.close()

        report = run_analysis(isolated_db, days=7)
        loops = [f for f in report["findings"] if f["type"] == "death_loop"]
        assert len(loops) >= 1
        assert loops[0]["repetitions"] >= 3

    def test_cyclic_death_loop_detected(self, isolated_db, clean_plugin):
        """Period-2 cycle repeating ≥3× → cyclic death_loop."""
        plugin_mod._ensure_schema_once()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        # 6 turns alternating between two different fingerprints: A,B,A,B,A,B
        fp_a = [("patch", "error"), ("read_file", "ok")]
        fp_b = [("terminal", "error"), ("write_file", "ok")]
        for i in range(6):
            turn_id = f"t{i+1}"
            fp = fp_a if i % 2 == 0 else fp_b
            for tool, status in fp:
                conn.execute(
                    """INSERT INTO tool_outcomes
                       (session_id, turn_id, tool_name, status, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("cyclic-session", turn_id, tool, status, now),
                )
        conn.commit()
        conn.close()

        report = run_analysis(isolated_db, days=7)
        loops = [f for f in report["findings"] if f["type"] == "death_loop"]
        assert len(loops) >= 1

    def test_no_false_positive_cross_turn(self, isolated_db, clean_plugin):
        """Different tools erroring in different turns → no cross_turn_retry."""
        plugin_mod._ensure_schema_once()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        for turn_id, tool, status in [
            ("t1", "patch", "error"),
            ("t1", "read_file", "ok"),
            ("t2", "terminal", "error"),  # different tool, different turn
            ("t2", "read_file", "ok"),
        ]:
            conn.execute(
                """INSERT INTO tool_outcomes
                   (session_id, turn_id, tool_name, status, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                ("clean-session", turn_id, tool, status, now),
            )
        conn.commit()
        conn.close()

        report = run_analysis(isolated_db, days=7)
        cross_turn = [
            f for f in report["findings"] if f["type"] == "cross_turn_retry"
        ]
        assert len(cross_turn) == 0
        loops = [f for f in report["findings"] if f["type"] == "death_loop"]
        assert len(loops) == 0

    def test_findings_written_to_findings_file(self, isolated_db, clean_plugin, monkeypatch, tmp_path):
        """Cross-turn findings should appear in findings.md."""
        monkeypatch.setattr(
            "hermes_constants.get_hermes_home",
            lambda: tmp_path,
        )
        plugin_mod._ensure_schema_once()
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(isolated_db))
        for turn_id in ["t1", "t2", "t3"]:
            for tool, status in [("patch", "error"), ("patch", "error")]:
                conn.execute(
                    """INSERT INTO tool_outcomes
                       (session_id, turn_id, tool_name, status, timestamp)
                       VALUES (?, ?, ?, ?, ?)""",
                    ("loop-session", turn_id, tool, status, now),
                )
        conn.commit()
        conn.close()

        report = run_analysis(isolated_db, days=7)
        analyze_mod.write_findings_file(report)
        content = (tmp_path / "outcomes" / "findings.md").read_text()
        assert "Cross-Turn Patterns" in content
