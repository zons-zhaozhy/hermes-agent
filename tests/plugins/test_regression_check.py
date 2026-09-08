"""Tests for discipline regression check (outcome feedback flywheel Layer 2.5).

Contract under test: a discipline whose violation density drops after its
effective date is judged `effective`; one that rises is `regressed`; windows
not yet fully past are `pending`. Expectations are derived independently
from the documented thresholds (70% / 100%), NOT from implementation output.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "outcome-collector"

_spec = importlib.util.spec_from_file_location(
    "regression_check", PLUGIN_DIR / "regression_check.py"
)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


@pytest.fixture
def db(tmp_path):
    """Build a minimal outcomes.db with violations + tool_outcomes tables."""
    db_path = tmp_path / "outcomes.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT NOT NULL, command TEXT NOT NULL, level TEXT,
            session_id TEXT, timestamp TEXT NOT NULL)"""
    )
    conn.execute(
        """CREATE TABLE tool_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL, turn_id TEXT, tool_call_id TEXT,
            tool_name TEXT NOT NULL, status TEXT NOT NULL,
            error_type TEXT, error_message TEXT, duration_ms INTEGER DEFAULT 0,
            args_summary TEXT, timestamp TEXT NOT NULL)"""
    )
    yield conn, db_path
    conn.close()


def _seed(conn, rule, day, n_viol, n_calls):
    for _ in range(n_viol):
        conn.execute(
            "INSERT INTO violations (rule, command, timestamp) VALUES (?, ?, ?)",
            (rule, "cmd", f"{day}T12:00:00Z"),
        )
    for _ in range(n_calls):
        conn.execute(
            "INSERT INTO tool_outcomes (session_id, tool_name, status, timestamp) "
            "VALUES ('s', 'terminal', 'ok', ?)",
            (f"{day}T12:00:00Z",),
        )


TODAY = "2026-09-20"


class TestVerdicts:
    def test_effective_when_density_drops(self, db):
        """Drops to 0 → effective. Baseline 10 viol / 1000 calls per day."""
        conn, _ = db
        # baseline 2026-09-01~07 (eff 09-08), after 09-09~15, today 09-20
        for d in range(1, 8):
            _seed(conn, "R6", f"2026-09-0{d}", 10, 1000)
        for d in range(9, 16):
            _seed(conn, "R6", f"2026-09-{d}", 0, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R6", "2026-09-08", TODAY)
        assert out["verdict"] == "effective", out
        assert out["ratio"] == 0.0

    def test_regressed_when_density_rises(self, db):
        conn, _ = db
        for d in range(1, 8):
            _seed(conn, "R5", f"2026-09-0{d}", 2, 1000)
        for d in range(9, 16):
            _seed(conn, "R5", f"2026-09-{d}", 5, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R5", "2026-09-08", TODAY)
        assert out["verdict"] == "regressed", out

    def test_zero_baseline_positive_after_is_inconclusive(self, db):
        """基线窗口零违规、验收窗口有违规 → 无可比基线,判 inconclusive 而非 inf 回退."""
        conn, _ = db
        for d in range(1, 8):
            _seed(conn, "R1", f"2026-09-0{d}", 0, 1000)
        for d in range(9, 16):
            _seed(conn, "R1", f"2026-09-{d}", 3, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R1", "2026-09-08", TODAY)
        assert out["verdict"] == "inconclusive", out
        assert out["ratio"] is None

    def test_zero_baseline_zero_after_is_effective(self, db):
        """基线与验收窗口都零违规 → 保持,判 effective."""
        conn, _ = db
        for d in range(1, 16):
            _seed(conn, "R2", f"2026-09-{d:02d}", 0, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R2", "2026-09-08", TODAY)
        assert out["verdict"] == "effective", out

    def test_regressed_ratio_value(self, db):
        conn, _ = db
        for d in range(1, 8):
            _seed(conn, "R5", f"2026-09-0{d}", 2, 1000)
        for d in range(9, 16):
            _seed(conn, "R5", f"2026-09-{d}", 5, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R5", "2026-09-08", TODAY)
        assert out["ratio"] == pytest.approx(2.5)

    def test_pending_when_window_incomplete(self, db):
        conn, _ = db
        _seed(conn, "R3", "2026-09-01", 5, 100)
        conn.commit()
        out = rc.check_rule(conn, "R3", "2026-09-18", TODAY)
        assert out["verdict"] == "pending", out

    def test_inconclusive_when_no_data(self, db):
        conn, _ = db
        out = rc.check_rule(conn, "R1", "2026-09-08", TODAY)
        assert out is None  # rule never seen → skipped

    def test_normalized_by_call_volume(self, db):
        """Same absolute violations but half the calls after → density rises."""
        conn, _ = db
        for d in range(1, 8):
            _seed(conn, "R4", f"2026-09-0{d}", 10, 2000)
        for d in range(9, 16):
            _seed(conn, "R4", f"2026-09-{d}", 10, 1000)
        conn.commit()
        out = rc.check_rule(conn, "R4", "2026-09-08", TODAY)
        assert out["verdict"] == "regressed", out
        assert out["ratio"] == pytest.approx(2.0)


class TestDatesFile:
    def test_load_valid(self, tmp_path):
        p = tmp_path / "d.json"
        p.write_text('{"R6": "2026-09-07"}')
        assert rc.load_discipline_dates(p) == {"R6": "2026-09-07"}

    def test_load_missing_or_corrupt(self, tmp_path):
        assert rc.load_discipline_dates(tmp_path / "nope.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert rc.load_discipline_dates(bad) == {}


class TestTextFormat:
    def test_no_crash_on_reason_only_rows(self):
        """inconclusive rows lack window keys — format must not KeyError."""
        report = {
            "generated_at": TODAY,
            "results": [{"rule": "R1", "effective": "2026-09-08",
                         "verdict": "inconclusive", "reason": "窗口内无数据"}],
            "untracked_rules": [],
            "dates_file": "x",
        }
        text = rc.format_as_text(report)
        assert "窗口内无数据" in text
