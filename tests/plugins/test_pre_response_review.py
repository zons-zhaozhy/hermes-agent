"""pre-response-review profile 隔离与可见降级契约。"""

from __future__ import annotations

import importlib
import logging
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

review = importlib.import_module("plugins.pre-response-review")


def _seed_message(db_path: Path, session_id: str, tool_calls: str | None) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                tool_calls TEXT,
                tool_name TEXT
            )
            """
        )
        conn.execute("INSERT INTO sessions VALUES (?)", (session_id,))
        conn.execute(
            "INSERT INTO messages(session_id, tool_calls, tool_name) VALUES (?, ?, ?)",
            (session_id, tool_calls, None),
        )


def test_profile_scoped_read_evidence_suppresses_false_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_id = f"pre-response-{uuid.uuid4()}"
    _seed_message(
        home / "state.db",
        session_id,
        '[{"function": {"name": "read_file", "arguments": "{}"}}]',
    )

    assert review._check_false_claims("改前验证✓", session_id) == []


def test_absent_read_evidence_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_id = f"pre-response-absent-{uuid.uuid4()}"
    (home / "state.db").touch()
    _seed_message(home / "state.db", session_id, None)

    warnings = review._check_false_claims("改前验证✓", session_id)
    assert len(warnings) == 1
    assert "无 read_file/search_files/terminal 调用记录" in warnings[0]


def test_regex_claim_with_re_usage_is_flagged_without_db() -> None:
    warnings = review._check_false_claims(
        "正则✓\n```python\nimport re\nre.match(r'a', value)\n```",
        None,
    )
    assert len(warnings) == 1
    assert "标记'正则✓'" in warnings[0]


def test_malformed_tool_calls_json_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    session_id = f"pre-response-malformed-{uuid.uuid4()}"
    _seed_message(home / "state.db", session_id, "{not-json")

    with caplog.at_level(logging.WARNING, logger="plugins.pre-response-review"):
        warnings = review._check_false_claims("改前验证✓", session_id)

    assert len(warnings) == 1
    assert "malformed tool_calls JSON" in caplog.text


def test_database_query_failure_logs_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db_path = home / "state.db"
    db_path.write_bytes(b"this is not sqlite3")
    session_id = f"pre-response-query-{uuid.uuid4()}"

    with caplog.at_level(logging.WARNING, logger="plugins.pre-response-review"):
        assert review._check_false_claims("改前验证✓", session_id)

    assert "failed to read session tool calls" in caplog.text
