"""Tests for db-safety plugin — SQL keyword false-positive fix.

Regression (Aug 2026): a Python verification script containing
``from agent.auxiliary_client import ...`` was blocked as SQL because
"FROM agent" matched the FROM-clause extractor and `.create(` matched the
CREATE keyword — the plugin only looked at keywords, never at whether the
command actually invokes a DB client shell. Additionally, schema
confirmation via sqlite3 (`.schema` / `.tables` / `PRAGMA table_info`) was
not recognized, so sessions that HAD confirmed the schema stayed blocked.
"""
import importlib.util as _ilu
import os

import pytest

_spec = _ilu.spec_from_file_location(
    "plugins.db_safety",
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "discipline", "db_safety.py"),
)
_ds = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ds)

on_pre_tool_call = _ds.on_pre_tool_call
on_post_tool_call = _ds.on_post_tool_call
_looks_like_db_shell = _ds._looks_like_db_shell

from plugins._shared_state import clear_session


@pytest.fixture(autouse=True)
def reset_state():
    clear_session("test-sid")
    yield
    clear_session("test-sid")


def _terminal(cmd: str):
    return on_pre_tool_call(session_id="test-sid", tool_name="terminal",
                            args={"command": cmd})


class TestNotSqlCommands:
    """Commands carrying SQL-shaped words but no DB client must pass."""

    def test_python_import_from_agent(self):
        cmd = ('cd /tmp && printf \'%s\\n\' \'import sys\' '
               '\'sys.path.insert(0, "/x")\' '
               '\'from agent.auxiliary_client import _get_task_extra_body\' '
               '| python3')
        assert _terminal(cmd) is None

    def test_python_openai_client_create(self):
        cmd = 'python3 -c "from openai import client; client.create()"'
        assert _terminal(cmd) is None

    def test_heredoc_python_with_select_word(self):
        cmd = 'python3 - <<\'EOF\'\nrows = [c for c in select_all()]\nEOF'
        assert _terminal(cmd) is None

    def test_grep_select_in_file(self):
        assert _terminal("grep 'SELECT' config.yaml") is None

    def test_echo_create_table(self):
        assert _terminal("echo 'CREATE TABLE foo'") is None


class TestRealSqlStillBlocked:
    """Genuine DB-shell SQL without schema confirmation must still block."""

    def test_psql_select_blocked(self):
        decision = _terminal('psql -c "SELECT * FROM users"')
        assert decision is not None and decision["action"] == "block"

    def test_mysql_select_blocked(self):
        decision = _terminal('mysql -e "SELECT * FROM orders"')
        assert decision is not None and decision["action"] == "block"

    def test_sqlite3_select_blocked(self):
        decision = _terminal('sqlite3 /tmp/x.db "SELECT * FROM tool_outcomes"')
        assert decision is not None and decision["action"] == "block"

    def test_env_prefixed_psql_blocked(self):
        decision = _terminal('PGPASSWORD=x psql -c "SELECT * FROM users"')
        assert decision is not None and decision["action"] == "block"

    def test_sqlite3_update_blocked(self):
        decision = _terminal('sqlite3 ~/.hermes/outcomes.db "UPDATE t SET a=1"')
        assert decision is not None and decision["action"] == "block"


class TestSchemaConfirmationRecognized:
    """sqlite3 .schema/.tables/PRAGMA must mark the session confirmed."""

    def test_sqlite_schema_then_query_passes(self):
        on_post_tool_call(session_id="test-sid", tool_name="terminal",
                          args={"command": 'sqlite3 /tmp/x.db ".schema"'})
        assert _terminal('sqlite3 /tmp/x.db "SELECT * FROM t"') is None

    def test_pragma_table_info_confirms(self):
        on_post_tool_call(session_id="test-sid", tool_name="terminal",
                          args={"command": 'sqlite3 x.db "PRAGMA table_info(messages)"'})
        assert _terminal('sqlite3 x.db "SELECT * FROM messages"') is None

    def test_dot_tables_confirms(self):
        on_post_tool_call(session_id="test-sid", tool_name="terminal",
                          args={"command": 'sqlite3 x.db ".tables"'})
        assert _terminal('sqlite3 x.db "SELECT 1 FROM messages"') is None


class TestDbShellDetector:
    def test_plain_psql(self):
        assert _looks_like_db_shell("psql -c 'select 1'") is True

    def test_env_prefixed(self):
        assert _looks_like_db_shell("PGPASSWORD=a psql -c 'select 1'") is True

    def test_sudo_mysql(self):
        assert _looks_like_db_shell("sudo mysql -e 'select 1'") is True

    def test_python_not_db_shell(self):
        assert _looks_like_db_shell("python3 script.py") is False

    def test_git_not_db_shell(self):
        assert _looks_like_db_shell("git log --oneline") is False

    def test_sqlite3(self):
        assert _looks_like_db_shell("sqlite3 db.sqlite '.tables'") is True
