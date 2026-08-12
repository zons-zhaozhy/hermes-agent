"""db-safety plugin — block SQL queries that guess table/column names.

Enforces the rule: before any SQL query (SELECT/INSERT/UPDATE/DELETE/ALTER),
the agent must have confirmed the schema with \dt (list tables), \d table_name
(describe columns), or information_schema queries.

ACTIVATION: ON by default. Set DB_SAFETY_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re  # noqa: R1 — SQL pattern matching requires regex (security scanner)
from typing import Any, Dict, Optional, Set

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

# ── Session-keyed state helpers ────────────────────────────────────────

_NAMESPACE = "db_safety"


def _get_schema_confirmed(sid: str) -> bool:
    """Whether schema has been confirmed in this session."""
    return bool(get_session_state(sid, _NAMESPACE).get("schema_confirmed", False))


def _get_confirmed_tables(sid: str) -> Set[str]:
    """Set of table names whose columns have been confirmed."""
    return get_session_state(sid, _NAMESPACE).setdefault("confirmed_tables", set())


def _get_recent_sql(sid: str) -> list:
    """Recent SQL commands in this session (for error pattern detection)."""
    return get_session_state(sid, _NAMESPACE).setdefault("recent_sql", [])


# ── Configuration ────────────────────────────────────────────────────

_SQL_KEYWORDS = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|ALTER|CREATE|DROP|TRUNCATE)\b",
    re.IGNORECASE,
)

_SCHEMA_CONFIRM_COMMANDS = re.compile(
    r"(\\dt|\\d\s|information_schema|SHOW\s+TABLES|DESCRIBE\s|SHOW\s+COLUMNS|"
    r"pg_catalog|\\d\+)",
    re.IGNORECASE,
)


def _plugin_disabled() -> bool:
    return os.environ.get("DB_SAFETY_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL (FROM/INTO/UPDATE/JOIN clauses)."""
    # FROM/JOIN table_name
    from_pattern = re.findall(
        r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE
    )
    # UPDATE table_name SET
    update_pattern = re.findall(
        r"\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+SET", sql, re.IGNORECASE
    )
    # INSERT INTO table_name
    insert_pattern = re.findall(
        r"\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, re.IGNORECASE
    )
    return set(from_pattern + update_pattern + insert_pattern)


def _extract_db_client(cmd: str) -> str:
    """Detect which DB client is being used (psql, mysql, psql -c, etc.)."""
    lower = cmd.strip().lower()
    if lower.startswith("psql") or lower.startswith("pg_"):
        return "postgresql"
    if lower.startswith("mysql") or lower.startswith("mariadb"):
        return "mysql"
    if lower.startswith("sqlite"):
        return "sqlite"
    if lower.startswith("python") and ("cursor" in lower or "execute" in lower):
        return "generic"
    return "unknown"


# ── post_tool_call hook ──────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Track schema confirmations and recent SQL commands."""
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")

    if tool_name == "terminal":
        args = kwargs.get("args") or {}
        cmd = str(args.get("command", ""))

        # Detect schema confirmation commands
        if _SCHEMA_CONFIRM_COMMANDS.search(cmd):
            get_session_state(sid, _NAMESPACE)["schema_confirmed"] = True

            # Extract table names from \d commands
            d_matches = re.findall(r"\\d\s+(\w+)", cmd)
            for table in d_matches:
                _get_confirmed_tables(sid).add(table.lower())

        # Track SQL queries
        if _SQL_KEYWORDS.search(cmd):
            tables = _extract_tables_from_sql(cmd)
            recent = _get_recent_sql(sid)
            recent.append({"cmd": cmd[:200], "tables": tables})
            # Keep only last 10
            if len(recent) > 10:
                recent[:] = recent[-10:]


# ── pre_tool_call hook ────────────────────────────────────────────────

def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Block SQL queries when schema hasn't been confirmed."""
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "terminal":
        return None

    args = kwargs.get("args") or {}
    cmd = str(args.get("command", "")).strip()

    # Not a SQL command
    if not _SQL_KEYWORDS.search(cmd):
        return None

    # Schema already confirmed in this session
    if _get_schema_confirmed(sid):
        return None

    # Check if it's a schema confirmation command itself (don't block it)
    if _SCHEMA_CONFIRM_COMMANDS.search(cmd):
        return None

    # Extract table names — queries without a concrete table target
    # (e.g. SELECT 1, SELECT version(), health checks) don't need schema
    tables = _extract_tables_from_sql(cmd)
    if not tables:
        return None

    # This is a SQL query against concrete tables without prior schema confirmation

    return {
        "action": "block",
        "message": (
            "[DBSafety] DB 操作铁律拦截：SQL 查询前必须先确认 schema。\n"
            f"  命令: {cmd[:100]}...\n"
            f"  涉及表: {', '.join(sorted(tables))}\\n"
            f"  schema 状态: 未确认\n"
            f"  修复: 先执行 schema 确认命令：\n"
            f"    PostgreSQL: \\dt (查表名) → \\d table_name (查列名)\n"
            f"    MySQL: SHOW TABLES → DESCRIBE table_name\n"
            f"    通用: SELECT * FROM information_schema.columns WHERE table_name='...'"
        ),
    }


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("db-safety registered")
