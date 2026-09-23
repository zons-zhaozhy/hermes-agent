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

# SQL runs through a DB client shell. Only commands that START with a DB
# client are candidates for SQL-keyword inspection; anything else carrying
# SELECT/CREATE/... words is ordinary code or prose (e.g. Python
# `from agent.x import ...`, `.create(...)`, `new Set(...)`) and must NOT
# be treated as SQL. This kills the dominant false-positive class where
# verification scripts got blocked on words like "agent"/"openai".
_DB_CLIENT_PREFIXES = (
    "psql", "pg_", "mysql", "mariadb", "sqlite3", "sqlplus",
    "disql", "duckdb",
)

# sqlite3 dot-commands (.schema/.tables) and PRAGMA table_info are schema
# confirmation commands too — previously only \dt / information_schema were
# recognized, so a session that HAD confirmed the schema via sqlite3 was
# still blocked on every subsequent query.
_SCHEMA_CONFIRM_COMMANDS = re.compile(
    r"(\.schema|\.tables|\\dt|\\d\s|information_schema|SHOW\s+TABLES|DESCRIBE\s|"
    r"SHOW\s+COLUMNS|pg_catalog|\\d\+|PRAGMA\s+table_info)",
    re.IGNORECASE,
)


def _looks_like_db_shell(cmd: str) -> bool:
    """True if *cmd* starts with a DB client (possibly after env/PATH noise).

    Handles leading ``ENV=...`` assignments and ``sudo``/``command``/``env``
    wrappers so ``PGPASSWORD=x psql -c ...`` still counts.
    """
    tokens = cmd.strip().split(None, 8)
    i = 0
    # skip VAR=value assignments
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-") \
            and tokens[i].split("=", 1)[0].replace("_", "").isalnum():
        i += 1
    # skip sudo / command / env wrappers
    while i < len(tokens) and tokens[i] in ("sudo", "command", "env"):
        i += 1
        # env may carry more VAR= assignments
        while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith("-"):
            i += 1
    if i >= len(tokens):
        return False
    first = tokens[i].rstrip(";|&")
    return any(first == p or first.startswith(p + " ") or first == p.strip() or first.startswith(p)
               for p in _DB_CLIENT_PREFIXES if p.strip())


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
    # DROP TABLE table_name / TRUNCATE [TABLE] table_name / ALTER TABLE table_name
    # 2026-08-28 审计修复：DDL 同样针对具体表，漏判使 DROP 绕过 schema 铁律
    ddl_pattern = re.findall(
        r"\b(?:DROP\s+TABLE|TRUNCATE(?:\s+TABLE)?|ALTER\s+TABLE)\s+"
        r"(?:IF\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_.]*)",
        sql,
        re.IGNORECASE,
    )
    return set(from_pattern + update_pattern + insert_pattern + ddl_pattern)


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

        # Detect schema confirmation commands (any command may embed one,
        # e.g. `sqlite3 db ".schema"` or `psql -c '\dt'`)
        if _SCHEMA_CONFIRM_COMMANDS.search(cmd):
            get_session_state(sid, _NAMESPACE)["schema_confirmed"] = True

            # Extract table names from \d commands
            d_matches = re.findall(r"\\d\s+(\w+)", cmd)
            for table in d_matches:
                _get_confirmed_tables(sid).add(table.lower())

        # Track SQL queries (DB client shells only — avoids logging Python
        # code that merely contains SQL-shaped words)
        if _looks_like_db_shell(cmd) and _SQL_KEYWORDS.search(cmd):
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

    # Not a DB client command — SQL keywords in code/prose are not SQL.
    if not _looks_like_db_shell(cmd):
        return None

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
