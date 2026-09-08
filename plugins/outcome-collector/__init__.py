"""outcome-collector plugin — structured tool-call outcome capture (Layer 0).

Records every tool_call's structured outcome (ok/error, error_type, duration,
diagnostic args) into a dedicated SQLite database (~/.hermes/outcomes.db).
This is the signal-capture layer of the self-evolving feedback flywheel:

    tool_call → post_tool_call hook → outcomes.db → analysis → memory → behavior change

DESIGN DECISIONS:
- Separate DB (outcomes.db), NOT the main state.db:
  Avoids schema migration complexity on the critical session store.
  Analysis queries don't contend with session reads/writes.
- Diagnostic args only (path, command prefix, function name):
  No full argument capture — prevents blob bloat and credential leakage.
- Session-scoped with on_session_end flush:
  In-memory buffer per session, bulk-write on session end for efficiency.
  Falls back to per-call write if buffer exceeds threshold.
- DISABLE: Set OUTCOME_COLLECTOR_DISABLE=1 to turn off.

See: skill "outcome-feedback-methodology" for the full architecture.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

_NAMESPACE = "outcome_collector"

# Flush when in-memory buffer reaches this many entries
_FLUSH_THRESHOLD = 50

# Tools whose args we never capture (privacy / noise)
_NO_ARG_TOOLS = frozenset({
    "browser_snapshot", "browser_console", "browser_scroll",
    "memory", "todo", "process",
})


def _plugin_disabled() -> bool:
    return os.environ.get("OUTCOME_COLLECTOR_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


# ── DB path resolution ────────────────────────────────────────────────

_db_lock = threading.Lock()
_db_path: Optional[Path] = None


def _get_db_path() -> Path:
    """Resolve outcomes.db path under the active HERMES_HOME."""
    global _db_path
    if _db_path is not None:
        return _db_path
    try:
        from hermes_constants import get_hermes_home
        _db_path = get_hermes_home() / "outcomes.db"
    except Exception:  # noqa: D5 — get_hermes_home fallback, non-critical
        # Fallback: construct from HERMES_HOME env directly
        home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        _db_path = Path(home) / "outcomes.db"
    return _db_path


def _get_conn() -> sqlite3.Connection:
    """Get a connection to outcomes.db (creates if not exists)."""
    db = _get_db_path()
    conn = sqlite3.connect(str(db), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _ensure_schema() -> None:
    """Create tables if they don't exist. Called once on first use."""
    with _db_lock:
        conn = _get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tool_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id TEXT,
                    tool_call_id TEXT,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_type TEXT,
                    error_message TEXT,
                    duration_ms INTEGER DEFAULT 0,
                    args_summary TEXT,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_to_session ON tool_outcomes(session_id);
                CREATE INDEX IF NOT EXISTS idx_to_tool ON tool_outcomes(tool_name);
                CREATE INDEX IF NOT EXISTS idx_to_status ON tool_outcomes(status);
                CREATE INDEX IF NOT EXISTS idx_to_timestamp ON tool_outcomes(timestamp);

                CREATE TABLE IF NOT EXISTS session_summaries (
                    session_id TEXT PRIMARY KEY,
                    total_calls INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    tool_breakdown TEXT,
                    first_call_at TEXT,
                    last_call_at TEXT,
                    updated_at TEXT
                );
            """)
            conn.commit()
        finally:
            conn.close()


_schema_ready = False


def _ensure_schema_once() -> None:
    global _schema_ready
    if _schema_ready:
        return
    try:
        _ensure_schema()
        _schema_ready = True
    except Exception as exc:
        logger.warning("outcome-collector: schema init failed: %s", exc)


# ── In-memory buffer ──────────────────────────────────────────────────

# session_id → list of outcome dicts
_buffers: Dict[str, List[Dict[str, Any]]] = {}
_buffer_lock = threading.Lock()


def _get_buffer(sid: str) -> List[Dict[str, Any]]:
    with _buffer_lock:
        return _buffers.setdefault(sid, [])


# ── Diagnostic arg extraction ─────────────────────────────────────────

def _extract_diagnostic_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the diagnostic-relevant args for each tool type.

    Captures enough to answer 'what was attempted?' without storing
    full arguments (which may contain credentials, large file contents, etc.)
    """
    if tool_name in _NO_ARG_TOOLS:
        return {}

    summary: Dict[str, Any] = {}

    if tool_name in ("patch", "write_file", "read_file"):
        path = args.get("path", "")
        if path:
            summary["path"] = str(path)[-200:]  # cap length
        if tool_name == "patch":
            old = str(args.get("old_string", ""))[:60]
            if old:
                summary["old_prefix"] = old

    elif tool_name == "terminal":
        cmd = str(args.get("command", ""))
        summary["cmd_prefix"] = cmd[:120]

    elif tool_name in ("browser_click", "browser_type"):
        ref = args.get("ref", "")
        if ref:
            summary["ref"] = str(ref)[:50]

    elif tool_name == "skill_view":
        summary["skill"] = str(args.get("name", ""))[:64]

    elif tool_name == "delegate_task":
        goal = str(args.get("goal", ""))
        summary["goal_prefix"] = goal[:80]

    elif tool_name == "web_search":
        summary["query_prefix"] = str(args.get("query", ""))[:80]

    elif tool_name == "search_files":
        summary["pattern_prefix"] = str(args.get("pattern", ""))[:60]
        tgt = args.get("target", "")
        if tgt:
            summary["target"] = str(tgt)[:60]

    elif tool_name == "execute_code":
        code = str(args.get("code", ""))
        # Just the first meaningful line (import or function call)
        for line in code.strip().split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                summary["code_prefix"] = stripped[:80]
                break

    elif tool_name == "clarify":
        q = str(args.get("question", ""))
        summary["q_prefix"] = q[:80]

    return summary


# ── Truncate error message ────────────────────────────────────────────

def _truncate(msg: Any, limit: int = 300) -> Optional[str]:
    if not msg:
        return None
    s = str(msg)
    return s[:limit] if len(s) > limit else s


# ── Write outcomes ────────────────────────────────────────────────────

def _flush_buffer(sid: str) -> None:
    """Bulk-write buffered outcomes for a session to the DB."""
    with _buffer_lock:
        buf = _buffers.get(sid, [])
        if not buf:
            return
        # Move out of the buffer under lock
        to_write = buf[:]
        _buffers[sid] = []

    if not to_write:
        return

    _ensure_schema_once()
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                conn.executemany(
                    """INSERT INTO tool_outcomes
                       (session_id, turn_id, tool_call_id, tool_name, status,
                        error_type, error_message, duration_ms, args_summary, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            r["session_id"],
                            r.get("turn_id"),
                            r.get("tool_call_id"),
                            r["tool_name"],
                            r["status"],
                            r.get("error_type"),
                            r.get("error_message"),
                            r.get("duration_ms", 0),
                            json.dumps(r.get("args_summary"), ensure_ascii=False)
                            if r.get("args_summary") else None,
                            r["timestamp"],
                        )
                        for r in to_write
                    ],
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("outcome-collector: flush failed: %s", exc)


def _write_outcome(record: Dict[str, Any]) -> None:
    """Buffer a single outcome, flushing if threshold reached."""
    sid = record.get("session_id", "")
    if not sid:
        return

    buf = _get_buffer(sid)
    buf.append(record)

    if len(buf) >= _FLUSH_THRESHOLD:
        _flush_buffer(sid)


# ── Session summary computation ───────────────────────────────────────

def _update_session_summary(sid: str) -> None:
    """Compute/update the aggregate summary for a session."""
    _ensure_schema_once()
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                rows = conn.execute(
                    """SELECT tool_name, status, timestamp
                       FROM tool_outcomes WHERE session_id = ?""",
                    (sid,),
                ).fetchall()

                if not rows:
                    return

                total = len(rows)
                errors = sum(1 for r in rows if r["status"] == "error")
                breakdown: Dict[str, int] = {}
                for r in rows:
                    name = r["tool_name"]
                    breakdown[name] = breakdown.get(name, 0) + 1

                timestamps = [r["timestamp"] for r in rows]
                first_at = min(timestamps)
                last_at = max(timestamps)

                conn.execute(
                    """INSERT INTO session_summaries
                       (session_id, total_calls, error_count, tool_breakdown,
                        first_call_at, last_call_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         total_calls=excluded.total_calls,
                         error_count=excluded.error_count,
                         tool_breakdown=excluded.tool_breakdown,
                         last_call_at=excluded.last_call_at,
                         updated_at=excluded.updated_at""",
                    (
                        sid,
                        total,
                        errors,
                        json.dumps(breakdown, ensure_ascii=False),
                        first_at,
                        last_at,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("outcome-collector: summary update failed: %s", exc)


# ── Layer 1: Task-level outcome inference ──────────────────────────────
#
# Pure objective signal — NO keyword detection. Infer task outcome from
# the tool-call sequence pattern within each turn.
#
# Patterns (all derivable from Layer 0 tool_outcomes data):
#   repeated_same_tool_error: same tool errors ≥2× → stuck loop (failure)
#   retry_then_success:       tool error → retry → ok → obstacle overcome (success)
#   high_error_density:       ≥50% error rate with ≥5 calls → debugging (partial)
#   clean_completion:         all calls ok → likely success
#   escalation:               user repeats similar request → prior incomplete (failure)

# Turn-level outcome tracking
# session_id → list of {turn_id, tool_calls: [...], outcome: str|None}
_turn_outcomes: Dict[str, List[Dict[str, Any]]] = {}


def _classify_tool_sequence(
    tool_calls: List[Dict[str, Any]],
) -> tuple[str, Optional[str]]:
    """Classify a turn's tool-call sequence into an outcome.

    Pure function — no NLP, no keywords, no external signals.
    Only uses the tool call status sequence (ok/error/blocked).

    Returns (outcome, pattern) where:
      outcome: 'success' | 'failure' | 'partial' | 'unknown'
      pattern: specific failure/success pattern or None
    """
    if not tool_calls:
        return "unknown", None

    total = len(tool_calls)
    # Guard blocks are by-design gating (agent must gather evidence and
    # retry), not tool failures. Count only genuine errors here; blocked
    # calls still feed the retry-then-success pattern below so that
    # [blocked, blocked, ok] classifies as success/retry_then_success.
    error_calls = [tc for tc in tool_calls if tc["status"] == "error"]
    error_count = len(error_calls)

    # Pattern: retry_then_success — at least one non-ok call followed by a
    # later ok on the same tool (agent overcame the obstacle, possibly via
    # a guard block that demanded more evidence first)
    tool_outcomes: dict[str, list[str]] = {}
    for tc in tool_calls:
        tool_outcomes.setdefault(tc["tool_name"], []).append(tc["status"])
    for tool_name, statuses in tool_outcomes.items():
        has_error = "error" in statuses or "blocked" in statuses
        has_later_ok = False
        seen_error = False
        for s in statuses:
            if s in ("error", "blocked"):
                seen_error = True
            elif s == "ok" and seen_error:
                has_later_ok = True
                break
        if has_error and has_later_ok:
            return "success", "retry_then_success"

    # Pattern: repeated_same_tool_error — same tool errored ≥2 times with no
    # later success on that tool (checked after retry_then_success so that
    # [blocked, blocked, ok] is not misread as a failure loop)
    if error_calls:
        from collections import Counter
        error_tool_counts = Counter(tc["tool_name"] for tc in error_calls)
        for tool, cnt in error_tool_counts.items():
            if cnt >= 2:
                return "failure", "repeated_same_tool_error"

    # Pattern: high_error_density — ≥50% error rate with enough calls to be meaningful
    if total >= 5 and error_count / total >= 0.5:
        return "partial", "high_error_density"

    # Pattern: clean_completion — no errors at all
    if error_count == 0:
        return "success", "clean_completion"

    # Some errors but low rate and no retry-success pattern
    return "partial", "low_error_minor"


def _persist_turn_outcome(sid: str, turn_data: Dict[str, Any]) -> None:
    """Write a turn-level outcome as a special marker row in tool_outcomes."""
    _ensure_schema_once()
    try:
        with _db_lock:
            conn = _get_conn()
            try:
                conn.execute(
                    """INSERT INTO tool_outcomes
                       (session_id, turn_id, tool_name, status, error_type,
                        error_message, duration_ms, args_summary, timestamp)
                       VALUES (?, ?, '_turn_outcome', ?, ?, ?, ?, ?, ?)""",
                    (
                        sid,
                        turn_data["turn_id"],
                        turn_data["outcome"],
                        turn_data.get("failure_pattern"),
                        turn_data.get("summary"),
                        turn_data.get("duration_ms", 0),
                        json.dumps({
                            "tool_count": turn_data.get("tool_count", 0),
                            "error_count": turn_data.get("error_count", 0),
                            "tools": turn_data.get("tools", []),
                        }, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception as exc:
        logger.warning("outcome-collector: turn outcome persist failed: %s", exc)


def _check_findings_for_injection() -> Optional[str]:
    """Layer 3: read findings.md, return context to inject if high-severity exists.

    This is the flywheel's closing loop — the agent SEES the findings as
    ephemeral context appended to the user message (cache-safe, not in
    system prompt). Returns None when there's nothing to inject.

    Injection is rate-limited to once per session to avoid nagging.
    """
    try:
        import importlib.util as _ilu
        _p = Path(__file__).parent / "analyze.py"
        _s = _ilu.spec_from_file_location("_oc_analyzer", _p)
        if _s is None or _s.loader is None:
            return None
        _m = _ilu.module_from_spec(_s)
        _s.loader.exec_module(_m)

        findings_path = _m.get_findings_path()
        if not findings_path.exists():
            return None

        content = findings_path.read_text(encoding="utf-8").strip()
        if not content or "⚠" not in content:
            return None  # No high-severity findings

        # Extract high-severity lines for a compact injection
        high_lines = [
            line.strip() for line in content.splitlines()
            if "⚠" in line and not line.startswith("#") and not line.startswith("<!--")
        ]
        if not high_lines:
            return None

        summary = "\n".join(f"  {line}" for line in high_lines[:5])

        return (
            "[Outcome Analysis] Recent tool-call patterns show high error rates:\n"
            f"{summary}\n"
            "Review these patterns and adjust your approach if you're about to "
            "use the same tools in similar ways."
        )
    except Exception as exc:
        logger.warning("outcome-collector: findings injection check failed: %s", exc)
        return None


# Track which sessions have already received the findings injection
_injected_sessions: set[str] = set()


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, str]]:
    """Layer 1: classify previous turn's tool sequence.
    Layer 3: inject findings context on first turn of each session.

    Returns {"context": "..."} when there's findings to inject — this gets
    appended to the user message (cache-safe). Returns None otherwise.
    """
    if _plugin_disabled():
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "") or ""

    # ── Layer 1: classify previous turn from tool sequence ──
    if sid:
        with _buffer_lock:
            turns = _turn_outcomes.get(sid, [])
            if turns:
                prev_turn = turns[-1]
                if not prev_turn.get("outcome_evaluated"):
                    prev_turn["outcome_evaluated"] = True
                    tool_calls = prev_turn.get("tool_calls", [])
                    outcome, pattern = _classify_tool_sequence(tool_calls)

                    if outcome != "unknown":
                        prev_turn["outcome"] = outcome
                        prev_turn["failure_pattern"] = pattern
                        prev_turn["tool_count"] = len(tool_calls)
                        prev_turn["error_count"] = sum(
                            1 for tc in tool_calls if tc["status"] == "error"
                        )
                        prev_turn["tools"] = list(set(tc["tool_name"] for tc in tool_calls))
                        prev_turn["summary"] = f"outcome={outcome}" + (
                            f", pattern={pattern}" if pattern else ""
                        )
                        _persist_turn_outcome(sid, prev_turn)

    # ── Layer 3: inject findings once per session (first turn only) ──
    is_first_turn = kwargs.get("is_first_turn", False)
    if is_first_turn and sid and sid not in _injected_sessions:
        _injected_sessions.add(sid)
        ctx = _check_findings_for_injection()
        if ctx:
            return {"context": ctx}

    return None


# ── Hooks ─────────────────────────────────────────────────────────────

def on_post_tool_call(**kwargs) -> None:
    """Capture every tool call outcome into the buffer."""
    if _plugin_disabled():
        return

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if not sid:
        return

    tool_name = kwargs.get("tool_name", "")
    status = kwargs.get("status", "ok") or "ok"
    args = kwargs.get("args") or {}

    # Truncate error message
    err_msg = _truncate(kwargs.get("error_message"))

    record = {
        "session_id": sid,
        "turn_id": kwargs.get("turn_id") or "",
        "tool_call_id": kwargs.get("tool_call_id") or "",
        "tool_name": tool_name,
        "status": status if status in ("ok", "error", "blocked") else "ok",
        "error_type": kwargs.get("error_type") or None,
        "error_message": err_msg,
        "duration_ms": kwargs.get("duration_ms", 0) or 0,
        "args_summary": _extract_diagnostic_args(tool_name, args),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _write_outcome(record)

    # Also track for Layer 1 turn-level analysis
    with _buffer_lock:
        turns = _turn_outcomes.setdefault(sid, [])
        if not turns or turns[-1].get("turn_id") != record["turn_id"]:
            turns.append({"turn_id": record["turn_id"], "tool_calls": []})
        turns[-1]["tool_calls"].append({
            "tool_name": tool_name,
            "status": record["status"],
        })

    # Layer 2.6: memory 纪律条目写入成功后自动登记生效日期（供 regression_check 验收）
    if tool_name == "memory" and status == "ok":
        _register_discipline_effective_date(args)


def _parse_discipline_tags(text: str) -> Dict[str, str]:
    """从纪律条目文本解析 {规则号: 生效日}。纯 str 方法,无正则。

    约定格式: "§R6 v3(0907):..." —— § 分段,段内 R<数字> 开头,首个括号内 MMDD。

    Contract:
      Preconditions: text 为单条 memory 条目字符串
      Postconditions: 命中约定格式返回非空 dict;日期非法(如0931)该段跳过;
                      MMDD 晚于今天属上一年
    """
    found: Dict[str, str] = {}
    now = datetime.now(timezone.utc)
    for part in text.split("§"):
        part = part.strip()
        if not part.startswith("R"):
            continue
        digits = ""
        for ch in part[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            continue
        rule = f"R{digits}"
        open_paren = part.find("(")
        close_paren = part.find(")", open_paren + 1)
        if open_paren < 0 or close_paren < 0:
            continue
        inner = part[open_paren + 1: close_paren]
        if len(inner) != 4 or not inner.isdigit():
            continue
        mm, dd = int(inner[:2]), int(inner[2:])
        try:
            cand = datetime(now.year, mm, dd)
        except ValueError:
            continue
        if cand.date() > now.date():
            cand = datetime(now.year - 1, mm, dd)
        found[rule] = cand.strftime("%Y-%m-%d")
    return found


def _register_discipline_effective_date(args: Dict[str, Any]) -> None:
    """memory 写入成功后,把条目里的 R<n>+MMDD 登记进 discipline_dates.json.

    Contract:
      Preconditions: args 是 memory 工具参数 dict,调用方仅在 status=ok 时进入
      Postconditions: 命中约定格式的规则写入 dates 文件(同规则号再写=修订生效,
                      重开验收窗口);无命中=静默返回(非纪律条目是常态);
                      IO 失败仅 warning 不抛,不阻断采集主链
    隐私: 只持久化 {规则号: 日期} 两个短字符串,不落任何条目内容。
    """
    import json as _json

    texts: List[str] = []
    content = args.get("content") or args.get("new_text") or ""
    if isinstance(content, str) and content:
        texts.append(content)
    for op in args.get("operations") or []:
        if isinstance(op, dict):
            c = op.get("content") or op.get("new_text") or ""
            if isinstance(c, str) and c:
                texts.append(c)

    found: Dict[str, str] = {}
    for text in texts:
        found.update(_parse_discipline_tags(text))
    if not found:
        return

    dates_path = _get_db_path().parent / "outcomes" / "discipline_dates.json"
    dates_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing: Dict[str, str] = {}
        if dates_path.exists():
            loaded = _json.loads(dates_path.read_text(encoding="utf-8"))
            assert isinstance(loaded, dict), "corrupt dates file"
            existing = {str(k): str(v) for k, v in loaded.items()}
        existing.update(found)  # 同规则号再写=修订版生效,重开窗口
        dates_path.write_text(
            _json.dumps(dict(sorted(existing.items())), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("outcome-collector: discipline dates registered: %s", found)
    except Exception as exc:
        logger.warning("outcome-collector: discipline date registration failed: %s", exc)


def on_session_end(**kwargs) -> None:
    """Flush remaining buffer and compute session summary on session end."""
    if _plugin_disabled():
        return

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if not sid:
        return

    _flush_buffer(sid)
    _update_session_summary(sid)


# ── Registration ──────────────────────────────────────────────────────

def register(ctx) -> None:
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
    logger.info("outcome-collector registered (Layer 0+1+3)")
