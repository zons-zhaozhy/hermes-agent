#!/usr/bin/env python3
"""outcome-analyzer — Layer 2 pattern analysis for the outcome feedback flywheel.

Analyzes accumulated tool_outcomes data to identify:
1. High-error tools (>30% error rate)
2. Error patterns by tool (top error types/messages)
3. Session-level failure clustering
4. Temporal trends (improving vs degrading)

Outputs structured findings to stdout (consumable by agent or cron).
Optionally writes high-signal patterns to Hermes memory for behavior change.

USAGE:
    python plugins/outcome-collector/analyze.py                    # Full analysis
    python plugins/outcome-collector/analyze.py --days 7           # Last 7 days
    python plugins/outcome-collector/analyze.py --memory           # Also write to memory
    python plugins/outcome-collector/analyze.py --json             # JSON output
    OUTCOME_COLLECTOR_DISABLE=1 python ...                         # no-op

DESIGN:
- Pure SQL aggregation (no ML, no LLM calls)
- Threshold-based pattern detection (>30% error rate, >5 occurrences)
- Memory output uses the memory tool's target=user for cross-session persistence
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_db_path() -> Path:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "outcomes.db"


def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ── Analysis queries ──────────────────────────────────────────────────

def analyze_tool_error_rates(conn: sqlite3.Connection, days: int) -> List[Dict]:
    """Find tools with >30% error rate (min 5 calls)."""
    rows = conn.execute(
        """
        SELECT tool_name,
               COUNT(*) as total,
               SUM(CASE WHEN status IN ('error', 'blocked') THEN 1 ELSE 0 END) as errors
        FROM tool_outcomes
        WHERE tool_name != '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        GROUP BY tool_name
        HAVING total >= 5
        ORDER BY errors * 1.0 / total DESC
        """,
        (f"-{days} days",),
    ).fetchall()

    findings = []
    for r in rows:
        total = r["total"]
        errors = r["errors"]
        rate = errors / total if total else 0
        if rate >= 0.3:
            findings.append({
                "type": "high_error_tool",
                "tool": r["tool_name"],
                "total_calls": total,
                "error_count": errors,
                "error_rate": round(rate * 100, 1),
                "severity": "high" if rate >= 0.5 else "medium",
            })
    return findings


def analyze_error_patterns(conn: sqlite3.Connection, days: int) -> List[Dict]:
    """Top error messages per tool (min 3 occurrences)."""
    rows = conn.execute(
        """
        SELECT tool_name,
               error_type,
               error_message,
               COUNT(*) as cnt
        FROM tool_outcomes
        WHERE status IN ('error', 'blocked')
          AND timestamp >= datetime('now', ?)
          AND error_message IS NOT NULL
        GROUP BY tool_name, error_type, error_message
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT 20
        """,
        (f"-{days} days",),
    ).fetchall()

    findings = []
    for r in rows:
        findings.append({
            "type": "recurring_error",
            "tool": r["tool_name"],
            "error_type": r["error_type"],
            "error_message": r["error_message"][:150],
            "count": r["cnt"],
        })
    return findings


def analyze_session_failure_clusters(conn: sqlite3.Connection, days: int) -> List[Dict]:
    """Sessions with high error density (potential stuck/debugging loops)."""
    rows = conn.execute(
        """
        SELECT session_id,
               COUNT(*) as total,
               SUM(CASE WHEN status IN ('error', 'blocked') THEN 1 ELSE 0 END) as errors,
               MIN(timestamp) as started,
               MAX(timestamp) as ended
        FROM tool_outcomes
        WHERE tool_name != '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        GROUP BY session_id
        HAVING total >= 10 AND errors * 1.0 / total >= 0.4
        ORDER BY errors * 1.0 / total DESC
        LIMIT 10
        """,
        (f"-{days} days",),
    ).fetchall()

    findings = []
    for r in rows:
        findings.append({
            "type": "high_error_session",
            "session_id": r["session_id"],
            "total_calls": r["total"],
            "error_count": r["errors"],
            "error_rate": round(r["errors"] / r["total"] * 100, 1),
            "timespan": f"{r['started']} → {r['ended']}",
        })
    return findings


def analyze_tool_usage_breakdown(conn: sqlite3.Connection, days: int) -> Dict:
    """Overall tool usage statistics."""
    rows = conn.execute(
        """
        SELECT tool_name,
               COUNT(*) as total,
               SUM(CASE WHEN status IN ('error', 'blocked') THEN 1 ELSE 0 END) as errors,
               AVG(duration_ms) as avg_ms
        FROM tool_outcomes
        WHERE tool_name != '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        GROUP BY tool_name
        ORDER BY total DESC
        """,
        (f"-{days} days",),
    ).fetchall()

    return {
        "type": "usage_breakdown",
        "tools": [
            {
                "tool": r["tool_name"],
                "total": r["total"],
                "errors": r["errors"],
                "error_rate": round(r["errors"] / r["total"] * 100, 1) if r["total"] else 0,
                "avg_duration_ms": round(r["avg_ms"]) if r["avg_ms"] else 0,
            }
            for r in rows
        ],
    }


def analyze_temporal_trend(conn: sqlite3.Connection, days: int) -> Dict:
    """Daily error rate trend — improving or degrading?"""
    rows = conn.execute(
        """
        SELECT DATE(timestamp) as day,
               COUNT(*) as total,
               SUM(CASE WHEN status IN ('error', 'blocked') THEN 1 ELSE 0 END) as errors
        FROM tool_outcomes
        WHERE tool_name != '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY day
        """,
        (f"-{days} days",),
    ).fetchall()

    daily = [
        {
            "date": r["day"],
            "total": r["total"],
            "errors": r["errors"],
            "error_rate": round(r["errors"] / r["total"] * 100, 1) if r["total"] else 0,
        }
        for r in rows
    ]

    # Simple trend: compare first half vs second half
    trend = "stable"
    if len(daily) >= 4:
        mid = len(daily) // 2
        first_half_rate = sum(d["error_rate"] for d in daily[:mid]) / mid
        second_half_rate = sum(d["error_rate"] for d in daily[mid:]) / (len(daily) - mid)
        if second_half_rate < first_half_rate - 5:
            trend = "improving"
        elif second_half_rate > first_half_rate + 5:
            trend = "degrading"

    return {
        "type": "temporal_trend",
        "trend": trend,
        "daily": daily,
    }


def analyze_turn_outcomes(conn: sqlite3.Connection, days: int) -> Dict:
    """Layer 1 analysis: aggregate turn-level outcomes (success/failure/partial)."""
    rows = conn.execute(
        """
        SELECT status as outcome,
               error_type as failure_pattern,
               COUNT(*) as count
        FROM tool_outcomes
        WHERE tool_name = '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        GROUP BY status, error_type
        """,
        (f"-{days} days",),
    ).fetchall()

    by_outcome: Dict[str, int] = {}
    by_pattern: Dict[str, int] = {}
    for r in rows:
        outcome = r["outcome"] or "unknown"
        by_outcome[outcome] = by_outcome.get(outcome, 0) + r["count"]
        if r["failure_pattern"]:
            by_pattern[r["failure_pattern"]] = (
                by_pattern.get(r["failure_pattern"], 0) + r["count"]
            )

    total_turns = sum(by_outcome.values())
    failure_rate = (
        round(by_outcome.get("failure", 0) / total_turns * 100, 1)
        if total_turns
        else 0
    )

    return {
        "type": "turn_outcomes",
        "total_turns": total_turns,
        "by_outcome": by_outcome,
        "by_failure_pattern": by_pattern,
        "turn_failure_rate": failure_rate,
    }


# ── Cross-turn pattern detection (Tier 2) ──────────────────────────────


def analyze_cross_turn_patterns(conn: sqlite3.Connection, days: int) -> List[Dict]:
    """Tier 2: detect cross-turn retry loops and death loops within sessions.

    These are the highest-signal patterns — they indicate the agent is stuck
    in a loop it can't break out of.

    Detection logic:
      1. Group tool calls by session_id, then by turn_id.
      2. For each session, build per-turn "fingerprints" (ordered tool+status tuples).
      3. Cross-turn retry: same tool errors appear in ≥2 distinct turns → agent
         keeps retrying across turns without success.
      4. Death loop: the same fingerprint (or a rotation of it) repeats ≥3 turns →
         agent is cycling through the same steps with no progress.
    """
    rows = conn.execute(
        """
        SELECT session_id, turn_id, tool_name, status, timestamp
        FROM tool_outcomes
        WHERE tool_name != '_turn_outcome'
          AND timestamp >= datetime('now', ?)
        ORDER BY session_id, timestamp, id
        """,
        (f"-{days} days",),
    ).fetchall()

    findings: List[Dict] = []

    # Group by session → turn → calls
    sessions: Dict[str, Dict[str, List[Dict]]] = {}
    for r in rows:
        sid = r["session_id"]
        tid = r["turn_id"] or "_unknown"
        sessions.setdefault(sid, {}).setdefault(tid, []).append({
            "tool": r["tool_name"],
            "status": r["status"],
        })

    for sid, turns in sessions.items():
        if len(turns) < 2:
            continue  # Need ≥2 turns for any cross-turn pattern

        turn_ids = sorted(turns.keys())

        # --- Cross-turn retry: same tool errors in ≥2 distinct turns ---
        error_tool_turns: Dict[str, List[str]] = {}  # tool_name → [turn_ids where it errored]
        for tid in turn_ids:
            errored_tools = {
                c["tool"] for c in turns[tid]
                if c["status"] in ("error", "blocked")
            }
            for tool in errored_tools:
                error_tool_turns.setdefault(tool, []).append(tid)

        for tool, tids in error_tool_turns.items():
            if len(tids) >= 2:
                findings.append({
                    "type": "cross_turn_retry",
                    "severity": "high",
                    "tool": tool,
                    "session_id": sid,
                    "retry_turns": len(tids),
                    "turn_ids": tids,
                    "description": (
                        f"Tool '{tool}' errored across {len(tids)} turns "
                        f"in session {sid[:8]} — agent is stuck retrying "
                        f"the same failing tool without resolution."
                    ),
                })

        # --- Death loop: same fingerprint repeats ≥3 turns ---
        # Fingerprint = tuple of (tool, status) pairs in call order.
        # We also detect cyclic rotations: [A,B,C,A,B,C] is a death loop
        # even if each turn's fingerprint differs slightly.
        fingerprints: List[tuple] = []
        for tid in turn_ids:
            fp = tuple((c["tool"], c["status"]) for c in turns[tid])
            fingerprints.append(fp)

        if len(fingerprints) >= 3:
            # Direct repetition: same fingerprint ≥3 times
            from collections import Counter
            fp_counts = Counter(fingerprints)
            for fp, cnt in fp_counts.items():
                if cnt >= 3:
                    tool_names = [t[0] for t in fp]
                    findings.append({
                        "type": "death_loop",
                        "severity": "high",
                        "session_id": sid,
                        "loop_length": len(fp),
                        "repetitions": cnt,
                        "tools_in_loop": tool_names,
                        "description": (
                            f"Death loop in session {sid[:8]}: identical "
                            f"tool sequence ({' → '.join(tool_names)}) "
                            f"repeated {cnt} times — agent is cycling "
                            f"without making progress."
                        ),
                    })

            # Cyclic death loop: detect if the last N turns are a rotation
            # of an earlier pattern (e.g., [A,B,C] then [A,B,C] then [A,B,C])
            # Check for the minimal repeating unit.
            if len(fingerprints) >= 4:
                _detect_cyclic_death_loop(findings, sid, fingerprints)

    return findings


def _detect_cyclic_death_loop(
    findings: List[Dict],
    sid: str,
    fingerprints: List[tuple],
) -> None:
    """Detect cyclic repetition in tool-call fingerprints.

    Checks if the sequence has a short period (2-4) that repeats ≥3 times.
    Example: [A,B,A,B,A,B] has period 2, repeated 3 times.
    """
    n = len(fingerprints)
    for period in (2, 3, 4):
        if n < period * 3:
            continue  # Need at least 3 full cycles
        # Extract the candidate repeating unit
        unit = fingerprints[:period]
        is_cyclic = True
        for i in range(period, n):
            if fingerprints[i] != unit[i % period]:
                is_cyclic = False
                break
        if is_cyclic:
            # Flatten all tool names from the repeating unit
            tool_names = [pair[0] for fp_tuple in unit for pair in fp_tuple]
            repetitions = n // period
            findings.append({
                "type": "death_loop",
                "severity": "high",
                "session_id": sid,
                "loop_length": period,
                "repetitions": repetitions,
                "tools_in_loop": tool_names,
                "description": (
                    f"Cyclic death loop in session {sid[:8]}: "
                    f"period-{period} sequence ({' → '.join(tool_names)}) "
                    f"repeated {repetitions} times — agent is trapped."
                ),
            })
            return  # One detection per session is enough


# ── Main ──────────────────────────────────────────────────────────────

def run_analysis(db_path: Path, days: int = 7) -> Dict[str, Any]:
    """Run full analysis and return structured findings."""
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}", "findings": []}

    conn = _get_conn(db_path)
    try:
        findings = []
        findings.extend(analyze_tool_error_rates(conn, days))
        findings.extend(analyze_error_patterns(conn, days))
        findings.extend(analyze_session_failure_clusters(conn, days))
        findings.extend(analyze_cross_turn_patterns(conn, days))

        usage = analyze_tool_usage_breakdown(conn, days)
        trend = analyze_temporal_trend(conn, days)
        turn_outcomes = analyze_turn_outcomes(conn, days)

        # Turn failure rate > 40% is a high-signal finding
        if turn_outcomes["total_turns"] >= 5 and turn_outcomes["turn_failure_rate"] > 40:
            findings.append({
                "type": "high_turn_failure_rate",
                "severity": "high",
                "turn_failure_rate": turn_outcomes["turn_failure_rate"],
                "total_turns": turn_outcomes["total_turns"],
                "by_pattern": turn_outcomes["by_failure_pattern"],
                "description": (
                    f"{turn_outcomes['turn_failure_rate']}% of turns "
                    f"({turn_outcomes['total_turns']} evaluated) resulted in failure. "
                    f"Top pattern: {max(turn_outcomes['by_failure_pattern'], key=turn_outcomes['by_failure_pattern'].get) if turn_outcomes['by_failure_pattern'] else 'N/A'}"
                ),
            })

        return {
            "generated_at": _ts(),
            "period_days": days,
            "total_findings": len(findings),
            "high_signal_findings": [f for f in findings if f.get("severity") == "high"],
            "findings": findings,
            "usage_breakdown": usage,
            "temporal_trend": trend,
            "turn_outcomes": turn_outcomes,
        }
    finally:
        conn.close()


def format_as_text(report: Dict) -> str:
    """Human-readable text format for terminal/cron output."""
    lines = []
    lines.append(f"=== Outcome Analysis Report ({report['generated_at']}) ===")
    lines.append(f"Period: last {report['period_days']} days")
    lines.append(f"Total findings: {report['total_findings']}")
    lines.append("")

    # High-signal findings first
    high = report.get("high_signal_findings", [])
    if high:
        lines.append("--- HIGH SEVERITY ---")
        for f in high:
            if f["type"] == "high_error_tool":
                lines.append(
                    f"  ⚠ {f['tool']}: {f['error_rate']}% error rate "
                    f"({f['error_count']}/{f['total_calls']} calls)"
                )
        lines.append("")

    # Recurring errors
    recurring = [f for f in report.get("findings", []) if f["type"] == "recurring_error"]
    if recurring:
        lines.append("--- RECURRING ERRORS ---")
        for f in recurring[:10]:
            lines.append(
                f"  {f['tool']}: \"{f['error_message']}\" "
                f"(×{f['count']})"
            )
        lines.append("")

    # Usage breakdown
    usage = report.get("usage_breakdown", {})
    if usage.get("tools"):
        lines.append("--- TOOL USAGE ---")
        for t in usage["tools"][:15]:
            marker = " ⚠" if t["error_rate"] >= 30 else ""
            lines.append(
                f"  {t['tool']:25s} {t['total']:5d} calls  "
                f"{t['error_rate']:5.1f}% err  {t['avg_duration_ms']:6d}ms avg{marker}"
            )
        lines.append("")

    # Cross-turn patterns (Tier 2)
    cross_turn = [f for f in report.get("findings", []) if f["type"] in ("cross_turn_retry", "death_loop")]
    if cross_turn:
        lines.append("--- CROSS-TURN PATTERNS (Tier 2) ---")
        for f in cross_turn[:10]:
            if f["type"] == "cross_turn_retry":
                lines.append(
                    f"  ⚠ RETRY: {f['tool']} errored across {f['retry_turns']} turns "
                    f"(session {f['session_id'][:8]})"
                )
            elif f["type"] == "death_loop":
                lines.append(
                    f"  ⚠ LOOP: {' → '.join(f['tools_in_loop'])} "
                    f"repeated {f['repetitions']}× "
                    f"(session {f['session_id'][:8]})"
                )
        lines.append("")

    # Trend
    trend = report.get("temporal_trend", {})
    if trend:
        arrow = {"improving": "↓", "degrading": "↑", "stable": "→"}.get(
            trend["trend"], "?"
        )
        lines.append(f"--- TREND: {arrow} {trend['trend'].upper()} ---")

    return "\n".join(lines)


def get_findings_path() -> Path:
    """Path to the findings file (read by on_session_start hook)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "outcomes" / "findings.md"


def write_findings_file(report: Dict) -> int:
    """Write analysis findings to ~/.hermes/outcomes/findings.md.

    This is a dedicated channel — does NOT pollute MEMORY.md (which has a
    tight char budget for durable facts, not volatile statistics).
    The on_session_start hook reads this file and notifies the agent.

    Returns the number of findings written.
    """
    high = report.get("high_signal_findings", [])
    recurring = [f for f in report.get("findings", []) if f["type"] == "recurring_error"]
    turn = report.get("turn_outcomes", {})

    if not high and not recurring and not turn.get("by_outcome"):
        print("  No findings to write.")
        return 0

    findings_path = get_findings_path()
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Outcome Analysis — {report['generated_at']}",
        f"Period: last {report['period_days']} days",
        "",
    ]

    # Layer 1: Turn outcomes summary
    if turn.get("total_turns", 0) > 0:
        lines.append("## Task-Level Outcomes")
        lines.append(f"Total turns evaluated: {turn['total_turns']}")
        for outcome, count in sorted(turn.get("by_outcome", {}).items()):
            lines.append(f"  {outcome}: {count}")
        if turn.get("turn_failure_rate", 0) > 30:
            lines.append(
                f"  ⚠ Failure rate {turn['turn_failure_rate']}% is high — "
                f"review failure patterns below"
            )
        if turn.get("by_failure_pattern"):
            lines.append("  Failure patterns:")
            for pattern, count in sorted(
                turn["by_failure_pattern"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {pattern}: {count}")
        lines.append("")

    # Layer 0: High-severity tool findings
    if high:
        lines.append("## High-Severity Findings")
        for f in high:
            if f["type"] == "high_error_tool":
                lines.append(
                    f"  ⚠ {f['tool']}: {f['error_rate']}% error rate "
                    f"({f['error_count']}/{f['total_calls']} calls)"
                )
            elif f["type"] == "high_turn_failure_rate":
                lines.append(f"  ⚠ {f['description']}")
        lines.append("")

    # Recurring errors (same error repeated ≥3 times)
    if recurring:
        lines.append("## Recurring Errors")
        for f in recurring[:10]:
            lines.append(
                f"  {f['tool']} (×{f['count']}): \"{f['error_message']}\""
            )
        lines.append("")

    # Cross-turn patterns (Tier 2)
    cross_turn = [f for f in report.get("findings", []) if f["type"] in ("cross_turn_retry", "death_loop")]
    if cross_turn:
        lines.append("## Cross-Turn Patterns (Tier 2)")
        lines.append("")
        for f in cross_turn[:10]:
            if f["type"] == "cross_turn_retry":
                lines.append(
                    f"- ⚠ RETRY: `{f['tool']}` errored across {f['retry_turns']} turns "
                    f"(session `{f['session_id'][:8]}`)"
                )
            elif f["type"] == "death_loop":
                lines.append(
                    f"- ⚠ LOOP: `{' → '.join(f['tools_in_loop'])}` "
                    f"repeated {f['repetitions']}× "
                    f"(session `{f['session_id'][:8]}`)"
                )
        lines.append("")

    # Trend
    trend = report.get("temporal_trend", {})
    if trend and trend.get("daily"):
        lines.append(f"## Trend: {trend['trend'].upper()}")
        lines.append("")

    lines.append("<!-- Auto-generated by outcome-collector analyze.py -->")

    content = "\n".join(lines)
    findings_path.write_text(content, encoding="utf-8")
    count = len(high) + len(recurring) + len(cross_turn)
    print(f"  ✓ Written {count} findings to {findings_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tool-call outcomes for self-evolving feedback."
    )
    parser.add_argument("--days", type=int, default=7, help="Analysis window (default: 7)")
    parser.add_argument("--write", action="store_true",
                        help="Write findings to ~/.hermes/outcomes/findings.md")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    db_path = _get_db_path()

    report = run_analysis(db_path, days=args.days)

    if "error" in report:
        print(report["error"])
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_as_text(report))

    if args.write:
        print("\n--- Writing findings ---")
        write_findings_file(report)


if __name__ == "__main__":
    main()
