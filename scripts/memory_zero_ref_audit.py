#!/usr/bin/env python3
"""Memory zero-reference audit — builtin MEMORY.md + hindsight facts.

Scans memory entries against session history (state.db FTS5) to find
entries that were never substantively referenced in any conversation.
Also reports hindsight facts with retrieval_count=0.

Zero-intrusion: read-only, does not modify any files.

Usage:
    python3 scripts/memory_zero_ref_audit.py                # stdout report
    python3 scripts/memory_zero_ref_audit.py --json         # machine-readable JSON
    python3 scripts/memory_zero_ref_audit.py --days 30      # last 30 days only
    python3 scripts/memory_zero_ref_audit.py --hindsight   # include hindsight facts
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ENTRY_DELIMITER = "§"


def get_hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def get_memory_dir() -> Path:
    return get_hermes_home() / "memories"


def parse_entries(text: str) -> list[str]:
    entries = []
    for chunk in text.split(ENTRY_DELIMITER):
        chunk = chunk.strip()
        if chunk:
            entries.append(chunk)
    return entries


def extract_key_phrases(entry: str, min_len: int = 4) -> list[str]:
    """Extract distinctive searchable phrases from a memory entry."""
    clauses = re.split(r'[。\n；;！!？？]', entry)
    phrases = []
    for clause in clauses:
        clause = clause.strip()
        if len(clause) < min_len:
            continue
        if re.search(r'[\u4e00-\u9fff]', clause):
            segments = re.split(r'[，,、：:\s（）()\[\]【】]', clause)
            for seg in segments:
                seg = seg.strip()
                if len(seg) >= min_len:
                    phrases.append(seg)
        else:
            phrases.append(clause)

    seen = set()
    unique = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:5]


def get_state_db_path() -> Path:
    home = get_hermes_home()
    candidates = [home / "state.db"]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return candidates[0]


def count_fts_hits(conn: sqlite3.Connection, phrase: str,
                   since_ts: float = 0) -> int:
    """Count distinct sessions that reference the phrase via FTS5."""
    escaped = phrase.replace('"', '""')

    if since_ts > 0:
        query = """
            SELECT count(DISTINCT m.session_id)
            FROM messages_fts fts
            JOIN messages m ON m.id = fts.rowid
            WHERE messages_fts MATCH ?
              AND m.timestamp >= ?
        """
        params = (escaped, since_ts)
    else:
        query = """
            SELECT count(DISTINCT m.session_id)
            FROM messages_fts fts
            JOIN messages m ON m.id = fts.rowid
            WHERE messages_fts MATCH ?
        """
        params = (escaped,)

    try:
        row = conn.execute(query, params).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        # FTS5 syntax error — try trigram as fallback
        try:
            if since_ts > 0:
                q = """
                    SELECT count(DISTINCT m.session_id)
                    FROM messages_fts_trigram fts
                    JOIN messages m ON m.id = fts.rowid
                    WHERE messages_fts_trigram MATCH ?
                      AND m.timestamp >= ?
                """
                p = (escaped, since_ts)
            else:
                q = """
                    SELECT count(DISTINCT m.session_id)
                    FROM messages_fts_trigram fts
                    JOIN messages m ON m.id = fts.rowid
                    WHERE messages_fts_trigram MATCH ?
                """
                p = (escaped,)
            row = conn.execute(q, p).fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0


def audit_builtin_memory(db_path: Path, days: int = 0) -> list[dict]:
    """Audit builtin MEMORY.md against session FTS."""
    memory_path = get_memory_dir() / "MEMORY.md"
    if not memory_path.exists():
        return []

    text = memory_path.read_text(encoding="utf-8")
    entries = parse_entries(text)
    if not entries:
        return []

    since_ts = 0.0
    if days > 0:
        import time
        since_ts = time.time() - days * 86400

    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        results = []
        for i, entry in enumerate(entries):
            phrases = extract_key_phrases(entry)
            if not phrases:
                results.append({
                    "index": i,
                    "preview": entry[:80],
                    "hit_sessions": 0,
                    "phrases_tested": 0,
                    "status": "no_phrases",
                    "source": "builtin",
                })
                continue

            max_hits = 0
            tested = 0
            for phrase in phrases:
                hits = count_fts_hits(conn, phrase, since_ts)
                tested += 1
                if hits > max_hits:
                    max_hits = hits
                if max_hits > 0:
                    break

            results.append({
                "index": i,
                "preview": entry[:80],
                "hit_sessions": max_hits,
                "phrases_tested": tested,
                "status": "referenced" if max_hits > 0 else "zero_ref",
                "source": "builtin",
            })
    finally:
        conn.close()

    return results


def audit_hindsight_facts(days: int = 0) -> list[dict]:
    """Report hindsight facts with retrieval_count statistics."""
    facts_db = get_hermes_home() / "memory_store.db"
    if not facts_db.exists() or facts_db.stat().st_size == 0:
        return []

    conn = sqlite3.connect(f"file:{facts_db}?mode=ro", uri=True)
    try:
        rows = conn.execute("""
            SELECT fact_id, content, retrieval_count, helpful_count,
                   created_at, updated_at
            FROM facts
            ORDER BY retrieval_count ASC, created_at DESC
        """).fetchall()

        results = []
        for r in rows:
            # Compute days since last update
            age_days = 0
            if r[5]:
                import time
                try:
                    from datetime import datetime
                    updated = datetime.strptime(r[5], "%Y-%m-%d %H:%M:%S")
                    age_days = (datetime.utcnow() - updated).days
                except (ValueError, TypeError):
                    pass

            entry = {
                "fact_id": r[0],
                "preview": r[1][:80],
                "retrieval_count": r[2],
                "helpful_count": r[3],
                "created_at": r[4],
                "updated_at": r[5],
                "age_days": age_days,
                "status": "zero_ref" if r[2] == 0 else "referenced",
                "source": "hindsight",
            }

            # Filter by days if specified
            if days > 0 and age_days > days:
                continue

            results.append(entry)
    finally:
        conn.close()

    return results


def print_report(builtin_results: list[dict], hindsight_results: list[dict],
                 days: int):
    print("Memory Zero-Reference Audit")
    print("=" * 60)

    # --- Builtin MEMORY.md ---
    total_builtin = len(builtin_results)
    zero_builtin = [r for r in builtin_results if r["status"] == "zero_ref"]
    ref_builtin = [r for r in builtin_results if r["status"] == "referenced"]

    print(f"\n[1] Builtin MEMORY.md ({get_memory_dir() / 'MEMORY.md'})")
    print(f"    Total entries: {total_builtin}")
    print(f"    Referenced (>=1 session): {len(ref_builtin)}")
    print(f"    Zero reference: {len(zero_builtin)}")

    if zero_builtin:
        print(f"\n    --- Zero-Reference Entries ({len(zero_builtin)}) ---")
        for r in zero_builtin:
            print(f"      [{r['index']:2d}] {r['preview']}")

    if ref_builtin:
        sorted_ref = sorted(ref_builtin, key=lambda x: x["hit_sessions"],
                            reverse=True)
        print(f"\n    --- Top Referenced ({len(ref_builtin)}) ---")
        for r in sorted_ref[:10]:
            print(f"      [{r['index']:2d}] {r['hit_sessions']:>3d} sessions | {r['preview']}")
        if len(ref_builtin) > 10:
            print(f"      ... and {len(ref_builtin) - 10} more")

    # --- Hindsight facts ---
    if hindsight_results is not None:
        total_h = len(hindsight_results)
        zero_h = [r for r in hindsight_results if r["status"] == "zero_ref"]
        ref_h = [r for r in hindsight_results if r["status"] == "referenced"]
        avg_ref = (sum(r["retrieval_count"] for r in hindsight_results)
                   / total_h) if total_h > 0 else 0

        print(f"\n[2] Hindsight facts (memory_store.db)")
        print(f"    Total facts: {total_h}")
        print(f"    Zero retrieval: {len(zero_h)} ({len(zero_h)*100//total_h if total_h else 0}%)")
        print(f"    Avg retrieval: {avg_ref:.1f}")

        if zero_h:
            print(f"\n    --- Zero-Retrieval Facts ({len(zero_h)}) ---")
            for r in zero_h[:15]:
                age = f"{r['age_days']}d old" if r['age_days'] > 0 else "age unknown"
                print(f"      [{r['fact_id']:>3d}] {age:>12s} | {r['preview']}")
            if len(zero_h) > 15:
                print(f"      ... and {len(zero_h) - 15} more")

        if ref_h:
            sorted_h = sorted(ref_h, key=lambda x: x["retrieval_count"],
                              reverse=True)
            print(f"\n    --- Top Retrieved Facts ({len(ref_h)}) ---")
            for r in sorted_h[:5]:
                print(f"      [{r['fact_id']:>3d}] ret={r['retrieval_count']:>3d} help={r['helpful_count']} | {r['preview']}")

    # --- Summary ---
    total_zero = len(zero_builtin) + (len(zero_h) if hindsight_results else 0)
    print(f"\n{'='*60}")
    print(f"Total zero-reference: {total_zero}")
    if total_zero > 0:
        print(f"Suggestion: review these entries and remove stale ones via")
        print(f"  /memory or the memory tool (action=remove).")


def main():
    parser = argparse.ArgumentParser(
        description="Audit memory for zero-reference entries")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON")
    parser.add_argument("--days", type=int, default=0,
                        help="Only entries from last N days (0=all time)")
    parser.add_argument("--hindsight", action="store_true",
                        help="Also audit hindsight facts (default: yes)")
    parser.add_argument("--no-hindsight", action="store_true",
                        help="Skip hindsight facts audit")
    args = parser.parse_args()

    db_path = get_state_db_path()
    if not db_path.exists() or db_path.stat().st_size == 0:
        print(f"ERROR: state.db not found or empty: {db_path}", file=sys.stderr)
        sys.exit(1)

    builtin_results = audit_builtin_memory(db_path, args.days)

    hindsight_results = None
    if not args.no_hindsight:
        hindsight_results = audit_hindsight_facts(args.days)

    if args.json:
        output = {"builtin": builtin_results}
        if hindsight_results is not None:
            output["hindsight"] = hindsight_results
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_report(builtin_results, hindsight_results, args.days)


if __name__ == "__main__":
    main()
