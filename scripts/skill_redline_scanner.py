#!/usr/bin/env python3
"""Red line scanner — extracts all red lines / hard rules from skills.

Scans all SKILL.md files and their references/ for red line patterns,
outputs a consolidated report. Used to audit "what can't I do" across
the entire skill library.

Usage:
    python scripts/skill_redline_scanner.py              # scan all skills
    python scripts/skill_redline_scanner.py --skill foo   # scan one skill
    python scripts/skill_redline_scanner.py --stats       # summary stats only
"""

import argparse
import re  # noqa: scanner tool — regex is essential
import sys
from pathlib import Path

SKILLS_ROOT = Path.home() / ".hermes" / "skills"

# Patterns that indicate red lines / hard rules / prohibitions
_PATTERNS = [
    (r"⛔\s*(.+?)(?:\n|$)", "critical"),
    (r"禁止[：:]\s*(.+?)(?:\n|$)", "prohibit"),
    (r"不允许[：:]\s*(.+?)(?:\n|$)", "prohibit"),
    (r"绝[对不]允许[：:，,]?\s*(.+?)(?:\n|$)", "prohibit"),
    (r"绝不[：:，,]?\s*(.+?)(?:\n|$)", "prohibit"),
    (r"铁律[：:]\s*(.+?)(?:\n|$)", "iron_rule"),
    (r"红线[：:]\s*(.+?)(?:\n|$)", "red_line"),
    (r"零容忍[：:]\s*(.+?)(?:\n|$)", "zero_tolerance"),
    (r"必须[：:]\s*(.+?)(?:\n|$)", "must"),
    (r"❌\s*(.+?)(?:\n|$)", "forbidden"),
]


def scan_file(file_path: Path) -> list[dict]:
    """Scan a single file for red line patterns."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    results = []
    for pattern, category in _PATTERNS:
        for m in re.finditer(pattern, content):
            text = m.group(1).strip()
            if len(text) < 5:
                continue
            line_num = content[:m.start()].count("\n") + 1
            results.append({
                "category": category,
                "text": text,
                "line": line_num,
                "file": file_path.name,
            })
    return results


def scan_skill(skill_dir: Path) -> list[dict]:
    """Scan a skill directory (SKILL.md + references/)."""
    results = []
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        for r in scan_file(skill_md):
            r["source"] = "SKILL.md"
            results.append(r)

    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for f in refs_dir.rglob("*.md"):
            for r in scan_file(f):
                r["source"] = str(f.relative_to(skill_dir))
                results.append(r)

    return results


def print_report(skill_name: str, results: list[dict], show_all: bool = False):
    """Print a report for one skill."""
    # Deduplicate by text (same rule may appear in SKILL.md and a reference)
    seen = set()
    unique = []
    for r in results:
        key = r["text"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return

    print(f"\n{'='*60}")
    print(f"  {skill_name}  ({len(unique)} rules)")
    print(f"{'='*60}")

    # Group by category
    by_cat: dict[str, list[dict]] = {}
    for r in unique:
        by_cat.setdefault(r["category"], []).append(r)

    # Order: critical > zero_tolerance > iron_rule > red_line > prohibit > forbidden > must
    order = ["critical", "zero_tolerance", "iron_rule", "red_line", "prohibit", "forbidden", "must"]
    for cat in order:
        items = by_cat.get(cat)
        if not items:
            continue
        print(f"\n  [{cat.upper()}] ({len(items)})")
        for r in items:
            rel = r["source"] if r["source"] != "SKILL.md" else ""
            suffix = f"  ({rel}:{r['line']})" if rel else f"  ({r['source']}:{r['line']})"
            # Truncate long rules
            text = r["text"][:120] + "..." if len(r["text"]) > 120 else r["text"]
            print(f"    • {text}{suffix}")

    # Show duplicates suppressed count
    dup_count = len(results) - len(unique)
    if dup_count > 0:
        print(f"\n  ({dup_count} duplicate(s) suppressed)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan skills for red lines / hard rules.")
    parser.add_argument("--skill", help="Scan only this skill")
    parser.add_argument("--stats", action="store_true", help="Summary stats only")
    args = parser.parse_args(argv)

    if args.skill:
        target = SKILLS_ROOT / args.skill
        if not (target / "SKILL.md").exists():
            target = Path(args.skill)
        if not (target / "SKILL.md").exists():
            print(f"Skill not found: {args.skill}", file=sys.stderr)
            return 1
        results = scan_skill(target)
        if args.stats:
            print(f"{len(set(r['text'][:60] for r in results))} unique rules")
        else:
            print_report(str(target.relative_to(SKILLS_ROOT)), results)
        return 0

    # All skills
    all_skills = []
    if SKILLS_ROOT.exists():
        for md in SKILLS_ROOT.rglob("SKILL.md"):
            all_skills.append(md.parent)

    all_skills.sort()
    total_rules = 0
    skill_count = 0

    for sd in all_skills:
        results = scan_skill(sd)
        unique_count = len(set(r["text"][:60] for r in results))
        if unique_count == 0:
            continue
        total_rules += unique_count
        skill_count += 1

    if args.stats:
        print(f"Skills with rules: {skill_count}/{len(all_skills)}")
        print(f"Total unique rules: {total_rules}")
        # Category breakdown
        cat_counts: dict[str, int] = {}
        for sd in all_skills:
            for r in scan_skill(sd):
                cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
        print(f"\nBy category:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:20s} {count}")
        return 0

    for sd in all_skills:
        results = scan_skill(sd)
        unique_count = len(set(r["text"][:60] for r in results))
        if unique_count == 0:
            continue
        print_report(str(sd.relative_to(SKILLS_ROOT)), results)

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_rules} unique rules across {skill_count} skills")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())