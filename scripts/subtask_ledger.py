#!/usr/bin/env python3
"""Subtask ledger — cross-session task persistence.

Manages subtask state in a JSON file (default: .specs/subtasks.json),
enabling AI to pick up where a previous session left off.

Usage:
    python scripts/subtask_ledger.py init "Feature Name"       # create new ledger
    python scripts/subtask_ledger.py add M1 "description"     # add subtask
    python scripts/subtask_ledger.py status                    # show all
    python scripts/subtask_ledger.py start M1                  # move to doing
    python scripts/subtask_ledger.py done M1 "abc12345"       # mark done + commit hash
    python scripts/subtask_ledger.py note M1 "progress note"  # append note
    python scripts/subtask_ledger.py show                      # full detail
    python scripts/subtask_ledger.py resume                    # resume summary for new session
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEDGER_DIR = Path.home() / ".hermes" / "specs"


def _ledger_path(spec_name: str | None = None) -> Path:
    """Resolve ledger file path."""
    if spec_name:
        DEFAULT_LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_LEDGER_DIR / f"{spec_name}.subtasks.json"
    # Find the most recent ledger
    if not DEFAULT_LEDGER_DIR.exists():
        print("No specs directory found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    ledgers = sorted(DEFAULT_LEDGER_DIR.glob("*.subtasks.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ledgers:
        print("No subtask ledgers found. Run 'init' first.", file=sys.stderr)
        sys.exit(1)
    return ledgers[0]


def _load(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"spec": path.stem.replace(".subtasks", ""), "created": None, "subtasks": []}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_init(args):
    path = DEFAULT_LEDGER_DIR / f"{args.name}.subtasks.json"
    if path.exists():
        print(f"Ledger already exists: {path}", file=sys.stderr)
        sys.exit(1)
    data = {
        "spec": args.name,
        "created": datetime.now(timezone.utc).isoformat(),
        "subtasks": [],
    }
    _save(path, data)
    print(f"Created: {path}")
    print(f"Next: subtask_ledger add M1 'description'")


def cmd_add(args):
    path = _ledger_path()
    data = _load(path)
    for t in data["subtasks"]:
        if t["id"] == args.id:
            print(f"Subtask {args.id} already exists.", file=sys.stderr)
            sys.exit(1)
    subtask = {
        "id": args.id,
        "title": args.title,
        "status": "pending",
        "stage": None,
        "commit": None,
        "notes": [],
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat(),
    }
    data["subtasks"].append(subtask)
    _save(path, data)
    print(f"Added {args.id}: {args.title}")


def cmd_start(args):
    path = _ledger_path()
    data = _load(path)
    for t in data["subtasks"]:
        if t["id"] == args.id:
            t["status"] = "doing"
            t["stage"] = args.stage
            t["updated"] = datetime.now(timezone.utc).isoformat()
            _save(path, data)
            print(f"{args.id} → doing ({args.stage})")
            return
    print(f"Subtask {args.id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_done(args):
    path = _ledger_path()
    data = _load(path)
    for t in data["subtasks"]:
        if t["id"] == args.id:
            t["status"] = "done"
            t["commit"] = args.commit
            t["updated"] = datetime.now(timezone.utc).isoformat()
            _save(path, data)
            print(f"{args.id} → done (commit: {args.commit or 'N/A'})")
            return
    print(f"Subtask {args.id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_note(args):
    path = _ledger_path()
    data = _load(path)
    for t in data["subtasks"]:
        if t["id"] == args.id:
            t["notes"].append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "text": args.text,
            })
            t["updated"] = datetime.now(timezone.utc).isoformat()
            _save(path, data)
            print(f"{args.id}: note appended")
            return
    print(f"Subtask {args.id} not found.", file=sys.stderr)
    sys.exit(1)


def cmd_status(args):
    path = _ledger_path()
    data = _load(path)
    spec = data.get("spec", path.stem)
    tasks = data.get("subtasks", [])
    if not tasks:
        print(f"{spec}: no subtasks")
        return
    pending = [t for t in tasks if t["status"] == "pending"]
    doing = [t for t in tasks if t["status"] == "doing"]
    done = [t for t in tasks if t["status"] == "done"]
    print(f"{spec}: {len(done)}/{len(tasks)} done")
    if doing:
        for t in doing:
            print(f"  [DOING] {t['id']} {t['title']}  (stage: {t.get('stage')})")
    if pending:
        for t in pending:
            print(f"  [  ]    {t['id']} {t['title']}")


def cmd_show(args):
    path = _ledger_path()
    data = _load(path)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_resume(args):
    """Print a resume summary for a new AI session to pick up."""
    path = _ledger_path()
    data = _load(path)
    spec = data.get("spec", path.stem)
    tasks = data.get("subtasks", [])
    if not tasks:
        print(f"{spec}: no subtasks")
        return
    done = [t for t in tasks if t["status"] == "done"]
    doing = [t for t in tasks if t["status"] == "doing"]
    pending = [t for t in tasks if t["status"] == "pending"]
    print(f"=== RESUME: {spec} ===")
    print(f"Progress: {len(done)}/{len(tasks)} done")
    if doing:
        t = doing[0]
        print(f"\nCurrently working on: {t['id']} {t['title']}")
        if t.get("notes"):
            print("Last notes:")
            for note in t["notes"][-3:]:
                print(f"  [{note['ts'][:10]}] {note['text']}")
    if pending:
        print(f"\nRemaining ({len(pending)}):")
        for t in pending:
            blocking = [d for d in tasks if (d["id"] in t.get("depends_on", [])) and (d["status"] != "done")]
            blocked = " (blocked)" if blocking else ""
            print(f"  {t['id']}: {t['title']}{blocked}")
    print(f"\nLedger: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-session subtask ledger")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init", help="Create a new ledger")
    p.add_argument("name", help="Feature/spec name")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("add", help="Add a subtask")
    p.add_argument("id", help="Subtask ID (e.g. M1, M2)")
    p.add_argument("title", help="Description")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("start", help="Mark subtask as doing")
    p.add_argument("id", help="Subtask ID")
    p.add_argument("--stage", default="implement", help="Current stage")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("done", help="Mark subtask as done")
    p.add_argument("id", help="Subtask ID")
    p.add_argument("commit", nargs="?", default=None, help="Associated commit hash")
    p.set_defaults(func=cmd_done)

    p = sub.add_parser("note", help="Append a note to a subtask")
    p.add_argument("id", help="Subtask ID")
    p.add_argument("text", help="Note content")
    p.set_defaults(func=cmd_note)

    p = sub.add_parser("status", help="Quick status overview")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="Full JSON output")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("resume", help="Resume summary for new session")
    p.set_defaults(func=cmd_resume)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
