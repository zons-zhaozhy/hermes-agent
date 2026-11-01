#!/usr/bin/env python3
"""Block owner-loop hazards in gateway async code.

A safe off-loop pattern often wraps the blocking call in a nested function and hands that closure to
``asyncio.to_thread``/executor. This gate therefore inspects only calls in an async function's OWN
scope; nested function/lambda bodies are skipped.

The reverse bug is equally fatal: a sync function that calls ``asyncio.create_task`` must not be
handed to ``asyncio.to_thread``, because worker threads have no running event loop.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOTS = ("gateway", "tui_gateway", "cron")
DB_HELPERS = {"_read_ctx", "_read_all", "_read_one", "_execute_write", "_write_sql", "_write_rowcount", "get_meta"}
DB_CURSOR = {"execute", "executemany", "fetchall", "fetchone", "fetchmany"}
FILE_METHODS = {"read_text", "write_text", "read_bytes", "write_bytes"}
FS_FUNCTIONS = {"rmtree", "copytree", "copy2", "move"}
SUBPROCESS_FUNCTIONS = {"run", "check_output", "check_call", "system"}
SYNC_CONFIG = {"load_gateway_config", "discover_plugins"}
SQLITE_MODULES = {"sqlite3", "sqlite3.dbapi2"}
ALLOW_MARK = "# owner-loop: allow "
# These helpers create asyncio tasks; handing them to a worker thread removes their loop.
SCHEDULING_WORKERS = {"recover_local_sessions", "recover_authority_api_turns"}
LOOP_ONLY_CALLS = {"create_task", "get_running_loop", "call_soon_threadsafe"}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    rule: str
    expr: str


def attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def own_scope_calls(function: ast.AsyncFunctionDef):
    """Yield calls in this coroutine, not in nested functions/lambdas (common to_thread closures)."""
    stack = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def classify(call: ast.Call) -> str | None:
    func = call.func
    chain = attr_chain(func)
    attrs = set(chain.split("."))
    if attrs & DB_HELPERS:
        return "owner-loop.db-helper"
    if attrs & DB_CURSOR and chain not in {"execute", "executemany"}:
        # Bare ``execute(...)`` is a callback convention (for example destructive slash confirm),
        # not evidence of a cursor. Cursor calls must have a receiver.
        return "owner-loop.db-cursor"
    if isinstance(func, ast.Name) and func.id == "open":
        return "owner-loop.file-open"
    if isinstance(func, ast.Attribute) and func.attr in FILE_METHODS:
        return "owner-loop.file-io"
    if isinstance(func, ast.Name) and func.id in FS_FUNCTIONS:
        return "owner-loop.fs-bulk"
    # Bare ``run()`` is commonly an async task closure; only a receiver proves subprocess.run.
    if isinstance(func, ast.Name) and func.id in (SUBPROCESS_FUNCTIONS - {"run"}):
        return "owner-loop.subprocess"
    if isinstance(func, ast.Name) and func.id in SYNC_CONFIG:
        return "owner-loop.sync-config"
    if isinstance(func, ast.Attribute) and func.attr in SUBPROCESS_FUNCTIONS:
        root = chain.split(".", 1)[0]
        if root in {"os", "subprocess"}:
            return "owner-loop.subprocess"
    if isinstance(func, ast.Attribute) and func.attr == "connect" and chain.split(".", 1)[0] in SQLITE_MODULES:
        return "owner-loop.sqlite-connect"
    if isinstance(func, ast.Attribute) and func.attr == "sleep":
        root = chain.split(".", 1)[0]
        if root == "time":
            return "owner-loop.sync-sleep"
    return None


def worker_target_forbidden(call: ast.Call) -> bool:
    """Protect scheduling helpers from the no-running-event-loop worker bug."""
    if attr_chain(call.func) != "asyncio.to_thread" or not call.args:
        return False
    target = call.args[0]
    return isinstance(target, ast.Name) and target.id in SCHEDULING_WORKERS


def scan_path(path: Path) -> list[Finding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        print(f"owner-loop-gate: cannot parse {path}: {exc}", file=sys.stderr)
        raise
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    findings: list[Finding] = []

    class Visitor(ast.NodeVisitor):
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            for call in own_scope_calls(node):
                rule = classify(call)
                if rule:
                    if ALLOW_MARK not in lines[call.lineno - 1]:
                        findings.append(Finding(path, call.lineno, rule, ast.unparse(call.func)))
                elif worker_target_forbidden(call):
                    findings.append(Finding(
                        path, call.lineno, "owner-loop.async-worker",
                        ast.unparse(call.func),
                    ))
            self.generic_visit(node)

    Visitor().visit(tree)
    return findings


def render(findings: list[Finding]) -> str:
    if not findings:
        return "owner-loop-gate: 0 owner-loop hazard call(s) in gateway async scopes"
    lines = [f"owner-loop-gate: {len(findings)} owner-loop hazard call(s) in gateway async scopes"]
    for item in sorted(findings, key=lambda x: (x.path, x.line, x.rule)):
        lines.append(f"  {item.path}:{item.line}: {item.rule}: {item.expr}")
    lines.append(
        "  Move SQLite/file/subprocess work behind asyncio.to_thread (often via a nested function); "
        "never move a helper that calls asyncio.create_task/get_running_loop into a worker thread. "
        "A measured exception takes `# owner-loop: allow <rule> -- why` on the call line."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("repo_root", nargs="?", default=Path.cwd(), type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root
    findings: list[Finding] = []
    for root_name in ROOTS:
        for path in (root / root_name).rglob("*.py"):
            if "__pycache__" not in path.parts:
                findings.extend(scan_path(path))
    print(render(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
