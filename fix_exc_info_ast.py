#!/usr/bin/env python3
"""AST-level fix: add exc_info=True to logger calls in except blocks, and add logger.warning to bare except blocks."""

import ast
import sys
import os
import re
from pathlib import Path

TARGET_DIRS = [
    "/Users/stan/code/ai/cnb/ontox/plugins/aml-platform/backend/app/",
    "/Users/stan/code/ai/cnb/ontox/plugins/aml-diagnostic/app/",
    "/Users/stan/code/ai/cnb/ontox/plugins/aml-diagnostic/services/",
    "/Users/stan/code/ai/cnb/ontox/plugins/ecom-adapter/app/",
    "/Users/stan/code/ai/cnb/ontox/plugins/sample-erp-sync/",
    "/Users/stan/code/ai/cnb/ontox/dbchat/backend/app/",
]

LOGGER_METHODS = {"error", "warning", "debug", "critical", "info"}

def collect_python_files(directories):
    files = []
    for d in directories:
        p = Path(d)
        if not p.exists():
            print(f"  WARNING: directory not found: {d}", file=sys.stderr)
            continue
        for f in sorted(p.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            files.append(f)
    return files

def find_logger_names(tree):
    names = {"logger", "log"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        if (isinstance(func, ast.Attribute) and func.attr == "getLogger" and
                            isinstance(func.value, ast.Name) and func.value.id == "logging"):
                            names.add(target.id)
                        if isinstance(func, ast.Name) and func.id == "getLogger":
                            names.add(target.id)
    return names

def analyze_file(filepath):
    """Analyze a file and return list of fixes to apply.
    Each fix is a dict with type, line info, and details."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.split('\n')
    logger_names = find_logger_names(tree)
    fixes = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not node.body:
            continue

        exc_var = node.name  # exception variable name

        # --- Rule 1: logger calls with exception var as last arg but no exc_info ---
        for stmt in node.body:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
                continue
            call = stmt.value
            func = call.func
            if not (isinstance(func, ast.Attribute) and func.attr in LOGGER_METHODS and
                    isinstance(func.value, ast.Name) and func.value.id in logger_names):
                continue
            # Already has exc_info?
            if any(kw.arg == "exc_info" for kw in call.keywords):
                continue
            # Last positional arg is exception variable?
            if call.args and isinstance(call.args[-1], ast.Name) and call.args[-1].id == exc_var:
                fixes.append({
                    'type': 'add_exc_info',
                    'lineno': call.lineno,
                    'end_lineno': call.end_lineno or call.lineno,
                    'end_col': call.end_col_offset,
                    'exc_var': exc_var,
                })

        # --- Rule 2: no logger before return/continue/break ---
        has_logger = False
        first_control_idx = None
        for i, stmt in enumerate(node.body):
            for n in ast.walk(stmt):
                if isinstance(n, ast.Call):
                    f2 = n.func
                    if (isinstance(f2, ast.Attribute) and f2.attr in LOGGER_METHODS and
                        isinstance(f2.value, ast.Name) and f2.value.id in logger_names):
                        has_logger = True
                        break
            if has_logger:
                break
            if isinstance(stmt, (ast.Return, ast.Continue, ast.Break)):
                first_control_idx = i
                break

        if not has_logger and first_control_idx is not None and exc_var:
            control_stmt = node.body[first_control_idx]
            first_body_line = node.body[0].lineno
            fixes.append({
                'type': 'add_logger_warning',
                'insert_before_line': control_stmt.lineno,
                'indent_line': first_body_line,
                'exc_var': exc_var,
            })

    return fixes


def apply_fixes(filepath, fixes):
    """Apply fixes to file. Returns number of changes made."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    lines = content.split('\n')
    changes = 0

    # Sort fixes: line-based inserts by line desc, then exc_info by line desc
    # Process from bottom to top to preserve line numbers
    fixes.sort(key=lambda f: (
        0 if f['type'] == 'add_exc_info' else 1,
        f.get('end_lineno', f.get('insert_before_line', 0))
    ), reverse=True)

    for fix in fixes:
        if fix['type'] == 'add_exc_info':
            lineno = fix['lineno']
            end_lineno = fix['end_lineno']
            end_col = fix['end_col']

            if end_lineno == lineno:
                # Single-line call
                li = lineno - 1
                line = lines[li]
                # Find the position to insert before closing )
                pos = min(end_col, len(line))
                # Walk backwards to find the closing paren
                found = False
                for p in range(min(pos, len(line)) - 1, max(0, pos - 200), -1):
                    if line[p] == ')':
                        line = line[:p] + ', exc_info=True)' + line[p+1:]
                        lines[li] = line
                        changes += 1
                        found = True
                        break
                if not found:
                    print(f"    WARN: could not find ) on line {lineno}: {line[:80]}", file=sys.stderr)
            else:
                # Multi-line call - insert before closing )
                li = end_lineno - 1
                line = lines[li]
                pos = min(end_col, len(line))
                found = False
                for p in range(min(pos, len(line)) - 1, max(0, pos - 200), -1):
                    if line[p] == ')':
                        line = line[:p] + ', exc_info=True)' + line[p+1:]
                        lines[li] = line
                        changes += 1
                        found = True
                        break
                if not found:
                    print(f"    WARN: could not find ) on line {end_lineno}: {line[:80]}", file=sys.stderr)

        elif fix['type'] == 'add_logger_warning':
            li = fix['insert_before_line'] - 1
            indent_li = fix['indent_line'] - 1
            indent = re.match(r'^(\s*)', lines[indent_li]).group(1)
            exc_var = fix['exc_var']
            new_line = f'{indent}logger.warning(f"{exc_var}", exc_info=True)'
            lines.insert(li, new_line)
            changes += 1

    if changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    return changes


def main():
    total_changes = 0
    total_files = 0

    py_files = collect_python_files(TARGET_DIRS)
    print(f"Found {len(py_files)} Python files to scan")

    for filepath in py_files:
        try:
            fixes = analyze_file(filepath)
        except Exception as e:
            print(f"  SKIP (error): {filepath}: {e}", file=sys.stderr)
            continue

        if fixes:
            n = apply_fixes(filepath, fixes)
            if n > 0:
                total_changes += n
                total_files += 1
                rel = os.path.relpath(filepath, "/Users/stan/code/ai/cnb/ontox/")
                print(f"  FIXED {n}: {rel}")
                for fix in fixes:
                    if fix['type'] == 'add_exc_info':
                        print(f"    [+exc_info] L{fix['lineno']} (var={fix['exc_var']})")
                    elif fix['type'] == 'add_logger_warning':
                        print(f"    [+logger.warning] before L{fix['insert_before_line']} (var={fix['exc_var']})")

    print(f"\n=== SUMMARY ===")
    print(f"Files modified: {total_files}")
    print(f"Total fixes applied: {total_changes}")


if __name__ == '__main__':
    main()
