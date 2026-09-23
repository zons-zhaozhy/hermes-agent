"""
indexer-sync — Incremental index sync for codegraph/gitnexus after file edits.

Hooks into post_tool_call; when the agent edits a file (patch/write_file/read_file
touching a codegraph- or gitnexus-tracked repo), debounce-syncs the corresponding
index in a background thread. Zero blocking on the agent loop.
"""

import os
import re
import time
import threading
import subprocess
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────
_DEBOUNCE_SEC = 8.0       # coalesce bursts of rapid edits
_SYNC_TIMEOUT_SMALL = 60   # < 500 files
_SYNC_TIMEOUT_LARGE = 300  # >= 500 files (ontoX 3522 files needs ~3-4 min)
_MIN_RESYNC_INTERVAL = 120  # don't re-sync same repo within 2 min
# Tools that MODIFY files → trigger index sync
_WRITE_TOOL_NAMES = frozenset({
    "patch", "write_file",
    "terminal", "execute_code",  # can write via shell/python
})
# Tools that only READ → never trigger sync
# (read_file, search_files, etc. don't change the index)

# ── Tracked repo registry (resolved at register time) ────────────────
# codegraph: {project_root: {"type": "codegraph", "sync_cmd": [...]}}
# gitnexus:  {project_root: {"type": "gitnexus", "name": "...", "path": "..."}}
_tracked: dict[str, dict] = {}

# Per-root debounce state: {root: {"last_trigger": float, "worker": Thread|None}}
_debounce: dict[str, dict] = {}
_lock = threading.Lock()


def _discover_tracked_repos():
    """Build the registry of codegraph and gitnexus tracked repos."""
    # 1. codegraph: scan for .codegraph/ directories (bounded find, fast)
    codegraph_roots = []
    home = Path.home()
    scan_roots = [
        home / "code",
        home / "Desktop" / "项目",
    ]
    for scan_root in scan_roots:
        if not scan_root.is_dir():
            continue
        try:
            result = subprocess.run(
                ["find", str(scan_root), "-maxdepth", "5",
                 "-name", ".codegraph", "-type", "d", "-not", "-path", "*/.codegraph/*"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                cg_dir = Path(line)
                if cg_dir.is_dir():
                    codegraph_roots.append(cg_dir.parent)
        except Exception:
            pass

    for root in codegraph_roots:
        root_key = str(root)
        # Quick file count via find (fast, bounded)
        file_count = 0
        try:
            r = subprocess.run(
                ["find", str(root), "-maxdepth", "4",
                 "-name", "*.py", "-o", "-name", "*.ts", "-o", "-name", "*.tsx",
                 "-o", "-name", "*.js", "-o", "-name", "*.java"],
                capture_output=True, text=True, timeout=3,
            )
            file_count = len(r.stdout.strip().splitlines()) if r.stdout.strip() else 0
        except Exception:
            pass
        _tracked[root_key] = {
            "type": "codegraph",
            "sync_cmd": ["codegraph", "sync", "-q", str(root)],
            "file_count": file_count,
            "last_sync": 0.0,
        }
        logger.debug("indexer-sync: tracking codegraph at %s (%d files)", root_key, file_count)

    # 2. gitnexus: read registry
    try:
        import json
        registry_path = Path.home() / ".gitnexus" / "registry.json"
        if registry_path.exists():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            # registry is a bare list of {"name", "path", ...} objects
            repos = registry if isinstance(registry, list) else registry.get("repos", [])
            for repo in repos:
                repo_path = repo.get("path", "")
                repo_name = repo.get("name", "")
                if repo_path and Path(repo_path).exists():
                    file_count = repo.get("stats", {}).get("files", 0)
                    _tracked[repo_path] = {
                        "type": "gitnexus",
                        "name": repo_name,
                        "path": repo_path,
                        "file_count": file_count,
                        "last_sync": 0.0,
                    }
                    logger.debug(
                        "indexer-sync: tracking gitnexus '%s' at %s",
                        repo_name, repo_path,
                    )
    except Exception as e:
        logger.debug("indexer-sync: gitnexus registry scan failed: %s", e)


def _extract_paths(tool_name: str, args: dict, result: Any = None) -> set[str]:
    """Extract file paths from tool call args."""
    paths = set()
    if tool_name not in _WRITE_TOOL_NAMES:
        return paths

    # patch/write_file/read_file/search_files → "path" arg
    for key in ("path", "workdir"):
        val = args.get(key)
        if val and isinstance(val, str):
            paths.add(val)

    # terminal/execute_code → scan for absolute paths in command/code strings
    for key in ("command", "code"):
        val = args.get(key)
        if val and isinstance(val, str):
            # Match /Users/.../something patterns
            found = re.findall(r"/Users/stan/[^\s\'\"<>|;&]+", val)
            paths.update(found)

    return paths


def _match_repo(file_path: str) -> list[str]:
    """Match a file path to tracked repo root(s). A file can be in multiple
    repos if nested (rare). Returns repo root keys."""
    matches = []
    for root_key in _tracked:
        if file_path.startswith(root_key + "/") or file_path == root_key:
            matches.append(root_key)
    return matches


def _do_sync(root_key: str):
    """Run the actual sync command for a repo root."""
    info = _tracked.get(root_key)
    if info is None:
        return
    root_type = info["type"]

    # Rate limit: skip if synced recently
    last = info.get("last_sync", 0.0)
    if last and (time.time() - last) < _MIN_RESYNC_INTERVAL:
        logger.debug("indexer-sync: skipping %s (synced %.0fs ago)", root_key, time.time() - last)
        return

    # Pick timeout by repo size
    file_count = info.get("file_count", 0)
    timeout = _SYNC_TIMEOUT_LARGE if file_count >= 500 else _SYNC_TIMEOUT_SMALL

    try:
        if root_type == "codegraph":
            cmd = info["sync_cmd"]
            logger.info("indexer-sync: codegraph sync %s (%d files, timeout=%ds)", root_key, file_count, timeout)
            subprocess.run(
                cmd, timeout=timeout,
                capture_output=True, text=True,
            )
        elif root_type == "gitnexus":
            repo_path = info["path"]
            cmd = ["gitnexus", "analyze", repo_path, "--skip-agents-md"]
            logger.info("indexer-sync: gitnexus analyze %s (%d files, timeout=%ds)", repo_path, file_count, timeout)
            subprocess.run(
                cmd, timeout=timeout,
                capture_output=True, text=True,
            )
        info["last_sync"] = time.time()
    except subprocess.TimeoutExpired:
        logger.warning(
            "indexer-sync: sync timed out for %s (>%ds)",
            root_key, timeout,
        )
    except FileNotFoundError:
        logger.debug(
            "indexer-sync: CLI not found for %s — skipping", root_key,
        )
    except Exception as e:
        logger.debug("indexer-sync: sync failed for %s: %s", root_key, e)


def _debounced_sync(root_key: str):
    """Debounce multiple rapid triggers into one sync per window."""
    with _lock:
        state = _debounce.setdefault(root_key, {
            "last_trigger": 0.0, "worker": None,
        })
        state["last_trigger"] = time.time()

        existing = state.get("worker")
        if existing is not None and existing.is_alive():
            return  # pending worker will pick up the latest trigger

        def _run():
            time.sleep(_DEBOUNCE_SEC)
            # Re-check: only sync if this was the latest trigger
            with _lock:
                t = _debounce.get(root_key, {})
                if t.get("last_trigger", 0) and (
                    time.time() - t["last_trigger"] < _DEBOUNCE_SEC
                ):
                    # Reset and wait again (burst still in progress)
                    pass
                _debounce[root_key] = {"last_trigger": 0.0, "worker": None}

            _do_sync(root_key)

        worker = threading.Thread(target=_run, daemon=True)
        state["worker"] = worker


def register(ctx):
    """Plugin entry point."""
    _discover_tracked_repos()
    logger.info(
        "indexer-sync: monitoring %d repos (%d codegraph, %d gitnexus)",
        len(_tracked),
        sum(1 for v in _tracked.values() if v["type"] == "codegraph"),
        sum(1 for v in _tracked.values() if v["type"] == "gitnexus"),
    )

    # Initial sync on load (backlog from stale indices)
    for root_key in list(_tracked.keys()):
        info = _tracked[root_key]
        staleness = time.time() - info.get("last_sync", 0)
        if staleness > 3600:  # only if last sync > 1h ago (or never)
            _debounced_sync(root_key)

    def _on_post_tool_call(
        tool_name: str = "",
        args: dict | None = None,
        result: Any = None,
        **kwargs,
    ):
        if not _tracked:
            return
        if args is None:
            args = {}

        paths = _extract_paths(tool_name, args, result)
        if not paths:
            return

        triggered = set()
        for p in paths:
            matched_roots = _match_repo(p)
            triggered.update(matched_roots)

        for root_key in triggered:
            _debounced_sync(root_key)

    ctx.register_hook("post_tool_call", _on_post_tool_call)