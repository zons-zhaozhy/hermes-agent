"""Curator snapshot + rollback. Before any mutating curator pass, ``~/.hermes/skills/`` is tar.gz'd under
``~/.hermes/skills/.curator_backups/<utc-iso>/`` with a ``manifest.json``. Rollback first snapshots the CURRENT tree (so it is
itself undoable), then extracts the chosen snapshot into place. Excluded: ``.curator_backups/``, ``.hub/`` (hub-managed), ``.git/``.
Included: skill dirs, ``.usage.json``, ``.archive/``, ``.curator_state`` (so rollback also restores last-run-at and the curator
doesn't re-fire), ``.bundled_manifest``, ``.curator_suppressed``. Each snapshot also copies ``~/.hermes/cron/jobs.json`` as
``cron-jobs.json``: the consolidation pass rewrites cron ``skills``/``skill`` references in place, so rollback restores those two
fields (only) — the rest is live state."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import tarfile
from datetime import datetime, timezone, UTC
from itertools import chain, count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from agent.skill_utils import is_excluded_skill_path
from agent.curator import _read_config_section
from hermes_cli.sizefmt import format_bytes

logger = logging.getLogger(__name__)

DEFAULT_KEEP = 2

# Never rolled into a snapshot: .hub/ is owned by the skills hub (rolling it back breaks lockfile invariants); .curator_backups
# is the backup dir itself; .git is repository metadata — rolling it back breaks git tracking, and snapshots that include it grow
# with the full history (once backups are committed back, each snapshot contains the prior ones: 38MB of skills inflated to 24GB
# in weeks); .locks holds skill_manage's per-skill lock files — restoring them would swap a lock out from under a waiting
# writer. The tar filter in ``snapshot_skills`` applies the same set to nested paths, so a nested ``.git`` is skipped too.
# See #91449. ``.curator_ledger.jsonl`` is the append-only audit log and ``.archive/`` the recoverable store the curator
# promises never to delete: rolling either back to an older copy LOSES entries/skills, and both grow without bound (a 650MB
# ledger made every snapshot 820MB — and every archive step gunzips the newest snapshot in full, so a pass that pruned 57
# skills held the CLI prompt for 6 minutes).
_EXCLUDE_TOP_LEVEL = {".curator_backups", ".hub", ".locks", ".git", ".archive", ".curator_ledger.jsonl"}


def _excluded_member(ti: tarfile.TarInfo) -> bool:
    """Skip excluded names anywhere in the path, plus regeneratable DIRECTORIES (venv, node_modules,
    caches — a 1.3 GB torch venv inside one skill made every snapshot 349 MB, #107539). Directory
    parts only: a plain file that happens to be called ``venv`` is skill content."""
    from tools.skill_ledger import TRANSIENT_DIRS

    parts = Path(ti.name).parts
    dir_parts = parts if ti.isdir() else parts[:-1]
    return any(p in _EXCLUDE_TOP_LEVEL for p in parts) or any(p in TRANSIENT_DIRS for p in dir_parts)

# Snapshot id: UTC ISO with colons replaced by dashes (Windows-safe filename); optional ``-NN`` suffix for same-second snapshots.
_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z(-\d{2})?$")

CRON_JOBS_FILENAME = "cron-jobs.json"
_ARCHIVE_NAME = "skills.tar.gz"
_STAGING_PREFIX = ".rollback-staging-"
_UNRESTORED_PREFIX = ".rollback-unrestored-"  # retained staging; outside _STAGING_PREFIX so _prune_old keeps it


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _backups_dir() -> Path:
    return _skills_dir() / ".curator_backups"


def _jobs_list(parsed: Any) -> Optional[list]:
    """jobs.json is ``{"jobs": [...], "updated_at": ...}``; also accept a bare list for forward compat. None otherwise."""
    parsed = parsed.get("jobs") if isinstance(parsed, dict) else parsed
    return parsed if isinstance(parsed, list) else None


def _backup_cron_jobs_into(dest: Path) -> dict[str, Any]:
    """Copy the live ``~/.hermes/cron/jobs.json`` into ``dest`` as ``cron-jobs.json``. Never raises: a missing/unreadable
    file yields ``backed_up=False`` plus a reason, and the snapshot proceeds."""
    src = get_hermes_home() / "cron" / "jobs.json"
    info: dict[str, Any] = {"backed_up": False, "jobs_count": 0}
    if not src.exists():
        return {**info, "reason": "no cron/jobs.json present"}
    try:
        # utf-8-sig, same dialect as cron/jobs.load_jobs: a Windows-editor BOM would otherwise break json.loads
        # AND be written into the backup.
        raw = src.read_text(encoding="utf-8-sig")
    except OSError as e:
        logger.debug("Failed to read cron/jobs.json for backup: %s", e)
        return {**info, "reason": f"read error: {e}"}
    try:  # jobs_count is a diagnostic only — an unparseable file is still stored raw.
        info["jobs_count"] = len(_jobs_list(json.loads(raw)) or [])
    except (json.JSONDecodeError, TypeError):
        info["parse_warning"] = "jobs.json was not valid JSON at snapshot time"
    try:
        (dest / CRON_JOBS_FILENAME).write_text(raw, encoding="utf-8")
    except OSError as e:
        logger.debug("Failed to write cron backup file: %s", e)
        return {**info, "reason": f"write error: {e}"}
    return {**info, "backed_up": True}


def _utc_id(now: Optional[datetime] = None) -> str:
    """UTC ISO-ish filesystem-safe timestamp: ``2026-05-01T13-05-42Z``."""
    s = (datetime.now(UTC) if now is None else now).replace(microsecond=0).isoformat()
    return s.removesuffix("+00:00").replace(":", "-") + "Z"


def _load_config() -> dict[str, Any]:
    return _read_config_section("curator", "backup", label="curator backup", log=logger)


def _unusable_exclude_reason(entry: str) -> str:
    """Why *entry* can never match a snapshot member — empty string when it can.

    Entries match by literal skills-relative path prefix (``_matches_exclude_prefix``) and are never
    read as globs, so a pattern-flavoured entry, an absolute path or a parent-relative path matches
    nothing and would silently leave the runtime dirs in every snapshot — the exact failure this
    setting exists to prevent. Validation runs on the raw entry: stripping a leading ``/`` first
    would turn a rejected absolute path into a plausible-looking relative one.
    """
    if any(char in entry for char in "*?["):
        return "glob patterns are not supported; use a literal skills-relative path"
    if entry.startswith("/"):
        return "absolute paths are not supported; use a skills-relative path"
    if entry.startswith("~"):
        return "home-relative paths are not supported; use a skills-relative path"
    parts = entry.split("/")
    if ".." in parts:
        return "parent-relative paths are not supported; use a skills-relative path"
    if not entry.strip("/"):
        return "empty path"
    if any(part == "" for part in parts[:-1]):
        return "contains an empty path segment; use a single-slash skills-relative path"
    return ""


def _exclude_raw_entries(raw: Any) -> list[Any]:
    """``curator.backup.exclude`` raw config value → candidate entries (never ``None``).

    One comma/newline separated string is tolerated because the CLI hands it in that way; anything
    that is neither string nor sequence is reported instead of being coerced into one.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        chunks = raw.replace(",", "\n").split("\n")
        return [chunk for chunk in chunks if chunk.strip()]
    if not isinstance(raw, (list, tuple)):
        logger.warning("curator.backup.exclude must be a list of skills-relative paths; ignoring %r", raw)
        return []
    return list(raw)


def _exclude_entry_to_prefix(item: Any) -> str:
    """One exclude entry → skills-relative prefix, or ``""`` when it can never match.

    Each rejection is logged with its reason here, so the caller stays a plain aggregation loop and
    no entry is ever dropped silently.
    """
    if not isinstance(item, str) or not item.strip():
        logger.warning("curator.backup.exclude entry %r is not a path; ignoring it", item)
        return ""
    stripped = item.strip()
    reason = _unusable_exclude_reason(stripped)
    if reason:
        logger.warning("curator.backup.exclude entry %r is unusable (%s); ignoring it", item, reason)
        return ""
    return stripped.strip("/")


def _exclude_prefixes() -> tuple[str, ...]:
    """``curator.backup.exclude``: skills-relative paths that must never enter a snapshot.

    A skill directory that doubles as a runtime workspace (build output, virtualenv, code index,
    logs) otherwise lands in every snapshot and ``keep`` multiplies that cost. Entries are paths
    relative to the skills root (``"ontox-ops/state"``) so an unrelated skill holding a same-named
    directory stays out of scope. Unusable entries are dropped with a warning, never guessed at.
    """
    prefixes: list[str] = []
    for item in _exclude_raw_entries(_load_config().get("exclude", [])):
        prefix = _exclude_entry_to_prefix(item)
        if prefix and prefix not in prefixes:
            prefixes.append(prefix)
    if prefixes:
        logger.debug("curator.backup.exclude active prefixes: %s", ", ".join(prefixes))
    return tuple(prefixes)


def _matches_exclude_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    """True when a snapshot member path IS a configured prefix or lives under it."""
    if not prefixes:
        return False
    path = name.strip("/")
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def is_enabled() -> bool:
    """Default ON — the whole point of the backup is safety by default."""
    return bool(_load_config().get("enabled", True))


def get_keep() -> int:
    try:
        return max(1, int(_load_config().get("keep", DEFAULT_KEEP)))
    except (TypeError, ValueError):
        return DEFAULT_KEEP


# --- Snapshot ---
def _count_skill_files(base: Path) -> int:
    try:
        return sum(1 for p in base.rglob("SKILL.md") if not is_excluded_skill_path(p))
    except OSError:
        return 0


def _write_manifest(dest: Path, reason: str, archive_path: Path, skills_counted: int, cron_info: dict[str, Any]) -> None:
    cron_jobs: dict[str, Any] = {"backed_up": bool(cron_info.get("backed_up", False)), "jobs_count": int(cron_info.get("jobs_count", 0))}
    if not cron_info.get("backed_up"):
        cron_jobs["reason"] = cron_info.get("reason", "not captured")
    if cron_info.get("parse_warning"):
        cron_jobs["parse_warning"] = cron_info["parse_warning"]
    manifest = {"id": dest.name, "reason": reason, "created_at": datetime.now(UTC).isoformat(), "archive": archive_path.name,
                "archive_bytes": archive_path.stat().st_size, "skill_files": skills_counted, "cron_jobs": cron_jobs}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _mkdir(path: Path, what: str, *, exist_ok: bool) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
        return True
    except OSError as e:
        logger.debug("Failed to create %s %s: %s", what, path, e)
        return False


def snapshot_skills(reason: str = "manual", *, protect_ids: Optional[set[str]] = None) -> Optional[Path]:
    """Create a tar.gz snapshot of ``~/.hermes/skills/`` and prune old ones. Returns the snapshot dir, or None when
    skipped (disabled, skills dir missing, IO error) — logged at debug so the curator never aborts a pass over a
    backup failure. ``protect_ids`` survive the prune step (rollback protects its target)."""
    if not is_enabled():
        logger.debug("Curator backup disabled by config; skipping snapshot")
        return None
    skills, backups = _skills_dir(), _backups_dir()
    if not skills.exists():
        logger.debug("No ~/.hermes/skills/ directory — nothing to back up")
        return None
    if not _mkdir(backups, "backups dir", exist_ok=True):
        return None

    base_id = _utc_id()  # Two curator runs in the same second must not clobber each other: -NN suffix.
    snap_id = next(i for i in chain([base_id], (f"{base_id}-{n:02d}" for n in count(1))) if not (backups / i).exists())
    dest = backups / snap_id
    if not _mkdir(dest, "snapshot dir", exist_ok=False):
        return None

    archive = dest / _ARCHIVE_NAME
    exclude_prefixes = _exclude_prefixes()

    def _keep_member(ti: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        """Snapshot filter: drop hardcoded exclusions and configured-away runtime paths."""
        if _excluded_member(ti) or _matches_exclude_prefix(ti.name, exclude_prefixes):
            return None
        return ti

    try:
        with tarfile.open(archive, "w:gz", compresslevel=6) as tf:
            for entry in sorted(skills.iterdir()):
                if entry.name in _EXCLUDE_TOP_LEVEL:
                    continue
                # arcname relative to skills/ so extraction drops back in cleanly; the filter excludes nested paths too.
                tf.add(str(entry), arcname=entry.name, recursive=True, filter=_keep_member)
        # Cron capture is additive and never fails the snapshot; the manifest records whether it happened so rollback can say "no cron data".
        _write_manifest(dest, reason, archive, _count_skill_files(skills), _backup_cron_jobs_into(dest))
    except (OSError, tarfile.TarError) as e:
        logger.debug("Curator snapshot failed: %s", e, exc_info=True)
        shutil.rmtree(dest, ignore_errors=True)  # clean up partial snapshot
        return None

    # A same-second id reuse after a prune (`...Z` next to a surviving `...Z-02`) sorts BELOW its sibling;
    # the snapshot just written must never be its own prune victim.
    _prune_old(keep=get_keep(), protect=(protect_ids or set()) | {snap_id})
    logger.info("Curator snapshot created: %s (%s)", snap_id, reason)
    return dest


def prune_old_snapshots() -> list[str]:
    """Apply ``curator.backup.keep`` without taking a new snapshot (the prune-only curator pass)."""
    return _prune_old(keep=get_keep())


def _prune_old(keep: int, protect: Optional[set[str]] = None) -> list[str]:
    """Delete regular snapshots beyond the newest *keep*; returns deleted ids. Ids in *protect* are never deleted —
    rollback() uses this so the mandatory pre-rollback safety snapshot cannot evict the snapshot being restored.
    Stale ``.rollback-staging-*`` dirs (crashed rollback) are cleaned up on every call; ``.rollback-unrestored-*`` is kept on purpose."""
    protect = protect or set()
    backups = _backups_dir()
    if not backups.exists():
        return []
    dirs = [c for c in backups.iterdir() if c.is_dir()]
    # Newest first (lexicographic works because the id is UTC ISO).
    entries = sorted((c for c in dirs if _ID_RE.match(c.name)), key=lambda c: c.name, reverse=True)
    doomed = [(p, "prune") for p in entries[keep:] if p.name not in protect]
    doomed += [(p, "clean stale staging dir") for p in dirs if p.name.startswith(_STAGING_PREFIX)]
    deleted: list[str] = []
    for path, what in doomed:
        try:
            shutil.rmtree(path)
            if what == "prune":
                deleted.append(path.name)
        except OSError as e:
            logger.debug("Failed to %s %s: %s", what, path, e)
    return deleted


# --- List + rollback ---
def _read_manifest(snap_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _is_restorable(child: Path) -> bool:
    """A real snapshot dir with a tarball (excludes ``.rollback-staging-*`` and retained ``.rollback-unrestored-*``)."""
    return bool(child.is_dir() and _ID_RE.match(child.name) and (child / _ARCHIVE_NAME).exists())


def _restorable_snapshots() -> list[Path]:
    """Restorable snapshot dirs, newest first."""
    backups = _backups_dir()
    return [c for c in sorted(backups.iterdir(), reverse=True) if _is_restorable(c)] if backups.exists() else []


def list_backups() -> list[dict[str, Any]]:
    """All restorable snapshots (manifest dicts), newest first."""
    out: list[dict[str, Any]] = []
    for child in _restorable_snapshots():
        mf = {"id": child.name, "path": str(child), **_read_manifest(child)}
        try:
            mf.setdefault("archive_bytes", (child / _ARCHIVE_NAME).stat().st_size)
        except OSError:
            mf.setdefault("archive_bytes", 0)
        out.append(mf)
    return out


def _resolve_backup(backup_id: Optional[str]) -> Optional[Path]:
    """Path of the requested backup (newest if *backup_id* is None); None if no match."""
    if backup_id:
        target = _backups_dir() / backup_id
        return target if _ID_RE.match(backup_id) and _is_restorable(target) else None
    return next(iter(_restorable_snapshots()), None)


def _restore_cron_skill_links(snapshot_dir: Path) -> dict[str, Any]:
    """Reconcile backed-up cron skill links into the live ``cron/jobs.json``. Only ``skills``/``skill`` are restored,
    and only on jobs that still exist live (by ``id``) — everything else is live state. Backup-only jobs are skipped
    and reported; live-only jobs untouched. Never raises; writes through ``cron.jobs`` under the scheduler's lock so
    we don't race tick()."""
    report: dict[str, Any] = {"attempted": False, "restored": [], "skipped_missing": [], "unchanged": 0, "error": None}
    backup_file = snapshot_dir / CRON_JOBS_FILENAME
    if not backup_file.exists():
        return {**report, "error": f"snapshot has no {CRON_JOBS_FILENAME}"}
    try:
        backup_jobs = _jobs_list(json.loads(backup_file.read_text(encoding="utf-8-sig")))
    except (OSError, json.JSONDecodeError) as e:
        return {**report, "error": f"failed to load backed-up jobs: {e}"}
    if backup_jobs is None:
        return {**report, "error": "backed-up cron-jobs.json has no jobs list"}

    # Backed-up skill state keyed by job id (legacy single + modern list field).
    backup_by_id: dict[str, dict[str, Any]] = {
        job["id"]: {"skills": job.get("skills"), "skill": job.get("skill"), "name": job.get("name") or job["id"]}
        for job in backup_jobs if isinstance(job, dict) and isinstance(job.get("id"), str) and job.get("id")
    }
    if not backup_by_id:
        return {**report, "attempted": True}  # we tried but there was nothing to do
    try:
        from cron.jobs import load_jobs, save_jobs, _jobs_lock
    except ImportError as e:
        return {**report, "error": f"cron module unavailable: {e}"}

    report["attempted"] = True
    try:
        with _jobs_lock():
            live_jobs = load_jobs()
            changed, live_ids = False, set()
            for live in live_jobs:
                jid = live.get("id") if isinstance(live, dict) else None
                if not isinstance(jid, str) or not jid:
                    continue
                live_ids.add(jid)
                backup = backup_by_id.get(jid)
                if backup is None:  # live job didn't exist at snapshot time
                    continue
                cur = {"skills": live.get("skills"), "skill": live.get("skill")}
                bkp = {"skills": backup.get("skills"), "skill": backup.get("skill")}
                if cur == bkp:
                    report["unchanged"] += 1
                    continue
                for key, value in bkp.items():  # Restore, preserving absence (don't add a key the backup lacked).
                    if value is None:
                        live.pop(key, None)
                    else:
                        live[key] = value
                report["restored"].append({"job_id": jid, "job_name": backup.get("name") or jid, "from": cur, "to": bkp})
                changed = True

            # Jobs in backup but not live = user deleted them after the snapshot.
            report["skipped_missing"] = [{"job_id": jid, "job_name": b.get("name") or jid}
                                         for jid, b in backup_by_id.items() if jid not in live_ids]
            if changed:
                save_jobs(live_jobs)
    except Exception as e:
        logger.debug("Cron skill-link restore failed: %s", e, exc_info=True)
        report["error"] = f"restore failed mid-flight: {e}"
    return report


def _remove_entry(entry: Path) -> None:
    if entry.is_dir() and not entry.is_symlink():
        shutil.rmtree(entry)
    elif entry.exists() or entry.is_symlink():
        entry.unlink()


def _rel_name(rel: Path, name: str) -> str:
    """Staging-relative path of *name* directly under *rel* (``rel`` is ``.`` at the staging root)."""
    return name if rel == Path(".") else f"{rel}/{name}"


def _is_carried_over(rel_path: str, name: str, prefixes: tuple[str, ...]) -> bool:
    """True when the entry at staging-relative *rel_path* (basename *name*) must be carried back.

    The hardcoded exclusions match a basename at any depth (``alpha/venv``, ``alpha/.git``), while
    ``curator.backup.exclude`` prefixes are matched against the skills-relative path because that is
    how the snapshot filter reads them. Both kinds are absent from the snapshot, so the staged live
    tree holds the only copy and dropping it with the staging dir loses it silently.
    """
    from tools.skill_ledger import TRANSIENT_DIRS

    if name in _EXCLUDE_TOP_LEVEL or name in TRANSIENT_DIRS:
        return True
    return _matches_exclude_prefix(rel_path, prefixes)


def _restore_excluded_subtrees(staged: Path, skills: Path, prefixes: tuple[str, ...]) -> list[str]:
    """Move snapshot-excluded entries from *staged* back under *skills* after a successful extract.

    Snapshots never contain these paths (``_EXCLUDE_TOP_LEVEL``, ``TRANSIENT_DIRS``, and every
    ``curator.backup.exclude`` prefix), so the staged copy of the live tree is the only source.
    ``.git`` may be a dir or a file (submodule / worktree ``gitdir:`` pointer) — both are moved.
    Best-effort and conditional: an entry is carried only when its parent skill dir was restored and
    nothing sits at the target. If the target snapshot predates the skill, the entry is dropped with
    the staging dir rather than left orphaned. Returns entries whose move failed, so their only
    surviving copies stay staged for recovery.
    """
    failed: list[str] = []
    for dirpath, dirnames, filenames in os.walk(staged):
        base, rel = Path(dirpath), Path(dirpath).relative_to(staged)
        carried = [base / n for n in filenames if _is_carried_over(_rel_name(rel, n), n, prefixes)]
        carried += [base / n for n in dirnames if _is_carried_over(_rel_name(rel, n), n, prefixes)]
        for src in carried:
            dest = skills / src.relative_to(staged)
            if dest.parent.is_dir() and not dest.exists():
                try:
                    shutil.move(str(src), str(dest))
                except OSError as e:
                    failed.append(str(src.relative_to(staged)))
                    logger.warning("Could not restore excluded entry %s: %s", src, e)
        dirnames[:] = [d for d in dirnames if not _is_carried_over(_rel_name(rel, d), d, prefixes)]
    return failed


def _retain_staging(staged: Path, unrestored: list[str], reason: str) -> tuple[bool, str, None]:
    """Keep a staging dir that still holds the only copy of *unrestored* entries and build rollback()'s failure result.
    The dir is renamed out of the prunable prefix (unique suffix on a same-second collision). If the rename fails, the
    original path is reported so recovery never raises out of rollback()."""
    base = _UNRESTORED_PREFIX + staged.name[len(_STAGING_PREFIX):]
    kept, n = staged.with_name(base), 1
    while kept.exists():
        kept, n = staged.with_name(f"{base}-{n}"), n + 1
    note = ""
    try:
        staged.rename(kept)
    except OSError as e:
        logger.warning("Could not retain rollback staging dir %s: %s", staged, e)
        kept, note = staged, f" (rename failed: {e}; move it before the next curator run or it will be pruned)"
    return (False, f"{reason} - could not restore {', '.join(sorted(unrestored))}; staged copies kept at {kept}{note}", None)


def _unstage(moved: list[tuple[Path, Path]]) -> list[str]:
    """Move staged entries back to their original paths; returns names that could not be restored. ``shutil.move``
    moves *into* an existing destination dir, so partial-extract debris would bury the real skill
    (``skills/foo/foo/``) — clear each original path first. The staged copy is authoritative."""
    failed: list[str] = []
    for orig, dest in moved:
        try:
            _remove_entry(orig)
            shutil.move(str(dest), str(orig))
        except OSError:
            failed.append(orig.name)
    return failed


def _cron_summary(cron_report: dict[str, Any]) -> Optional[str]:
    if not cron_report.get("attempted"):
        return None
    if cron_report.get("error"):
        return f"cron links: error — {cron_report['error']}"
    # (attempted with nothing matched — empty snapshot or no overlapping ids — says nothing)
    parts = [f"{n} {label}" for n, label in (
        (len(cron_report.get("restored") or []), "job(s) had skill links restored"),
        (len(cron_report.get("skipped_missing") or []), "backed-up job(s) no longer exist (skipped)"),
        (cron_report.get("unchanged", 0), "already matched"),
    ) if n]
    return "cron links: " + ", ".join(parts) if parts else None


def rollback(backup_id: Optional[str] = None) -> tuple[bool, str, Optional[Path]]:
    """Restore ``~/.hermes/skills/`` from a snapshot (explicit id or newest): safety-snapshot the CURRENT tree; stage
    current top-level entries; extract; on failure move staged entries back. Returns ``(ok, message, snapshot_path)``."""
    target = _resolve_backup(backup_id)
    if target is None:
        return (False, "no matching backup found" + (f" for id '{backup_id}'" if backup_id else "")
                + " (use `hermes curator rollback --list` to see available snapshots)", None)
    archive = target / _ARCHIVE_NAME
    if not archive.exists():
        return (False, f"snapshot {target.name} has no skills.tar.gz — corrupted?", None)

    skills, backups = _skills_dir(), _backups_dir()
    backups.mkdir(parents=True, exist_ok=True)  # parents=True also creates skills/

    # Safety snapshot FIRST; bail if it fails, else a failed extract could leave the user with no skills. Protect the target from its prune.
    try:
        safety_snapshot = snapshot_skills(reason=f"pre-rollback to {target.name}", protect_ids={target.name})
    except Exception as e:
        return (False, f"pre-rollback safety snapshot failed: {e}", None)
    if safety_snapshot is None:
        return (False, "pre-rollback safety snapshot failed; backups may be disabled "
                "or unavailable, and current skills were not changed", None)

    # Stage current entries so the extract lands in an empty tree; the safety snapshot above (not staging) is the user-facing undo handle.
    staged = backups / f"{_STAGING_PREFIX}{_utc_id()}"
    try:
        staged.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        return (False, f"failed to create staging dir: {e}", None)

    moved: list[tuple[Path, Path]] = []
    try:
        for entry in list(skills.iterdir()):
            if entry.name not in _EXCLUDE_TOP_LEVEL:
                shutil.move(str(entry), str(staged / entry.name))
                moved.append((entry, staged / entry.name))
    except OSError as e:
        unrestored = _unstage(moved)
        if unrestored:
            return _retain_staging(staged, unrestored, f"failed to stage current skills: {e}")
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"failed to stage current skills: {e}", None)

    try:
        with tarfile.open(archive, "r:gz") as tf:
            # Reject absolute paths and ".." defensively; Python 3.12+ also gets filter='data', older interpreters fall back unfiltered.
            for member in tf.getmembers():
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise tarfile.TarError(f"refusing to extract unsafe path: {member.name!r}")
            try:
                tf.extractall(str(skills), filter="data")  # type: ignore[call-arg]
            except TypeError:
                tf.extractall(str(skills))  # Python < 3.12 — no filter kwarg
    except (OSError, tarfile.TarError) as e:
        # A partial extract can leave entries the original tree never had; drop those first or the "restored" tree is skills + a slice of snapshot.
        keep = _EXCLUDE_TOP_LEVEL | {orig.name for orig, _ in moved}
        for entry in [e for e in skills.iterdir() if e.name not in keep]:
            with contextlib.suppress(OSError):
                _remove_entry(entry)
        unrestored = _unstage(moved)
        if unrestored:  # Don't claim a clean restore; keep the staging dir for hand recovery.
            return _retain_staging(staged, unrestored, f"snapshot extract failed: {e}")
        shutil.rmtree(staged, ignore_errors=True)
        return (False, f"snapshot extract failed (state restored): {e}", None)

    # Snapshots never contain excluded subtrees (nested ``.git``, ``.hub``, ...), so carry them over from the staged live tree
    # (top-level ``.git`` is never staged). Then staging is done; the undo handle is the safety snapshot.
    unrestored = _restore_excluded_subtrees(staged, skills, _exclude_prefixes())
    if unrestored:
        return _retain_staging(staged, unrestored, "snapshot extracted but excluded entries were not carried over")
    shutil.rmtree(staged, ignore_errors=True)

    # Cron reconciliation failures don't fail the rollback — the skills tree (the main guarantee) is already restored.
    cron_report = _restore_cron_skill_links(target)
    logger.info("Curator rollback: restored from %s (cron_report=%s)", target.name, cron_report)
    return (True, "; ".join(filter(None, [f"restored from snapshot {target.name}", _cron_summary(cron_report)])), target)


# --- Human-readable summary for CLI ---
def summarize_backups() -> str:
    rows = list_backups()
    if not rows:
        return "No curator snapshots yet."
    header = f"{'id':<24}  {'reason':<40}  {'skills':>6}  {'size':>8}"
    return "\n".join([header, "─" * len(header)] + [
        f"{r.get('id','?'):<24}  {(r.get('reason','?') or '?')[:40]:<40}  "
        f"{r.get('skill_files', 0):>6}  {format_bytes(int(r.get('archive_bytes', 0))):>8}" for r in rows])
