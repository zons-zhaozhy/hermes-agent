#!/usr/bin/env python3
"""Skill SHA drift detector — checks if source files referenced by skills have changed.

Detects stale skills by comparing cached source-file SHAs against current SHAs.
Outputs a triage report (new/changed/deleted) so the reviewer can update
skills in one pass.

Usage:
    python scripts/skill_sha_drift.py              # check all skills
    python scripts/skill_sha_drift.py --skill foo  # check one skill
    python scripts/skill_sha_drift.py --update     # update cache after review

Cache file: ~/.hermes/cache/skill_sha_baseline.json
    Format: {"skill_path": {"/abs/path/to/source.py": "sha_hex", ...}, ...}
"""

import argparse
import json
import logging
import re  # noqa: scanner tool — regex is essential
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SKILLS_ROOT = Path.home() / ".hermes" / "skills"
CACHE_FILE = Path.home() / ".hermes" / "cache" / "skill_sha_baseline.json"

# Pattern to find file references in skill content.
# Matches:
#   /Users/stan/code/.../file.py
#   relative paths in code blocks: file.py, path/to/file.py
#   hermes-agent specific: agent/foo.py, tools/bar.py, gateway/baz.py
_FILE_REF_PATTERN = re.compile(
    r"(?:"
    r"(?:/[\w/.-]+\.(?:py|ts|tsx|js|yaml|yml|json|md|sh|toml))"  # absolute paths
    r"|(?:\b(?:agent|tools|gateway|hermes_cli|ui-tui|tui_gateway|plugins|cron|acp_adapter|tests|skills)/(?:[\w/.-]+\.(?:py|ts|tsx|js|yaml|yml|json|sh|toml)))"  # hermes-specific relative
    r"|(?:\b[\w][\w./-]*\.(?:py|ts|tsx|js|yaml|yml|json|sh|toml)\b)"  # any file.ext
    r")"
)


def git_sha(file_path: Path) -> Optional[str]:
    """Get the git SHA of a file. Returns None if not tracked in git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD:", str(file_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = result.stdout.strip().split("\n")[0]
        # Must be a valid 40-char hex SHA (git rev-parse outputs error
        # messages to stdout for untracked paths — e.g. "HEAD:/tmp/foo")
        if result.returncode == 0 and len(sha) == 40 and all(
            c in "0123456789abcdef" for c in sha
        ):
            return sha
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def resolve_path(ref: str, skill_dir: Path) -> Optional[Path]:
    """Try to resolve a file reference to an absolute path within the project git repo.

    Only resolves paths inside the git repo found by walking up from cwd.
    Skips files inside ~/.hermes/skills/ (skill-internal scripts are
    part of the skill, not external source references).
    """
    # Absolute path — only resolve if inside a git repo (and not in skills dir)
    p = Path(ref)
    if p.is_absolute():
        if _is_in_project_git(p) and p.exists() and not _is_in_skills_dir(p):
            return p
        return None

    # Relative to skill directory — skip (skill-internal files)
    candidate = skill_dir / ref
    if candidate.exists() and _is_in_skills_dir(candidate):
        return None

    # Relative to project git root (if we're in one)
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".git").exists():
            candidate = parent / ref
            if candidate.exists() and not _is_in_skills_dir(candidate):
                return candidate
            break

    return None


def _is_in_project_git(file_path: Path) -> bool:
    """Check if a file is inside the project git repo (found from cwd)."""
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".git").exists():
            try:
                file_path.resolve().relative_to(parent.resolve())
                return True
            except ValueError:
                logger.warning("Path %s not in git repo %s", file_path, parent)
                return False
    return False


def _is_in_skills_dir(file_path: Path) -> bool:
    """Check if a file is inside ~/.hermes/skills/."""
    skills = Path.home() / ".hermes" / "skills"
    try:
        file_path.resolve().relative_to(skills.resolve())
        return True
    except ValueError:
        logger.warning("Path %s not in skills dir", file_path)
        return False


def extract_file_refs(content: str, skill_dir: Path) -> List[Path]:
    """Extract file references from skill content, resolve to absolute paths."""
    refs = set(_FILE_REF_PATTERN.findall(content))
    resolved = []
    for ref in refs:
        p = resolve_path(ref, skill_dir)
        if p is not None:
            resolved.append(p)
    return resolved


def load_cache() -> Dict[str, Dict[str, str]]:
    """Load the SHA baseline cache."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read cache %s, starting fresh", CACHE_FILE)
    return {}


def save_cache(cache: Dict[str, Dict[str, str]]) -> None:
    """Save the SHA baseline cache."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def check_skill(
    skill_dir: Path,
    baseline: Dict[str, Dict[str, str]],
) -> Tuple[List[str], List[str], List[str]]:
    """Check one skill for drift. Returns (new_files, changed_files, deleted_files)."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [], [], []

    content = skill_md.read_text(encoding="utf-8")

    # Also scan references/ directory
    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for f in refs_dir.glob("*.md"):
            try:
                content += "\n" + f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.warning("Failed to read %s, skipping", f)

    current_files = extract_file_refs(content, skill_dir)
    cached = baseline.get(str(skill_dir), {})

    new_files = []
    changed_files = []
    deleted_files = []

    current_shas: Dict[str, str] = {}

    for f in current_files:
        f_str = str(f)
        sha = git_sha(f)
        if sha is None:
            # Not in git repo — skip (external files)
            continue
        current_shas[f_str] = sha

        if f_str not in cached:
            new_files.append(f_str)
        elif cached[f_str] != sha:
            changed_files.append(f_str)

    for f_str in cached:
        if f_str not in current_shas:
            deleted_files.append(f_str)

    # Update cache with current state
    if current_shas:
        baseline[str(skill_dir)] = current_shas

    return new_files, changed_files, deleted_files


def check_all_skills(baseline: Dict[str, Dict[str, str]]) -> int:
    """Check all skills for drift. Returns count of skills with changes."""
    total_changes = 0

    # Collect all skill directories
    skill_dirs = []
    if SKILLS_ROOT.exists():
        for md in SKILLS_ROOT.rglob("SKILL.md"):
            skill_dirs.append(md.parent)

    skill_dirs.sort()

    report_new: List[Tuple[str, str]] = []
    report_changed: List[Tuple[str, str, str, str]] = []
    report_deleted: List[Tuple[str, str]] = []
    stale_skills: List[str] = []

    for sd in skill_dirs:
        new_f, changed_f, deleted_f = check_skill(sd, baseline)

        skill_name = sd.relative_to(SKILLS_ROOT)
        total = len(new_f) + len(changed_f) + len(deleted_f)

        if total == 0:
            continue

        total_changes += 1
        stale_skills.append(str(skill_name))

        for f in new_f:
            report_new.append((str(skill_name), f))

        for f in changed_f:
            old_sha = baseline.get(str(sd), {}).get(f, "?")
            new_sha = git_sha(Path(f)) or "?"
            # Show short SHAs for readability
            report_changed.append((
                str(skill_name), f,
                old_sha[:8], new_sha[:8],
            ))

        for f in deleted_f:
            report_deleted.append((str(skill_name), f))

    # Print triage report
    if report_new:
        print("\n🟢 NEW file references (source exists, not in baseline):")
        for skill, f in report_new:
            print(f"  [{skill}] {f}")

    if report_changed:
        print("\n🟡 CHANGED file references (SHA mismatch — review skill):")
        for skill, f, old_sha, new_sha in report_changed:
            print(f"  [{skill}] {f}  {old_sha} → {new_sha}")

    if report_deleted:
        print("\n🔴 DELETED file references (in baseline, not in skill):")
        for skill, f in report_deleted:
            print(f"  [{skill}] {f}")

    if stale_skills:
        print(f"\n📊 {len(stale_skills)} skill(s) with drift: {', '.join(stale_skills)}")
    else:
        print("\n✅ All skills up to date — no drift detected.")

    return total_changes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check if source files referenced by skills have changed (SHA drift detection).",
    )
    parser.add_argument("--skill", help="Check only this skill (name or path)")
    parser.add_argument("--update", action="store_true", help="Update baseline cache after review")
    parser.add_argument("--list", action="store_true", help="List all tracked file refs (no diff)")
    args = parser.parse_args(argv)

    baseline = load_cache()

    if args.list:
        # List mode — show all tracked references
        for skill_path, files in sorted(baseline.items()):
            print(f"\n{skill_path}:")
            for f, sha in sorted(files.items()):
                print(f"  {sha[:12]}  {f}")
        return 0

    if args.skill:
        # Single skill mode
        target = SKILLS_ROOT / args.skill
        if not (target / "SKILL.md").exists():
            # Try as a direct path
            target = Path(args.skill)
        if not (target / "SKILL.md").exists():
            logger.error("Skill not found: %s", args.skill)
            return 1

        new_f, changed_f, deleted_f = check_skill(target, baseline)
        total = len(new_f) + len(changed_f) + len(deleted_f)
        if total > 0:
            for f in new_f:
                print(f"  NEW: {f}")
            for f in changed_f:
                old_sha = baseline.get(str(target), {}).get(f, "?")
                new_sha = git_sha(Path(f)) or "?"
                print(f"  CHANGED: {f}  {old_sha[:8]} → {new_sha[:8]}")
            for f in deleted_f:
                print(f"  DELETED: {f}")
            print(f"\n  {total} drift(s) detected in {target}")
        else:
            print(f"✅ {target} is up to date.")

        if args.update:
            save_cache(baseline)
            print(f"  Baseline updated: {CACHE_FILE}")
    else:
        # All skills mode
        total_changes = check_all_skills(baseline)
        if args.update:
            save_cache(baseline)
            print(f"\n  Baseline updated: {CACHE_FILE} ({total_changes} skills recorded)")
        elif total_changes > 0:
            print(f"\n  Run with --update to save the new baseline.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())