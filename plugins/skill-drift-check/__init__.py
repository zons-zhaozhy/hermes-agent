"""skill-drift-check plugin — auto-detect skill source-file drift on session start.

Hooks into on_session_start to compare cached source-file SHAs against current
SHAs.  Only logs when drift is detected — zero noise on clean state.

The core logic delegates to scripts/skill_sha_drift.py so the CLI tool and the
plugin never diverge.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the standalone script — imported lazily so a broken script
# doesn't crash the plugin system.
_SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "skill_sha_drift.py"


def _import_drift_checker():
    """Lazily import the drift check logic from the standalone script."""
    if not _SCRIPT.exists():
        # Script not in repo — try user-installed copy
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("skill_sha_drift", _SCRIPT)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        logger.warning("skill_sha_drift.py failed to load", exc_info=True)
        return None


def on_session_start(**kwargs):
    """Check skill SHA drift at session start. Zero output when clean."""
    mod = _import_drift_checker()
    if mod is None:
        return  # Script unavailable — silent

    try:
        baseline = mod.load_cache()
        if not baseline:
            # First run or cache cleared — build baseline silently
            return

        # Run drift check, collect stale skills
        skill_dirs = []
        if mod.SKILLS_ROOT.exists():
            for md in mod.SKILLS_ROOT.rglob("SKILL.md"):
                skill_dirs.append(md.parent)

        stale = []
        for sd in sorted(skill_dirs):
            new_f, changed_f, deleted_f = mod.check_skill(sd, baseline)
            total = len(new_f) + len(changed_f) + len(deleted_f)
            if total > 0:
                skill_name = sd.relative_to(mod.SKILLS_ROOT)
                stale.append((str(skill_name), total))

        if stale:
            summary = ", ".join(f"{name}({n})" for name, n in stale[:5])
            extra = f" +{len(stale)-5} more" if len(stale) > 5 else ""
            logger.info(
                "[Skill Drift] %d skill(s) have stale source refs: %s%s. "
                "Run: python scripts/skill_sha_drift.py",
                len(stale), summary, extra,
            )
    except Exception:
        logger.warning("[Skill Drift] check failed", exc_info=True)


def register(ctx):
    """Plugin registration."""
    ctx.register_hook("on_session_start", on_session_start)
    logger.debug("skill-drift-check plugin registered")
