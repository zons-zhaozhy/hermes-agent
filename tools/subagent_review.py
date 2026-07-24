"""
Subagent Two-Stage Review — independent verification of delegation results.

Inspired by Superpowers' subagent-driven-development pattern: every task gets
(1) implementer self-review, (2) independent task reviewer (spec + quality),
(3) final whole-branch review.  The key insight is that subagent summaries
are SELF-REPORTS — an independent reviewer catches issues the implementer
missed.

This module provides:
- ``should_review()`` — quick gate: skip review for tasks that didn't
  modify files (pure research / analysis tasks).
- ``review_child_output()`` — dispatch a fresh reviewer subagent that
  examines the implementing subagent's diff (via git) and reports issues
  by severity (Critical / Important / Minor).
- ``fix_and_re_review()`` — if the reviewer finds Critical/Important issues,
  dispatch a fix subagent then re-review.

Config (config.yaml ``delegation:`` section):

.. code-block:: yaml

    delegation:
      review_enabled: true          # default: false (opt-in)
      review_max_iterations: 1      # max review-fix cycles before giving up
      review_model: null            # override model for reviewer (default: parent model)

The review is OPT-IN (``delegation.review_enabled``) because it doubles the
subagent count per task.  The model decides whether to use it based on the
tool description, which surfaces the config state.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_review_config() -> dict:
    """Load review-related config from the delegation section."""
    try:
        from tools.delegate_tool import _load_config

        cfg = _load_config()
        return {
            "enabled": bool(cfg.get("review_enabled", False)),
            "max_iterations": int(cfg.get("review_max_iterations", 1)),
            "model": cfg.get("review_model"),
            "provider": cfg.get("review_provider"),
            "base_url": cfg.get("review_base_url"),
        }
    except Exception:
        logger.warning("Failed to load review config", exc_info=True)
        return {"enabled": False, "max_iterations": 1, "model": None}


def should_review(task_result: Dict[str, Any]) -> bool:
    """Quick gate: should this task result be reviewed?

    Skip review for tasks that:
    - Didn't complete successfully
    - Didn't modify any files (pure research / analysis)
    - Have no summary (empty/failed)

    Returns True only when there's something meaningful to review.
    """
    if task_result.get("status") != "completed":
        return False
    if not task_result.get("summary"):
        return False
    # Check if files were written — file_state tracks this
    files_written = task_result.get("files_written") or []
    if not files_written:
        return False
    return True


def _build_reviewer_prompt(
    goal: str,
    task_summary: str,
    files_written: List[str],
    diff_content: str,
) -> str:
    """Build the system prompt for the reviewer subagent.

    The reviewer is a fresh agent with NO context from the parent session.
    It gets: the task goal, the implementer's summary, the list of modified
    files, and the git diff.  It must report issues by severity.
    """
    return (
        "You are an independent code reviewer. Your job is to verify that a "
        "subagent's implementation is correct and complete.\n\n"
        "## Task Goal\n"
        f"{goal}\n\n"
        "## Implementer Summary\n"
        f"{task_summary}\n\n"
        "## Modified Files\n"
        + "\n".join(f"- {f}" for f in files_written)
        + "\n\n## Git Diff\n"
        "```diff\n"
        f"{diff_content[:8000]}\n"
        "```\n\n"
        "## Your Review\n"
        "Check the diff against the task goal. Report issues by severity:\n"
        "- CRITICAL: bugs that break functionality or cause data loss\n"
        "- IMPORTANT: missing requirements, spec violations, security issues\n"
        "- MINOR: style, naming, minor improvements\n\n"
        "If the implementation is correct and complete, respond with:\n"
        "REVIEW: APPROVED\n\n"
        "If there are issues, respond with:\n"
        "REVIEW: NEEDS_FIX\n"
        "Then list each issue with its severity and a one-line description.\n"
        "Be concise. Only report real issues — do not nitpick."
    )


def _build_fixer_prompt(
    goal: str,
    review_issues: str,
    files_to_fix: List[str],
) -> str:
    """Build the prompt for the fix subagent."""
    return (
        "A code review found issues in your previous work. Fix them.\n\n"
        "## Original Task\n"
        f"{goal}\n\n"
        "## Review Issues\n"
        f"{review_issues}\n\n"
        "## Files to Fix\n"
        + "\n".join(f"- {f}" for f in files_to_fix)
        + "\n\n"
        "Fix each CRITICAL and IMPORTANT issue. Run the relevant tests after "
        "fixing. Commit your fixes. Report what you changed."
    )


def _get_git_diff(files: List[str]) -> str:
    """Get the git diff for the specified files.

    Returns an empty string if git is unavailable or the files are not
    in a git repo.  Uses uncommitted diff (staged + unstaged) since the
    subagent may not have committed yet.
    """
    if not files:
        return ""
    try:
        import subprocess

        # Try to get diff — both staged and unstaged
        result = subprocess.run(
            ["git", "diff", "HEAD", "--"] + files,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
        # Fall back to untracked files content
        result2 = subprocess.run(
            ["git", "diff", "--no-index", "/dev/null", "--"] + files[:1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result2.stdout if result2.returncode == 1 else ""
    except Exception:
        logger.warning("git diff unavailable for review", exc_info=True)
        return ""


def review_child_output(
    task_result: Dict[str, Any],
    goal: str,
    parent_agent=None,
    task_index: int = 0,
) -> Dict[str, Any]:
    """Review a child agent's output by dispatching a fresh reviewer.

    Returns a dict with:
    - ``approved``: True if the reviewer approved, False if issues found.
    - ``issues``: The reviewer's issue text (empty if approved).
    - ``review_summary``: Short summary from the reviewer.

    If review is disabled or the reviewer subagent fails, returns
    ``{"approved": True, "issues": "", "review_summary": "review skipped"}``
    — fail-open so the parent isn't blocked.
    """
    review_cfg = _load_review_config()
    if not review_cfg["enabled"]:
        return {
            "approved": True,
            "issues": "",
            "review_summary": "review disabled (delegation.review_enabled=false)",
        }

    if not should_review(task_result):
        return {
            "approved": True,
            "issues": "",
            "review_summary": "no files modified, review skipped",
        }

    files_written = task_result.get("files_written") or []
    diff_content = _get_git_diff(files_written)
    if not diff_content:
        return {
            "approved": True,
            "issues": "",
            "review_summary": "no diff available, review skipped",
        }

    # Import here to avoid circular import at module load
    from tools.delegate_tool import _build_child_agent, _run_single_child

    reviewer_prompt = _build_reviewer_prompt(
        goal=goal,
        task_summary=task_result.get("summary", ""),
        files_written=files_written,
        diff_content=diff_content,
    )

    try:
        reviewer_child = _build_child_agent(
            task_index=task_index,
            goal=reviewer_prompt,
            context=(
                "You are reviewing another agent's work. Use git and read_file "
                "to verify the implementation. Do NOT write or modify any files."
            ),
            toolsets=None,
            model=review_cfg.get("model") or getattr(parent_agent, "model", None),
            max_iterations=15,
            task_count=1,
            parent_agent=parent_agent,
            role="leaf",
        )
        result = _run_single_child(
            task_index=task_index,
            goal=reviewer_prompt,
            child=reviewer_child,
            parent_agent=parent_agent,
        )

        review_text = result.get("summary", "") or ""
        approved = "REVIEW: APPROVED" in review_text.upper()

        issues_text = ""
        if not approved:
            # Extract everything after "REVIEW: NEEDS_FIX"
            parts = review_text.upper().split("NEEDS_FIX")
            if len(parts) > 1:
                issues_text = review_text[review_text.upper().index("NEEDS_FIX") + len("NEEDS_FIX"):]
            else:
                issues_text = review_text

        return {
            "approved": approved,
            "issues": issues_text.strip(),
            "review_summary": review_text[:500],
            "review_status": result.get("status"),
        }
    except Exception:
        logger.warning(
            "Review subagent failed for task %d — failing open",
            task_index,
            exc_info=True,
        )
        return {
            "approved": True,
            "issues": "",
            "review_summary": "review subagent failed, failing open",
        }


def fix_and_re_review(
    goal: str,
    issues: str,
    files_to_fix: List[str],
    parent_agent=None,
    task_index: int = 0,
    max_cycles: int = 1,
) -> Dict[str, Any]:
    """Dispatch a fix subagent, then re-review.

    Loops up to ``max_cycles`` times.  Each cycle:
    1. Dispatch a fixer subagent with the review issues
    2. Re-review the fixer's output

    Returns the final review result.
    """
    from tools.delegate_tool import _build_child_agent, _run_single_child

    review_cfg = _load_review_config()
    cycles_remaining = max_cycles

    while cycles_remaining > 0:
        cycles_remaining -= 1

        fixer_prompt = _build_fixer_prompt(goal, issues, files_to_fix)
        try:
            fixer_child = _build_child_agent(
                task_index=task_index,
                goal=fixer_prompt,
                context=(
                    "Fix the issues found in code review. Read the files, "
                    "understand the original intent, apply minimal fixes, "
                    "and test."
                ),
                toolsets=None,
                model=getattr(parent_agent, "model", None),
                max_iterations=20,
                task_count=1,
                parent_agent=parent_agent,
                role="leaf",
            )
            fix_result = _run_single_child(
                task_index=task_index,
                goal=fixer_prompt,
                child=fixer_child,
                parent_agent=parent_agent,
            )
        except Exception:
            logger.warning(
                "Fix subagent failed for task %d",
                task_index,
                exc_info=True,
            )
            break

        # Re-review
        re_review = review_child_output(
            task_result=fix_result,
            goal=goal,
            parent_agent=parent_agent,
            task_index=task_index,
        )
        if re_review.get("approved"):
            return re_review
        issues = re_review.get("issues", issues)

    return {
        "approved": False,
        "issues": issues,
        "review_summary": f"fix-review cycle exhausted ({max_cycles} cycles)",
    }
