"""
Subagent Two-Stage Review — independent verification of delegation results.

Builder-Judge-Manager pattern (inspired by CyrilXBT's self-correction architecture):
- Builder (child agent) produces deliverables + uncertainty declaration.
- Judge (independent reviewer) checks output against **ground truth** (git diff,
  test results, file-system evidence) and returns structured per-item verdicts.
- Manager (routing logic in delegate_tool.py) decides: approve / fix / escalate.

Ground truth principle: a Judge that only sees Builder output can only judge
internal consistency (format, logic flow), not correctness (does it actually work,
does it actually solve the goal).  Every check must reference independently
verifiable evidence — git diff, test output, file existence, command results.

This module provides:
- ``should_review()`` — quick gate: skip review for tasks that didn't
  modify files (pure research / analysis tasks).
- ``collect_ground_truth()`` — gather independently-verifiable evidence
  from verification_evidence.db and the file system.
- ``review_child_output()`` — dispatch a fresh reviewer subagent that
  examines the implementing subagent's output against ground truth and
  returns structured per-item verdicts (PASS/FAIL per check).
- ``fix_and_re_review()`` — if the reviewer finds Critical/Important issues,
  dispatch a fix subagent then re-review.

Config (config.yaml ``delegation:`` section):

.. code-block:: yaml

    delegation:
      review_enabled: true          # default: false (opt-in)
      review_max_iterations: 1      # max review-fix cycles before escalate
      review_model: null            # override model for reviewer (default: parent model)
      review_ground_truth: true     # inject ground truth evidence into Judge prompt
      review_cost_limit: 0.5        # USD ceiling for review-fix cycles
      review_timeout_seconds: 300   # wall-clock ceiling per review-fix cycle

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
            "ground_truth": bool(cfg.get("review_ground_truth", True)),
            "cost_limit": float(cfg.get("review_cost_limit", 0.5)),
            "timeout_seconds": int(cfg.get("review_timeout_seconds", 300)),
        }
    except Exception:
        logger.warning("Failed to load review config", exc_info=True)
        return {
            "enabled": False,
            "max_iterations": 1,
            "model": None,
            "ground_truth": True,
            "cost_limit": 0.5,
            "timeout_seconds": 300,
        }


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
    ground_truth: str = "",
) -> str:
    """Build the system prompt for the reviewer subagent.

    The reviewer is a fresh agent with NO context from the parent session.
    It gets: the task goal, the implementer's summary, the list of modified
    files, the git diff, and independently-verifiable ground truth.
    It returns structured per-item verdicts.
    """
    ground_truth_block = ""
    if ground_truth and ground_truth != "(no independent ground truth available)":
        ground_truth_block = (
            "\n## Ground Truth (independently verified)\n"
            "This evidence was gathered independently from the implementer's "
            "report. Trust it over the implementer's self-description.\n\n"
            f"{ground_truth}\n"
        )

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
        "```\n"
        f"{ground_truth_block}\n"
        "## Your Review\n"
        "Check the diff and ground truth against the task goal. "
        "Return structured per-item verdicts:\n"
        "- CRITICAL: bugs that break functionality or cause data loss\n"
        "- IMPORTANT: missing requirements, spec violations, security issues\n"
        "- MINOR: style, naming, minor improvements\n\n"
        "For EACH check, state PASS or FAIL with evidence.\n"
        "If ALL checks pass, respond with:\n"
        "REVIEW: APPROVED\n\n"
        "If ANY check fails, respond with:\n"
        "REVIEW: NEEDS_FIX\n"
        "Then list each failed check:\n"
        "- [SEVERITY] description of the issue\n\n"
        "WARNING: If ground truth shows a command failed (non-zero exit code) "
        "or a deliverable file does not exist, that is an automatic FAIL "
        "regardless of what the implementer's summary claims. "
        "Trust independently verified evidence over self-reported claims."
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


def collect_ground_truth(
    files_written: List[str],
    task_result: Dict[str, Any],
) -> str:
    """Gather independently-verifiable evidence for Judge.

    Ground truth principle: Judge must see evidence that Builder cannot
    fabricate.  Sources (by priority):

    1. Handoff cross-verification — Builder claims to have run commands;
       we cross-check those claims against verification_evidence.db.
       Claimed but unverified = red flag for Judge.
    2. Verification evidence — command outputs recorded by the terminal
       tool during Builder execution (test results, build status, lint
       output).
    3. File-system checks — os.path.exists() for deliverables.
    4. Uncertainty from handoff — what the Builder admits it doesn't know.

    Returns a formatted block for injection into the Judge prompt.
    """
    sections: list[str] = []

    # 1. Handoff cross-verification
    handoff = _parse_handoff(task_result.get("summary", ""))
    cross_check = _cross_verify_handoff(handoff, task_result)
    if cross_check:
        sections.append(cross_check)

    # 2. Verification evidence (command results from Builder's session)
    evidence_section = _collect_verification_evidence(task_result)
    if evidence_section:
        sections.append(evidence_section)

    # 3. File-system checks
    file_checks = _check_files_exist(files_written)
    if file_checks:
        sections.append(file_checks)

    # 4. Builder's own uncertainty (honesty check for Judge)
    uncertainty_section = _format_uncertainty(handoff)
    if uncertainty_section:
        sections.append(uncertainty_section)

    if not sections:
        return "(no independent ground truth available)"

    return "\n".join(sections)


def _collect_verification_evidence(task_result: Dict[str, Any]) -> str:
    """Read command results from verification_evidence.db."""
    session_id = task_result.get("session_id", "")
    if not session_id:
        return ""
    try:
        from agent.verification_evidence import verification_status
        from agent.coding_context import project_facts_for

        # Use cwd from task_result if available, else current dir
        cwd = task_result.get("cwd") or task_result.get("working_directory", "")
        if not cwd:
            return ""

        status = verification_status(session_id=session_id, cwd=cwd)
        if status.get("status") == "not_applicable":
            return ""

        evidence = status.get("evidence")
        if not evidence:
            return ""

        lines = [
            "### Verification Evidence (from command execution ledger)",
            f"Last command: `{evidence.get('command', 'N/A')}`",
            f"Kind: {evidence.get('kind', 'unknown')}",
            f"Status: {evidence.get('status', 'unknown')}",
            f"Exit code: {evidence.get('exit_code', 'N/A')}",
        ]
        output_summary = evidence.get("output_summary", "")
        if output_summary:
            lines.append(f"Output: {output_summary[:2000]}")
        return "\n".join(lines)
    except Exception:
        logger.warning("verification_evidence lookup failed for review", exc_info=True)
        return ""


def _check_files_exist(files_written: List[str]) -> str:
    """Check which deliverables actually exist on disk."""
    if not files_written:
        return ""
    results: list[str] = []
    for f in files_written:
        if os.path.exists(f):
            size = os.path.getsize(f)
            results.append(f"- PASS: `{f}` exists ({size} bytes)")
        else:
            results.append(f"- FAIL: `{f}` does NOT exist")
    if not results:
        return ""
    return "### File Existence Check\n" + "\n".join(results)


def _parse_handoff(summary_text: str) -> Dict[str, Any]:
    """Parse the Builder's Deliverable Handoff block from the summary.

    Extracts three sections from a markdown-formatted handoff:
    - files: list of file paths
    - commands: list of (command, exit_code) tuples
    - uncertainty: list of known-unknown items

    Returns a dict with keys 'files', 'commands', 'uncertainty'.
    Each value is a list (possibly empty if the Builder omitted that section).

    The handoff format is:

        ## Deliverable Handoff
        ### Files
        - path/to/file1.py
        - path/to/file2.yaml
        ### Commands Executed
        - pytest tests/test_x.py -q  (exit 0)
        - python -c '...' (exit 1)
        ### Uncertainty
        - something not verified
    """
    import re  # noqa: R1 — kept for backward compat, not used
    # All parsing uses str methods only — no regex, no backtracking risk.

    result: Dict[str, Any] = {
        "files": [],
        "commands": [],
        "uncertainty": [],
        "raw_block": "",
    }

    if not summary_text:
        return result

    # Find "## Deliverable Handoff" (case-insensitive)
    lower_text = summary_text.lower()
    header_tag = "## deliverable handoff\n"
    header_idx = lower_text.find(header_tag)
    if header_idx == -1:
        return result

    # Extract block: from header to next "## " heading (not "### ")
    block = summary_text[header_idx:]
    # Skip past the header line
    header_len = block.index('\n') + 1
    body = block[header_len:]

    # Cut at next ## heading (double hash, not triple)
    # Search for "\n## " that is NOT followed by "#"
    next_h2_idx = -1
    for i in range(len(body)):
        if body[i:i+4] == '\n## ' and (i+4 >= len(body) or body[i+4] != '#'):
            next_h2_idx = i
            break
    if next_h2_idx != -1:
        body = body[:next_h2_idx]

    result["raw_block"] = block[:block.index('\n') + 1].strip() + '\n' + body.strip() if body.strip() else block.strip()

    # Split by "### " headers and process each section
    # Use "\n### " as delimiter to avoid splitting on "#" inside text
    sections = ('\n' + body).split('\n### ')
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Get section name (first line or up to first \n after any trailing chars)
        first_nl = section.find('\n')
        header = section[:first_nl].strip() if first_nl != -1 else section.strip()
        content = section[first_nl+1:].strip() if first_nl != -1 else ''

        if header.lower().startswith('files'):
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    file_path = line[2:].strip()
                    if file_path:
                        result["files"].append(file_path)

        elif header.lower().startswith('commands executed'):
            for line in content.split('\n'):
                line = line.strip()
                if not line.startswith('- '):
                    continue
                cmd_text = line[2:].strip()
                exit_code = None
                command = cmd_text
                # Parse "- command (exit N)" or "- command (exit: N)"
                if cmd_text.endswith(')'):
                    paren_start = cmd_text.rfind('(exit')
                    if paren_start != -1:
                        paren_content = cmd_text[paren_start+1:-1].strip()
                        parts = paren_content.split()
                        if len(parts) >= 2 and parts[0] == 'exit':
                            try:
                                exit_code = int(parts[1].strip(':'))
                            except ValueError:
                                logger.warning("subagent review: failed to parse exit code from '%s'", parts[1])
                        command = cmd_text[:paren_start].strip()
                result["commands"].append((command, exit_code))

        elif header.lower().startswith('uncertainty'):
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    item = line[2:].strip()
                    if item and not (item.startswith('(') and item.endswith(')')):
                        result["uncertainty"].append(item)

    return result


def _cross_verify_handoff(
    handoff: Dict[str, Any],
    task_result: Dict[str, Any],
) -> str:
    """Cross-check Builder's claimed commands against verification_evidence.db.

    For each command the Builder claims to have run, check if there is a
    matching entry in the command execution ledger.  Commands the Builder
    claims but cannot be independently verified are flagged.
    """
    commands_claimed = handoff.get("commands", [])
    if not commands_claimed:
        return ""

    session_id = task_result.get("session_id", "")
    if not session_id:
        return _format_handoff_commands(commands_claimed, [])

    try:
        from agent.verification_evidence import verification_status

        cwd = task_result.get("cwd") or task_result.get("working_directory", "")
        if not cwd:
            return _format_handoff_commands(commands_claimed, [])

        status = verification_status(session_id=session_id, cwd=cwd)
        evidence = status.get("evidence")
        ledger_command = evidence.get("command", "") if evidence else ""

        # Simple matching: check if any claimed command substring appears
        # in the ledger command, or vice versa.
        verified = []
        for cmd, exit_code in commands_claimed:
            cmd_short = cmd.split()[0] if cmd else ""
            if cmd and (cmd_short in ledger_command or ledger_command in cmd):
                verified.append(cmd)

        return _format_handoff_commands(commands_claimed, verified)
    except Exception:
        logger.warning("Handoff cross-verification failed", exc_info=True)
        return _format_handoff_commands(commands_claimed, [])


def _format_handoff_commands(
    commands_claimed: list,
    verified_commands: list,
) -> str:
    """Format cross-verification results for the Judge prompt."""
    if not commands_claimed:
        return ""

    lines = ["### Handoff Cross-Verification"]
    for cmd, exit_code in commands_claimed:
        status = "VERIFIED" if cmd in verified_commands else "UNVERIFIED"
        ec_str = f" (claimed exit={exit_code})" if exit_code is not None else ""
        lines.append(f"- [{status}] `{cmd}`{ec_str}")

    unverified = [c for c, _ in commands_claimed if c not in verified_commands]
    if unverified:
        lines.append(
            "\nWARNING: The Builder claims to have run commands that could "
            "not be independently verified. Treat the Builder's self-reported "
            "results with extra scrutiny."
        )

    return "\n".join(lines)


def _format_uncertainty(handoff: Dict[str, Any]) -> str:
    """Format Builder's uncertainty list for the Judge prompt."""
    uncertainty = handoff.get("uncertainty", [])
    if not uncertainty:
        return ""

    lines = [
        "### Builder's Stated Uncertainty",
        "The Builder admits it is unsure about:",
    ]
    for item in uncertainty:
        lines.append(f"- {item}")
    lines.append(
        "\nPay extra attention to these areas — the Builder has already "
        "flagged them as potentially wrong or incomplete."
    )
    return "\n".join(lines)


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

    # Collect ground truth (independently-verifiable evidence)
    ground_truth = ""
    if review_cfg.get("ground_truth", True):
        ground_truth = collect_ground_truth(files_written, task_result)

    # When no git diff is available (e.g. subagent ran in isolated
    # working directory), degrade to ground-truth-only review instead
    # of skipping entirely.  If ground truth is also empty, skip.
    if not diff_content:
        if not ground_truth or ground_truth == "(no independent ground truth available)":
            return {
                "approved": True,
                "issues": "",
                "review_summary": "no diff and no ground truth available, review skipped",
            }
        # Ground-truth-only review: still spawn the reviewer with GT evidence
        logger.info(
            "No git diff available for review — degrading to ground-truth-only mode"
        )

    # Import here to avoid circular import at module load
    from tools.delegate_tool import _build_child_agent, _run_single_child

    reviewer_prompt = _build_reviewer_prompt(
        goal=goal,
        task_summary=task_result.get("summary", ""),
        files_written=files_written,
        diff_content=diff_content,
        ground_truth=ground_truth,
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
            override_provider=review_cfg.get("provider") or getattr(parent_agent, "provider", None),
            override_base_url=review_cfg.get("base_url") or getattr(parent_agent, "base_url", None),
            override_api_key=review_cfg.get("api_key") or getattr(parent_agent, "api_key_ref", None),
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


def run_review_loop(
    task_result: Dict[str, Any],
    goal: str,
    *,
    parent_agent=None,
    task_index: int = 0,
    review_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full Builder-Judge-Manager review loop.

    After Builder finishes, run Judge (review_child_output), then route
    based on verdict. If issues are found, enter a fix-review cycle
    bounded by ``review_cfg.max_iterations``.

    Updates ``task_result['_review']`` in place with review metadata:
    - iteration: number of fix-review cycles attempted
    - approved: final verdict
    - escalate: True when max iterations exhausted without approval
    - escalate_reason: human-readable reason for escalation

    Returns the (possibly mutated) task_result.

    This is the entry point for both delegate_tool and goal Judge —
    any code path that wants Builder-Judge-Manager review after a
    subagent completes.
    """
    if review_cfg is None:
        review_cfg = _load_review_config()

    if not review_cfg.get("enabled", False):
        return task_result

    review_result = review_child_output(
        task_result=task_result,
        goal=goal,
        parent_agent=parent_agent,
        task_index=task_index,
    )

    iteration_count = 0
    max_iterations = review_cfg.get("max_iterations", 1)

    task_result["_review"] = {
        "initial_review": review_result.get("review_summary", ""),
        "iteration": 0,
        "approved": review_result.get("approved", True),
    }

    if review_result.get("approved"):
        return task_result

    # Issues found — enter fix-review loop
    issues = review_result.get("issues", "")
    files_written = task_result.get("files_written") or []

    while iteration_count < max_iterations:
        iteration_count += 1
        task_result["_review"]["iteration"] = iteration_count

        fix_outcome = fix_and_re_review(
            goal=goal,
            issues=issues,
            files_to_fix=files_written,
            parent_agent=parent_agent,
            task_index=task_index,
            max_cycles=1,
        )

        task_result["_review"]["final_review"] = fix_outcome.get("review_summary", "")
        task_result["_review"]["approved"] = fix_outcome.get("approved", False)

        if fix_outcome.get("approved"):
            return task_result
        else:
            issues = fix_outcome.get("issues", issues)
            if iteration_count >= max_iterations:
                task_result["_review"]["escalate"] = True
                task_result["_review"]["escalate_reason"] = (
                    f"Max review iterations ({max_iterations}) exhausted — "
                    "task result returned as-is, manual review recommended"
                )
                break

    return task_result
