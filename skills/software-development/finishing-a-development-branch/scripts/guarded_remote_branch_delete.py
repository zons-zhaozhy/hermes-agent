#!/usr/bin/env python3
"""Delete remote branches only after checking them against open pull requests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from typing import Any, Protocol
from urllib.parse import urlparse


class Runner(Protocol):
    def __call__(self, command: list[str], *, check: bool, text: bool,
                 capture_output: bool) -> subprocess.CompletedProcess[str]: ...


_SCP_REMOTE_RE = re.compile(r"^[^/@\s]+@([^:/\s]+):([^/\s]+/[^/\s]+?)(?:\.git)?$")
_REMOTE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _github_repo(url: str) -> str | None:
    """Return ``owner/repo`` for a GitHub HTTPS or SCP-style remote URL."""
    if url.startswith("https://") or url.startswith("ssh://"):
        parsed = urlparse(url)
        if (parsed.hostname or "").lower() != "github.com":
            return None
        parts = [part for part in parsed.path.split("/") if part]
    else:
        match = _SCP_REMOTE_RE.match(url)
        if match is None or match.group(1).lower() != "github.com":
            return None
        parts = [part for part in match.group(2).split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1].removesuffix('.git')}"


def _parse_confirmations(values: Iterable[str], branches: set[str]) -> dict[str, set[int]]:
    confirmations: dict[str, set[int]] = {}
    for value in values:
        branch, separator, number = value.partition("=")
        if not separator or branch not in branches or not number.isdigit() or int(number) <= 0:
            raise ValueError(f"invalid confirmation {value!r}; expected <branch>=<open PR number>")
        confirmations.setdefault(branch, set()).add(int(number))
    return confirmations


def _run_checked(command: list[str], runner: Runner) -> subprocess.CompletedProcess[str]:
    result = runner(command, check=True, text=True, capture_output=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result


def _open_prs(
    pr_remote: str, branches: Iterable[str], runner: Runner
) -> list[dict[str, Any]]:
    remote_result = _run_checked(["git", "remote", "get-url", pr_remote], runner)
    repo = _github_repo(remote_result.stdout.strip())
    if repo is None:
        raise ValueError(f"remote {pr_remote!r} is not a GitHub repository")
    pulls: list[dict[str, Any]] = []
    for branch in branches:
        result = _run_checked(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--head",
                branch,
                "--limit",
                "1000",
                "--json",
                "number,url,headRefName",
            ],
            runner,
        )
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"gh returned invalid JSON: {error}") from error
        if not isinstance(parsed, list):
            raise ValueError("gh returned an unexpected pull-request list")
        for pull in parsed:
            number = pull.get("number") if isinstance(pull, dict) else None
            head = pull.get("headRefName") if isinstance(pull, dict) else None
            url = pull.get("url") if isinstance(pull, dict) else None
            if (
                not isinstance(pull, dict)
                or isinstance(number, bool)
                or not isinstance(number, int)
                or number <= 0
                or not isinstance(head, str)
                or not head
                or not isinstance(url, str)
            ):
                raise ValueError("gh returned an unexpected pull-request list")
            pulls.append(pull)
    return pulls


def _select_deletions(
    branches: list[str], open_prs: list[dict[str, Any]], confirmations: dict[str, set[int]]
) -> tuple[list[str], list[str]]:
    by_branch: dict[str, list[dict[str, Any]]] = {branch: [] for branch in branches}
    for pull in open_prs:
        head = pull.get("headRefName")
        if head in by_branch:
            by_branch[head].append(pull)

    deletions: list[str] = []
    skipped: list[str] = []
    for branch in branches:
        pulls = by_branch[branch]
        numbers = {int(pull["number"]) for pull in pulls if int(pull.get("number", 0)) > 0}
        if not numbers:
            deletions.append(branch)
            continue
        if confirmations.get(branch) == numbers:
            numbers_text = ", ".join(f"#{number}" for number in sorted(numbers))
            print(f"CONFIRMED {branch}: open PR(s) {numbers_text}")
            deletions.append(branch)
            continue
        skipped.append(branch)
        for pull in pulls:
            print(f"SKIPPED {branch}: open PR #{pull['number']} {pull.get('url', '')}".rstrip())
    return deletions, skipped


def main(argv: list[str] | None = None, *, run: Runner = subprocess.run) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-remote", required=True, help="remote containing branches to delete")
    parser.add_argument("--pr-remote", required=True, help="GitHub remote whose open PRs are checked")
    parser.add_argument("branches", nargs="+", metavar="BRANCH")
    parser.add_argument(
        "--confirm-pr",
        action="append",
        default=[],
        metavar="BRANCH=PR",
        help="separate user confirmation for an open-PR head; repeat for every open PR",
    )
    args = parser.parse_args(argv)

    if not _REMOTE_NAME_RE.fullmatch(args.branch_remote) or not _REMOTE_NAME_RE.fullmatch(args.pr_remote):
        parser.error("remote names may contain only letters, digits, dot, underscore, and dash")
    if any(branch.startswith("-") for branch in args.branches):
        parser.error("branch names may not start with '-'")
    branches = list(dict.fromkeys(args.branches))
    try:
        confirmations = _parse_confirmations(args.confirm_pr, set(branches))
        open_prs = _open_prs(args.pr_remote, branches, run)
    except (ValueError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or error
        print(f"REFUSED: open-PR query failed; no branch deleted: {detail}", file=sys.stderr)
        return 1

    actual_numbers = {
        branch: {
            int(pull["number"])
            for pull in open_prs
            if pull.get("headRefName") == branch and int(pull.get("number", 0)) > 0
        }
        for branch in branches
    }
    for branch, confirmed in confirmations.items():
        numbers = actual_numbers[branch]
        if confirmed != numbers:
            actual = f"open PR #{next(iter(numbers))}" if numbers else "no open PR"
            print(
                f"REFUSED: confirmation for {branch} does not match {actual}; no branch deleted",
                file=sys.stderr,
            )
            return 2

    deletions, _skipped = _select_deletions(branches, open_prs, confirmations)
    if not deletions:
        return 0
    try:
        _run_checked(["git", "push", args.branch_remote, "--delete", *deletions], run)
    except subprocess.CalledProcessError as error:
        print(f"git push failed: {getattr(error, 'stderr', '') or error}", file=sys.stderr)
        return 1
    print("DELETED " + ", ".join(deletions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
