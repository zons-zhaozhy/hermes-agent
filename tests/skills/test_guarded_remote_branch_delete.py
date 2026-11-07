"""The safe cleanup helper refuses open-PR heads unless separately confirmed."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock


SKILL_DIR = (
    Path(__file__).parents[2]
    / "skills"
    / "software-development"
    / "finishing-a-development-branch"
)
HELPER = SKILL_DIR / "scripts" / "guarded_remote_branch_delete.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("guarded_remote_branch_delete", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(outputs):
    calls = []

    def run(command, *, check, text, capture_output):
        calls.append(
            {
                "command": command,
                "check": check,
                "text": text,
                "capture_output": capture_output,
            }
        )
        result = MagicMock(returncode=0)
        if command[:2] == ["git", "remote"]:
            result.stdout = outputs["remote"]
        elif command[0] == "gh":
            head = command[command.index("--head") + 1]
            pulls = [pull for pull in json.loads(outputs["prs"]) if pull["headRefName"] == head]
            result.stdout = json.dumps(pulls)
        return result

    return run, calls


def test_queries_open_prs_and_skips_blocked_heads(capsys):
    module = _load_helper()
    run, calls = _runner(
        {
            "remote": "ssh://git@github.com:443/NousResearch/hermes-agent.git",
            "prs": json.dumps(
                [
                    {
                        "number": 143665,
                        "url": "https://github.com/NousResearch/hermes-agent/pull/143665",
                        "headRefName": "fix/open-pr",
                    }
                ]
            ),
        }
    )

    exit_code = module.main(
        [
            "--branch-remote",
            "origin",
            "--pr-remote",
            "upstream",
            "fix/open-pr",
            "fix/safe",
        ],
        run=run,
    )

    assert exit_code == 0
    assert calls[0]["command"] == ["git", "remote", "get-url", "upstream"]
    assert calls[1]["command"] == [
        "gh",
        "pr",
        "list",
        "--repo",
        "NousResearch/hermes-agent",
        "--state",
        "open",
        "--head",
        "fix/open-pr",
        "--limit",
        "1000",
        "--json",
        "number,url,headRefName",
    ]
    assert calls[2]["command"][:9] == [
        "gh",
        "pr",
        "list",
        "--repo",
        "NousResearch/hermes-agent",
        "--state",
        "open",
        "--head",
        "fix/safe",
    ]
    assert calls[-1]["command"] == [
        "git",
        "push",
        "origin",
        "--delete",
        "fix/safe",
    ]
    output = capsys.readouterr().out
    assert "SKIPPED fix/open-pr: open PR #143665" in output
    assert "https://github.com/NousResearch/hermes-agent/pull/143665" in output


def test_exact_pr_confirmation_is_required_for_open_head(capsys):
    module = _load_helper()
    prs = json.dumps(
        [
            {
                "number": 143665,
                "url": "https://example/143665",
                "headRefName": "fix/open-pr",
            }
        ]
    )
    run, calls = _runner(
        {"remote": "https://github.com/NousResearch/hermes-agent.git", "prs": prs}
    )

    exit_code = module.main(
        [
            "--branch-remote",
            "origin",
            "--pr-remote",
            "upstream",
            "--confirm-pr",
            "fix/open-pr=143666",
            "fix/open-pr",
        ],
        run=run,
    )

    assert exit_code == 2
    assert [call["command"] for call in calls] == [calls[0]["command"], calls[1]["command"]]
    assert "confirmation for fix/open-pr does not match open PR #143665" in capsys.readouterr().err


def test_exact_pr_confirmation_allows_confirmed_head():
    module = _load_helper()
    prs = json.dumps(
        [
            {
                "number": 143665,
                "url": "https://example/143665",
                "headRefName": "fix/open-pr",
            }
        ]
    )
    run, calls = _runner(
        {"remote": "git@github.com:NousResearch/hermes-agent.git", "prs": prs}
    )

    exit_code = module.main(
        [
            "--branch-remote",
            "origin",
            "--pr-remote",
            "upstream",
            "--confirm-pr",
            "fix/open-pr=143665",
            "fix/open-pr",
        ],
        run=run,
    )

    assert exit_code == 0
    assert calls[-1]["command"] == [
        "git",
        "push",
        "origin",
        "--delete",
        "fix/open-pr",
    ]


def test_gh_failure_never_reaches_git_push():
    module = _load_helper()

    def run(command, *, check, text, capture_output):
        result = MagicMock(returncode=1)
        if command[:2] == ["git", "remote"]:
            result.stdout = "git@github.com:NousResearch/hermes-agent.git\n"
        else:
            result.stderr = "auth failed"
        return result

    exit_code = module.main(
        ["--branch-remote", "origin", "--pr-remote", "upstream", "fix/example"],
        run=run,
    )

    assert exit_code == 1


def test_malformed_pr_record_never_reaches_git_push(capsys):
    module = _load_helper()
    run, _calls = _runner(
        {
            "remote": "ssh://git@github.com:443/NousResearch/hermes-agent.git",
            "prs": json.dumps([{"url": "https://example/1", "headRefName": "fix/example"}]),
        }
    )

    exit_code = module.main(
        ["--branch-remote", "origin", "--pr-remote", "upstream", "fix/example"],
        run=run,
    )

    assert exit_code == 1
    assert "unexpected pull-request list" in capsys.readouterr().err
