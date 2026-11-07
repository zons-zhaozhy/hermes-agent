"""Hard gate for remote branch deletion through the Hermes terminal tool."""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_tool_guards import remote_branch_delete_block


class TestRemoteBranchDeleteDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "git push origin --delete fix/example",
            "git push origin -d fix/example",
            "git push --delete origin fix/example",
            "git -C /repo push origin --delete fix/example",
            "git push origin :fix/example",
            "git push origin :refs/heads/fix/example",
            "zsh -lc 'git push origin --delete fix/example'",
            "git status && git push origin --delete fix/example",
        ],
    )
    def test_blocks_native_remote_branch_deletion(self, command):
        blocked = remote_branch_delete_block(command)

        assert blocked is not None
        result = json.loads(blocked)
        assert result["status"] == "blocked"
        assert "gh pr list --state open" in result["error"]
        assert "guarded_remote_branch_delete.py" in result["error"]

    @pytest.mark.parametrize(
        "command",
        [
            "git push origin fix/example",
            "git push origin HEAD:refs/heads/fix/example",
            "git branch --delete fix/example",
            "gh pr list --state open --json number,url,headRefName",
            "echo 'git push origin --delete fix/example'",
            "git status",
        ],
    )
    def test_allows_non_remote_deletion(self, command):
        assert remote_branch_delete_block(command) is None


def _run_terminal(command, monkeypatch, cwd):
    from tools.terminal_tool import terminal_tool

    config = {
        "env_type": "local",
        "timeout": 30,
        "cwd": cwd,
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    mock_env.cwd = cwd

    with ExitStack() as stack:
        stack.enter_context(
            patch("tools.terminal_tool._get_env_config", return_value=config)
        )
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(
            patch("tools.terminal_tool._active_environments", {"default": mock_env})
        )
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._session_cwd", {}))
        stack.enter_context(
            patch(
                "tools.terminal_tool._check_all_guards",
                return_value={"approved": True},
            )
        )
        return json.loads(terminal_tool(command=command)), mock_env


def test_terminal_tool_blocks_remote_branch_delete_before_execution(tmp_path, monkeypatch):
    result, env = _run_terminal(
        "git push origin --delete fix/example", monkeypatch, str(tmp_path)
    )

    assert result["status"] == "blocked"
    assert "gh pr list --state open" in result["error"]
    env.execute.assert_not_called()
