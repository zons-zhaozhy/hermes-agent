"""`hermes pm install --test-environment` on an index that reports no upload times."""

import pytest

import pm
import pm.cli
import pm.testenv
from pm.package import InstallError, UNDATED_INDEX_REMEDY


@pytest.fixture()
def environments(monkeypatch):
    """No real lock check, no real environment: record what the caller decided."""
    built: list[str] = []
    monkeypatch.setattr(pm.cli, "repo_root", lambda: "REPO")
    monkeypatch.setattr(pm.testenv, "parse_extras", lambda value: value or ["all"])
    monkeypatch.setattr(pm.testenv, "ensure_testenv", lambda root, extras: built.append(root))
    return built


@pytest.fixture()
def make_lock_check_fail(monkeypatch):
    """Make pm.check_project_lock raise the given error."""
    def install_lock_failure(error: InstallError) -> None:
        def raise_error(*args, **kwargs):
            raise error
        monkeypatch.setattr(pm, "check_project_lock", raise_error)
    return install_lock_failure


def test_undated_index_skips_the_lock_check_and_builds_the_test_environment(
        environments, make_lock_check_fail, capsys):
    """The check resolves under the 3-day release-age rule, which a mirror without
    upload times cannot answer; the environment is built frozen from uv.lock anyway."""
    make_lock_check_fail(InstallError("venv", "uv lock exited 1", UNDATED_INDEX_REMEDY))

    assert pm.cli._install_python_environments(["all"], sync=False, test_environment="") == 0
    assert environments == ["REPO"]
    assert "lock check skipped" in capsys.readouterr().out


def test_any_other_lock_failure_still_fails_the_install(
        environments, make_lock_check_fail, capsys):
    """A genuinely stale or unreadable lock is not an undated index: it stays a failure."""
    make_lock_check_fail(InstallError("venv", "uv lock exited 1: lockfile is out of date"))

    assert pm.cli._install_python_environments(["all"], sync=False, test_environment="") == 1
    assert environments == []
