"""Fixtures shared across hermes_cli kanban tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )


@pytest.fixture
def no_real_launchd():
    """Keep the update pipeline's gateway-restart phase off THIS machine's
    real launchd services.

    On a macOS dev host with real ``ai.hermes.gateway*`` LaunchAgents (e.g.
    a sibling profile), the update flow's genuine launchctl discovery and
    restart leaks live services into tests and the fail-closed restart
    contract makes every update test exit(1). Update-flow test files opt in
    with ``pytestmark = pytest.mark.usefixtures("no_real_launchd")`` so the
    isolation logic lives in ONE place instead of being copy-pasted per file.
    """
    from unittest.mock import patch

    with patch("hermes_cli.gateway.find_gateway_pids", return_value=[]), \
         patch(
             "hermes_cli.gateway.find_profile_gateway_processes",
             return_value=[],
         ), \
         patch(
             # The plist lookup derives from the DEFAULT install root (not the
             # sandboxed HERMES_HOME), so a real plist takes the live launchctl
             # restart path and fails closed. No-op instead.
             "hermes_cli.gateway.get_launchd_plist_path",
             return_value=Path(os.environ.get("HERMES_HOME", "/tmp")) / "nonexistent-launchd-plist.plist",
         ), \
         patch(
             "hermes_cli.gateway.launchd_gateway_labels_for_install",
             return_value=[],
         ), \
         patch(
             "hermes_cli.update_cmd._restart_launchd_gateway_after_update",
             return_value=([], []),
         ), \
         patch(
             # Runtime inventory reads control sockets / PID files from the
             # DEFAULT install root — a dev box with a live gateway yields a
             # non-empty plan, fleet rows are then "expected", and the zero-row
             # fail-closed contract (#93406) exits 1. No plan → no expectation.
             # NOTE: report_unaccounted_runtimes / collect_fleet_versions are
             # deliberately NOT mocked — some update tests assert on their
             # real output.
             "hermes_cli.update_inventory.collect_runtime_inventory",
             return_value=None,
         ), \
         patch.object(
             # The stale-module purge evicts ``hermes_cli.gateway`` from
             # sys.modules mid-update; the restart phase's fresh ``from
             # hermes_cli.gateway import ...`` then loads an UNPATCHED copy,
             # silently discarding every mock above (see test_update_autostash's
             # fixture for the pioneering comment).
             __import__("hermes_cli.main", fromlist=["_purge_stale_hermes_modules"]),
             "_purge_stale_hermes_modules",
             lambda *a, **kw: None,
         ):
        yield
