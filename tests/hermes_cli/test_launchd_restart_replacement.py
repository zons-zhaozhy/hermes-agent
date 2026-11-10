"""launchd restarts must prove a replacement gateway exists before success."""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli


def _patch_forced_kickstart(
    monkeypatch, *, graceful_exit: bool, gateway_exits: bool, supervised: bool
) -> list:
    """Drive the drain-timeout -> kickstart path and return its ordered calls."""
    calls: list = []
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **k: 654)
    monkeypatch.setattr(gateway_cli, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr(gateway_cli, "_launchctl_supervised_pid", lambda label: 655)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
    monkeypatch.setattr(
        gateway_cli,
        "probe_gateway_loop_liveness",
        lambda pid, **kw: gateway_cli.GATEWAY_LOOP_ALIVE,
    )
    monkeypatch.setattr(gateway_cli, "_get_restart_exit_wait_budget", lambda: 27.0)
    monkeypatch.setattr(
        gateway_cli,
        "_graceful_restart_via_sigusr1",
        lambda pid, timeout, **_: graceful_exit,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_launchd_service_pid",
        lambda label, old_pid, timeout=10.0, *, domain: False,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kw: calls.append(("gateway_exit", kw)) or gateway_exits,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_api_server_port_free",
        lambda **kw: calls.append(("port_free", kw)) or gateway_exits,
    )
    monkeypatch.setattr(
        gateway_cli,
        "wait_for_launchd_gateway_supervision",
        lambda **kw: calls.append(("supervision", kw)) or supervised,
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda *a, **k: calls.append(("kickstart", a[0])) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(gateway_cli, "_clear_launchd_unsupported_marker", lambda: None)
    return calls


def test_forced_kickstart_reports_success_after_fresh_supervision(monkeypatch, capsys):
    """A graceful exit with no immediate KeepAlive revival can still recover via kickstart."""
    calls = _patch_forced_kickstart(
        monkeypatch, graceful_exit=True, gateway_exits=True, supervised=True
    )

    gateway_cli.launchd_restart()

    assert [call[0] for call in calls] == [
        "kickstart",
        "gateway_exit",
        "port_free",
        "supervision",
    ]
    out = capsys.readouterr().out
    assert "did not revive" in out
    assert "✓ Service restarted" in out


def test_forced_kickstart_fails_when_old_gateway_still_owns_runtime(monkeypatch, capsys):
    """A kickstart that leaves the old gateway draining must not return success.

    ``kickstart -k`` can kill the launchd wrapper while its detached gateway child keeps draining.
    Exit 0 at that point prints success over a draining gateway and a spawn-scheduled service.
    """
    calls = _patch_forced_kickstart(
        monkeypatch, graceful_exit=False, gateway_exits=False, supervised=False
    )

    with pytest.raises(SystemExit) as excinfo:
        gateway_cli.launchd_restart()

    assert excinfo.value.code == 1
    assert [call[0] for call in calls] == [
        "kickstart",
        "gateway_exit",
        "port_free",
        "supervision",
    ]
    assert calls[-1][1]["old_pid"] == 655
    out = capsys.readouterr().out
    assert "✗ launchd did not supervise a fresh gateway" in out
    assert "✓ Service restarted" not in out
