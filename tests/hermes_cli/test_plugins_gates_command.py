"""`hermes plugins gates` exposes the shared Gate Plane report."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace


REPORT = {
    "overall_status": "unknown",
    "guard_count": 2,
    "active_guard_count": 1,
    "manual_guards": [{
        "name": "fork-guard",
        "policy": "required",
        "reason": "no fresh audit evidence",
        "status": "unknown",
    }],
    "suites": [
        {
            "name": "intent",
            "hooks": ["pre_llm_call"],
            "status": "unknown",
            "guards": ["intent-guard"],
            "missing": [{
                "name": "dead-guard",
                "reason": "not enabled",
                "hooks": ["pre_llm_call"],
                "policy": "required",
            }],
            "inactive_advisory": [{
                "name": "style-guard",
                "reason": "not enabled",
                "hooks": ["pre_llm_call"],
                "policy": "advisory",
            }, {
                "name": "reply-certainty-checker",
                "reason": "disabled via config",
                "hooks": ["pre_llm_call"],
                "policy": "advisory",
            }],
            "runtime_unknown": 1,
        },
    ],
    "events": [{
        "hook": "pre_llm_call",
        "callback": "plugins.dead_guard:check",
        "session_id": "session-1",
        "call_id": "turn-1",
        "status": "unknown",
        "detail": "TimeoutError: judge timed out",
        "duration_ms": 12.5,
    }],
}


def _parse(argv):
    from hermes_cli.plugins_cmd import plugins_command
    from hermes_cli.subcommands.plugins import build_plugins_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_plugins_parser(parser.add_subparsers(dest="command"), cmd_plugins=plugins_command)
    return parser.parse_args(["plugins", *argv])


def test_parser_registers_gates_options() -> None:
    args = _parse(["gates", "--json", "--session", "session-1"])

    assert args.plugins_action == "gates"
    assert args.json is True
    assert args.session == "session-1"


def test_gates_text_report_names_missing_guards_and_unknown_events(monkeypatch, capsys) -> None:
    from hermes_cli import plugins_cmd
    import hermes_cli.plugins as plugins_mod

    calls = []
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: SimpleNamespace(
        discover_and_load=lambda force=False: None,
        gate_plane_report=lambda session_id=None: calls.append(session_id) or REPORT,
    ))
    plugins_cmd.cmd_gates(session_id="session-1")
    output = capsys.readouterr().out

    assert calls == ["session-1"]
    assert "Gate suites: 0/5 healthy" in output
    assert "Guards inactive: 2 rows (1 via config, 1 not enabled)" in output
    assert "via config: reply-certainty-checker" in output
    assert "not enabled: style-guard" in output
    assert "dead-guard" in output and "not enabled" in output
    assert "advisory disabled style-guard: not enabled" in output
    assert "manual unknown fork-guard: required; no fresh audit evidence" in output
    assert "plugins.dead_guard:check" in output
    assert "TimeoutError: judge timed out" in output


def test_gates_json_emits_report_verbatim(monkeypatch, capsys) -> None:
    from hermes_cli import plugins_cmd
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: SimpleNamespace(
        discover_and_load=lambda force=False: None,
        gate_plane_report=lambda session_id=None: REPORT,
    ))
    plugins_cmd.cmd_gates(json_output=True)

    assert json.loads(capsys.readouterr().out) == REPORT


def test_gates_discovers_the_profile_plugin_surface_before_reporting(monkeypatch, capsys) -> None:
    from hermes_cli import plugins_cmd
    import hermes_cli.plugins as plugins_mod

    discovered = []
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: SimpleNamespace(
        discover_and_load=lambda force=False: discovered.append(True),
        gate_plane_report=lambda session_id=None: REPORT,
    ))
    plugins_cmd.cmd_gates(json_output=True)

    assert discovered == [True]
    assert json.loads(capsys.readouterr().out) == REPORT
