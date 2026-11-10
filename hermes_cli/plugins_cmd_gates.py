"""``hermes plugins gates`` — diagnostics for the shared Gate Plane report."""

from __future__ import annotations

import json
import sys

from hermes_cli.gate_plane import GATE_SUITES


def cmd_gates(session_id: str | None = None, *, json_output: bool = False) -> None:
    """Print gate-suite health, manual receipts, and recent unknown callback evidence.

    The command is read-only: it renders the profile-scoped manager report and never
    performs filesystem writes or network access. Manual evidence comes from durable
    plugin state and never turns an imported manual guard into runtime protection.
    """
    from hermes_cli.plugins import _ensure_plugins_discovered

    report = _ensure_plugins_discovered().gate_plane_report(session_id=session_id)
    if json_output:
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return

    suites = report.get("suites", [])
    from hermes_cli import plugins_cmd

    console = plugins_cmd._console()
    healthy = sum(suite.get("status") != "unknown" for suite in suites)
    console.print(
        f"Gate status: [bold]{report.get('overall_status', 'unknown')}[/bold] "
        f"[dim](guards active: {report.get('active_guard_count', 0)}/"
        f"{report.get('guard_count', 0)})[/dim]"
    )
    inactive = [
        guard
        for suite in suites
        for guard in suite.get("inactive_advisory", ())
    ]
    via_config = sorted(
        guard.get("name", "unknown")
        for guard in inactive
        if "disabled via config" in guard.get("reason", "")
    )
    not_enabled = sorted(
        guard.get("name", "unknown")
        for guard in inactive
        if guard.get("name", "unknown") not in via_config
    )
    console.print(
        f"Guards inactive: {len(inactive)} "
        f"({len(via_config)} via config, {len(not_enabled)} not enabled)"
    )
    console.print(f"  via config: {', '.join(via_config) or 'none'}")
    console.print(f"  not enabled: {', '.join(not_enabled) or 'none'}")
    console.print(f"Gate suites: {healthy}/{len(GATE_SUITES)} healthy")
    for suite in suites:
        hooks = ", ".join(suite.get("hooks", ()))
        console.print(
            f"  [bold]{suite.get('name', 'unknown')}[/bold]: "
            f"{suite.get('status', 'unknown')} [dim]({hooks or 'lifecycle'})[/dim]"
        )
        for guard in suite.get("missing", ()):
            console.print(
                f"    missing {guard.get('name', 'unknown')}: "
                f"{guard.get('reason', 'unknown')}"
            )
        for guard in suite.get("inactive_advisory", ()):
            console.print(
                f"    advisory disabled {guard.get('name', 'unknown')}: "
                f"{guard.get('reason', 'unknown')}"
            )
    for guard in report.get("manual_guards", ()):
        console.print(
            f"    manual {guard.get('status', 'unknown')} "
            f"{guard.get('name', 'unknown')}: "
            f"{guard.get('policy', 'required')}; {guard.get('reason', 'unknown')}"
        )

    unknown_events = [
        event for event in report.get("events", ())
        if event.get("status") == "unknown"
    ]
    console.print("Recent unknown events:")
    if not unknown_events:
        console.print("  none")
        return
    for event in unknown_events:
        call_id = event.get("call_id") or "no-call-id"
        console.print(f"  {event.get('hook', 'unknown')} ({call_id})", markup=False)
        console.print(f"    {event.get('callback', 'unknown')}")
        console.print(f"    {event.get('detail', 'unknown')}")
