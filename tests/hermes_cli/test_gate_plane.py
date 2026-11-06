"""Gate Plane stores safe guard evidence and computes lifecycle suite health."""

from __future__ import annotations

from datetime import datetime, UTC
from types import SimpleNamespace

import pytest

from hermes_cli.gate_plane import GatePlane, GateStatus, build_gate_report, result_status


def test_result_status_classifies_normal_and_policy_directives() -> None:
    assert result_status(None) is GateStatus.PASS
    assert result_status({"context": "reminder"}) is GateStatus.PASS
    assert result_status({"action": "block", "message": "blocked"}) is GateStatus.BLOCK
    assert result_status({"action": "modify", "args": {}}) is GateStatus.MODIFY
    assert result_status("answer") is GateStatus.PASS


def _callback():
    """A named local callback for stable identity assertions."""


def test_gate_plane_records_only_safe_diagnostic_fields() -> None:
    plane = GatePlane()

    plane.record(
        "pre_tool_call",
        _callback,
        session_id="session-1",
        call_id="call-1",
        status=GateStatus.BLOCK,
        detail="action=block",
        duration_ms=1.25,
    )

    event = plane.events()[0].as_dict()
    assert event == {
        "hook": "pre_tool_call",
        "callback": f"{__name__}:_callback",
        "session_id": "session-1",
        "call_id": "call-1",
        "status": "block",
        "detail": "action=block",
        "duration_ms": 1.25,
    }


def test_gate_plane_history_is_bounded() -> None:
    plane = GatePlane(max_events=3)

    for index in range(5):
        plane.record(
            "pre_llm_call", _callback, call_id=f"turn-{index}",
            status=GateStatus.PASS,
        )

    assert [event.call_id for event in plane.events()] == [
        "turn-2", "turn-3", "turn-4",
    ]


def _plugin(
    name: str,
    *,
    key: str | None = None,
    enabled: bool = True,
    error: str | None = None,
    hooks: tuple[str, ...] = ("pre_verify",),
    registered: tuple[str, ...] | None = None,
    tags: tuple[str, ...] = ("guard",),
    policy: str | None = "required",
    kind: str | None = None,
    gate_audit: dict[str, object] | None = None,
) -> SimpleNamespace:
    manifest = SimpleNamespace(
        name=name,
        key=key or name,
        tags=list(tags),
        provides_hooks=list(hooks),
    )
    if policy is not None:
        manifest.gate_policy = policy
    if kind is not None:
        manifest.gate_kind = kind
    if gate_audit is not None:
        manifest.gate_audit = gate_audit
    return SimpleNamespace(
        manifest=manifest,
        enabled=enabled,
        error=error,
        hooks_registered=list(hooks if registered is None else registered),
    )


def test_hook_free_guard_is_manual_not_runtime_active() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("intent-guard", hooks=("pre_llm_call",)),
        _plugin("fork-guard", hooks=(), kind="manual"),
    ]

    report = build_gate_report(plugins, plane)

    assert report["guard_count"] == 2
    assert report["active_guard_count"] == 1
    assert report["manual_guards"] == [{
        "name": "fork-guard",
        "policy": "required",
        "reason": "no fresh audit evidence",
        "status": "unknown",
    }]
    assert all("fork-guard" not in suite["guards"] for suite in report["suites"])


def test_required_manual_guard_without_evidence_keeps_report_unknown() -> None:
    report = build_gate_report(
        [_plugin("fork-guard", hooks=(), kind="manual")], GatePlane(),
    )

    assert report["overall_status"] == "unknown"
    assert report["manual_guards"][0]["status"] == "unknown"


def _receipt(
    *,
    status: str = "pass",
    subject: str = "repo:/tmp/project",
    checked_at: str = "2026-11-05T00:00:00+00:00",
    version: int = 1,
) -> dict[str, object]:
    return {
        "version": version,
        "status": status,
        "subject": subject,
        "checked_at": checked_at,
        "fingerprint": "sha256:test",
        "detail": "",
    }


def test_fresh_matching_manual_audit_pass_is_visible() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("intent-guard", hooks=("pre_llm_call",)),
        _plugin(
            "fork-guard",
            hooks=(),
            kind="manual",
            gate_audit={"subject": "repo:/tmp/project", "ttl_seconds": 60},
        ),
    ]
    now = datetime(2026, 11, 5, 0, 0, 59, tzinfo=UTC).timestamp()

    report = build_gate_report(
        plugins, plane,
        manual_evidence={"fork-guard": _receipt()},
        now=now,
    )

    assert report["active_guard_count"] == 1
    assert report["manual_guards"] == [{
        "name": "fork-guard",
        "policy": "required",
        "reason": "",
        "status": "pass",
        "subject": "repo:/tmp/project",
        "checked_at": "2026-11-05T00:00:00+00:00",
        "fingerprint": "sha256:test",
    }]
    assert all("fork-guard" not in suite["guards"] for suite in report["suites"])


@pytest.mark.parametrize(
    ("receipt", "expected_reason"),
    [
        (
            _receipt(checked_at="2026-11-04T23:00:00+00:00"),
            "audit evidence stale",
        ),
        (_receipt(subject="repo:/other"), "audit subject mismatch"),
        (_receipt(version=2), "audit evidence malformed"),
        (None, "no fresh audit evidence"),
    ],
)
def test_manual_guard_degrades_on_failed_stale_wrong_or_missing_evidence(
    receipt: dict[str, object] | None, expected_reason: str,
) -> None:
    plugin = _plugin(
        "fork-guard",
        hooks=(),
        kind="manual",
        gate_audit={"subject": "repo:/tmp/project", "ttl_seconds": 60},
    )
    now = datetime(2026, 11, 5, 0, 0, 0, tzinfo=UTC).timestamp()

    report = build_gate_report(
        [plugin], GatePlane(), manual_evidence={"fork-guard": receipt}, now=now,
    )

    item = report["manual_guards"][0]
    assert report["overall_status"] == "unknown"
    assert item["status"] == "unknown"
    assert item["reason"] == expected_reason


def test_fresh_failed_manual_audit_is_fail_not_unknown() -> None:
    plugin = _plugin(
        "fork-guard",
        hooks=(),
        kind="manual",
        gate_audit={"subject": "repo:/tmp/project", "ttl_seconds": 60},
    )
    now = datetime(2026, 11, 5, 0, 0, 0, tzinfo=UTC).timestamp()

    report = build_gate_report(
        [plugin], GatePlane(),
        manual_evidence={"fork-guard": _receipt(status="fail")},
        now=now,
    )

    item = report["manual_guards"][0]
    assert report["overall_status"] == "unknown"
    assert item["status"] == "fail"
    assert item["reason"] == "latest audit failed"


def test_disabled_manual_guard_with_receipt_stays_unknown() -> None:
    plugin = _plugin(
        "fork-guard",
        enabled=False,
        error="disabled via config",
        hooks=(),
        kind="manual",
        gate_audit={"subject": "repo:/tmp/project", "ttl_seconds": 60},
    )
    now = datetime(2026, 11, 5, 0, 0, 0, tzinfo=UTC).timestamp()

    report = build_gate_report(
        [plugin], GatePlane(),
        manual_evidence={"fork-guard": _receipt()},
        now=now,
    )

    assert report["manual_guards"][0]["status"] == "unknown"
    assert report["manual_guards"][0]["reason"] == "disabled via config"


def test_observer_hooks_map_to_integrity_and_registration_is_checked() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("engine-invariants", hooks=("post_llm_call",), kind="observer"),
        _plugin(
            "tool-observer", hooks=("post_tool_call",),
            registered=(), kind="observer",
        ),
    ]

    report = build_gate_report(plugins, plane)

    integrity = next(suite for suite in report["suites"] if suite["name"] == "integrity")
    assert integrity["guards"] == ["engine-invariants"]
    assert [item["name"] for item in integrity["missing"]] == ["tool-observer"]


def test_observer_unknown_event_degrades_integrity() -> None:
    plane = GatePlane()
    plane.record(
        "post_tool_call", _callback, session_id="session-1", call_id="call-1",
        status=GateStatus.UNKNOWN, detail="RuntimeError: observer failed",
    )

    report = build_gate_report(
        [_plugin("tool-observer", hooks=("post_tool_call",), kind="observer")],
        plane,
        session_id="session-1",
    )

    integrity = next(suite for suite in report["suites"] if suite["name"] == "integrity")
    assert integrity["status"] == "unknown"
    assert integrity["runtime_unknown"] == 1


def test_report_builds_healthy_suites_from_real_registrations() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("intent-guard", hooks=("pre_llm_call",)),
        _plugin("completion-guard", hooks=("pre_verify",)),
    ]

    report = build_gate_report(plugins, plane)

    assert report["overall_status"] == "unknown"
    assert report["guard_count"] == 2
    assert report["active_guard_count"] == 2
    assert {suite["name"]: suite["status"] for suite in report["suites"]} == {
        "intent": "healthy",
        "execution": "unknown",
        "output": "unknown",
        "completion": "healthy",
        "integrity": "unknown",
    }
    assert report["suites"][0]["guards"] == ["intent-guard"]


def test_disabled_error_and_hollow_guards_make_their_suites_unknown() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("disabled-guard", enabled=False, hooks=("pre_tool_call",)),
        _plugin("broken-guard", error="import failed", hooks=("transform_llm_output",)),
        _plugin("hollow-guard", registered=(), hooks=("pre_verify",)),
    ]

    report = build_gate_report(plugins, plane)

    assert report["overall_status"] == "unknown"
    statuses = {suite["name"]: suite for suite in report["suites"]}
    assert statuses["execution"]["status"] == "unknown"
    assert statuses["output"]["status"] == "unknown"
    assert statuses["completion"]["status"] == "unknown"
    assert {item["name"] for item in statuses["execution"]["missing"]} == {"disabled-guard"}
    assert statuses["output"]["missing"][0]["reason"] == "import failed"
    assert statuses["completion"]["missing"][0]["reason"] == "declared hooks not registered: pre_verify"


def test_disabled_advisory_guard_is_visible_without_degrading_its_suite() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("baseline-guard", hooks=("pre_llm_call",)),
        _plugin("style-guard", enabled=False, hooks=("pre_llm_call",), policy="advisory"),
    ]

    report = build_gate_report(plugins, plane)

    intent = next(suite for suite in report["suites"] if suite["name"] == "intent")
    assert intent["status"] == "healthy"
    assert intent["guards"] == ["baseline-guard"]
    assert intent["missing"] == []
    assert intent["inactive_advisory"] == [{
        "name": "style-guard",
        "reason": "not enabled",
        "hooks": ["pre_llm_call"],
        "policy": "advisory",
    }]


def test_explicitly_disabled_advisory_is_inactive_even_when_manager_records_a_reason() -> None:
    plane = GatePlane()
    plugin = _plugin(
        "style-guard", enabled=False, error="disabled via config",
        hooks=("pre_llm_call",), policy="advisory",
    )

    report = build_gate_report([plugin], plane)

    intent = next(suite for suite in report["suites"] if suite["name"] == "intent")
    assert intent["missing"] == []
    assert [item["name"] for item in intent["inactive_advisory"]] == ["style-guard"]


def test_enabled_advisory_guard_that_is_broken_or_hollow_still_degrades_health() -> None:
    plane = GatePlane()
    plugins = [
        _plugin(
            "broken-advisory", error="import failed",
            hooks=("transform_llm_output",), policy="advisory",
        ),
        _plugin(
            "hollow-advisory", registered=(),
            hooks=("pre_verify",), policy="advisory",
        ),
    ]

    report = build_gate_report(plugins, plane)

    statuses = {suite["name"]: suite for suite in report["suites"]}
    assert statuses["output"]["status"] == "unknown"
    assert statuses["completion"]["status"] == "unknown"
    assert statuses["output"]["missing"][0]["policy"] == "advisory"
    assert statuses["completion"]["missing"][0]["policy"] == "advisory"
    assert all(not suite["inactive_advisory"] for suite in report["suites"])


def test_legacy_guards_default_to_required_and_hook_derived_suites() -> None:
    plane = GatePlane()
    legacy = SimpleNamespace(
        manifest=SimpleNamespace(
            name="legacy-guard",
            key="legacy-guard",
            tags=["guard"],
            provides_hooks=["pre_tool_call"],
        ),
        enabled=False,
        error=None,
        hooks_registered=["pre_tool_call"],
    )

    report = build_gate_report([legacy], plane)

    execution = next(suite for suite in report["suites"] if suite["name"] == "execution")
    assert execution["status"] == "unknown"
    assert execution["missing"] == [{
        "name": "legacy-guard",
        "reason": "not enabled",
        "hooks": ["pre_tool_call"],
        "policy": "required",
    }]


def test_runtime_unknown_events_degrade_their_hook_suites() -> None:
    plane = GatePlane()
    plane.record(
        "pre_llm_call", _callback, session_id="session-1", call_id="turn-1",
        status=GateStatus.UNKNOWN, detail="TimeoutError: judge timed out",
    )

    report = build_gate_report([_plugin("intent-guard")], plane, session_id="session-1")

    intent = next(suite for suite in report["suites"] if suite["name"] == "intent")
    assert intent["status"] == "unknown"
    assert intent["runtime_unknown"] == 1
    assert report["events"][0]["detail"] == "TimeoutError: judge timed out"


def test_no_guard_surface_is_explicit_unknown() -> None:
    report = build_gate_report([], GatePlane())

    assert report["overall_status"] == "unknown"
    assert report["guard_count"] == 0
    assert all(suite["status"] == "unknown" for suite in report["suites"])
