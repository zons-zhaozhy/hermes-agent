"""Gate Plane stores safe guard evidence and computes lifecycle suite health."""

from __future__ import annotations

from types import SimpleNamespace

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
    enabled: bool = True,
    error: str | None = None,
    hooks: tuple[str, ...] = ("pre_verify",),
    registered: tuple[str, ...] | None = None,
    tags: tuple[str, ...] = ("guard",),
    policy: str | None = "required",
) -> SimpleNamespace:
    manifest = SimpleNamespace(
        name=name,
        key=name,
        tags=list(tags),
        provides_hooks=list(hooks),
    )
    if policy is not None:
        manifest.gate_policy = policy
    return SimpleNamespace(
        manifest=manifest,
        enabled=enabled,
        error=error,
        hooks_registered=list(hooks if registered is None else registered),
    )


def test_report_builds_healthy_suites_from_real_registrations() -> None:
    plane = GatePlane()
    plugins = [
        _plugin("intent-guard", hooks=("pre_llm_call",)),
        _plugin("completion-guard", hooks=("pre_verify",)),
    ]

    report = build_gate_report(plugins, plane)

    assert report["overall_status"] == "healthy"
    assert report["guard_count"] == 2
    assert report["active_guard_count"] == 2
    assert {suite["name"]: suite["status"] for suite in report["suites"]} == {
        "intent": "healthy",
        "execution": "unknown",
        "output": "unknown",
        "completion": "healthy",
        "integrity": "healthy",
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
