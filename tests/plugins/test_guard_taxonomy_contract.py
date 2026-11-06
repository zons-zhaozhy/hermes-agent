"""Bundled guard taxonomy declarations must match executable hook surfaces."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.gate_plane import (
    GATE_ENFORCEMENT_HOOKS,
    GATE_KINDS,
    GATE_OBSERVER_HOOKS,
    GATE_POLICIES,
    GATE_SUITES,
    GATE_SUITE_NAMES,
)
from hermes_cli.plugins_manifest import parse_manifest_file


_HOOK_SUITES = {
    hook: suite
    for suite, hooks in GATE_SUITES.items()
    if suite != "integrity"
    for hook in hooks
}
_BASELINE_GUARDS = {
    "block_escalation",
    "discipline",
    "engine-invariants",
    "finish_guard",
    "guard-sentinel",
    "guards",
    "read-think-gate",
}


def _bundled_guard_manifests():
    repo = Path(__file__).resolve().parents[2]
    for plugin_dir in sorted((repo / "plugins").iterdir()):
        manifest_file = plugin_dir / "plugin.yaml"
        if not manifest_file.is_file():
            continue
        manifest = parse_manifest_file(manifest_file, plugin_dir, "bundled", "")
        assert manifest is not None
        if "guard" in manifest.tags:
            yield manifest


def _expected_gate_kind(manifest) -> str:
    hooks = set(manifest.provides_hooks)
    if hooks & GATE_ENFORCEMENT_HOOKS:
        return "enforcement"
    if hooks & GATE_OBSERVER_HOOKS:
        return "observer"
    return "manual"


def test_every_bundled_guard_declares_hook_aligned_taxonomy_and_policy() -> None:
    manifests = list(_bundled_guard_manifests())

    assert manifests
    for manifest in manifests:
        expected = {
            _HOOK_SUITES[hook]
            for hook in manifest.provides_hooks
            if hook in GATE_ENFORCEMENT_HOOKS
        }
        actual = set(manifest.gate_suites)
        if expected:
            assert expected <= actual, manifest.name
            assert actual - expected <= {"integrity"}, manifest.name
        else:
            assert actual == {"integrity"}, manifest.name
        assert actual <= GATE_SUITE_NAMES
        assert manifest.gate_policy in GATE_POLICIES, manifest.name
        assert manifest.gate_kind in GATE_KINDS, manifest.name
        assert manifest.gate_kind == _expected_gate_kind(manifest), manifest.name


def test_baseline_guard_owners_are_required() -> None:
    policies = {
        manifest.name: manifest.gate_policy
        for manifest in _bundled_guard_manifests()
    }

    missing = _BASELINE_GUARDS - set(policies)
    assert not missing
    assert all(policies[name] == "required" for name in _BASELINE_GUARDS)
