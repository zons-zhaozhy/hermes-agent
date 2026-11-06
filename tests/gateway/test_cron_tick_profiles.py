"""``cron.tick_profiles`` scopes ONE gateway's tick set, so a split install can run an
interactive gateway and a scheduled-fleet gateway side by side.

Regression for the 2026-11-06 stall: a single multiplexer ticked 30 profiles, and the scheduled
fleet's turns (1445 model calls, 81.7% of the process total) inflated every interactive call's
latency (r=+0.69 between concurrent sessions and latency) while sharing its event loop. The split
needs each process to own a disjoint tick set; ``_cron_profile_gate`` still holds a profile owned
by another live gateway, so an overlap degrades to one ticker instead of a double fire.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.run import _cron_tick_profile_homes
from gateway.run_cron_tick import tick_profile_override


class _Config:
    """Stand-in for the gateway config object (only ``_runtime_profile_homes`` is read)."""


@pytest.fixture
def profiles(monkeypatch, tmp_path):
    """Serve three named profiles and resolve their homes under a temp root."""
    homes = {name: tmp_path / "profiles" / name for name in ("alpha", "beta", "gamma")}
    for home in homes.values():
        home.mkdir(parents=True)
    served = [(name, home) for name, home in homes.items()]

    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name in homes, raising=True)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda name: homes[name], raising=True)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "alpha", raising=True)
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes", lambda config: list(served), raising=True)
    return homes


def _set_tick_profiles(monkeypatch, value):
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"cron": {"tick_profiles": value}}, raising=True)
    monkeypatch.setattr("hermes_cli.config.cfg_get", _cfg_get, raising=True)


def _cfg_get(cfg, *keys, default=None):
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def test_absent_key_ticks_every_served_profile(profiles, monkeypatch):
    _set_tick_profiles(monkeypatch, None)
    assert tick_profile_override(_Config()) == []
    assert [name for name, _home in _cron_tick_profile_homes(_Config())] == [
        "alpha", "beta", "gamma",
    ]


def test_tick_profiles_replaces_the_served_set(profiles, monkeypatch):
    """The scheduled-fleet gateway names its own profiles; the interactive one names only its own."""
    _set_tick_profiles(monkeypatch, ["beta", "gamma"])
    assert tick_profile_override(_Config()) == ["beta", "gamma"]
    assert [name for name, _home in _cron_tick_profile_homes(_Config())] == ["beta", "gamma"]

    _set_tick_profiles(monkeypatch, ["alpha"])
    assert [name for name, _home in _cron_tick_profile_homes(_Config())] == ["alpha"]


def test_comma_string_is_accepted(profiles, monkeypatch):
    """``config.yaml`` may hold either a list or a comma string."""
    _set_tick_profiles(monkeypatch, " beta , gamma ")
    assert tick_profile_override(_Config()) == ["beta", "gamma"]


def test_unknown_names_are_dropped_not_fatal(profiles, monkeypatch):
    """A typo must not silently stop the rest of the fleet (or the jobs of the typo'd profile)."""
    _set_tick_profiles(monkeypatch, ["beta", "ghost"])
    assert [name for name, _home in _cron_tick_profile_homes(_Config())] == ["beta"]


def test_all_unknown_falls_back_to_the_served_set(profiles, monkeypatch):
    """Nothing resolvable = misconfiguration: keep ticking rather than go silently idle."""
    _set_tick_profiles(monkeypatch, ["ghost", "phantom"])
    assert [name for name, _home in _cron_tick_profile_homes(_Config())] == [
        "alpha", "beta", "gamma",
    ]


def test_empty_list_keeps_the_served_set(profiles, monkeypatch):
    _set_tick_profiles(monkeypatch, [])
    assert tick_profile_override(_Config()) == []
    assert len(_cron_tick_profile_homes(_Config())) == 3
