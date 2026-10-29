"""Multiplexed hosted-room supervisor scheduling contracts."""

from __future__ import annotations

from pathlib import Path

from tui_gateway.hosted_room_driver import _poll_phase_seconds


def test_multiplexed_runtime_poll_phases_are_stable_and_staggered(tmp_path: Path):
    interval = 5.0
    paths = [tmp_path / f"profile-{index}" / "state.db" for index in range(29)]

    phases = [_poll_phase_seconds(path, interval) for path in paths]

    assert all(0.0 <= phase < interval for phase in phases)
    assert len(set(phases)) == len(paths), (
        "each served profile store must contribute a distinct startup phase"
    )
    assert _poll_phase_seconds(paths[0], interval) == phases[0]
