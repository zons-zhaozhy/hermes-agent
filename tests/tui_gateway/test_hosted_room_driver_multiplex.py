"""Multiplexed hosted-room supervisor scheduling contracts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from tui_gateway.hosted_room_driver import HostedRoomRuntime, _idle_delay_seconds, _poll_phase_seconds


def test_multiplexed_runtime_poll_phases_are_stable_and_staggered(tmp_path: Path):
    interval = 5.0
    paths = [tmp_path / f"profile-{index}" / "state.db" for index in range(29)]

    phases = [_poll_phase_seconds(path, interval) for path in paths]

    assert all(0.0 <= phase < interval for phase in phases)
    assert len(set(phases)) == len(paths), (
        "each served profile store must contribute a distinct startup phase"
    )
    assert _poll_phase_seconds(paths[0], interval) == phases[0]


def test_empty_multiplexed_store_backs_off_but_stays_bounded():
    assert [_idle_delay_seconds(5.0, cycle) for cycle in range(6)] == [
        5.0, 10.0, 20.0, 40.0, 60.0, 60.0]


def test_empty_multiplexed_cycle_reports_no_rooms(tmp_path: Path):
    @contextmanager
    def turn_lock(_profile):
        yield

    runtime = HostedRoomRuntime(
        db_path=tmp_path / "state.db", rooms=[], rpc=object, turn_lock=turn_lock)

    assert runtime._run_cycle() is False
