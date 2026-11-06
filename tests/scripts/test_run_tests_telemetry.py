"""Telemetry contracts for the per-file parallel test runner.

The runner already reports what happened after a run. These contracts make it
report whether the run is healthy while it is happening: throughput, ETA,
parallel utilization, and host-load risk.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
_TELEMETRY_PATH = REPO_ROOT / "scripts" / "run_tests_telemetry.py"


def _load_telemetry():
    spec = importlib.util.spec_from_file_location("run_tests_telemetry", _TELEMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_progress_reports_throughput_eta_and_measured_parallelism() -> None:
    telemetry = _load_telemetry()

    details = telemetry.format_progress_telemetry(
        runner_elapsed_seconds=120.0,
        tests_done=60,
        completed_subprocess_wall=180.0,
        remaining_estimated_work=900.0,
    )

    assert "30.0 tests/min" in details
    assert "ETA 10m 00s" in details
    assert "eff 1.5x" in details


def test_progress_does_not_invent_eta_before_work_completes() -> None:
    telemetry = _load_telemetry()

    details = telemetry.format_progress_telemetry(
        runner_elapsed_seconds=0.5,
        tests_done=2,
        completed_subprocess_wall=0.0,
        remaining_estimated_work=120.0,
    )

    assert "tests/min" in details
    assert "ETA" in details
    assert "ETA --" in details


def test_parallel_utilization_uses_worker_capacity() -> None:
    telemetry = _load_telemetry()

    assert telemetry.parallel_utilization(
        total_subprocess_wall=300.0,
        runner_wall=200.0,
        workers=2,
    ) == 0.75


def test_duration_estimate_distinguishes_cached_and_default_work() -> None:
    telemetry = _load_telemetry()
    files = [Path("/repo/tests/cached.py"), Path("/repo/tests/new.py")]

    estimate = telemetry.estimate_run_work(
        files=files,
        durations={str(files[0]): 8.0},
        default_duration_seconds=2.0,
    )

    assert estimate.total_seconds == 10.0
    assert estimate.cached_files == 1
    assert telemetry.format_run_estimate(estimate, workers=2) == (
        "Duration estimate: 10.0s subprocess work (1/2 files cached) -> ~5.0s at 2 workers"
    )


def test_local_long_run_advice_names_ci_and_slices() -> None:
    telemetry = _load_telemetry()

    advice = telemetry.local_run_advice(estimated_wall_seconds=3_600.0)

    assert "Local estimate is 1h 00m" in advice
    assert "CI/slice" in advice


def test_host_load_warning_uses_worker_pressure_not_absolute_load() -> None:
    telemetry = _load_telemetry()

    assert telemetry.host_load_warning(load_1m=3.0, workers=2) is None
    warning = telemetry.host_load_warning(load_1m=8.0, workers=2)

    assert warning is not None
    assert "host load 8.0" in warning
    assert "2 workers" in warning


def test_runner_prints_live_and_final_telemetry(tmp_path: Path) -> None:
    root = tmp_path / "runner-root"
    scripts = root / "scripts"
    (scripts / "ci").mkdir(parents=True)
    real_scripts = REPO_ROOT / "scripts"
    for name in (
        "run_tests_parallel.py",
        "run_tests_launch.py",
        "run_tests_fork_child.py",
        "run_tests_telemetry.py",
    ):
        shutil.copy2(real_scripts / name, scripts / name)
    shutil.copy2(real_scripts / "ci" / "list_os_marked_tests.py", scripts / "ci")

    probe = tmp_path / "test_telemetry_probe.py"
    probe.write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n",
        encoding="utf-8",
    )
    (root / "test_durations.json").write_text(
        json.dumps({str(probe): 4.0}) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(scripts / "run_tests_parallel.py"),
            "--files",
            str(probe),
            "-j",
            "1",
            "--file-timeout",
            "30",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout
    assert "Duration estimate: 4.0s subprocess work (1/1 files cached)" in proc.stdout
    progress = next(line for line in proc.stdout.splitlines() if "tests/min" in line)
    assert "ETA" in progress
    assert "Throughput:" in proc.stdout
    assert "Average file:" in proc.stdout
    assert "Parallel utilization:" in proc.stdout
    if hasattr(os, "getloadavg"):
        assert "Host load:" in proc.stdout
