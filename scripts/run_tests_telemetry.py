"""Pure timing telemetry for the per-file parallel test runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_LOCAL_RUN_ADVICE_SECONDS = 30 * 60


@dataclass(frozen=True)
class RunWorkEstimate:
    total_seconds: float
    cached_files: int
    total_files: int


def format_duration(seconds: float | None) -> str:
    """Render a duration compactly without wall-clock dependencies."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--"
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def estimate_run_work(
    files: Sequence[Path | str],
    durations: Mapping[str, float],
    default_duration_seconds: float,
) -> RunWorkEstimate:
    """Estimate serial subprocess work from cache entries and a stated default."""
    total = 0.0
    cached = 0
    for file in files:
        duration = durations.get(str(file))
        if duration is None:
            total += default_duration_seconds
        else:
            cached += 1
            total += float(duration)
    return RunWorkEstimate(total, cached, len(files))


def format_run_estimate(estimate: RunWorkEstimate, workers: int) -> str:
    estimated_wall = estimate.total_seconds / max(workers, 1)
    return (
        f"Duration estimate: {estimate.total_seconds:.1f}s subprocess work "
        f"({estimate.cached_files}/{estimate.total_files} files cached) "
        f"-> ~{estimated_wall:.1f}s at {workers} workers"
    )


def run_start_lines(estimate: RunWorkEstimate, workers: int) -> list[str]:
    lines = [format_run_estimate(estimate, workers)]
    advice = local_run_advice(estimate.total_seconds / max(workers, 1))
    if advice:
        lines.append(advice)
    return lines


def local_run_advice(estimated_wall_seconds: float) -> str | None:
    if estimated_wall_seconds < _LOCAL_RUN_ADVICE_SECONDS:
        return None
    return (
        "Local estimate is "
        f"{format_duration(estimated_wall_seconds)}; for a full batch prefer CI/slice "
        "and keep this host on targeted tests."
    )


def summary_lines(
    tests_collected: int,
    runner_wall: float,
    total_subprocess_wall: float,
    file_count: int,
    workers: int,
    load_1m: float | None,
) -> list[str]:
    throughput = tests_per_minute(tests_collected, runner_wall)
    utilization = parallel_utilization(total_subprocess_wall, runner_wall, workers)
    effective_parallelism = (
        total_subprocess_wall / runner_wall if runner_wall > 0 else 0.0
    )
    lines = [
        "=== Test telemetry ===",
        (
            f"  Throughput: {throughput:.1f} tests/min "
            f"({tests_collected} tests / {runner_wall:.1f}s)"
        ),
        f"  Average file: {total_subprocess_wall / file_count:.2f}s",
        (
            f"  Effective parallelism: {effective_parallelism:.2f}x "
            f"(subprocess wall {total_subprocess_wall:.1f}s / "
            f"runner wall {runner_wall:.1f}s)"
        ),
        f"  Parallel utilization: {utilization * 100:.1f}% of {workers} workers",
    ]
    if load_1m is not None:
        warning = host_load_warning(load_1m, workers)
        suffix = f" {warning}" if warning else ""
        lines.append(f"  Host load: {load_1m:.1f} (1m){suffix}")
    return lines


def tests_per_minute(tests_done: int, runner_elapsed_seconds: float) -> float:
    if runner_elapsed_seconds <= 0:
        return 0.0
    return tests_done * 60.0 / runner_elapsed_seconds


def eta_seconds(
    runner_elapsed_seconds: float,
    completed_subprocess_wall: float,
    remaining_estimated_work: float,
) -> float | None:
    """Project remaining wall time from measured effective parallelism."""
    if runner_elapsed_seconds <= 0 or completed_subprocess_wall <= 0:
        return None
    effective_parallelism = completed_subprocess_wall / runner_elapsed_seconds
    if effective_parallelism <= 0:
        return None
    return remaining_estimated_work / effective_parallelism


def format_progress_telemetry(
    runner_elapsed_seconds: float,
    tests_done: int,
    completed_subprocess_wall: float,
    remaining_estimated_work: float,
) -> str:
    throughput = tests_per_minute(tests_done, runner_elapsed_seconds)
    effective_parallelism = (
        completed_subprocess_wall / runner_elapsed_seconds
        if runner_elapsed_seconds > 0 and completed_subprocess_wall > 0
        else 0.0
    )
    remaining = eta_seconds(
        runner_elapsed_seconds,
        completed_subprocess_wall,
        remaining_estimated_work,
    )
    eta = format_duration(remaining)
    return (
        f"{throughput:.1f} tests/min | ETA {eta} | "
        f"eff {effective_parallelism:.1f}x"
    )


def parallel_utilization(
    total_subprocess_wall: float,
    runner_wall: float,
    workers: int,
) -> float:
    if runner_wall <= 0 or workers <= 0:
        return 0.0
    return total_subprocess_wall / runner_wall / workers


def host_load_warning(load_1m: float, workers: int) -> str | None:
    """Warn when runnable pressure is far above requested worker slots."""
    capacity = max(workers, 1)
    if load_1m <= capacity * 3:
        return None
    return (
        f"⚠ host load {load_1m:.1f} is far above {workers} workers; "
        "reduce -j or move the batch to CI"
    )
