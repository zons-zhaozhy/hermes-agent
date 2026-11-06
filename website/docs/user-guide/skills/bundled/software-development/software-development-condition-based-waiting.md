---
title: "Condition Based Waiting — Wait on events and observable conditions with deadlines"
sidebar_label: "Condition Based Waiting"
description: "Wait on events and observable conditions with deadlines"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Condition Based Waiting

Wait on events and observable conditions with deadlines.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/condition-based-waiting` |
| Version | `1.0.0` |
| Author | Stan + Hermes Agent (adapted from Codex engineering practice) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `waiting`, `polling`, `deadline`, `events`, `async`, `reliability` |
| Related skills | [`test-driven-development`](../../bundled/software-development/software-development-test-driven-development.md), [`systematic-debugging`](../../bundled/software-development/software-development-systematic-debugging.md), [`repository-investigation`](../../bundled/software-development/software-development-repository-investigation.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Condition-Based Waiting

Wait for observable state, not for a guessed number of seconds. A correct wait has a condition,
a monotonic deadline, bounded polling, cancellation, and a useful timeout error.

## When to Use

- tests or checks depend on another process, thread, event loop, service, or filesystem;
- polling an API, job, lock, process, log marker, or eventual-consistency result;
- implementing retry or readiness logic;
- a timing bug is intermittent and tempting to hide with a larger delay.

Do not use waiting to mask an unknown root cause; load `systematic-debugging` first when the
condition should already be true.

## Prerequisites

- An observable predicate that can safely be evaluated repeatedly.
- A deadline appropriate to the slowest legitimate completion, not the happy path.
- A cancellation mechanism for interactive or shutdown paths.
- Hermes `terminal` for commands and `read_file`/`search_files` for implementation review.

## How to Run

1. Define success as a predicate over observable state.
2. Prefer an event, future, callback, condition variable, or readiness endpoint.
3. Otherwise poll the predicate with a monotonic deadline and modest interval.
4. Yield correctly on the event loop and remain cancellable.
5. On timeout, report elapsed time, last observed state, deadline, and next diagnosis.

## Quick Reference

| Need | Correct shape |
|---|---|
| Thread finished | Wait on its completion event or join with a deadline |
| Async task finished | Await the task under `wait_for` or equivalent |
| External service ready | Poll a readiness endpoint or documented state object |
| File appears | Poll existence plus expected content or inode state |
| Process exits | Wait on process handle or monitored lifecycle event |
| Two workers converge | Use barriers/counts, not independent clock guesses |
| Test assertion | `wait_until(predicate, timeout=...)`, never a fixed sleep |

## Procedure

### 1. Define the condition

Write the exact predicate and its source of truth: process exit code, task state, queue depth,
database row, readiness response, file content, or event. "Some seconds passed" is not a
condition.

### 2. Select the strongest primitive

Use an event or future when the producer can signal completion. Use a readiness/state API when
one exists. Poll only when there is no signal, the producer is outside the process, or the
source is intentionally eventually consistent.

### 3. Bound the wait

Set a deadline with a monotonic clock. Choose the timeout from measured slow-path behavior and
say the number's source. Poll often enough to notice progress, rarely enough to starve the
system under test. Every loop iteration must re-check cancellation and deadline.

### 4. Preserve progress evidence

Retain the last predicate result, error, state snapshot, sequence number, or timestamp. On
timeout, this evidence distinguishes "never started", "in progress", "failed", and "wrong
condition".

### 5. Avoid race-shaped assertions

Do not assert immediately after spawning work. Do not turn a green fixed delay into proof. If
multiple conditions must hold, wait for the conjunction and report the first failing component.

### 6. Diagnose timeout as a symptom

On timeout, inspect the producer and condition path with `systematic-debugging`. Do not merely
increase the delay, retry forever, or downgrade the assertion.

## Pitfalls

1. Using wall-clock time, which jumps during DST or clock synchronization.
2. Sleeping first and then checking, adding latency to every successful run.
3. Busy polling at high frequency.
4. Missing cancellation and blocking shutdown.
5. Treating a HTTP 200 from the wrong process as readiness.
6. Checking file existence before required content or metadata is durable.
7. Making a flaky test less frequent without identifying the race.

## Verification

Before leaving this skill:

- [ ] Success is an observable condition, not elapsed time alone.
- [ ] The wait uses the strongest available signal.
- [ ] Deadline and poll interval are explicit and bounded.
- [ ] Interactive and shutdown paths remain cancellable.
- [ ] Timeout reports last state and elapsed time.
- [ ] Timing assumptions cite measurements or project limits.
