---
title: "Systematic Debugging — Find root causes with evidence before fixing"
sidebar_label: "Systematic Debugging"
description: "Find root causes with evidence before fixing"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Systematic Debugging

Find root causes with evidence before fixing.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/systematic-debugging` |
| Version | `2.0.0` |
| Author | Stan + Hermes Agent (adapted from obra/superpowers) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `debugging`, `troubleshooting`, `root-cause`, `investigation`, `evidence` |
| Related skills | [`test-driven-development`](../../bundled/software-development/software-development-test-driven-development.md), [`repository-investigation`](../../bundled/software-development/software-development-repository-investigation.md), [`condition-based-waiting`](../../bundled/software-development/software-development-condition-based-waiting.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Systematic Debugging

Diagnose the exact mechanism before changing code. Build a tight feedback loop, read the full
error, trace data to its source, test one falsifiable hypothesis at a time, and add defense in
depth after the root cause is proven.

## When to Use

- any error, test failure, crash, hang, performance anomaly, or unexpected behavior;
- a guard, migration, service, browser flow, or platform-specific path fails;
- an earlier fix did not work or exposed another failure.

Use `repository-investigation` when the main question is how an unfamiliar system is designed.
Use `condition-based-waiting` when diagnosing a race or readiness condition.

## Prerequisites

- Original error text, log, trace, test output, or user-visible symptom.
- Hermes `terminal`, `read_file`, and `search_files`.
- A way to observe the failing behavior repeatedly, or a plan to build one.

## How to Run

1. Read and preserve the original error evidence.
2. Build or identify a red-capable feedback loop.
3. Inspect recent changes and trace data flow across boundaries.
4. Compare against a working sibling or reference.
5. Rank and minimally test falsifiable hypotheses.
6. Fix the proven root cause through TDD.
7. Add proportional defense in depth and fresh regression evidence.

## Quick Reference

| Phase | Exit condition |
|---|---|
| 1. Evidence | Exact symptom, path, and likely mechanism are understood |
| 2. Pattern | A working comparison explains the meaningful difference |
| 3. Hypothesis | One prediction is confirmed or falsified minimally |
| 4. Fix | Regression test was red, root-cause fix is green |
| 5. Defense | Siblings, guards, observability, and docs are proportionally covered |

## Procedure

### 1. Read the original evidence

Copy the complete error, stack, status code, timestamp, process/profile identity, and command.
Extract file, symbol, line, host, and user/session scope before searching. Never paraphrase an
error into a generic keyword and guess.

Check the newest relevant logs before changing code. When logs lack a field needed to locate
the event, treat that as evidence and improve diagnosis before proposing a product fix.

### 2. Build a tight feedback loop

Create or identify one command that fails on the exact symptom and can pass only after the
defect is fixed. Prefer, in order:

1. a targeted real-seam test;
2. a CLI or HTTP probe against the smallest live slice;
3. a deterministic script or replay from captured input;
4. a temporary diagnostic harness with tagged output;
5. a stress or repetition loop for an intermittent symptom.

For timing behavior, wait on events or observable predicates with deadlines. Do not use fixed
delays to make a race less frequent.

Record the red output once before changing production code.

### 3. Inspect change and state

Compare current work to the last known-good revision. Inspect configuration, schema, secrets
presence, environment scope, migration state, cache shape, process identity, and platform.
Never infer infrastructure location or tenant from a name; read the active configuration and
logs.

### 4. Trace boundaries

Follow the bad value or failed transition upstream and downstream. At every component boundary,
name input, output, error, retry, timeout, cancellation, persistence, and ownership. Continue
to the source that first creates or fails to validate the bad state.

Search exact symbols, error strings, old names, sibling paths, and consumers. A local patch at
the deepest crash site is a symptom patch if the origin remains unguarded.

### 5. Minimize and compare

Reduce the repro one element at a time while it remains red. Find a working sibling or
reference and list meaningful differences. Read a referenced implementation completely before
adapting it.

### 6. Test hypotheses scientifically

Write three to five hypotheses when the cause is not obvious. For each, state a prediction and
the cheapest observation that can falsify it. Probe one variable at a time.

Temporary diagnostics need a unique tag and must be removed after evidence is captured. Two
failed correction attempts require returning to evidence; three indicate an architectural
assumption that must be questioned.

### 7. Fix through TDD

Turn the minimized repro into a regression test and observe red. Fix the proven source with the
smallest change, then observe green. In this repository, run targeted tests through:

```bash
HERMES_PYTHON="$PWD/.venv/bin/python" scripts/run_tests.sh tests/<area>/test_file.py -q
```

Do not bundle unrelated refactors or speculative cleanup with the fix.

### 8. Add proportional defense in depth

After the root cause is fixed, ask which layer is the single best enforcement point and what
evidence would detect recurrence:

- search and fix sibling paths in the same class;
- add an invariant test at the public seam;
- add a guard only where it is precise, testable, and has a real escape path;
- log the decision point with enough scope to diagnose recurrence;
- document the invariant where future contributors will read it.

Do not add hooks, flags, retries, or telemetry without a concrete failure class and consumer.
Avoid silent fallback; if degradation is intentional, warn and leave an audit trail.

## Pitfalls

1. Proposing a fix before reading the original error.
2. Treating a nearby generic failure as the user's symptom.
3. Testing only whether a call no longer raises, not whether the contract is restored.
4. Changing two variables at once.
5. Fixing the downstream crash instead of the upstream source.
6. Hiding a race with a larger delay.
7. Adding a broad guard that creates false positives.
8. Declaring recovery without rerunning the original repro.

## Verification

Before leaving this skill:

- [ ] Full original error and relevant log evidence were read.
- [ ] The tight loop was red on the exact symptom.
- [ ] Root cause is stated with file, symbol, and data flow.
- [ ] A prediction was confirmed before the fix.
- [ ] Regression test was observed red, then green.
- [ ] Sibling paths and recurrence detection were considered.
- [ ] Fresh targeted and regression-scope results are recorded.
