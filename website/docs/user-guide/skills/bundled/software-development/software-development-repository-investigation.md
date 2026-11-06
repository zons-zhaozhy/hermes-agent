---
title: "Repository Investigation — Map code, callers, history, and invariants before changes"
sidebar_label: "Repository Investigation"
description: "Map code, callers, history, and invariants before changes"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Repository Investigation

Map code, callers, history, and invariants before changes.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/repository-investigation` |
| Version | `1.0.0` |
| Author | Stan + Hermes Agent (adapted from Codex engineering practice) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `investigation`, `repository`, `architecture`, `callers`, `history`, `invariants` |
| Related skills | [`brainstorming`](../../bundled/software-development/software-development-brainstorming.md), [`systematic-debugging`](../../bundled/software-development/software-development-systematic-debugging.md), [`codebase-inspection`](../../bundled/software-development/software-development-codebase-inspection.md), [`writing-plans`](../../bundled/software-development/software-development-writing-plans.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Repository Investigation

Build an evidence-backed model of the repository before proposing or making a change. This
skill produces an investigation result; it does not itself authorize editing.

## When to Use

- before a refactor, interface change, migration, or security-sensitive modification;
- when a request mentions a subsystem but the exact files, callers, or state are unknown;
- when documentation, memory, or a prior conversation disagrees with current behavior;
- before writing an implementation plan or estimating blast radius.

Use `codebase-inspection` for size and language metrics. Use `systematic-debugging` when an
observed failure is the primary subject.

## Prerequisites

- `read_file`, `search_files`, and `terminal` for repository evidence.
- `skill_view` for follow-up workflow skills.
- Access to current tests, configuration, logs, and version history when they exist.

## How to Run

1. Define the exact question and change boundary.
2. Locate the current implementation and authoritative interfaces.
3. Trace callers, data flow, sibling paths, and lifecycle boundaries.
4. Identify tests, fixtures, configuration, migrations, and rollback behavior.
5. Check recent history for intentional invariants.
6. Record evidence, uncertainties, and the smallest safe verification loop.
7. Hand off to `brainstorming`, `systematic-debugging`, or `writing-plans` as appropriate.

## Quick Reference

| Question | Evidence to collect |
|---|---|
| Where does behavior live? | Exact files and symbols, not just similarly named modules |
| Who depends on it? | Callers, imports, routes, jobs, UI consumers, persisted formats |
| Why is it this way? | Commit or issue that introduced the current contract |
| How is it verified? | Existing tests, canonical runner, fixtures, manual repro |
| What can break? | Sibling paths, migration/state, rollback, platform/profile scope |
| What remains unknown? | Explicit uncertainty and the cheapest probe that resolves it |

## Procedure

### 1. Frame the investigation

Write the concrete question, desired outcome, authorization boundary, and repository state.
Distinguish "explain the current system" from "change the current system"; investigation alone
is not permission to edit.

### 2. Locate authoritative implementation

Use `search_files` for exact symbol names, error strings, route strings, config keys, and
schema fields. Then use `read_file` on the defining module and its local area. Do not infer a
facade's behavior from its re-exports: locate the sibling or implementation that owns the
behavior.

For each load-bearing claim, record file and symbol. A claim without a current source,
command output, log, schema, or document is labeled inference or unverified.

### 3. Trace the full path

Map at least:

- entry points and callers;
- inputs, outputs, errors, retries, cancellation, and persistence;
- background jobs, caches, queues, hooks, and lifecycle callbacks;
- profile, platform, permission, and tenant scope;
- sibling implementations that share the bug or invariant;
- consumers of public interfaces and persisted formats.

Search by all stable spellings: symbol, route, config key, old name, error text, and schema
field. A rename may leave documentation, tests, migrations, or telemetry behind.

### 4. Find the intended contract

Read the nearby tests and area instructions before treating behavior as accidental. Check
history with `terminal` when a design choice is surprising. An absence may be load-bearing,
and an invariant may be documented in a test rather than prose.

### 5. Map verification and rollback

Identify the canonical test command and the narrow red-capable repro. Find the tests that
must change and the regression scope that should stay green. For state or interface changes,
identify compatibility, migration, backup, and rollback paths before design.

### 6. Produce the evidence map

Report:

| Claim | Evidence | Confidence |
|---|---|---|
| Current behavior | file/symbol or command output | verified |
| Consumer impact | caller/test/route | verified or inferred |
| Historical intent | commit/issue/test | verified |
| Risk | explicit failure mode | inferred |

Separate facts, hypotheses, and blocked evidence. Name the next cheapest probe for every
important hypothesis rather than jumping to a patch.

### 7. Choose the next workflow

- Ambiguous product or architecture choice → `brainstorming`.
- Observed defect or performance anomaly → `systematic-debugging`.
- Approved multi-step design → `writing-plans`.
- Only repository size/language statistics → `codebase-inspection`.

## Pitfalls

1. Editing after one file read because the change looks local.
2. Trusting memory, stale documentation, or another session's summary over current disk.
3. Searching only the exact new spelling after a rename.
4. Ignoring tests that encode intended behavior as implementation accidents.
5. Treating a facade as the implementation.
6. Forgetting profile, platform, background, and persisted-state boundaries.
7. Turning a partial investigation into a confident design recommendation.

## Verification

Before leaving this skill:

- [ ] Exact files and symbols for the target behavior are identified.
- [ ] Callers, sibling paths, tests, configuration, and state effects were searched.
- [ ] Historical intent was checked where behavior is surprising.
- [ ] Every material claim is marked verified, inferred, or unverified.
- [ ] A red-capable verification path and regression scope are named.
- [ ] No edit was made without separate authorization.
