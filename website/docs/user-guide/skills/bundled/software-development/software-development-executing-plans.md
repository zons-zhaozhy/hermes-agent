---
title: "Executing Plans — Run written plans with checkpoints and evidence"
sidebar_label: "Executing Plans"
description: "Run written plans with checkpoints and evidence"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Executing Plans

Run written plans with checkpoints and evidence.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/executing-plans` |
| Version | `1.0.0` |
| Author | Stan + Hermes Agent (adapted from obra/superpowers) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `plan-execution`, `implementation`, `checkpoints`, `tdd`, `verification`, `change-control` |
| Related skills | [`writing-plans`](../../bundled/software-development/software-development-writing-plans.md), [`test-driven-development`](../../bundled/software-development/software-development-test-driven-development.md), [`systematic-debugging`](../../bundled/software-development/software-development-systematic-debugging.md), [`requesting-code-review`](../../bundled/software-development/software-development-requesting-code-review.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Executing Plans Skill

Execute a written implementation plan one task at a time, with TDD evidence and explicit
stops for blockers. This skill controls scope and sequencing; it does not grant authority to
commit, publish, spend, or mutate remote systems.

## When to Use

Use when:

- a written plan and its spec are ready;
- an approved implementation spans multiple testable tasks;
- another agent produced a plan that must be validated against the current repository.

Use `systematic-debugging` before continuing when an unexpected failure invalidates the
plan's assumptions.

## Prerequisites

- A readable plan file.
- The cited spec file for Architectural work.
- `read_file`, `search_files`, `terminal`, `patch`, and `write_file`.
- The session task tracker when available.

## How to Run

1. Read the spec and plan completely.
2. Review the plan against the current disk state.
3. Create one tracked item per task.
4. Execute each task through its red-green steps.
5. Stop on ambiguity, repeated failure, or scope expansion.
6. Run final verification and hand off to review.

## Quick Reference

| Situation | Action |
|---|---|
| Plan matches reality | Execute the next unchecked task |
| Current file differs from plan assumptions | Stop and update the plan or request direction |
| Test fails for an unexpected reason | Load `systematic-debugging` |
| New work is outside the plan | Stop and request a scope decision |
| Verification fails | Report actual output; do not claim completion |
| Commit, push, PR, send, spend, or remote write is next | Stop for user authorization |

## Procedure

### 1. Load the complete contract

Read the plan and the cited spec. For each task, inspect the current versions of every file
named by that task immediately before editing. Do not patch from memory or from the plan's
quoted old text when the file has changed.

### 2. Review before execution

Check that:

- required files and interfaces exist;
- the plan's assumptions match current disk state;
- test commands are runnable;
- task order respects dependencies;
- no task silently omits a requirement from the spec.

Report a blocker before implementation if the plan cannot be executed as written.

### 3. Track the plan

Create one task-tracker item per plan task. Move one item to in progress at a time. Mark it
complete only after its own tests and verification step pass.

### 4. Execute with TDD

For each behaviour slice:

1. add the exact failing test from the plan;
2. run it with the plan's command;
3. confirm the failure reason matches the expected red signal;
4. implement only enough to pass;
5. rerun the targeted test;
6. refactor while green;
7. run the task's regression checks.

If the red result differs from the plan, stop and investigate. A typo or broken fixture is
not the expected red state.

### 5. Control scope

Keep edits inside the files and interfaces owned by the current task. Record an unexpected
issue that does not block the current task; do not perform unrelated cleanup or "while I am
here" refactoring.

### 6. Stop for the right reasons

Stop immediately when:

- the plan contradicts the repository;
- an instruction is ambiguous;
- a dependency or permission is unavailable;
- a precondition test fails;
- verification fails repeatedly;
- the root cause remains unknown after two failed correction attempts;
- the task requires a new interface, migration, destructive action, spend, or external
  effect outside the approved plan.

Ask for direction instead of inventing scope.

### 7. Use debugging when assumptions fail

Load `skill_view("systematic-debugging")` after an unexpected failure. Build a tight feedback
loop and find the root cause before changing the plan or production code.

### 8. Delegate only with authorization

Do not dispatch subagents unless the user or project instruction explicitly authorizes
delegation. When delegation is authorized, give the child exact task boundaries, files, tests,
and constraints; independently verify its changes and claims.

### 9. Preserve approval boundaries

Never infer permission to commit, push, open a pull request, release, publish, send a message,
spend money, or write remotely from approval of the implementation plan.

Stop before each such effect and show the exact target and action.

### 10. Finish with evidence

Run the plan's final verification commands. Load
`skill_view("verification-before-completion")` before summarizing success. Report what was
verified, what was not, and any remaining blocker from fresh command output.

## Pitfalls

1. Executing the plan without reading the cited spec.
2. Assuming the plan's quoted context is still current.
3. Replacing the planned test with a test derived from the implementation.
4. Continuing after an unexpected red failure.
5. Treating a local task completion as permission to commit or push.
6. Expanding scope because an unrelated defect was noticed.
7. Delegating work without explicit authorization or without verifying the result.
8. Claiming the plan is complete before final verification.

## Verification

Before leaving this skill:

- [ ] The complete spec and plan were read.
- [ ] The plan was checked against current disk state.
- [ ] Every task was tracked and completed in order.
- [ ] Every behaviour slice was observed red before implementation and green afterward.
- [ ] Regression checks for touched areas passed.
- [ ] Unexpected failures used root-cause investigation.
- [ ] Scope stayed within the approved plan.
- [ ] No external effect or irreversible action bypassed user approval.
- [ ] Final claims cite fresh command output.
