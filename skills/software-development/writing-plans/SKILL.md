---
name: writing-plans
description: "Turn approved specs into test-first task plans."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, implementation-plan, spec, tdd, interfaces, verification]
    related_skills: [brainstorming, executing-plans, test-driven-development]
---

# Writing Plans Skill

Turn an approved specification into an execution plan with exact files, interfaces, tests,
and verification commands. This skill owns decomposition and plan quality; it does not
implement the change.

## When to Use

Use after:

- an Architectural spec has been approved by the user;
- a well-understood Bounded change benefits from explicit task decomposition;
- another agent needs a plan it can execute without reconstructing the original intent.

Do not use a written plan as a substitute for reading the repository at execution time.

## Prerequisites

- The approved spec path for Architectural work, or a concrete Bounded request.
- `read_file` and `search_files` access to map the real file structure.
- The repository's canonical test and quality commands.

## How to Run

1. Read the spec or request completely.
2. Check that the scope belongs to one implementation plan.
3. Map exact files and interfaces.
4. Split the work into independently testable tasks.
5. Write each task as a RED-GREEN-REFACTOR execution sequence.
6. Self-review coverage, placeholders, and type consistency.
7. Save an Architectural plan and hand execution to `executing-plans`.

## Quick Reference

Every task names:

- exact files to create, modify, and test;
- consumed and produced interfaces;
- the failing test to write;
- the command and expected red failure;
- the minimal implementation;
- the command and expected green result;
- the verification command.

Every plan excludes:

- placeholders or deferred details;
- unstated interfaces;
- tests derived from the finished implementation;
- unauthorized commit, push, PR, publish, spend, or external effects.

## Procedure

### 1. Read before decomposing

Read the full spec and inspect every file named by it. Confirm that the current disk state
still matches the spec's assumptions. Report a blocker when the spec is stale or ambiguous.

If the spec covers multiple independent subsystems, ask for separate plans, one per
subsystem.

### 2. Map files to responsibilities

Before writing tasks, list:

- files to create;
- files to modify with approximate line anchors when useful;
- files to test;
- the single responsibility of each file;
- interfaces passed from one task to the next.

Prefer focused files with clear boundaries. In an existing codebase, follow its established
patterns rather than imposing a new structure.

### 3. Right-size tasks

A task is the smallest unit that has its own test cycle and is worth an independent review.
Fold required setup, configuration, and documentation into the deliverable that needs them.
Split when a reviewer could accept one task and reject its neighbor.

### 4. Use the required plan header

Start an Architectural plan with:

`# <Feature> Implementation Plan`

`> For agentic workers: use executing-plans.`

`Goal: one sentence`

`Architecture: two or three sentences`

`Tech Stack: exact technologies`

`Spec: path to the approved spec`

`Global Constraints: exact project-wide requirements`

Copy exact values from the spec. Do not summarize constraints into vague intent.

### 5. Write exact task contracts

For every task include:

- **Files:** exact create, modify, and test paths.
- **Interfaces:** exact functions, methods, types, inputs, and outputs consumed and produced.
- **Steps:** one checkbox per concrete action, each small enough to complete and verify.

Do not write "similar to Task N." Repeat the exact content so tasks can be read out of order.

### 6. Make every step executable

For a test step, include the actual test code. For an implementation step, include the
minimal code or exact patch target. For a command step, include the exact command and the
expected failure or success signal.

Each behaviour slice follows:

1. write one failing test;
2. run it and observe the expected red result;
3. add the minimal implementation;
4. run it and observe green;
5. refactor only while green;
6. run the relevant regression checks.

Use `scripts/run_tests.sh` in the Hermes repository. Use the repository's own canonical
command elsewhere.

### 7. Protect authorization boundaries

A plan may prepare a commit, push, deployment, external call, publication, or spend, but it
must stop for explicit user authorization before executing that effect.

Do not add a commit step unless the user has already authorized committing. Do not turn plan
completion into permission to push or open a PR.

### 8. Save the plan

Save Architectural plans to
`docs/superpowers/plans/YYYY-MM-DD-<topic>.md`.

Bounded work may use an in-response checklist or the session task tracker. Create a plan file
for Bounded work only when the user asks for one or a fresh executor needs the artifact.

### 9. Self-review

Check:

1. **Spec coverage:** every requirement maps to at least one task.
2. **Placeholder scan:** no TBD, TODO, "add validation", "handle edge cases", or deferred
   interface.
3. **Consistency:** names, signatures, paths, types, and commands match across tasks.
4. **TDD integrity:** every behaviour test appears before its implementation.
5. **Authorization:** no external or irreversible effect lacks an approval stop.

Fix issues inline.

### 10. Hand off

After saving the plan, load `executing-plans`. Do not begin implementation as part of planning.

## Pitfalls

1. Writing tasks from memory instead of reading the current files.
2. Creating a horizontal test pile before any implementation slice.
3. Omitting the exact interface between neighboring tasks.
4. Treating "run tests" as enough without the command and expected signal.
5. Including placeholders that force the executor to invent scope.
6. Adding commit or deployment commands as if plan approval authorized them.
7. Producing one giant task whose failure cannot be isolated.
8. Copying the upstream workflow's host commands instead of Hermes commands.

## Verification

Before leaving this skill:

- [ ] The approved spec or bounded request was read completely.
- [ ] The scope belongs to one plan.
- [ ] Every task has exact files and interfaces.
- [ ] Every behaviour slice contains a red command and a green command.
- [ ] Every command is repository-native and exact.
- [ ] Placeholder and type-consistency scans are clean.
- [ ] Every spec requirement maps to a task.
- [ ] No commit, push, PR, publish, spend, or external effect is executed without user
      authorization.
- [ ] The plan is saved at the required path for Architectural work.
