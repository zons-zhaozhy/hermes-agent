---
name: receiving-code-review
description: "Verify review feedback before changing code."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, feedback, verification, fixes, technical-judgement]
    related_skills: [requesting-code-review, test-driven-development, systematic-debugging]
---

# Receiving Code Review Skill

Process review feedback as technical claims that must be checked against the repository, not
as instructions to implement blindly. This skill owns feedback evaluation and fix ordering;
`requesting-code-review` owns producing the review.

## When to Use

Use when:

- a reviewer reports defects, missing tests, security issues, or design concerns;
- feedback is unclear or internally contradictory;
- an external review conflicts with repository invariants or the user's decisions;
- multiple review items need a safe implementation order.

Do not use performative agreement as a substitute for verification.

## Prerequisites

- The complete review feedback.
- Access to the current diff, source files, tests, configuration, and project history.
- The user's prior decisions and the approved change contract when applicable.

## How to Run

1. Read all feedback without editing.
2. Restate unclear requirements.
3. Verify each claim against the current repository.
4. Decide whether to implement, reject, or escalate.
5. Fix in blocker-first order with TDD.
6. Rerun targeted and regression checks.

## Quick Reference

| Feedback state | Response |
|---|---|
| Correct and in scope | Fix with a failing test, then verify |
| Correct but out of scope | Report the issue and ask whether to expand scope |
| Unclear | Ask for clarification before implementing any dependent items |
| Incorrect for this codebase | Push back with file, test, invariant, or compatibility evidence |
| Conflicts with user decision | Stop and ask the user |
| Cannot verify | State the missing evidence and requested direction |

## Procedure

### 1. Read everything first

Read all comments and identify their relationships. Do not start the easy items while an
unclear item may change the design or make some fixes obsolete.

### 2. Restate the technical requirement

For each item, state the required behavior rather than echoing the reviewer's wording. If a
claim is unclear, ask one focused question and wait before implementing dependent work.

### 3. Verify against reality

Inspect the relevant code, tests, callers, and history. Check:

- whether the reported failure exists;
- whether the proposed fix preserves existing behavior;
- whether the code currently relies on the present pattern;
- whether alternatives are needed for compatibility, security, or platform support;
- whether the reviewer had enough context.

Use `systematic-debugging` when a reported defect is not reproducible or its cause is
unclear.

### 4. Respond technically

For correct feedback, state the verified defect and the fix. Do not use canned gratitude,
empty agreement, or praise. For incorrect feedback, cite concrete evidence and propose the
safer path.

When feedback conflicts with an approved architectural decision, stop and raise the conflict
to the user instead of silently choosing a side.

### 5. Order the work

Resolve issues in this order:

1. security, data-loss, or blocking correctness defects;
2. simple high-confidence fixes;
3. complex logic or refactoring;
4. non-blocking cleanup.

Clarify all unclear items before implementing a batch whose pieces may depend on each other.

### 6. Use TDD for each material fix

Add or identify a test that reproduces the reviewed defect. Observe it fail for the expected
reason, apply the smallest root-cause fix, and observe it pass. Do not create tests that only
mirror the implementation.

### 7. Verify regressions

Run the targeted tests after each item and the relevant regression scope after the batch.
Report any feedback item that remains unresolved.

### 8. Respect authorization boundaries

Handling review feedback does not authorize commit, push, PR edits, external replies, spend,
or publication. Stop for user approval before any of those effects.

## Pitfalls

1. Replying with "you're absolutely right" before checking the code.
2. Implementing all comments in registration order.
3. Partially implementing a set whose unclear items may change the design.
4. Accepting a generic best practice that breaks this codebase's compatibility.
5. Adding a fix with no reproducing test.
6. Treating a non-blocking suggestion as mandatory scope.
7. Using a review response as permission to push or publish.

## Verification

Before leaving this skill:

- [ ] Every review item was read before edits began.
- [ ] Every unclear item was clarified before dependent implementation.
- [ ] Every accepted claim was verified against the current repository.
- [ ] Incorrect suggestions received evidence-backed pushback.
- [ ] Conflicts with user decisions were escalated.
- [ ] Fixes were ordered by risk and implemented with tests.
- [ ] Targeted and regression checks passed for the reviewed changes.
- [ ] No external reply, commit, push, PR, spend, or publication occurred without approval.
