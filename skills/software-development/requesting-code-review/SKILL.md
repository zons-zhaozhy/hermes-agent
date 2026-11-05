---
name: requesting-code-review
description: "Review changes before commit with fresh evidence."
version: 3.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, security, verification, quality, diff]
    related_skills: [receiving-code-review, verification-before-completion, systematic-debugging]
    category: software-development
---

# Requesting Code Review

Review the work product against requirements, repository invariants, security, and fresh test
evidence before integration. Review does not authorize committing, pushing, publishing, or any
other external effect.

## When to Use

- after a major feature, security change, migration, or cross-interface bug fix;
- before merge or integration;
- when project instructions require a second review;
- when implementation required more than a trivial local edit.

Use `receiving-code-review` once findings arrive. Use `systematic-debugging` when a finding is
not reproducible.

## Prerequisites

- Exact base and head revisions, or the exact current diff.
- The approved requirement, plan, or issue contract.
- Fresh output from the canonical test and quality commands.
- `read_file`, `search_files`, and `terminal` for evidence.

## How to Run

1. Define the change contract and review scope.
2. Collect base, head, diff, status, and untracked files.
3. Review requirements and architecture before line-by-line details.
4. Run security, correctness, test, and maintainability checks.
5. Request an independent reviewer only when explicitly authorized.
6. Send findings through `receiving-code-review`; fix with TDD.
7. Rerun review and verification after fixes.

## Quick Reference

| Review dimension | Questions |
|---|---|
| Requirements | Does the diff implement every approved outcome and nothing else? |
| Architecture | Does it preserve boundaries, caching invariants, and rollback safety? |
| Correctness | Are errors, state, cancellation, concurrency, and platform handled? |
| Security | Are trust boundaries, authorization, secrets, input, and output safe? |
| Evidence | Were exact commands run on the final tree? |
| Maintainability | Do tests cover behavior rather than implementation shape? |

## Procedure

### 1. Freeze the review object

Record the base revision, head revision, changed files, staged/unstaged state, and untracked
files. A review of an ambiguous or changing tree is not evidence. Do not stage, commit, or
discard anything merely to make review easier.

### 2. Review the contract first

Re-read the requirement, plan, issue, or user decision. List each expected outcome and map it
to a diff hunk and verification result. Report omitted requirements before discussing style.

### 3. Review the architecture

Check the repository's stated invariants: narrow core surfaces, profile scope, prompt-cache
stability, plugin boundaries, dependency limits, public compatibility, data migration, and
rollback. Verify that behavior lives in the owning module rather than a facade or duplicate.

### 4. Review correctness and failure paths

For each changed path, trace inputs, callers, errors, retries, cancellation, state transitions,
permissions, platform differences, and observability. Search sibling implementations and old
symbol names for the same bug or missed migration.

### 5. Review security

Name the trust boundary and check authorization, input validation, output encoding, path
containment, SQL parameterization, secret handling, command construction, and tenant/profile
isolation. Do not paste secret values into review output; describe their origin and scope.

### 6. Check evidence honestly

Run the repository's canonical targeted tests and quality command after the final edit. Record
scope, counts, skipped tests, and exit code. Never present a linter as a build, a targeted run
as the full suite, or a diff inspection as runtime proof.

### 7. Obtain independent review only with authorization

Delegation requires explicit user or project authorization. When authorized, provide only the
base/head revisions, requirement, diff, and verification output—not session history—and
independently inspect the result. Without delegation, perform a fresh requirements-and-diff
review and state that it was not independent.

### 8. Process findings safely

Classify each finding as blocker, important, minor, incorrect, or unclear. Do not automatically
dispatch a fix agent. Apply accepted fixes one at a time through `test-driven-development`, then
rerun review and verification.

### 9. Preserve authorization boundaries

A passing review does not authorize commit, push, PR creation, publication, messaging, spend,
credential use, or remote writes. Stop and request the exact external effect if the user wants
one.

## Pitfalls

1. Reviewing only lines changed while missing callers and migration consumers.
2. Letting a generic best practice override a documented repository invariant.
3. Trusting a prior test run after another edit.
4. Treating skipped tests or warnings as irrelevant without reading them.
5. Exposing secrets in review notes.
6. Asking a reviewer for session history instead of a precise work contract.
7. Turning review findings into unauthorized automatic fixes.
8. Treating reviewer agreement as permission to integrate.

## Verification

Before leaving this skill:

- [ ] Base, head, diff, status, and untracked files were recorded.
- [ ] Every requirement maps to implementation and fresh evidence.
- [ ] Architecture, correctness, security, and maintainability checks are complete.
- [ ] Findings are classified and processed through review discipline.
- [ ] Post-fix verification is fresh.
- [ ] No unauthorized commit, push, publication, message, spend, or remote write occurred.
