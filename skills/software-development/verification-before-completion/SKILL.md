---
name: verification-before-completion
description: "Prove completion with fresh command output."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [verification, completion, testing, evidence, quality, red-green]
    related_skills: [test-driven-development, executing-plans, requesting-code-review, systematic-debugging]
---

# Verification Before Completion Skill

Prevent completion claims from getting ahead of observable evidence. This skill governs
status language, not implementation; it requires a fresh run of the command that proves each
claim.

## When to Use

Use immediately before saying or implying that work is:

- complete;
- fixed;
- passing;
- ready to merge or ship;
- verified end to end;
- safe to commit, publish, or hand off.

Use it before reporting task completion, after review fixes, and whenever a previous claim
relied on stale or partial output.

## Prerequisites

- A concrete claim to verify.
- The repository's canonical test, build, lint, or verification command.
- Access to fresh stdout, stderr, and the process exit code.

## How to Run

1. Identify the exact claim.
2. Identify the command that proves it.
3. Run the full command in the current workspace.
4. Read the complete output and exit code.
5. Compare the evidence with the claim.
6. State the actual result, including failure or unverified scope.

## Quick Reference

| Claim | Required evidence |
|---|---|
| Targeted test passes | Fresh targeted test run with expected pass count |
| Full suite passes | Fresh full-suite run with zero failures |
| Lint is clean | Fresh lint command with zero blocking findings |
| Build succeeds | Fresh build with exit code zero |
| Bug is fixed | Original repro now passes and regression test is green |
| Regression test is meaningful | Test was observed red before the fix |
| Requirements are met | Requirement checklist checked against fresh behavior |

## Procedure

### 1. Name the claim

Write down the exact claim, such as "the five workflow skills load" or "the scoped tests
pass." Avoid broad claims that combine verified and unverified work.

### 2. Select the proving command

Choose the command that directly exercises the claim. Use `scripts/run_tests.sh` for the
Hermes repository. Use the project's documented command elsewhere.

Do not use a nearby weaker signal as proof:

- a linter does not prove a build;
- a targeted test does not prove a full suite;
- a diff does not prove runtime behavior;
- code presence does not prove execution;
- a subagent report does not prove its changes.

### 3. Run it fresh

Run the complete command in the current state after the latest modification. Do not use:

- a previous run from before the final edit;
- a partial invocation presented as the full command;
- cached output with no current exit code;
- confidence, expectation, or inference as evidence.

### 4. Read the entire result

Read the full output, not only a success banner. Check:

- exit code;
- failure and error counts;
- skipped or deselected tests that change the claim;
- warnings that materially affect correctness;
- the paths or targets actually exercised.

If the output is truncated, rerun with a scope or reporter that preserves the deciding
evidence.

### 5. Verify requirements

Re-read the request or plan and create a short requirement checklist. For each item, name
the fresh behavior, test, or artifact that proves it. Passing tests alone do not prove an
omitted requirement.

### 6. Report the truth

When evidence passes, state the claim and cite the command result. When evidence fails, lead
with the failure and report the actual state. When evidence is unavailable, say what remains
unverified and why.

Separate:

- verified;
- failed;
- not run;
- blocked;
- outside the current scope.

### 7. Stop before unauthorized effects

Passing verification does not authorize a commit, push, pull request, release, publication,
message, spend, or remote write. Request approval for the exact external effect.

## Pitfalls

1. Saying "should pass" or "looks good" instead of running the command.
2. Reusing output from before the last edit.
3. Presenting a targeted test as full-suite evidence.
4. Treating a code diff as proof that runtime behavior changed.
5. Ignoring skipped tests or warnings that invalidate the claim.
6. Trusting a subagent's success report without checking the diff and result.
7. Claiming requirements are met because implementation tasks are checked off.
8. Treating a verified implementation as permission to push or publish.

## Verification

Before leaving this skill:

- [ ] Every completion claim names the proving command.
- [ ] Every proving command was run after the final relevant edit.
- [ ] The complete output and exit code were read.
- [ ] Targeted and full-suite scopes are labeled accurately.
- [ ] A requirement checklist is checked against evidence.
- [ ] Failed, unrun, blocked, and out-of-scope work is reported explicitly.
- [ ] No unauthorized commit, push, PR, publish, send, spend, or remote write occurred.
