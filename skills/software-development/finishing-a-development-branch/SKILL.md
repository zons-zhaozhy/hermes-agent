---
name: finishing-a-development-branch
description: "Branch integration verification and safe cleanup."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from Codex engineering practice)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [git, branch, integration, verification, cleanup, worktree]
    related_skills: [verification-before-completion, requesting-code-review, receiving-code-review, using-git-worktrees]
    category: software-development
---

# Finishing a Development Branch

Finish only from fresh evidence. Verify the exact tree, identify the true fork base, let the
user choose integration, and preserve work whenever cleanup is refused.

## When to Use

- implementation is complete and its targeted verification has passed;
- the user asks to merge, integrate, finalize, clean up, or hand off a branch;
- a linked worktree needs a completion decision.

Use `verification-before-completion` first. Use `receiving-code-review` when review findings
remain open.

## Prerequisites

- Current diff, branch, fork base, and worktree state.
- The repository's canonical test and quality commands.
- Explicit user choice for merge, push/PR, or keep-as-is.

## How to Run

1. Re-read the approved requirements and final diff.
2. Run fresh verification on the exact tree being integrated.
3. Identify the fork base and linked-worktree ownership.
4. Present integration choices without executing an external effect.
5. Execute only the user-selected local operation.
6. Clean only managed, safe worktrees; preserve refused cleanup.

## Quick Reference

| Choice | Local merge | Push/PR | Preserve worktree | Delete branch |
|---|---:|---:|---:|---:|
| Merge locally | yes | no | after green merged result | only safe delete |
| Push / create PR | no | with approval | yes | no |
| Keep as-is | no | no | yes | no |
| Discard | no | no | only after explicit confirmation | only after explicit confirmation |

## Procedure

### 1. Prove the exact tree

Run the required final checks after the last edit. Record commands, scopes, pass/fail counts,
skips, and exit codes. A previous green run is historical context, not completion evidence.

If verification fails, stop at diagnosis. Do not offer integration choices as a way to skip the
failure.

### 2. Establish repository state

Capture the branch, upstream, fork base, detached/linked state, uncommitted files, untracked
files, and unpushed commits. Confirm the fork base from the plan, branch history, or configured
upstream; a wrong-base merge is expensive to reverse.

### 3. Review requirements

Check the approved request or plan line by line. Separate verified, failed, not run, blocked,
and out-of-scope items. Request code review first for a major change or when project rules
require it.

### 4. Ask for the integration decision

Unless the user already selected the exact operation, present:

1. merge locally into the confirmed base;
2. push and create a pull/merge request;
3. keep the branch and worktree unchanged.

Push, PR creation, publication, and external messaging require explicit authorization. Passing
tests never grants that authority.

### 5. Merge locally only after selection

Update the base using the repository's normal non-destructive workflow, merge the feature
branch, and rerun the required checks on the merged result. If anything fails, leave the branch
and worktree in place and investigate.

Delete a branch only with a safe delete after the merged result is green and no unique commits
remain.

### 6. Preserve cleanup refusals

When worktree removal refuses because it contains uncommitted or untracked files, never
force-remove it. Show the status, explain that unique work would be lost, and ask whether to
commit, move, archive, or explicitly delete it. Use the Hermes worktree lifecycle command when
the tree is Hermes-managed.

Discarding work is a separate explicit request. It must name the branch, worktree, commits, and
uncommitted files, and receive unambiguous confirmation.

## Pitfalls

1. Reusing a green result from before the final edit.
2. Guessing the base branch.
3. Treating targeted tests as full-suite evidence.
4. Turning implementation approval into push or publication authority.
5. Force-removing a dirty worktree to finish quickly.
6. Deleting a branch that still holds unique review fixes.
7. Leaving review findings unmentioned while claiming completion.

## Verification

Before leaving this skill:

- [ ] Fresh verification ran on the exact final or merged tree.
- [ ] The integration decision was explicit.
- [ ] Base branch, diff, untracked files, and unpushed commits were inspected.
- [ ] Review findings were resolved or explicitly reported.
- [ ] No unauthorized push, PR, publication, message, spend, or remote write occurred.
- [ ] Refused cleanup preserved all unique work.
