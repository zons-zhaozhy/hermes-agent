---
name: using-git-worktrees
description: "Use Git worktree isolation with a verified baseline."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from Codex engineering practice)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [git, worktree, isolation, baseline, parallel-work]
    related_skills: [executing-plans, finishing-a-development-branch, repository-investigation]
    category: software-development
---

# Using Git Worktrees

Protect unrelated work by detecting existing isolation first, using Hermes' managed worktree
surface when possible, and proving the new baseline before implementation.

## When to Use

- before executing an implementation plan in a dirty shared checkout;
- when parallel work must not share a branch or environment;
- before a risky refactor, migration, or long experiment;
- when project instructions require an isolated workspace.

Do not create a second tree when the session is already isolated. Do not create one for a
read-only investigation unless the probe itself writes.

## Prerequisites

- A Git repository and access to `terminal`, `read_file`, and `search_files`.
- Hermes `/worktree new` in an interactive CLI session when available.
- The project's canonical setup and baseline test commands.

## How to Run

1. Detect the current checkout and whether it is already isolated.
2. Preserve the current branch's dirty and untracked work.
3. Prefer `/worktree new` or `hermes -w`; use manual Git only when no managed surface exists.
4. Run project setup in the isolated tree.
5. Verify a bounded baseline before changing code.

## Quick Reference

| State | Action |
|---|---|
| Linked worktree | Reuse it; inspect branch and dirty state |
| Submodule | Treat as its relevant checkout; do not infer linked-worktree isolation |
| Clean dedicated branch | Work in place when isolation was not requested |
| Dirty shared checkout | Isolate or preserve work before edits |
| Hermes CLI available | Prefer `/worktree new` or `hermes -w` |
| No managed surface | Use the project's documented worktree command |
| Baseline red | Stop and report; do not attribute later failures to new work |

## Procedure

### 1. Detect before creating

Through `terminal`, resolve the Git directory, common Git directory, current branch, repository
root, and superproject when present. Different Git-directory and common-directory values
indicate a linked worktree only after excluding a submodule. Also inspect status, including
untracked files.

Record:

- checkout root;
- branch or detached HEAD;
- clean/dirty state;
- whether the host manages this workspace;
- setup and test commands that apply to this tree.

### 2. Preserve existing work

Never discard, stash away from view, or overwrite dirty/untracked files to make isolation
convenient. If existing work belongs to the user and blocks a safe baseline, report it and ask
for a direction. Moving or archiving user work requires explicit confirmation.

### 3. Prefer managed isolation

In a Hermes interactive CLI, use `/worktree new <name>` or launch the task with `hermes -w`.
These paths own placement, branch creation, terminal/file retargeting, recording, and exit
lifecycle. A manually created tree can be invisible to those controls.

Use a manual project-documented worktree path only when no managed surface is available. Verify
the destination is outside tracked source, uses the intended base revision, and does not run
untrusted repository configuration.

### 4. Set up the tree

Follow repository instructions exactly. For Hermes development, use the documented PM bootstrap
and test interpreter rather than mutating an environment ad hoc. Keep generated dependencies
inside the isolated workspace or a project-approved shared location.

### 5. Prove the baseline

Run the smallest suite that proves the area under work. For a broad architectural change, run
the repository's canonical broader check before editing. If the baseline fails, capture the
exact failure and count; implementation has not started until that baseline is understood or
the user explicitly accepts proceeding from a red baseline.

### 6. Report the boundary

State the worktree path, branch, base revision, setup result, and baseline result. Later edits
must stay within that tree until `finishing-a-development-branch` takes over.

## Pitfalls

1. Creating a worktree while already running in one.
2. Treating a submodule as a linked worktree.
3. Bypassing `/worktree new` when it is available.
4. Assuming dependency setup can be skipped because the main checkout has it.
5. Starting from a red baseline and later calling those failures regressions.
6. Cleaning a host-managed tree without its lifecycle tool.

## Verification

Before leaving this skill:

- [ ] Existing isolation and dirty state were detected.
- [ ] User-owned uncommitted work was preserved.
- [ ] A managed Hermes worktree path was preferred when available.
- [ ] Isolated setup completed or its blocker is reported.
- [ ] The baseline command, scope, pass/fail counts, and exit status are recorded.
- [ ] No destructive cleanup occurred.
