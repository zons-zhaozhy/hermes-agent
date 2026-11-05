---
name: brainstorming
description: "Design architecture and changes before code."
version: 1.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [planning, architecture, design, change-control, investigation, approval, workflow]
    related_skills: [spike, writing-plans, grill-me, test-driven-development]
---

# Brainstorming Skill

Turn a change request into an evidence-backed decision about what to build before production
code is changed. This skill owns classification, investigation, approach selection, and the
design authorization gate; it delegates feasibility probes, planning, and implementation to
the existing Hermes skills.

## When to Use

Use before:

- adding a feature or changing behaviour;
- refactoring code;
- changing an interface, data format, migration path, security boundary, or rollback story;
- implementing a request whose intended outcome is ambiguous.

Use `spike` instead when the deliverable is a disposable answer to a feasibility question.
Use `systematic-debugging` when the primary task is diagnosing an observed failure.

## Prerequisites

- `read_file`, `search_files`, and `terminal` for repository investigation.
- `skill_view` for loading the named follow-up skills.
- User input only when a decision or missing business context is required.

## How to Run

1. Classify the request as Spike, Bounded, or Architectural.
2. Establish intent and authorization.
3. Investigate the facts in the repository.
4. Compare approaches when the decision is Architectural.
5. Present the required design and stop at the authorization boundary.
6. For Architectural work, write and self-review the spec, then request user review.
7. After approval, hand off to `writing-plans`.

## Quick Reference

| Class | Use it when | Required output |
|---|---|---|
| Spike | The answer matters, not retained code. | Question, minimal probe plan, findings |
| Bounded | An existing flow needs a local, reversible change. | Short design, investigation evidence, tests |
| Architectural | Components or interfaces are restructured, added, or hard to roll back. | Approved spec and implementation plan |

When unsure between Bounded and Architectural, choose Architectural. A discovered
complication upgrades the class immediately.

## Procedure

### 1. State the class

Say which path applies before applying its process. Give a one-sentence reason when the
boundary is not obvious.

### 2. Establish intent

Restate the concrete outcome, scope, and success criterion. Separate it from related work
the user did not request.

The user's request to investigate, explain, review, or diagnose is not authorization to
modify code. A clear request to fix or implement the same scope is authorization.

### 3. Investigate before asking

Resolve discoverable facts yourself with `read_file`, `search_files`, `terminal`,
`web_search`, and `web_extract`. Inspect:

- the current implementation and the exact symbols or files involved;
- callers, consumers, sibling paths, tests, and configuration;
- recent history when it explains a load-bearing invariant;
- the change's verification, blast radius, and rollback path.

Ask the user one decision question at a time only when the answer cannot be found and two
reasonable choices would materially change the result. Do not ask for facts the environment
can answer.

### 4. Apply the authorization matrix

Proceed autonomously for a Spike or Bounded change when it is:

- inside the authorized workspace;
- reversible;
- local in impact;
- supported by clear user intent;
- fully investigated;
- covered by a concrete verification path.

A Bounded change becomes Architectural when it alters a public interface, data format,
migration path, security boundary, or non-local rollback story.

Stop for explicit user approval before:

- executing an Architectural design;
- spending money or expanding a paid/credit budget;
- sending or publishing externally;
- writing to a remote system;
- creating an issue, PR, comment, or release;
- using or changing credentials or privilege;
- deleting important data or performing an irreversible broad operation.

### 5. Compare approaches

For Architectural work, present two or three viable approaches with trade-offs and a
recommendation. Apply YAGNI. Do not fabricate alternatives that differ only in style.

For Bounded work, state the approach compactly:

`Class: Bounded | Change: ... | Files: ... | Verification: ... | Rollback: ...`

### 6. Present the design

For Bounded work, a short design in the response is sufficient. For Architectural work,
cover:

- architecture and component boundaries;
- data or control flow;
- error handling and failure semantics;
- testing and evidence;
- rollout, compatibility, and rollback.

Do not write an Architectural spec until the user agrees with the design direction.

### 7. Write and self-review a spec

Save the validated Architectural design to
`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.

Check the written spec for placeholders, contradictions, ambiguous requirements, and scope
that needs decomposition. Fix issues inline.

### 8. Stop for spec review

Tell the user where the spec was written and ask them to review it. Do not begin planning or
implementation until the user approves the written spec.

### 9. Hand off

After approval, load `writing-plans` with `skill_view("writing-plans")`. Do not invoke an
implementation skill directly from this skill.

## Pitfalls

1. Treating a clear-looking request as enough evidence to skip repository investigation.
2. Calling every change Bounded to avoid design work.
3. Asking the user factual questions that `read_file` or `search_files` can answer.
4. Writing code while the design is still awaiting approval.
5. Preserving a stale class after discovering cross-module or interface impact.
6. Writing a spec with vague acceptance criteria or unresolved placeholders.
7. Treating analysis, diagnosis, or planning as permission to edit.

## Verification

Before leaving this skill:

- [ ] The class is stated and its reason is clear.
- [ ] The concrete outcome and authorization boundary are established.
- [ ] Discoverable facts were investigated rather than asked of the user.
- [ ] The blast radius, verification path, and rollback are understood.
- [ ] A clear Bounded internal change can proceed without unnecessary approval.
- [ ] Every Architectural, spend, external, credential, or destructive action stops at the
      required approval boundary.
- [ ] An Architectural spec exists and passed self-review before spec approval is requested.
- [ ] No implementation started before the required authorization.
