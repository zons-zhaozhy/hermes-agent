---
name: test-driven-development
description: "Drive changes with RED-GREEN-REFACTOR tests first."
version: 2.0.0
author: "Stan + Hermes Agent (adapted from obra/superpowers)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [testing, tdd, development, quality, red-green-refactor]
    related_skills: [systematic-debugging, condition-based-waiting, verification-before-completion]
    category: software-development
---

# Test-Driven Development

Write one meaningful failing test, observe the expected failure, implement the smallest correct
change, then refactor while green. A test that was never red cannot prove that it catches the
behavior it claims to cover.

## When to Use

- any behavior change, bug fix, refactor, or new function;
- a reviewed defect that needs a regression test;
- a repair whose original symptom can be expressed as an executable predicate.

Ask before using this skill for a throwaway probe, generated fixture, or pure configuration
change that has no testable seam.

## Prerequisites

- A concrete behavioral requirement independent of the intended implementation.
- Hermes `terminal` for the canonical test command.
- `read_file` and `search_files` to place the test and understand the production seam.

## How to Run

1. Choose one smallest valuable behavior slice.
2. Write a failing test whose expectation comes from the requirement.
3. Run the targeted canonical test and confirm the expected red reason.
4. Implement only enough production code to pass.
5. Rerun the same test and confirm green.
6. Refactor in small steps while the test stays green.
7. Run the relevant regression scope after the final edit.

In this repository, targeted tests use:

```bash
HERMES_PYTHON="$PWD/.venv/bin/python" scripts/run_tests.sh tests/<area>/test_file.py -q
```

Never replace the repository runner with a direct framework invocation.

## Quick Reference

| Stage | Required evidence |
|---|---|
| RED | Fresh targeted run fails for the missing expected behavior |
| GREEN | Same run passes after the minimal change |
| REFACTOR | Behavior remains green while structure improves |
| REGRESSION | Relevant scope passes after the last edit |

## Procedure

### 1. Define behavior independently

Name the input, action, output, error, timing condition, and authorization boundary. Derive the
expectation from the requirement, observed defect, interface contract, or user decision—not
from code already written for the fix.

### 2. Choose a vertical slice

Test one end-to-end behavior at a time. Do not write a pile of imagined tests before any
implementation; each slice should teach the next interface decision.

For bug fixes, first reproduce the user-visible symptom at the narrowest real seam. Mock only
an external boundary that cannot safely run, and assert real outputs rather than mock calls.

### 3. Observe RED

Run the exact targeted command. Read the failure and confirm it is the expected missing
behavior, not a typo, missing fixture, collection error, or wrong test path. Record the failing
assertion or error signal.

If the new test passes immediately, it does not cover a missing behavior. Replace or narrow it
before writing production code.

### 4. Implement minimally

Change only the production path needed by the current test. Avoid extra flags, speculative
abstractions, broad refactors, and unrelated cleanup. Keep edits inside the authorized scope.

### 5. Observe GREEN

Run the same targeted command. Read the complete summary, including skipped tests, warnings,
and exit status. A warning that changes correctness is not pristine evidence.

### 6. Refactor while green

After green, improve names, remove duplication, extract a focused helper, or move behavior to
the owning sibling. Rerun the targeted test after each small refactor. If it fails, undo that
step and take a smaller one.

### 7. Check neighboring behavior

Search for sibling paths and callers that share the changed invariant. Add separate red-green
cycles for each materially different behavior; do not hide several outcomes in one assertion.

### 8. Run the relevant regression scope

Use the repository runner for the touched area, then the project's broader quality command when
the change crosses interfaces. Label exact scopes; never describe a targeted run as full-suite
evidence.

## Pitfalls

1. Writing the implementation first and then fitting a test to it.
2. Treating a collection error as a valid red result.
3. Asserting internal calls instead of observable behavior.
4. Bundling multiple behavior changes into one edit.
5. Skipping the rerun after refactoring.
6. Turning a slow or flaky test into a fixed clock sleep instead of `condition-based-waiting`.
7. Claiming completion without `verification-before-completion`.

## Verification

Before leaving this skill:

- [ ] Each production behavior has a test that was observed red.
- [ ] Each red failure was for the expected missing behavior.
- [ ] Minimal implementation made the same test green.
- [ ] Refactoring stayed green.
- [ ] Sibling paths were searched for the same class of issue.
- [ ] Relevant regression commands and scopes were run fresh.
- [ ] No unauthorized external effect occurred.
