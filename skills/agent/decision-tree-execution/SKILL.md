---
name: decision-tree-execution
description: Belief-based decision tree for complex multi-step tasks.
version: "1.0.0"
author: "Hermes Agent"
license: "MIT"
metadata:
  hermes:
    tags: [execution, planning, decision-making, self-check]
    category: agent
    related_skills:
      - strategic-execution
      - systematic-debugging
      - decision-framework
    config:
      skills.config.decision_tree.stagnation_window:
        type: integer
        default: 3
        description: "Rounds without improvement before forcing branch switch"
---

# Decision-Tree-Driven Execution Skill

Use a persistent, belief-based decision tree (K-Search co-evolving world model)
to track strategies, evidence, and beliefs across turns of a complex task.
The tree persists on disk and can be resumed across sessions.

## When to Use

- Multi-step task with 3+ non-trivial steps
- Task where you need to try multiple approaches and compare
- Debugging a complex issue where hypotheses need tracking
- Any task where you've previously gotten stuck in a loop of micro-optimizations
- Tasks spanning multiple sessions that need cross-session continuity

Do NOT use for: single tool call, simple query, one-step file edit.

## Prerequisites

The `decision-tree` plugin must be enabled:
```bash
hermes plugins enable decision-tree
```
Or add `decision-tree` to `plugins.enabled` in `config.yaml`.

The tree is persisted at `~/.hermes/decision_tree/<profile>/<session_id>.json`.

## Three-Layer Workflow

After K-Search (arXiv 2602.19128), complex execution is split into three
strictly isolated layers:

### Layer 1: World Model (Planning — no code)

Before touching any code, init the decision tree:

1. `world_model_init(task_summary="...", open_questions=["Q1?", "Q2?", ...])`
2. Propose 2-3 initial action alternatives via `world_model_update(action="add_action", ...)`:
   - Each action must be a SINGLE concrete change (one knob, one dimension)
   - Difficulty 1-5: prefer ≤3. Actions with difficulty >3 are deferred.
   - Portfolio: at least 2 structural/exploration actions, at most 2 pure-tuning actions
3. Rate each action: confidence and expected improvement

### Layer 2: Execute (Codegen — code-only)

Pick the highest-rated open action, then implement it:

1. `world_model_view(mode="open_actions")` to see candidates
2. Implement ONE action. Do NOT bundle multiple large changes.
3. After implementation and testing, attach the result:
   ```
   world_model_update(action="attach_result", node_id="...", status="PASSED|FAILED",
                      latency_ms=..., speedup=...)
   ```

### Layer 3: Verify (Anti-Self-Deception — no code)

After every action result, run the check:

```
world_model_check(node_id="...", claimed_intent="What I claim this action achieved")
```

The check enforces three anti-self-deception mechanisms:

1. **FOLLOW_THROUGH**: Did implementation actually match the intent?
   - If action title and claimed intent have zero word overlap → misalignment
   - If you said you'd "fix login" but only changed CSS → flagged

2. **PERF_GAP**: Expected improvement vs observed result
   - If predicted 2x speedup but got 1.1x → gap detected
   - Must explain the gap in node notes (UPDATE_BELIEF)

3. **Stagnation Detection**: N consecutive non-PASSED rounds
   - Default N=3 rounds → alert triggers, recommending branch switch
   - Must actively jump to a sibling branch or return to root

## Tool Reference

### world_model_init
Create a fresh decision tree. Call ONCE at task start.
```
world_model_init(
  task_summary="Fix login timeout on mobile — 500ms+ in production, target <200ms",
  open_questions=["Is network latency or server processing?", "Does async help?"]
)
```

### world_model_view
See current state before deciding next step.
- `mode="compact"` — prompt-friendly projection (default)
- `mode="open_actions"` — ranked list of pending actions
- `mode="summary"` — one-line node count summary
- `mode="full"` — complete JSON tree

### world_model_update
The main mutation tool. Five sub-actions:

- `action="add_action"` — Plan a new action
- `action="attach_result"` — Record evaluation outcome
- `action="update_belief"` — Revise ratings based on evidence
- `action="set_active"` — Switch to a different branch
- `action="advance_round"` — Start a new round

### world_model_check
Run all anti-self-deception checks. MUST call after every action result.
Returns: `has_gaps`, `stagnation_alert`, `recommendation` (CONTINUE/INVESTIGATE_GAPS/SWITCH_BRANCH).

## Difficulty Portfolio (Hard Constraint)

When proposing actions:
- At least 2 must be structural/exploration (trying different approaches)
- At most 2 can be pure tuning (micro-optimizations)
- Prefer difficulty ≤3. Defer difficulty 4-5 until base solution improves.

## Stagnation Rule (Hard Constraint)

After 3 consecutive non-PASSED rounds on the same branch:
1. `world_model_view(mode="open_actions")` to see alternatives
2. `world_model_update(action="set_active", node_id="<alternative>")`
3. OR return to root and try a completely different family

Do NOT continue tweaking the same branch past stagnation.

## Self-Check on "Done"

Before claiming a task is complete, run:
```
world_model_check(node_id="<active>", claimed_intent="<what I achieved>")
```

If `has_gaps=true` or `stagnation_alert=true`, address them before claiming done.

## Pitfalls

- **Don't skip init**: Without `world_model_init`, the tree doesn't exist and hooks can't auto-track.
- **Don't skip check**: Without `world_model_check` after result, you're flying blind on FOLLOW_THROUGH and PERF_GAP.
- **Don't bundle actions**: One action = one concrete change. "Refactor auth AND add caching AND fix timeout" is three actions.
- **Don't leave zero-confidence nodes**: If an action has confidence=0 and rating=0, either fill it with evidence or prune it.
- **Don't ignore stagnation**: If the check says SWITCH_BRANCH, switch. Doing another micro-tweak on the same branch is grinding.
- **Don't skip check before claiming done**: The pre_verify hook will BLOCK your stop if there are unresolved gaps. Run the check proactively.

## Verification

After task completion, verify:
```bash
# Check the persisted tree
ls -la ~/.hermes/decision_tree/
python3 -c "
from agent.decision_tree import load_or_create
wm = load_or_create('<session_id>')
print(wm.summary())
# All filled nodes should have non-zero confidence
for n in wm.nodes.values():
    if n.is_filled and n.confidence == 0:
        print(f'WARNING: {n.node_id} filled but zero confidence')
"
```
