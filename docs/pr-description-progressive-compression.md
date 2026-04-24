## Summary

Progressive tool-result compression — an ephemeral, per-API-call optimization that compresses old tool results to one-line summaries before each LLM call. Complements the existing `ContextCompressor` (persistent, threshold-triggered LLM summarization).

## Problem

In long conversations (40+ turns), old tool results — file contents, command outputs, search results — consume thousands of tokens that the model no longer needs verbatim. The model only needs: **WHAT** tool was called, and the **OUTCOME**.

Before this change, every API call sends the full verbatim content of *all* historical tool results. In a 50-turn coding session with 30 tool calls, this is easily 50K+ tokens of stale output.

## Solution

Before each API call, identify old tool results and replace them with compact one-line summaries:

```
# Before (2,847 chars)
{"role": "tool", "content": "  def process_transaction(txn):\n      if txn.amount > THRESHOLD:\n          ...140 lines of code...\n      return result\n\n# After (89 chars)
{"role": "tool", "content": "[read_file] OK (2847 chars) | def process_transaction(txn): → return result"
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Operates on `api_messages` copy only | `self.messages` never touched — fully reversible, zero risk |
| Regex-based summary, not LLM | Zero latency, zero cost per call |
| Respects `compression.enabled` | Users who opt out aren't silently opted in |
| `recent_tool_keep` defaults to `protect_last_n` | Consistent with existing "preserve recent context" intent |
| All thresholds configurable via `compression.progressive.*` | No hardcoded behavior |

### Relationship to existing compression

| | ContextCompressor | Progressive (this PR) |
|---|---|---|
| Trigger | After 413/overflow | Before every API call |
| Persistence | Permanent (mutates history) | Ephemeral (API copy only) |
| Method | LLM summarization | Regex one-line summary |
| Cost | API call per compression | Zero |
| Reversible | No | Yes |

## Evidence

Token savings measured with simulated coding sessions:

| Scenario | Tool calls | Before | After | Saved | Reduction |
|---|---|---|---|---|---|
| Small (10 calls, 2KB each) | 10 | 3,253 | 2,673 | 580 | 17.8% |
| Medium (20 calls, 5KB each) | 20 | 16,138 | 6,835 | 9,303 | **57.6%** |
| Large (30 calls, 8KB each) | 30 | 39,048 | 11,196 | 27,852 | **71.3%** |
| XLarge (50 calls, 10KB each) | 50 | 81,493 | 14,370 | 67,123 | **82.4%** |

Per-result compression ratio: **64.7x** (5,144 chars → 80 chars).

## Configuration

```yaml
agent:
  compression:
    progressive:
      enabled: true           # defaults to compression.enabled
      recent_tool_keep: 20    # defaults to compression.protect_last_n
      min_messages: 16        # only activate in long conversations
      max_compressed_len: 300 # skip results shorter than this
```

## Changes

### `run_agent.py`
- **`__init__`**: Read `compression.progressive.*` config with safe defaults that respect parent `compression.enabled` and `protect_last_n`
- **`_progressive_tool_result_compress`**: New `@staticmethod` — identifies old tool messages, builds `[tool_name] status (chars) | first_line → last_line` summaries
- **Agent loop**: Call progressive compression on `api_messages` before prompt caching, after prefill injection

### `tests/run_agent/test_progressive_tool_compress.py` (new)
32 tests: no-op paths (5), core behavior (6), result detection (4), threshold boundaries (3), immutability (1), edge cases (13)

### `tests/run_agent/test_compression_boundary.py`
- Remove unused `pytest` and `MagicMock` imports (lint cleanup)

## Testing

```
68 passed (32 progressive + 16 feasibility + 6 boundary + 14 upstream 413)
```
