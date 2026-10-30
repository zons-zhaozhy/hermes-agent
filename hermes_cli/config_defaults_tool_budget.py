"""Defaults for the tool-output budget blocks of DEFAULT_CONFIG (sibling of config_defaults.py).

Moved into a sibling to offset the parent file's line ratchet; comments have room here.
Runtime reader: tools/budget_config.py (``_tool_budget_block``); enforcement:
tools/tool_result_storage.py; docs: user-guide/configuration.md § Tool-Result Spillover Budget.
"""


def _tool_output_budgets() -> dict:
    """``tool_output`` (in-tool truncation) and ``tool_budget`` (spillover thresholds).

    The tool_budget values equal the hardcoded historical budget, so an untouched
    config resolves to the DEFAULT_BUDGET singleton (tests/tools/test_budget_config.py).
    """
    return {
        # Tool-output truncation. max_bytes: terminal_tool output cap in chars (head+tail kept; 50_000 ≈
        # 12-15K tokens). max_lines: max `limit` one read_file call may request before clamping.
        # max_line_length: per-line cap in read_file's line-numbered view (chars).
        "tool_output": {"max_bytes": 50000, "max_lines": 2000, "max_line_length": 2000},
        # Tool-result spillover budgets (chars). result_size_chars: generic per-result spillover
        # threshold, clamped to [8_000, 100_000] (out-of-range values warn once and clamp).
        # turn_budget_chars: per-turn aggregate for one tool batch, clamped to [16_000, 200_000].
        # mcp_result_size_chars: per-result threshold for mcp_* tools (capped by the generic one).
        # tool_overrides: free-form per-tool threshold mapping — names are user-chosen, so an
        # empty-dict seed also marks the sub-tree open for config-set key validation.
        "tool_budget": {
            "result_size_chars": 100000,
            "turn_budget_chars": 200000,
            "mcp_result_size_chars": 50000,
            "tool_overrides": {},
        },
    }
