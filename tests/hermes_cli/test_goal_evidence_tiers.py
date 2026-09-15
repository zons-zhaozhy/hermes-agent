"""Evidence-tier annotation in extract_tool_calls_summary (prepared ≠ observed).

Derived from the OMH 'prepared is not observed' contract: the judge must be
able to distinguish turns that only staged state (write_file/patch) from turns
that produced runtime output (terminal/execute_code). Expectations here are
derived from that contract independently of the implementation.
"""
from hermes_cli.goals import extract_tool_calls_summary


def _assistant_msg(*tool_names: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": name, "arguments": "{}"}} for name in tool_names
        ],
    }


def test_pure_preparatory_calls_have_zero_observed():
    # A turn that only edited files staged state; observed_evidence must be 0
    # so the judge treats any completion claim as unverified.
    history = [
        {"role": "user", "content": "go"},
        _assistant_msg("read_file"),
        _assistant_msg("patch", "write_file"),
    ]
    out = extract_tool_calls_summary(history)
    assert out is not None
    assert "observed_evidence=0" in out
    assert "preparatory=3" in out


def test_runtime_calls_count_as_observed():
    history = [
        {"role": "user", "content": "go"},
        _assistant_msg("patch"),
        _assistant_msg("terminal", "execute_code"),
    ]
    out = extract_tool_calls_summary(history)
    assert out is not None
    assert "observed_evidence=2" in out
    assert "preparatory=1" in out


def test_zero_calls_marker_unchanged():
    # Backward compatibility: text-only turns keep the exact legacy marker.
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "all done"},
    ]
    assert extract_tool_calls_summary(history) == "0 calls (text-only response)"


def test_none_history_still_returns_none():
    assert extract_tool_calls_summary(None) is None
    assert extract_tool_calls_summary([]) is None


def test_counts_sum_to_total_calls():
    # Invariant: observed + preparatory == number of tool calls in the turn.
    history = [
        {"role": "user", "content": "go"},
        _assistant_msg("terminal", "patch", "browser_exec", "write_file", "web_search"),
    ]
    out = extract_tool_calls_summary(history)
    assert out is not None
    assert out.startswith("5 call(s):")
    assert "observed_evidence=3" in out
    assert "preparatory=2" in out
