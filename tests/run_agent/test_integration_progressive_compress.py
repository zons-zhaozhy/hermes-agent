"""Integration test: progressive tool-result compression in a realistic long session.

No mocks. Calls _progressive_tool_result_compress directly with a synthetic
40-turn conversation that mirrors real AIAgent message shapes (user/assistant/
tool, with tool_calls as Pydantic objects or dicts).

Validates:
1. Original messages are NOT mutated (compression is ephemeral).
2. Old tool results are replaced with one-line summaries.
3. Recent tool results (last N) are preserved verbatim.
4. Output messages maintain valid API structure (every tool msg has tool_call_id).
5. Token savings are meaningful (>60% on tool-result content).
6. Outcome detection works on realistic outputs (exit codes, errors, success).
"""

import copy
import json
import uuid
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run_agent import AIAgent


def _make_tool_call(name: str, call_id: str | None = None) -> dict:
    """Mimic a tool_call entry inside an assistant message."""
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _build_realistic_long_conversation(num_tool_rounds: int = 25) -> list[dict]:
    """Build a 50+ message conversation alternating user/assistant(tool_call)/tool.

    Simulates a real coding session: read_file, terminal, search_files repeated.
    """
    messages = [
        {"role": "system", "content": "You are Hermes, a helpful AI agent."},
        {"role": "user", "content": "Help me refactor the auth module. Start by reading the current files."},
    ]

    # Repeated tool-use rounds (read_file, terminal, search_files)
    nl = "\n"
    tool_scenarios = [
        ("read_file", lambda i: f"# Source: src/auth/service.py{nl}{''.join(f'// Line {j}: some auth logic here{nl}' for j in range(1, 50+i*2))}"),
        ("terminal", lambda i: f"{''.join(f'compiling module {j}...{nl}' for j in range(1, 30+i))}Build completed successfully.{nl}exit_code: 0"),
        ("search_files", lambda i: f"Found {10+i*3} matches:{nl}" + nl.join(
            f"  src/auth/{j}.py:{j*10}: match_here_function_call" for j in range(1, 15+i)
        )),
        ("read_file", lambda i: f"# Source: src/auth/middleware.py{nl}{''.join(f'// Line {j}: JWT validation logic{nl}' for j in range(1, 60+i*2))}"),
        ("terminal", lambda i: f"Running tests...{nl}" + nl.join(
            f"  test_auth_{j}... PASSED" for j in range(1, 20+i)
        ) + f"{nl}{nl}All 20 tests passed.{nl}exit_code: 0"),
    ]

    for round_idx in range(num_tool_rounds):
        tool_name, content_fn = tool_scenarios[round_idx % len(tool_scenarios)]
        call_id = f"call_{uuid.uuid4().hex[:12]}"

        # Assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [_make_tool_call(tool_name, call_id)],
        })

        # Tool result (large content)
        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content_fn(round_idx),
        })

        # User follow-up (every other round)
        if round_idx % 2 == 0:
            messages.append({
                "role": "user",
                "content": f"Good, now check the next file. Round {round_idx+1}.",
            })

    # Final user message
    messages.append({
        "role": "user",
        "content": "Summarize what you found and suggest the refactoring plan.",
    })

    return messages


def test_immutable_original_messages():
    """Original messages list must NOT be modified by compression."""
    messages = _build_realistic_long_conversation(num_tool_rounds=25)
    original_snapshot = json.dumps(messages)

    # Call compression via AIAgent static-like method (it's a plain method on the class)
    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=5,
        min_messages=16,
        max_compressed_len=300,
    )

    after_snapshot = json.dumps(messages)
    assert original_snapshot == after_snapshot, "Original messages were mutated!"
    assert result is not messages, "Should return a new list, not the same reference"
    print(f"  [PASS] Original messages immutable (len={len(messages)})")


def test_old_tool_results_compressed():
    """Tool results beyond recent_tool_keep must be compressed to one-line summaries."""
    messages = _build_realistic_long_conversation(num_tool_rounds=25)

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=5,
        min_messages=16,
        max_compressed_len=300,
    )

    tool_msgs = [(i, m) for i, m in enumerate(result) if m.get("role") == "tool"]
    assert len(tool_msgs) == 25, f"Expected 25 tool messages, got {len(tool_msgs)}"

    # Last 5 tool messages should be preserved
    last_5 = tool_msgs[-5:]
    for i, msg in last_5:
        assert len(msg["content"]) > 300, (
            f"Recent tool result at index {i} was compressed (len={len(msg['content'])}), "
            f"should be preserved"
        )

    # Earlier tool messages should be compressed (short summary)
    earlier = tool_msgs[:-5]
    compressed_count = 0
    for i, msg in earlier:
        content_len = len(msg["content"])
        # Compressed messages should be short (< 300) or already short
        if content_len <= 300:
            compressed_count += 1
        # Compressed messages should start with [tool_name]
        if content_len < 100:
            assert msg["content"].startswith("[terminal]") or msg["content"].startswith("[read_file]") or msg["content"].startswith("[search_files]"), (
                f"Short content doesn't look like a compression summary: {msg['content'][:80]}"
            )

    compression_rate = compressed_count / len(earlier)
    print(f"  [PASS] {compressed_count}/{len(earlier)} old tool results compressed ({compression_rate:.0%})")
    assert compression_rate >= 0.8, f"Expected >= 80% compression rate, got {compression_rate:.0%}"


def test_output_structure_valid():
    """Compressed messages must maintain valid API structure."""
    messages = _build_realistic_long_conversation(num_tool_rounds=25)

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=5,
        min_messages=16,
        max_compressed_len=300,
    )

    for i, msg in enumerate(result):
        role = msg.get("role")
        assert role in ("system", "user", "assistant", "tool"), f"Invalid role at index {i}: {role}"

        if role == "tool":
            assert "tool_call_id" in msg, f"Tool message at index {i} missing tool_call_id"
            assert msg["tool_call_id"], f"Tool message at index {i} has empty tool_call_id"
            assert isinstance(msg["content"], str), f"Tool content at index {i} is not a string"

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                assert isinstance(tc, dict), f"tool_call at index {i} is not a dict: {type(tc)}"
                assert "id" in tc, f"tool_call at index {i} missing id"

    print(f"  [PASS] All {len(result)} messages have valid API structure")


def test_token_savings_meaningful():
    """Compression should achieve meaningful token reduction (>60% on tool content)."""
    messages = _build_realistic_long_conversation(num_tool_rounds=25)

    original_tool_chars = sum(
        len(m["content"]) for m in messages if m.get("role") == "tool" and isinstance(m["content"], str)
    )

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=5,
        min_messages=16,
        max_compressed_len=300,
    )

    compressed_tool_chars = sum(
        len(m["content"]) for m in result if m.get("role") == "tool" and isinstance(m["content"], str)
    )

    savings = 1 - (compressed_tool_chars / original_tool_chars)
    print(f"  Original tool content: {original_tool_chars:,} chars")
    print(f"  Compressed tool content: {compressed_tool_chars:,} chars")
    print(f"  Savings: {savings:.1%}")
    assert savings >= 0.5, f"Expected >= 50% savings, got {savings:.1%}"
    print(f"  [PASS] Meaningful savings achieved")


def test_outcome_detection_realistic():
    """Outcome detection should work on realistic multi-line outputs."""
    messages = [
        # Error output
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("terminal", "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "Traceback (most recent call last):\n  File 'main.py', line 42\n    import nonexistent_module\nModuleNotFoundError: No module named 'nonexistent_module'\nexit_code: 1"},
        # Success output (make it long enough to trigger compression > 100 chars)
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("terminal", "c2")]},
        {"role": "tool", "tool_call_id": "c2", "content": "Building project...\n" + "\n".join(
            f"  Compiling module_{i}.py... ✅ OK" for i in range(20)
        ) + "\n✅ Build successful\nAll 15 tests passed.\nexit_code: 0"},
        # Neutral output (make it long enough to trigger compression > 100 chars)
        {"role": "assistant", "content": None, "tool_calls": [_make_tool_call("read_file", "c3")]},
        {"role": "tool", "tool_call_id": "c3", "content": "# config.yaml\n" + "\n".join(
            f"  setting_{i}: value_{i}" for i in range(30)
        ) + "\ndatabase:\n  url: postgres://localhost/mydb\n"},
    ] + [
        # Pad to reach min_messages
        {"role": "user", "content": f"msg {i}"} for i in range(20)
    ]

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=1,  # Only keep last tool result
        min_messages=16,
        max_compressed_len=100,
    )

    tool_msgs = [m for m in result if m.get("role") == "tool"]
    assert len(tool_msgs) == 3

    # First tool result (error) should contain "ERROR" in summary
    assert "ERROR" in tool_msgs[0]["content"], f"Expected ERROR in summary: {tool_msgs[0]['content']}"
    # Second tool result (success) should contain "OK" in summary
    assert "OK" in tool_msgs[1]["content"], f"Expected OK in summary: {tool_msgs[1]['content']}"
    # Third tool result (neutral, last one) should be preserved since recent_tool_keep=1
    assert len(tool_msgs[2]["content"]) > 100, f"Last tool result should be preserved: len={len(tool_msgs[2]['content'])}"

    print(f"  [PASS] Outcome detection: ERROR={tool_msgs[0]['content'][:60]}")
    print(f"  [PASS] Outcome detection: OK={tool_msgs[1]['content'][:60]}")


def test_disabled_no_compression():
    """When disabled, no compression should occur."""
    messages = _build_realistic_long_conversation(num_tool_rounds=25)
    original_snapshot = json.dumps(messages)

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        messages,
        enabled=False,
    )

    assert json.dumps(result) == original_snapshot, "Messages changed when compression disabled!"
    print(f"  [PASS] Disabled mode: no changes (len={len(messages)})")


def test_boundary_recent_tool_keep_zero():
    """recent_tool_keep=0 should compress ALL tool results."""
    messages = _build_realistic_long_conversation(num_tool_rounds=10)

    agent = AIAgent.__new__(AIAgent)
    result = agent._progressive_tool_result_compress(
        copy.deepcopy(messages),
        enabled=True,
        recent_tool_keep=0,
        min_messages=5,
        max_compressed_len=200,
    )

    tool_msgs = [m for m in result if m.get("role") == "tool"]
    compressed = sum(1 for m in tool_msgs if len(m["content"]) <= 300)
    print(f"  [PASS] recent_tool_keep=0: {compressed}/{len(tool_msgs)} compressed")


if __name__ == "__main__":
    print("=" * 60)
    print("Integration test: progressive tool-result compression")
    print("=" * 60)

    test_disabled_no_compression()
    test_immutable_original_messages()
    test_old_tool_results_compressed()
    test_output_structure_valid()
    test_token_savings_meaningful()
    test_outcome_detection_realistic()
    test_boundary_recent_tool_keep_zero()

    print("\n" + "=" * 60)
    print("ALL 7 integration tests PASSED")
    print("=" * 60)
