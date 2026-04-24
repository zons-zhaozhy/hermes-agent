"""Benchmark: progressive tool-result compression token savings.

Simulates realistic long conversations and measures token reduction
from progressive compression. No API calls needed — uses rough
char-to-token estimation (1 token ≈ 4 chars for English, 2 chars for CJK).
"""
import json
import statistics
from run_agent import AIAgent


def _build_conversation(num_tool_pairs: int, avg_result_chars: int) -> list:
    """Build a realistic conversation with tool calls."""
    messages = [{"role": "system", "content": "You are a helpful coding assistant."}]
    for i in range(num_tool_pairs):
        # User asks a question
        messages.append({
            "role": "user",
            "content": f"Can you check the implementation of module_{i}?",
        })
        # Assistant calls a tool
        tool_call_id = f"call_{i:04d}"
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/module_{i}.py"})},
            }],
        })
        # Tool returns a file content (simulated)
        lines = [f"  {'def' if i % 3 == 0 else 'class'} func_{j}(self, x):" for j in range(avg_result_chars // 40)]
        content = "\n".join(lines)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
    return messages


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars/token."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return total_chars // 4


def benchmark():
    scenarios = [
        ("Small (10 tool calls, 2KB each)", 10, 2000),
        ("Medium (20 tool calls, 5KB each)", 20, 5000),
        ("Large (30 tool calls, 8KB each)", 30, 8000),
        ("XLarge (50 tool calls, 10KB each)", 50, 10000),
    ]

    recent_keep = 8
    min_msgs = 16
    max_len = 300

    print(f"{'Scenario':<40} {'Before':>10} {'After':>10} {'Saved':>10} {'Reduction':>10}")
    print("-" * 85)

    for label, num_tools, avg_chars in scenarios:
        messages = _build_conversation(num_tools, avg_chars)
        original = messages[:]
        before_tokens = _estimate_tokens(messages)

        compressed = AIAgent._progressive_tool_result_compress(
            messages,
            enabled=True,
            recent_tool_keep=recent_keep,
            min_messages=min_msgs,
            max_compressed_len=max_len,
        )
        after_tokens = _estimate_tokens(compressed)

        saved = before_tokens - after_tokens
        pct = (saved / before_tokens * 100) if before_tokens > 0 else 0
        print(f"{label:<40} {before_tokens:>10,} {after_tokens:>10,} {saved:>10,} {pct:>9.1f}%")

    # Extra: show per-tool-result compression ratio
    print("\n--- Per-result detail (Large scenario) ---")
    messages = _build_conversation(30, 8000)
    compressed = AIAgent._progressive_tool_result_compress(
        messages, enabled=True, recent_tool_keep=8, min_messages=16, max_compressed_len=300,
    )
    tool_msgs_orig = [m for m in messages if m.get("role") == "tool"]
    tool_msgs_comp = [m for m in compressed if m.get("role") == "tool"]
    compressed_count = sum(1 for o, c in zip(tool_msgs_orig, tool_msgs_comp) if o["content"] != c["content"])
    print(f"Total tool results: {len(tool_msgs_orig)}")
    print(f"Compressed: {compressed_count}")
    print(f"Preserved (recent): {len(tool_msgs_orig) - compressed_count}")
    if compressed_count > 0:
        orig_sizes = [len(m["content"]) for m in tool_msgs_orig[:compressed_count]]
        comp_sizes = [len(m["content"]) for m in tool_msgs_comp[:compressed_count]]
        print(f"Original avg size: {statistics.mean(orig_sizes):,.0f} chars")
        print(f"Compressed avg size: {statistics.mean(comp_sizes):,.0f} chars")
        print(f"Compression ratio: {statistics.mean(orig_sizes)/statistics.mean(comp_sizes):.1f}x")


if __name__ == "__main__":
    benchmark()
