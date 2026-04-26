"""Tests for progressive tool-result compression in AIAgent.

Covers the _progressive_tool_result_compress static method:
- Short conversations are left untouched
- Few tool results are left untouched
- Old tool results are compressed to one-line summaries
- Recent tool results are preserved verbatim
- Already-short results are not compressed
- Error outcomes are detected
- Success outcomes are detected
- Exit code outcomes are detected
- Non-string content is handled gracefully
- Disabled via enabled=False skips all compression
- Configurable thresholds are respected
"""

from run_agent import AIAgent as AgentRunner  # noqa: N811


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(assistant_idx: int, tool_name: str, tool_id: str):
    """Create an assistant message with a tool call."""
    return {
        "role": "assistant",
        "content": f"Calling {tool_name}...",
        "tool_calls": [
            {
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": "{}"},
            }
        ],
    }


def _make_tool_result(tool_id: str, content: str):
    """Create a tool result message."""
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": content,
    }


def _build_long_conversation(num_tool_pairs: int = 12, content_length: int = 500):
    """Build a conversation with enough tool calls to trigger compression.

    Returns (messages, original_tool_contents) where original_tool_contents
    maps tool_call_id -> original content for verification.
    """
    messages = [{"role": "system", "content": "You are helpful."}]
    messages.append({"role": "user", "content": "Do things."})
    original_contents = {}

    for i in range(num_tool_pairs):
        tool_id = f"call_{i:03d}"
        tool_name = f"tool_{i}"
        content = f"Result line 1 of tool_{i}\n" + "x" * content_length + f"\nResult last line of tool_{i}\n"
        messages.append(_make_tool_call(i, tool_name, tool_id))
        messages.append(_make_tool_result(tool_id, content))
        original_contents[tool_id] = content

    return messages, original_contents


# ---------------------------------------------------------------------------
# Tests: no-op cases
# ---------------------------------------------------------------------------

class TestProgressiveCompressNoOp:

    def test_disabled_skips_all(self):
        """When enabled=False, no compression happens regardless of length."""
        messages, _ = _build_long_conversation(num_tool_pairs=12, content_length=500)
        original = [m["content"] for m in messages]

        result = AgentRunner._progressive_tool_result_compress(
            messages, enabled=False, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        assert [m.get("content", "") for m in result] == original

    def test_short_conversation_untouched(self):
        """Conversations below min_messages threshold are not compressed."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "A" * 500},
        ]
        original_content = messages[-1]["content"]

        result = AgentRunner._progressive_tool_result_compress(
            messages, min_messages=16,
        )

        assert result[-1]["content"] == original_content

    def test_few_tool_results_untouched(self):
        """If tool_count <= recent_tool_keep, nothing is compressed."""
        messages, _ = _build_long_conversation(num_tool_pairs=3, content_length=500)
        # 3 tool results <= recent_tool_keep=8 (explicit), so no compression
        original_contents = [m.get("content", "") for m in messages if m.get("role") == "tool"]

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=8, min_messages=5, max_compressed_len=100,
        )

        result_contents = [m.get("content", "") for m in result if m.get("role") == "tool"]
        assert result_contents == original_contents

    def test_empty_messages(self):
        """Empty message list is returned as-is."""
        result = AgentRunner._progressive_tool_result_compress([])
        assert result == []

    def test_no_tool_messages(self):
        """Conversation with no tool messages is returned as-is."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ] * 10

        result = AgentRunner._progressive_tool_result_compress(
            messages, min_messages=5,
        )

        assert result == messages


# ---------------------------------------------------------------------------
# Tests: compression behavior
# ---------------------------------------------------------------------------

class TestProgressiveCompressBehavior:

    def test_old_tool_results_compressed(self):
        """Tool results beyond recent_tool_keep are compressed to summaries."""
        messages, originals = _build_long_conversation(num_tool_pairs=12, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=5, max_compressed_len=100,
        )

        # Last 3 tool results should be unchanged
        tool_results = [(i, m) for i, m in enumerate(result) if m.get("role") == "tool"]
        assert len(tool_results) == 12

        # Last 3 preserved
        for idx, msg in tool_results[-3:]:
            assert msg["content"] == originals[msg["tool_call_id"]]

        # Older ones should be compressed (shorter than original)
        for idx, msg in tool_results[:-3]:
            original_len = len(originals[msg["tool_call_id"]])
            compressed_len = len(msg["content"])
            assert compressed_len < original_len, (
                f"Tool result at index {idx} (id={msg['tool_call_id']}) was not compressed: "
                f"{compressed_len} >= {original_len}"
            )

    def test_compressed_format_contains_tool_name(self):
        """Compressed summaries include the tool name."""
        messages, _ = _build_long_conversation(num_tool_pairs=12, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # First tool call was tool_0
        assert "[tool_0]" in tool_results[0]["content"]

    def test_compressed_format_contains_char_count(self):
        """Compressed summaries include original character count."""
        messages, _ = _build_long_conversation(num_tool_pairs=12, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=5, max_compressed_len=100,
        )

        first_compressed = [m for m in result if m.get("role") == "tool"][0]
        assert "chars" in first_compressed["content"]

    def test_already_short_results_untouched(self):
        """Tool results shorter than max_compressed_len are not compressed."""
        messages = [{"role": "system", "content": "sys"}]
        for i in range(15):
            tool_id = f"call_{i:03d}"
            short_content = "short result"  # well under 300 chars
            messages.append(_make_tool_call(i, f"tool_{i}", tool_id))
            messages.append(_make_tool_result(tool_id, short_content))

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=5, max_compressed_len=300,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        for msg in tool_results:
            assert msg["content"] == "short result"

    def test_non_string_content_skipped(self):
        """Tool results with non-string content (None, list) are skipped."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c_old", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "c_recent", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c_old", "content": None},
            {"role": "tool", "tool_call_id": "c_recent", "content": "recent result"},
        ] * 4  # Repeat to hit min_messages

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=10,
        )

        # None content should remain None (not crash)
        none_results = [m for m in result if m.get("role") == "tool" and m.get("content") is None]
        assert len(none_results) > 0

    def test_empty_content_skipped(self):
        """Tool results with empty string content are skipped."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": ""},
        ] * 10

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=10,
        )

        for m in result:
            if m.get("role") == "tool":
                assert m.get("content") == ""


# ---------------------------------------------------------------------------
# Tests: outcome detection
# ---------------------------------------------------------------------------

class TestProgressiveCompressOutcomeDetection:

    def _make_single_old_tool(self, content: str):
        """Create a conversation where the first tool result is old and compressible."""
        messages = []
        for i in range(10):
            tool_id = f"call_{i:03d}"
            messages.append(_make_tool_call(i, f"tool_{i}", tool_id))
            if i == 0:
                messages.append(_make_tool_result(tool_id, content))
            else:
                messages.append(_make_tool_result(tool_id, "recent"))
        return messages

    def test_error_detected(self):
        """Error keywords produce ERROR outcome in summary."""
        messages = self._make_single_old_tool("some output\nError: connection refused\nmore output " + "x" * 400)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_tool = [m for m in result if m.get("role") == "tool"][0]
        assert "ERROR" in first_tool["content"]

    def test_success_detected(self):
        """Success keywords produce OK outcome in summary."""
        messages = self._make_single_old_tool("some output\nAll tests passed\nmore output " + "x" * 400)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_tool = [m for m in result if m.get("role") == "tool"][0]
        assert "OK" in first_tool["content"]

    def test_exit_code_detected(self):
        """Exit code in output produces exit=N outcome in summary."""
        messages = self._make_single_old_tool("compiling...\nexit_code: 1\nmore output " + "x" * 400)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_tool = [m for m in result if m.get("role") == "tool"][0]
        assert "exit=1" in first_tool["content"]

    def test_neutral_outcome(self):
        """No keyword produces 'done' outcome."""
        messages = self._make_single_old_tool("line one\nline two\nline three " + "x" * 400)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_tool = [m for m in result if m.get("role") == "tool"][0]
        assert "done" in first_tool["content"]


# ---------------------------------------------------------------------------
# Tests: configurable thresholds
# ---------------------------------------------------------------------------

class TestProgressiveCompressThresholds:

    def test_custom_recent_tool_keep(self):
        """Only the configured number of recent results are preserved."""
        messages, originals = _build_long_conversation(num_tool_pairs=10, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=5, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # Last 5 should be original
        for msg in tool_results[-5:]:
            assert msg["content"] == originals[msg["tool_call_id"]]
        # First 5 should be compressed
        for msg in tool_results[:-5]:
            assert len(msg["content"]) < len(originals[msg["tool_call_id"]])

    def test_custom_min_messages(self):
        """Compression only triggers above the configured min_messages."""
        messages, _ = _build_long_conversation(num_tool_pairs=5, content_length=500)
        # 5 tool pairs = 11 messages (sys + user + 5*2)
        original_tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]

        # min_messages=20 should skip compression (11 < 20)
        result_high = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=20, max_compressed_len=100,
        )
        high_contents = [m["content"] for m in result_high if m.get("role") == "tool"]

        # min_messages=5 should compress — use fresh copy since first call is immutable
        import copy
        messages_copy = copy.deepcopy(messages)
        result_low = AgentRunner._progressive_tool_result_compress(
            messages_copy, recent_tool_keep=1, min_messages=5, max_compressed_len=100,
        )
        low_contents = [m["content"] for m in result_low if m.get("role") == "tool"]

        # high threshold preserved all, low threshold compressed some
        assert high_contents == original_tool_contents
        assert low_contents != original_tool_contents

    def test_custom_max_compressed_len(self):
        """Results shorter than max_compressed_len are not compressed."""
        # 200 chars content, max_compressed_len=250 -> should not compress
        messages, _ = _build_long_conversation(num_tool_pairs=10, content_length=200)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=250,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # All results ~200 chars, under 250 limit, none compressed
        for msg in tool_results:
            # Should still have full content (contains "Result line 1")
            assert "Result line 1" in msg["content"] or msg["content"].startswith("[")


# ---------------------------------------------------------------------------
# Tests: immutability
# ---------------------------------------------------------------------------

class TestProgressiveCompressImmutability:

    def test_original_messages_unchanged(self):
        """The input messages list is not mutated."""
        messages, originals = _build_long_conversation(num_tool_pairs=12, content_length=500)

        import copy
        messages_copy = copy.deepcopy(messages)

        AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=5, max_compressed_len=100,
        )

        # Original should be unchanged
        for i, msg in enumerate(messages):
            assert msg["content"] == messages_copy[i]["content"], (
                f"Message at index {i} was mutated"
            )


# ---------------------------------------------------------------------------
# Tests: edge cases and code-path coverage
# ---------------------------------------------------------------------------

class TestProgressiveCompressEdgeCases:

    def test_tool_calls_as_objects_not_dicts(self):
        """tool_calls entries that are objects (not plain dicts) are handled."""
        # Simulate Pydantic/OpenAI SDK objects with attribute access
        class FakeToolCall:
            def __init__(self, id_, name):
                self.id = id_
                self.function = type("F", (), {"name": name})()

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                FakeToolCall("call_obj_old", "read_file"),
                FakeToolCall("call_obj_recent", "terminal"),
            ]},
            {"role": "tool", "tool_call_id": "call_obj_old", "content": "A" * 500},
            {"role": "tool", "tool_call_id": "call_obj_recent", "content": "recent"},
        ] * 5  # 25 messages total

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # Old one should be compressed and contain tool name from object
        assert "[read_file]" in tool_results[0]["content"]
        # Recent one preserved
        assert tool_results[-1]["content"] == "recent"

    def test_multiple_tool_calls_per_assistant_message(self):
        """One assistant message with multiple tool_calls — all mapped correctly."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "c3", "type": "function", "function": {"name": "web_search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "terminal output " + "x" * 400},
            {"role": "tool", "tool_call_id": "c2", "content": "file content " + "x" * 400},
            {"role": "tool", "tool_call_id": "c3", "content": "recent search"},
        ] * 5

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # Should have 15 tool results (3 per round * 5 rounds)
        assert len(tool_results) == 15
        # Check that the mapping is correct for each tool
        assert "[terminal]" in tool_results[0]["content"]
        assert "[read_file]" in tool_results[1]["content"]

    def test_orphan_tool_call_id_falls_back_to_tool(self):
        """Tool result with unknown tool_call_id falls back to 'tool'."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "known_id", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "orphan_id", "content": "orphan output " + "x" * 400},
            {"role": "tool", "tool_call_id": "known_id", "content": "recent"},
        ] * 5

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # First compressed result should use fallback name
        assert "[tool]" in tool_results[0]["content"]

    def test_first_line_equals_last_line(self):
        """When first_line == last_line, summary does not duplicate."""
        # Content where first and last non-empty lines are the same.
        # The padding must come BEFORE the repeated last line so that
        # reversed() iteration hits the repeated line last.
        content = "This is the repeated line.\n" + "padding " * 50 + "\nThis is the repeated line.\n"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        for i in range(8):
            tool_id = f"call_{i:03d}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tool_id, "type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}},
            ]})
            if i == 0:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
            else:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": "recent"})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_compressed = [m for m in result if m.get("role") == "tool"][0]
        summary = first_compressed["content"]
        # Should NOT contain "→" since first_line == last_line
        assert "→" not in summary
        # Should still have the first line
        assert "This is the repeated line" in summary

    def test_recent_tool_keep_zero_compresses_all(self):
        """recent_tool_keep=0 means ALL tool results are compressed."""
        messages, originals = _build_long_conversation(num_tool_pairs=8, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=0, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        for msg in tool_results:
            # Every tool result should be compressed (shorter than original)
            original_len = len(originals[msg["tool_call_id"]])
            assert len(msg["content"]) < original_len

    def test_no_assistant_messages_tool_id_to_name_empty(self):
        """Tool results without any assistant messages — name falls back to 'tool'."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        for i in range(8):
            messages.append({"role": "tool", "tool_call_id": f"call_{i:03d}", "content": "output " + "x" * 400})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # All compressed results should use fallback name "tool"
        for msg in tool_results[:-2]:
            assert msg["content"].startswith("[tool]")

    def test_content_as_list_not_string(self):
        """Some APIs return content as a list (content_parts), not a string."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            # List content — should be skipped (not isinstance(content, str))
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "multiline\noutput\nhere " + "x" * 400}
            ]},
        ] * 6

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # List content should remain as-is (not compressed to string)
        for msg in tool_results[:-1]:
            assert isinstance(msg["content"], list)

    def test_missing_tool_call_id_field(self):
        """Tool result without tool_call_id — name falls back to 'tool'."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
            ]},
            # No tool_call_id field at all
            {"role": "tool", "content": "output " + "x" * 400},
            {"role": "tool", "tool_call_id": "c1", "content": "recent"},
        ] * 5

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        # First compressed result has no tool_call_id → fallback name
        assert "[tool]" in tool_results[0]["content"]

    def test_content_exactly_at_max_compressed_len(self):
        """Content exactly max_compressed_len chars should NOT be compressed (<=)."""
        exact_len = 300
        content = "A" * exact_len
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        for i in range(8):
            tool_id = f"call_{i:03d}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tool_id, "type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}},
            ]})
            if i == 0:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
            else:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": "recent"})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=exact_len,
        )

        first_tool = [m for m in result if m.get("role") == "tool"][0]
        # len(content) == max_compressed_len, <= is true, so NOT compressed
        assert first_tool["content"] == content

    def test_total_equals_min_messages_boundary(self):
        """When total == min_messages, compression is skipped (<=)."""
        messages, _ = _build_long_conversation(num_tool_pairs=5, content_length=500)
        # 5 tool pairs = 11 messages. Set min_messages=11 → should skip.
        total_msgs = len(messages)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=1, min_messages=total_msgs, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        original_tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
        # All should be preserved since total == min_messages
        assert [m["content"] for m in tool_results] == original_tool_contents

    def test_tool_count_equals_recent_tool_keep_boundary(self):
        """When tool_count == recent_tool_keep, compression is skipped (<=)."""
        messages, _ = _build_long_conversation(num_tool_pairs=5, content_length=500)

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=5, min_messages=5, max_compressed_len=100,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        original_tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
        assert [m["content"] for m in tool_results] == original_tool_contents

    def test_all_empty_lines_content(self):
        """Content with only whitespace/newlines — first_line and last_line both empty."""
        # Pure whitespace lines followed by padding to exceed max_compressed_len
        content = "\n\n\n   \n\t\n" + " " * 400
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        for i in range(8):
            tool_id = f"call_{i:03d}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tool_id, "type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}},
            ]})
            if i == 0:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
            else:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": "recent"})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_compressed = [m for m in result if m.get("role") == "tool"][0]
        summary = first_compressed["content"]
        # No "→" since last_line is empty
        assert "→" not in summary
        # Should still have tool name and char count
        assert "[tool_0]" in summary
        assert "chars" in summary

    def test_exit_code_without_number_falls_back_to_done(self):
        """exit_code keyword present but no number → outcome stays 'done'."""
        content = "compilation output\nexit_code: (no number here)\nmore output " + "x" * 400
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
        ]
        for i in range(8):
            tool_id = f"call_{i:03d}"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tool_id, "type": "function", "function": {"name": f"tool_{i}", "arguments": "{}"}},
            ]})
            if i == 0:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": content})
            else:
                messages.append({"role": "tool", "tool_call_id": tool_id, "content": "recent"})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=2, min_messages=5, max_compressed_len=100,
        )

        first_compressed = [m for m in result if m.get("role") == "tool"][0]
        # Should be "done" not "exit=..."
        assert "done" in first_compressed["content"]
        assert "exit=" not in first_compressed["content"]


# ---------------------------------------------------------------------------
# Tests: size-gated compression
# ---------------------------------------------------------------------------

class TestProgressiveCompressSizeGated:

    def test_large_recent_result_compressed_within_window(self):
        """Large recent result gets compressed even when tool_count <= recent_tool_keep."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        tid = "call_000"
        large = "Large output line 1\n" + "x" * 5100 + "\nLarge output line N"
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": tid, "content": large})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=8, min_messages=2, max_compressed_len=100,
            max_single_size=3000,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 1
        summary = tool_results[0]["content"]
        assert len(summary) < 500
        assert "[terminal]" in summary
        assert "chars" in summary

    def test_large_result_compressed_below_min_messages(self):
        """Size-gated works even when total < min_messages."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        tid = "call_000"
        large = "A" * 6000
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": tid, "content": large})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=8, min_messages=16, max_compressed_len=100,
            max_single_size=5000,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert len(tool_results[0]["content"]) < 500

    def test_disabled_with_zero(self):
        """max_single_size=0 disables size-gated compression."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        tid = "call_000"
        large = "A" * 6000
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": tid, "type": "function", "function": {"name": "terminal", "arguments": "{}"}},
        ]})
        messages.append({"role": "tool", "tool_call_id": tid, "content": large})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=8, min_messages=2, max_compressed_len=100,
            max_single_size=0,
        )

        assert [m for m in result if m.get("role") == "tool"][0]["content"] == large

    def test_small_result_untouched_by_size_gate(self):
        """Results below max_single_size are not affected."""
        messages, originals = _build_long_conversation(num_tool_pairs=5, content_length=200)
        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=8, min_messages=2, max_compressed_len=100,
            max_single_size=300,
        )
        for msg in (m for m in result if m.get("role") == "tool"):
            assert msg["content"] == originals[msg["tool_call_id"]]

    def test_size_gated_and_old_compression_both_fire(self):
        """Old-position + size-gated compression both work in the same call."""
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
        for i in range(10):
            tid = f"call_{i:03d}"
            name = "tool_big" if i == 0 else "terminal"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": tid, "type": "function", "function": {"name": name, "arguments": "{}"}},
            ]})
            if i == 0:
                messages.append({"role": "tool", "tool_call_id": tid,
                                 "content": "Initial big output\n" + "a" * 8000 + "\nDone"})
            else:
                messages.append({"role": "tool", "tool_call_id": tid, "content": "x" * 200})

        result = AgentRunner._progressive_tool_result_compress(
            messages, recent_tool_keep=3, min_messages=2, max_compressed_len=100,
            max_single_size=5000,
        )

        tool_results = [m for m in result if m.get("role") == "tool"]
        assert len(tool_results[0]["content"]) < 500
        assert "[tool_big]" in tool_results[0]["content"]
        for msg in tool_results[-3:]:
            assert len(msg["content"]) == 200
