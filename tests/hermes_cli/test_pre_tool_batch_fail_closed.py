"""Fail-closed regression tests for the ``pre_tool_batch`` policy hook (2026-09-24).

The batch gate hook is a POLICY hook like ``pre_tool_call``: a callback that
raises or times out made no decision, and "no decision" from a guard must
surface as a NAMED block directive — never a silent allow. Before this fix
``pre_tool_batch`` was in neither timeout set: a crash was swallowed and a hang
blocked the whole agent forever.

Expectations derive from the dispatcher's documented policy-hook contract
(#109624: a raising policy guard fails closed; #105223: a hung one does not
fail closed forever — suppression window + bounded abandoned workers).
"""

import logging
import threading

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli import plugins_dispatch as dispatch


class TestPreToolBatchFailClosed:
    def test_raising_callback_fails_closed_with_named_block(self):
        """A crashing batch-gate callback must produce a block directive that
        names the callback and the error (visible in the tool result)."""
        mgr = PluginManager()

        def gate_exploded(**_kwargs):
            raise RuntimeError("gate exploded")

        mgr._hooks["pre_tool_batch"] = [gate_exploded]

        results = mgr.invoke_hook("pre_tool_batch", tool_calls=[])

        assert len(results) == 1
        directive = results[0]
        assert directive["action"] == "block"
        assert "gate_exploded" in directive["message"]
        assert "gate exploded" in directive["message"]

    def test_timed_out_callback_fails_closed(self, monkeypatch):
        """A hung batch-gate callback is abandoned at the timeout and fails
        closed with a block directive naming the hook."""
        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
        )

        mgr = PluginManager()
        mgr._hook_timeout_suppression_seconds = 0.0  # isolate from suppression back-off

        hold = threading.Event()

        def gate_hung(**_kwargs):
            hold.wait(timeout=10.0)
            return None

        mgr._hooks["pre_tool_batch"] = [gate_hung]

        results = mgr.invoke_hook("pre_tool_batch", tool_calls=[])

        assert len(results) == 1
        directive = results[0]
        assert directive["action"] == "block"
        assert directive["message"] == dispatch._hook_timeout_block_message("pre_tool_batch")
        assert "pre_tool_batch" in directive["message"]
        hold.set()

    def test_healthy_callback_still_decides(self):
        """Fail-closed is failure-path only: a healthy gate's allow/block
        verdicts pass through unchanged."""
        mgr = PluginManager()

        def healthy_allow(**_kwargs):
            return None

        def healthy_block(**_kwargs):
            return {"action": "block", "message": "blocked on merit"}

        mgr._hooks["pre_tool_batch"] = [healthy_allow]
        assert mgr.invoke_hook("pre_tool_batch", tool_calls=[]) == []

        mgr._hooks["pre_tool_batch"] = [healthy_block]
        assert mgr.invoke_hook("pre_tool_batch", tool_calls=[]) == [
            {"action": "block", "message": "blocked on merit"}
        ]

    def test_pre_tool_call_timeout_message_byte_stable(self):
        """The pre_tool_call timeout block message keeps its exact historical
        wording — existing tests and operators match on it."""
        msg = dispatch._hook_timeout_block_message("pre_tool_call")
        assert msg == "pre_tool_call plugin callback timed out or is still running"
        assert msg == dispatch._PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

    def test_policy_hook_classification(self):
        """pre_tool_batch is classified as a policy hook (fail-closed) and is
        therefore also timeout-bounded — a hang can no longer freeze the agent."""
        assert "pre_tool_batch" in dispatch._HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        assert dispatch._hook_uses_callback_timeout(
            "pre_tool_batch", 30.0
        ), "policy hooks must run under the callback timeout"

    def test_raising_callback_fails_closed_async_too(self):
        """The async dispatch path shares the failure contract."""
        import asyncio

        mgr = PluginManager()

        def gate_exploded(**_kwargs):
            raise RuntimeError("gate exploded")

        mgr._hooks["pre_tool_batch"] = [gate_exploded]

        results = asyncio.run(mgr.ainvoke_hook("pre_tool_batch", tool_calls=[]))

        assert len(results) == 1
        assert results[0]["action"] == "block"
        assert "gate exploded" in results[0]["message"]
