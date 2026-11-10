"""Queue priority: a user's typed input outranks queued automation prompts.

Goal/loop continuations and background-process notices share ``_pending_input`` with human
input. When the user interrupts a goal turn (busy_input_mode=interrupt), their message must
run NEXT, not behind the automation backlog, and the live-work dock's queued count must not
present automation prompts as user work waiting in line.
"""

from __future__ import annotations

import queue


def _make_pending():
    return queue.Queue()


class TestPromoteUserInput:
    def test_user_message_moves_ahead_of_automation_prompts(self):
        from hermes_cli.cli_queue_priority import promote_user_input_over_automation

        pending = _make_pending()
        pending.put("[Continuing toward your standing goal]\nGoal: build a thing\n\nContinue...")
        pending.put("[IMPORTANT: 2 background process results arrived while this session was busy.")
        pending.put("咋回事，这么慢？")
        promote_user_input_over_automation(pending)

        order = [pending.get_nowait() for _ in range(3)]
        assert order[0] == "咋回事，这么慢？"  # 期望: 用户消息排最前
        assert order[1].startswith("[Continuing toward your standing goal")  # 期望: 自动化项保持原相对顺序
        assert order[2].startswith("[IMPORTANT: ")

    def test_tuple_payload_with_images_stays_intact(self):
        from hermes_cli.cli_queue_priority import promote_user_input_over_automation

        pending = _make_pending()
        pending.put("[/loop wakeup #3 · every 5m]\nThis is an automatic wakeup from the /loop the user set.")
        pending.put(("看一下这个截图", ["a.png"]))
        promote_user_input_over_automation(pending)

        first = pending.get_nowait()
        assert isinstance(first, tuple) and first[0] == "看一下这个截图"
        assert pending.get_nowait().startswith("[/loop wakeup #")

    def test_all_automation_or_all_user_order_unchanged(self):
        from hermes_cli.cli_queue_priority import promote_user_input_over_automation

        pending = _make_pending()
        for text in ("第一条", "第二条", "第三条"):
            pending.put(text)
        promote_user_input_over_automation(pending)
        assert [pending.get_nowait() for _ in range(3)] == ["第一条", "第二条", "第三条"]

    def test_empty_queue_is_a_noop(self):
        from hermes_cli.cli_queue_priority import promote_user_input_over_automation

        pending = _make_pending()
        promote_user_input_over_automation(pending)
        assert pending.empty()


class _StubCli:
    def __init__(self, items):
        self.session_id = None  # gateway_store() path skipped
        self._pending_input = _make_pending()
        for item in items:
            self._pending_input.put(item)


class TestDockHidesAutomationPrompts:
    def test_queued_prompts_counts_only_human_input(self):
        from hermes_cli.cli_session_dock import queued_prompts

        cli = _StubCli([
            "[Continuing toward your standing goal]\nGoal: build a thing",
            "先看我的问题",
            "[IMPORTANT: 1 background process results arrived while this session was busy.",
        ])
        rows = queued_prompts(cli)
        assert rows == ["先看我的问题"]  # 期望: 排队计数只反映真人输入

    def test_queued_prompts_without_automation_unchanged(self):
        from hermes_cli.cli_session_dock import queued_prompts

        cli = _StubCli(["问题一", "问题二"])
        assert queued_prompts(cli) == ["问题一", "问题二"]
