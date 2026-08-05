#!/usr/bin/env python3
"""
Tests for sandwich-inject plugin (三明治架构).

Verifies:
1. Pre-hook: scene keyword match activates injection (SOP + knowledge + 守则)
2. No match = zero behavior (returns None)
3. Post-hook: required-tool contract check via conversation_history
4. Contract satisfied resets violation counter
5. Escalates to NEED_HUMAN_INTERVENTION after max_retries
6. Knowledge files loaded; missing files don't crash
"""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_PATH = (
    Path(__file__).parent.parent.parent
    / "plugins" / "sandwich-inject" / "__init__.py"
)


def load_plugin():
    spec = importlib.util.spec_from_file_location("sandwich_inject", PLUGIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_scene(**overrides):
    base = {
        "name": "test-scene",
        "match_keywords": ["反洗钱", "AML"],
        "sop": "1. 先查数据库\n2. 引用监管条款",
        "knowledge": "AML 知识库内容",
        "required_tools": ["mcp__dbhub__execute_sql_aml_v7"],
        "max_retries": 2,
    }
    base.update(overrides)
    return base


class TestSandwichInject(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmpdir.name)
        self._patcher = patch.dict(os.environ, {"HERMES_HOME": str(self.tmpdir)})
        self._patcher.start()
        self._plugin = load_plugin()
        self._plugin._VIOLATION_TRACKER.clear()

    def tearDown(self):
        self._plugin._VIOLATION_TRACKER.clear()
        self._patcher.stop()
        self._tmpdir.cleanup()

    # ── Pre-hook: 场景匹配 ──────────────────────────────────

    def test_keyword_hit_activates(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene()]}
        result = self._plugin.on_pre_llm_call(
            user_message="反洗钱审查这条交易",
            session_id="s1",
            conversation_history=[{"role": "user", "content": "反洗钱审查这条交易"}],
        )
        self.assertIsNotNone(result)
        ctx = result["context"]
        self.assertIn("【强制SOP - 必须严格遵循】", ctx)
        self.assertIn("先查数据库", ctx)
        self.assertIn("【已注入的最新知识 - 禁止自行猜测】", ctx)
        self.assertIn("必需工具契约", ctx)
        self.assertIn("NEED_HUMAN_INTERVENTION", ctx)

    def test_no_keyword_no_behavior(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene()]}
        result = self._plugin.on_pre_llm_call(
            user_message="写个 hello world",
            session_id="s1",
            conversation_history=[],
        )
        self.assertIsNone(result)

    def test_empty_keywords_never_match(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene(match_keywords=[])]}
        result = self._plugin.on_pre_llm_call(
            user_message="反洗钱", session_id="s1", conversation_history=[]
        )
        self.assertIsNone(result)

    def test_no_config_no_behavior(self):
        self._plugin._load_config = lambda: None
        result = self._plugin.on_pre_llm_call(
            user_message="反洗钱", session_id="s1", conversation_history=[]
        )
        self.assertIsNone(result)

    # ── Post-hook: 工具契约校验 ─────────────────────────────

    def test_missing_required_tool_flags_violation(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene()]}
        result = self._plugin.on_pre_llm_call(
            user_message="反洗钱审查",
            session_id="s2",
            conversation_history=[{"role": "user", "content": "反洗钱审查"}],
        )
        self.assertIsNotNone(result)
        self.assertIn("校验报错", result["context"])
        state = self._plugin._VIOLATION_TRACKER["s2"]["test-scene"]
        self.assertEqual(state["violations"], 1)

    def test_contract_satisfied_resets_violations(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene()]}
        # 第一轮违规
        self._plugin.on_pre_llm_call(
            user_message="反洗钱审查", session_id="s3",
            conversation_history=[{"role": "user", "content": "反洗钱审查"}],
        )
        # 第二轮调用了必需工具
        result = self._plugin.on_pre_llm_call(
            user_message="反洗钱审查", session_id="s3",
            conversation_history=[{
                "role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "mcp__dbhub__execute_sql_aml_v7", "arguments": "{}"}},
                ],
            }],
        )
        self.assertIsNotNone(result)
        self.assertNotIn("校验报错", result["context"])
        self.assertEqual(self._plugin._VIOLATION_TRACKER["s3"]["test-scene"]["violations"], 0)

    def test_escalates_to_human_after_max_retries(self):
        self._plugin._load_config = lambda: {"scenes": [make_scene(max_retries=2)]}
        r1 = self._plugin.on_pre_llm_call(
            user_message="反洗钱审查", session_id="s4",
            conversation_history=[{"role": "user", "content": "反洗钱审查"}],
        )
        r2 = self._plugin.on_pre_llm_call(
            user_message="反洗钱审查", session_id="s4",
            conversation_history=[{"role": "user", "content": "反洗钱审查"}],
        )
        self.assertIn("校验报错", r1["context"])
        self.assertIn("NEED_HUMAN_INTERVENTION", r2["context"])
        self.assertIn("转人工", r2["context"])

    # ── 辅助函数 ────────────────────────────────────────────

    def test_last_turn_tool_calls_picks_recent(self):
        history = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "terminal", "arguments": "{}"}},
            ]},
        ]
        self.assertEqual(self._plugin._last_turn_tool_calls(history), ["terminal"])

    def test_last_turn_tool_calls_no_recent_tool_turn(self):
        history = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "web_search", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        self.assertEqual(self._plugin._last_turn_tool_calls(history), [])

    def test_knowledge_file_loaded(self):
        kb = self.tmpdir / "kb.md"
        kb.write_text("KB CONTENT", encoding="utf-8")
        result = self._plugin._load_knowledge_files([str(kb)])
        self.assertEqual(result, "KB CONTENT")

    def test_missing_knowledge_file_no_crash(self):
        result = self._plugin._load_knowledge_files([str(self.tmpdir / "nope.md")])
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
