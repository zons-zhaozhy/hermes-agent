#!/usr/bin/env python3
"""
Tests for sandwich-inject plugin (三明治架构).

Verifies:
1. Pre-hook: scene keyword match activates injection (SOP + knowledge + 守则)
2. No match = zero behavior (returns None)
3. Required-tool contract: violation counter increments, resets on coverage
4. NEED_HUMAN_INTERVENTION after max_retries consecutive violations
5. Scene locking: once matched, subsequent turns keep injection (no re-match)
6. Knowledge-file mtime cache: file read once, refresh on change
7. Bounded session tracking: oldest sessions evicted past capacity
8. Violation counter per session per scene (isolated)
"""

import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN_ROOT = Path(__file__).parent.parent.parent / "plugins" / "sandwich-inject"


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "sandwich_inject", PLUGIN_ROOT / "__init__.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SandwichInjectTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.config = self.tmpdir / "sandwich.yaml"
        self._env_patcher = patch.dict(
            "os.environ", {"HERMES_SANDWICH_CONFIG": str(self.config)}
        )
        self._env_patcher.start()
        self.mod = _load_plugin()
        # 清空跨用例状态
        self.mod._VIOLATION_TRACKER.clear()
        self.mod._SESSION_SCENES.clear()
        self.mod._KB_CACHE.clear()
        self.mod._CONFIG_CACHE.clear()

    def tearDown(self):
        self._env_patcher.stop()
        self._tmp.cleanup()

    def _write_config(self, scenes):
        import yaml

        self.config.write_text(yaml.safe_dump({"scenes": scenes}), encoding="utf-8")

    def _call(self, user_message, history=None, session_id="sess-1"):
        kwargs = {
            "user_message": user_message,
            "session_id": session_id,
            "conversation_history": history or [],
        }
        return self.mod.on_pre_llm_call(**kwargs)

    # ── 1. 基础注入 ────────────────────────────────────────────────

    def test_injects_sop_and_knowledge_on_match(self):
        self._write_config([
            {
                "name": "aml",
                "match_keywords": ["AML", "可疑交易"],
                "sop": "1. 先查库\n2. 再分析",
                "knowledge": "大额标准: 20万",
            }
        ])
        result = self._call("帮我分析这笔可疑交易")
        self.assertIsNotNone(result)
        ctx = result["context"]
        self.assertIn("【强制SOP - 必须严格遵循】", ctx)
        self.assertIn("先查库", ctx)
        self.assertIn("大额标准: 20万", ctx)
        self.assertIn("禁止自行猜测", ctx)
        self.assertIn("NEED_HUMAN_INTERVENTION", ctx)  # 工作守则兜底

    def test_no_match_returns_none(self):
        self._write_config([
            {"name": "aml", "match_keywords": ["AML"], "sop": "x"}
        ])
        result = self._call("今天天气怎么样")
        self.assertIsNone(result)

    def test_empty_config_returns_none(self):
        result = self._call("AML 分析")
        self.assertIsNone(result)

    # ── 2. 工具契约 ────────────────────────────────────────────────

    def _aml_scene(self):
        return [{
            "name": "aml",
            "match_keywords": ["AML"],
            "sop": "查库",
            "required_tools": ["mcp__dbhub__execute_sql_aml_v8"],
            "max_retries": 2,
        }]

    def _history_with_tool(self, tool_name):
        return [
            {"role": "user", "content": "AML 分析"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": tool_name, "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "t1", "content": "{}"},
        ]

    def test_violation_increments_on_missing_tool(self):
        self._write_config(self._aml_scene())
        # 上轮是普通回答（无工具调用）→ 违规 +1
        result = self._call(
            "AML 分析",
            history=[{"role": "user", "content": "x"},
                     {"role": "assistant", "content": "直接回答"}],
        )
        self.assertIn("校验报错", result["context"])
        self.assertEqual(
            self.mod._VIOLATION_TRACKER["sess-1"]["aml"]["violations"], 1
        )

    def test_contract_met_resets_violations(self):
        self._write_config(self._aml_scene())
        # 先违规一次
        self._call("AML 分析", history=[
            {"role": "assistant", "content": "直接回答"}
        ])
        # 再合规一次
        result = self._call(
            "AML 分析", history=self._history_with_tool(
                "mcp__dbhub__execute_sql_aml_v8"
            )
        )
        self.assertNotIn("校验报错", result["context"])
        self.assertEqual(
            self.mod._VIOLATION_TRACKER["sess-1"]["aml"]["violations"], 0
        )

    def test_need_human_intervention_after_max_retries(self):
        self._write_config(self._aml_scene())
        for i in range(2):
            result = self._call("AML 分析", history=[
                {"role": "assistant", "content": f"回答{i}"}
            ])
        self.assertIn("NEED_HUMAN_INTERVENTION", result["context"])
        self.assertFalse(self.mod._VIOLATION_TRACKER["sess-1"]["aml"]["active"])

    def test_scene_activation_locked_for_session(self):
        """场景首轮命中后锁定：后续轮次关键词消失也继续注入"""
        self._write_config([
            {"name": "aml", "match_keywords": ["AML"], "sop": "查库"}
        ])
        r1 = self._call("请做 AML 分析")
        self.assertIsNotNone(r1)
        # 第二轮用户不再提 AML
        r2 = self._call("继续", session_id="sess-1")
        self.assertIsNotNone(r2)
        self.assertIn("查库", r2["context"])
        # 另一个会话不受影响
        r3 = self._call("继续", session_id="other-sess")
        self.assertIsNone(r3)

    def test_per_session_violation_isolation(self):
        self._write_config(self._aml_scene())
        self._call("AML", history=[
            {"role": "assistant", "content": "不查库直接答"}
        ], session_id="sess-A")
        self._call("AML", history=[
            {"role": "assistant", "content": "不查库直接答"}
        ], session_id="sess-B")
        self.assertEqual(
            self.mod._VIOLATION_TRACKER["sess-A"]["aml"]["violations"], 1
        )
        self.assertEqual(
            self.mod._VIOLATION_TRACKER["sess-B"]["aml"]["violations"], 1
        )

    # ── 3. 知识文件缓存 ─────────────────────────────────────────────

    def test_kb_file_cached_and_refreshed(self):
        kb = self.tmpdir / "kb.md"
        kb.write_text("版本1", encoding="utf-8")
        self._write_config([
            {
                "name": "aml",
                "match_keywords": ["AML"],
                "sop": "查库",
                "knowledge_files": [str(kb)],
            }
        ])
        r1 = self._call("AML 分析")
        self.assertIn("版本1", r1["context"])
        # 未修改 → 缓存命中
        r2 = self._call("AML 分析")
        self.assertIn("版本1", r2["context"])
        self.assertEqual(len(self.mod._KB_CACHE), 1)
        # 修改后 → mtime 变化 → 重新读取
        time.sleep(0.02)
        kb.write_text("版本2", encoding="utf-8")
        r3 = self._call("AML 分析")
        self.assertIn("版本2", r3["context"])

    def test_missing_kb_file_skipped(self):
        self._write_config([
            {
                "name": "aml",
                "match_keywords": ["AML"],
                "sop": "查库",
                "knowledge_files": [str(self.tmpdir / "不存在.md")],
            }
        ])
        result = self._call("AML 分析")
        self.assertIsNotNone(result)
        self.assertNotIn("不存在", result["context"])

    # ── 4. 有界会话追踪 ────────────────────────────────────────────

    def test_bounded_session_tracking(self):
        self.mod._MAX_TRACKED_SESSIONS = 3
        self._write_config([
            {"name": "aml", "match_keywords": ["AML"], "sop": "查库"}
        ])
        for i in range(5):
            self._call("AML", session_id=f"s{i}")
        self.assertLessEqual(len(self.mod._SESSION_SCENES), 3)
        # 最旧的被淘汰
        self.assertNotIn("s0", self.mod._SESSION_SCENES)
        self.assertIn("s4", self.mod._SESSION_SCENES)

    # ── 5. 并发安全 ────────────────────────────────────────────────

    def test_concurrent_violation_counting_caps_at_max_retries(self):
        """多线程并发违规：无计数丢失；每会话计数封顶在 max_retries"""
        self._write_config(self._aml_scene())
        import threading

        def worker(i):
            sid = f"conc-{i % 5}"
            hist = [
                {"role": "user", "content": "AML 分析"},
                {"role": "assistant", "content": "直接答"},
            ]
            self.mod.on_pre_llm_call(
                user_message="AML 分析", session_id=sid, conversation_history=hist
            )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        self.assertEqual(len(self.mod._VIOLATION_TRACKER), 5)
        for sid in [f"conc-{i}" for i in range(5)]:
            st = self.mod._VIOLATION_TRACKER[sid]["aml"]
            # 计数封顶在 max_retries（转人工后不再自增）
            self.assertEqual(st["violations"], 2)
            self.assertFalse(st["active"])


if __name__ == "__main__":
    unittest.main()
