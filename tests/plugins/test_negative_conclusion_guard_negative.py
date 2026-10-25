"""negative_conclusion_guard 负例契约测试——零命中探测积累必须触发权威入口提醒（违规必红）。

Contract:
    Preconditions: plugins/negative-conclusion-guard/__init__.py 存在
    Postconditions: 探测计数达阈值且无权威验证时 pre_llm_call 注入提醒
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "negative-conclusion-guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("neg_conclusion_guard_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_probe_accumulation_triggers_reminder():
    guard = _load_guard()
    sid = "neg-test-s1"
    for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
        guard.on_post_tool_call(
            tool_name="search_files", args={"pattern": "zzz_absent"},
            result={"total_count": 0}, session_id=sid,
        )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is not None and "权威" in out.get("context", "")  # 期望: 注入权威入口提醒（拦截裸断言）


def test_negative_authority_check_settles_reminder():
    guard = _load_guard()
    sid = "neg-test-s2"
    for _ in range(guard._MAX_PROBE_HITS_BEFORE_NUDGE):
        guard.on_post_tool_call(
            tool_name="search_files", args={"pattern": "zzz_absent"},
            result={"total_count": 0}, session_id=sid,
        )
    guard.on_post_tool_call(
        tool_name="terminal", args={"command": "hermes plugins list"},
        result={}, session_id=sid,
    )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is None  # 期望: None=权威验证后不再提醒


def test_negative_below_threshold_no_reminder():
    guard = _load_guard()
    sid = "neg-test-s3"
    guard.on_post_tool_call(
        tool_name="search_files", args={"pattern": "zzz"},
        result={"total_count": 0}, session_id=sid,
    )
    out = guard.on_pre_llm_call(session_id=sid)
    assert out is None  # 期望: None=未达阈值不提醒
