"""守卫 R5a 误拦根修验证：ledger tuple 格式 vs 守卫 str 比对漂移。red/green 双验。"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "spg_under_test", REPO / "plugins/scientific-programming-guard/__init__.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["spg_under_test"] = m
    spec.loader.exec_module(m)
    return m


def test_tuple_read_history_hit():
    from tools.file_tools_read_tracking import _read_tracker
    _read_tracker.clear()
    guard = _load_guard()
    from tools.file_tools_read_tracking import _read_tracker
    target = REPO / "cron/scheduler.py"
    _read_tracker["t1"] = {"read_history": {(str(target), 1, 30)}}
    # 期望: ledger 事实格式 tuple(path,offset,limit) 修后命中(file_tools.py:557 原样)
    assert guard._session_has_read(target) is True


def test_plain_string_history_still_hits():
    from tools.file_tools_read_tracking import _read_tracker
    _read_tracker.clear()
    guard = _load_guard()
    from tools.file_tools_read_tracking import _read_tracker
    target = REPO / "cron/scheduler.py"
    _read_tracker["t2"] = {"read_history": {str(target)}}
    # 期望: 纯字符串记录仍命中(兼容旧形态)
    assert guard._session_has_read(target) is True


def test_no_history_miss():
    from tools.file_tools_read_tracking import _read_tracker
    _read_tracker.clear()
    guard = _load_guard()
    from tools.file_tools_read_tracking import _read_tracker
    target = REPO / "cron/scheduler.py"
    _read_tracker["t3"] = {"read_history": set()}
    # 期望: 无记录不命中(防线本职, fail-closed 对未读文件)
    assert guard._session_has_read(target) is False
