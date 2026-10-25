"""block_escalation 负例契约测试——同一写意图连续被拦必须升级终止提示（违规必红）。

Contract:
    Preconditions: plugins/block_escalation/__init__.py 存在
    Postconditions: 同指纹第 2 次被拦后 transform 输出带 ESCALATION_MARKER
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard(tmp_db: Path) -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "block_escalation" / "__init__.py"
    spec = importlib.util.spec_from_file_location("block_escalation_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._db_path = tmp_db  # 隔离持久化库到测试沙盒
    return mod


def test_negative_second_block_escalates(tmp_path: Path):
    guard = _load_guard(tmp_path / "be1.db")
    guard._escalated.clear()
    args = {"path": "/tmp/zz_target.py"}
    guard._on_post_tool_call(tool_name="write_file", status="blocked",
                             args=args, session_id="be-s1")
    guard._on_post_tool_call(tool_name="patch", status="blocked",
                             args=args, session_id="be-s1")
    out = guard._on_transform_llm_output(response_text="我换个通道再试", session_id="be-s1")
    assert guard.ESCALATION_MARKER in out  # 期望: 第 2 次同因被拦后输出终止提示（拦截换通道）


def test_negative_readonly_block_not_escalated(tmp_path: Path):
    guard = _load_guard(tmp_path / "be2.db")
    guard._escalated.clear()
    args = {"command": "ls -la /somewhere"}
    guard._on_post_tool_call(tool_name="terminal", status="blocked",
                             args=args, session_id="be-s2")
    guard._on_post_tool_call(tool_name="terminal", status="blocked",
                             args=args, session_id="be-s2")
    out = guard._on_transform_llm_output(response_text="继续", session_id="be-s2")
    assert guard.ESCALATION_MARKER not in out  # 期望: 只读命令豁免不升级


def test_negative_single_block_no_escalation(tmp_path: Path):
    guard = _load_guard(tmp_path / "be3.db")
    guard._escalated.clear()
    guard._on_post_tool_call(tool_name="write_file", status="blocked",
                             args={"path": "/tmp/one.py"}, session_id="be-s3")
    out = guard._on_transform_llm_output(response_text="重试一次", session_id="be-s3")
    assert guard.ESCALATION_MARKER not in out  # 期望: 单次被拦不升级


def test_negative_success_clears_streak(tmp_path: Path):
    guard = _load_guard(tmp_path / "be4.db")
    guard._escalated.clear()
    args = {"path": "/tmp/clear.py"}
    guard._on_post_tool_call(tool_name="write_file", status="blocked",
                             args=args, session_id="be-s4")
    guard._on_post_tool_call(tool_name="write_file", status="ok",
                             args=args, session_id="be-s4")
    guard._on_post_tool_call(tool_name="write_file", status="blocked",
                             args=args, session_id="be-s4")
    out = guard._on_transform_llm_output(response_text="再次尝试", session_id="be-s4")
    assert guard.ESCALATION_MARKER not in out  # 期望: 成功落地清账后重新计数，不升级
