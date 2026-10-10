"""guards.batch_write 行为契约测试。

红线（2026-10-10 违反实录的机器化复现）：execute_code 循环内拼接路径批量
改 26 个 profile SOUL.md——本套件断言该形态被 block；同时断言四类放行边界
（单文件/新建文件/动态路径/临时目录）不误拦。
"""
import pytest

from plugins.guards import batch_write


@pytest.fixture
def no_exempt(monkeypatch):
    """block 用例前置：清空豁免表——pytest tmp_path 落在 macOS 系统临时目录
    （/private/var/tmp），属生产豁免区；block 语义测试须在非豁免前提下构造。"""
    monkeypatch.setattr(batch_write, "_EXEMPT_PREFIXES", ())


@pytest.fixture
def two_existing_files(tmp_path: pytest.TempPathFactory) -> tuple[str, str]:
    """两个磁盘上真实存在的文件（block 判定的最小充分条件）。"""
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_text("old-a", encoding="utf-8")
    b.write_text("old-b", encoding="utf-8")
    return str(a), str(b)


def _run(code: str, session_id: str = "s-test") -> dict | None:
    return batch_write._on_pre_tool_call(
        tool_name="execute_code", args={"code": code}, session_id=session_id,
    )


def test_batch_literal_paths_blocked(no_exempt, two_existing_files):
    """两条字面量 write_text 已有文件 → block。"""
    a, b = two_existing_files
    code = f"""
p1 = "{a}"
p2 = "{b}"
open(p1, "w").write("x")
open(p2, "w").write("y")
"""
    verdict = _run(code)
    assert verdict is not None and verdict["action"] == "block"  # 期望: 2 个已有文件=批量
    assert "patch" in verdict["message"]


def test_batch_loop_concat_blocked(no_exempt, two_existing_files, tmp_path):
    """循环内 f-string 拼接路径批量写（2026-10-10 事故形态）→ block。"""
    code = f"""
from pathlib import Path
base = "{tmp_path}"
for name in ("a.md", "b.md"):
    p = Path(base) / name
    p.write_text("x", encoding="utf-8")
"""
    verdict = _run(code)
    assert verdict is not None and verdict["action"] == "block"  # 期望: 拼接可解析=2 个已有文件


def test_binop_concat_blocked(no_exempt, two_existing_files):
    """字符串加法拼接路径 → block。"""
    a, b = two_existing_files
    code = f"""
base_a, base_b = "{a}", "{b}"
open(base_a + "", "w").write("x")
open(base_b + "", "w").write("y")
"""
    verdict = _run(code)
    assert verdict is not None and verdict["action"] == "block"  # 期望: 加法拼接两侧可解析


def test_single_existing_file_allowed(two_existing_files):
    """只写 1 个已有文件 → 放行（非批量语义）。"""
    a, _ = two_existing_files
    verdict = _run(f'open("{a}", "w").write("x")\n')
    assert verdict is None  # 期望: 1 < 阈值 2


def test_new_files_allowed(tmp_path):
    """目标均为不存在的新建文件 → 放行（无改前状态可丢）。"""
    new1, new2 = tmp_path / "n1.md", tmp_path / "n2.md"
    code = f"""
open("{new1}", "w").write("x")
open("{new2}", "w").write("y")
"""
    verdict = _run(code)
    assert verdict is None  # 期望: os.path.exists=False 不计数


def test_dynamic_path_allowed(two_existing_files):
    """路径含函数返回值等动态形态 → 放行（不猜）。"""
    a, b = two_existing_files
    code = f"""
def pick(i):
    return "{a}" if i == 0 else "{b}"
for i in range(2):
    open(pick(i), "w").write("x")
"""
    verdict = _run(code)
    assert verdict is None  # 期望: 动态路径不可静态解析=未证明批量


def test_exempt_dirs_allowed(two_existing_files):
    """写目标在 /tmp（豁免目录）→ 放行。"""
    code = """
open("/tmp/audit1.diff", "w").write("x")
open("/tmp/audit2.diff", "w").write("y")
"""
    verdict = _run(code)
    assert verdict is None  # 期望: /tmp 命中豁免前缀不计数


def test_cron_session_exempt(two_existing_files):
    """cron 会话豁免（与 diff_debt 同判据）。"""
    a, b = two_existing_files
    code = f'open("{a}", "w").write("x")\nopen("{b}", "w").write("y")\n'
    assert _run(code, session_id="cron_x") is None  # 期望: cron_ 前缀放行


def test_non_execute_code_ignored(two_existing_files):
    """非 execute_code 工具不进射程。"""
    a, b = two_existing_files
    verdict = batch_write._on_pre_tool_call(
        tool_name="terminal", args={"command": f"echo {a} {b}"}, session_id="s",
    )
    assert verdict is None  # 期望: 射程仅 execute_code


def test_syntax_error_fail_open():
    """语法错误源码放行（fail-open），不抛异常。"""
    assert _run("def broken(:\n  pass") is None  # 期望: 解析失败=放行不抛
