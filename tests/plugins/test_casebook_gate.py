"""casebook_gate 子护栏行为验证（红绿测试）。

规则来源（独立期望）：ontox-casebook skill 纪律 1「写代码前先查病例库」——
OntoX 路径写操作前，本会话必须存在一次 casebook 检索证据；拦截消息须给出
修复指引；库不可达时 fail-open（放行但有 warning）；非 OntoX 路径零打扰。

期望值全部从上述规则文本推导，不从实现反推。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import casebook_gate 


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    """隔离环境：假 OntoX 根 + 假病例库。"""
    ontox = tmp_path / "ontox"
    cb = ontox / "education" / "casebook"
    cb.mkdir(parents=True)
    (cb / "INDEX.md").write_text("# 病例索引\n", encoding="utf-8")
    monkeypatch.setattr(casebook_gate, "_ONTOX_ROOT", str(ontox))
    monkeypatch.setattr(casebook_gate, "_CASEBOOK_DIR", str(cb))
    yield ontox, cb


def _clear_probes(sid: str = "s1") -> None:
    st = casebook_gate._get_state(sid)
    st.pop("probes", None)


def test_write_to_ontox_without_probe_blocked(env: tuple[Path, Path]) -> None:
    ontox, cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="patch", args={"path": f"{ontox}/apps/aml/main.py"}, session_id="s1"
    )
    # 期望: 规则「未查病例库即拦」→ 返回 block 指令
    assert v is not None  # 期望: 规则要求未查病例库即拦——None=漏拦
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）
    assert "INDEX.md" in v["message"]  # 期望: 修复指引须含索引文件名（引导可执行）
    assert str(cb) in v["message"]  # 期望: 修复指引须含库绝对路径（可直接复制执行）


def test_probe_via_search_then_write_allowed(env: tuple[Path, Path]) -> None:
    ontox, cb = env
    _clear_probes()
    casebook_gate.on_post_tool_call(
        tool_name="search_files", args={"path": str(cb), "pattern": "吞异常"}, session_id="s1"
    )
    v = casebook_gate.on_pre_tool_call(
        tool_name="write_file", args={"path": f"{ontox}/tools/x.py"}, session_id="s1"
    )
    assert v is None  # 期望: 检索证据在先——规则满足即放行（None=放行）


def test_probe_via_read_index_allowed(env: tuple[Path, Path]) -> None:
    ontox, cb = env
    _clear_probes()
    casebook_gate.on_post_tool_call(
        tool_name="read_file", args={"path": f"{cb}/INDEX.md"}, session_id="s1"
    )
    v = casebook_gate.on_pre_tool_call(
        tool_name="patch", args={"path": f"{ontox}/apps/aml/api.py"}, session_id="s1"
    )
    assert v is None  # 期望: 读索引=查病例——规则满足即放行


def test_non_ontox_write_untouched(env: tuple[Path, Path]) -> None:
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="write_file", args={"path": "/tmp/notes/x.md"}, session_id="s1"
    )
    assert v is None  # 期望: 门只管 OntoX 仓——外部路径零打扰


def test_casebook_missing_fail_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _clear_probes()
    monkeypatch.setattr(casebook_gate, "_CASEBOOK_DIR", str(tmp_path / "nope"))
    v = casebook_gate.on_pre_tool_call(
        tool_name="patch", args={"path": "/anything/a.py"}, session_id="s1"
    )
    assert v is None  # 期望: 库不可达→fail-open 放行（禁死锁，日志层留痕）


def test_terminal_write_to_ontox_blocked(env: tuple[Path, Path]) -> None:
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"sed -i 's/a/b/' {ontox}/apps/aml/main.py"},
        session_id="s1",
    )
    # 期望: terminal 绕道写 OntoX 同样受门禁（防旁路）
    assert v is not None  # 期望: 规则要求未查即拦——None=旁路漏洞
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）


def test_disabled_env_short_circuits(
    env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ontox, _cb = env
    _clear_probes()
    monkeypatch.setenv("CASEBOOK_GATE_DISABLE", "1")
    v = casebook_gate.on_pre_tool_call(
        tool_name="patch", args={"path": f"{ontox}/a.py"}, session_id="s1"
    )
    assert v is None  # 期望: 显式关闭开关→一切放行（逃生门）
