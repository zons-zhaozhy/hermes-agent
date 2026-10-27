"""tool_classification 单一 owner + AST 门禁的负例覆盖契约测试。

guard 资产=guards/tool_classification（C2 收敛产物，2026-10-27）。
配套门禁=scripts/check_tool_classification.py。负例=注入漂移字面量
时门禁必须报违规（违规必红），豁免标记必须放行，owner 分层恒等。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from check_tool_classification import check_file

from plugins.guards.tool_classification import (
    CODE_WRITE_TOOLS,
    FILE_AND_SKILL_WRITE_TOOLS,
    FILE_WRITE_TOOLS,
    MAYBE_WRITE_TOOLS,
    SKILL_WRITE_TOOLS,
    STANDARD_WRITE_TOOLS,
)


def _probe(tmp_path: Path, body: str) -> list[str]:
    """写探针文件并返回门禁违规清单。"""
    f = tmp_path / "_probe.py"
    f.write_text(body, encoding="utf-8")
    return check_file(f)


def test_gate_blocked_on_drift_literal(tmp_path: Path) -> None:
    """负例：非 owner 模块持有 ≥2 写工具名字面量必须被门禁拦下。"""
    violations = _probe(
        tmp_path,
        "_LOCAL = frozenset({'write_file', 'patch', 'execute_code'})\n",
    )
    assert violations and "write_file" in violations[0]  # 期望: 漂移字面量被拦
    assert "tool_classification" in violations[0]  # 期望: 违规信息指向 owner


def test_gate_blocked_on_set_literal_drift(tmp_path: Path) -> None:
    """负例：set 字面量形态的写工具名集合同样被拦（禁换容器逃逸）。"""
    violations = _probe(tmp_path, "_LOCAL = {'patch', 'skill_manage'}\n")
    assert violations  # 期望: set 字面量形态同样被拦


def test_gate_passes_clean_and_exempt(tmp_path: Path) -> None:
    """正例：无写字面量放行；# tools-ok: 豁免放行。"""
    clean = _probe(tmp_path, "_LOCAL = frozenset({'read_file'})\n")
    assert not clean  # 期望: 只读集合放行
    exempt = _probe(
        tmp_path,
        "_LOCAL = frozenset({'write_file', 'patch'})  # tools-ok: 语义豁免\n",
    )
    assert not exempt  # 期望: 豁免标记放行


def test_owner_layers_semantics() -> None:
    """owner 分层语义断言：三基集与组合层恒等关系。"""
    assert FILE_WRITE_TOOLS == frozenset({"write_file", "patch"})  # 期望: 文件写基集
    assert CODE_WRITE_TOOLS == frozenset({"execute_code"})  # 期望: 代码写基集
    assert SKILL_WRITE_TOOLS == frozenset({"skill_manage"})  # 期望: 技能写基集
    assert STANDARD_WRITE_TOOLS == FILE_WRITE_TOOLS | CODE_WRITE_TOOLS  # 期望: 通用层
    assert (
        FILE_AND_SKILL_WRITE_TOOLS == FILE_WRITE_TOOLS | SKILL_WRITE_TOOLS
    )  # 期望: 文件+技能层
    assert MAYBE_WRITE_TOOLS == STANDARD_WRITE_TOOLS | {"terminal"}  # 期望: 静态可能写层
    assert "terminal" not in STANDARD_WRITE_TOOLS  # 期望: terminal 不入拦截面
