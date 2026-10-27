#!/usr/bin/env python3
"""工具写语义单一 owner 门禁（C2 收敛执法，AST 结构化解析）。

规则：plugins/ agent/ tools/ acp_adapter/ 下任何非 owner 模块出现
「frozenset/set 字面量同时含 ≥2 个写工具名」的赋值即 FAIL——写工具集
语义唯一 owner = plugins/guards/tool_classification.py，消费方必须
import 常量，禁止持有工具名字面量集合。

豁免标记：行内 `# tools-ok: <理由>` 豁免该赋值（语义确属别类：如
_PATH_SCOPED_WRITERS 路径域语义、_CODIFICATION_TOOLS 沉淀语义）。

用法：
  python3 scripts/check_tool_classification.py             # 检查
  python3 scripts/check_tool_classification.py --self-test # 负例注入自测
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER = "plugins/guards/tool_classification.py"
SCAN_ROOTS = ("plugins", "agent", "tools", "acp_adapter")
WRITE_TOOL_NAMES = frozenset(
    {"write_file", "patch", "execute_code", "skill_manage"}
)


def _iter_target_names(node: ast.AST) -> list[str]:
    """提取赋值目标的简单名字（支持 Tuple/List 解包）。"""
    names: list[str] = []
    if isinstance(node, ast.Name):
        names.append(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.extend(_iter_target_names(elt))
    return names


def _collection_string_items(node: ast.AST) -> list[str]:
    """提取 frozenset({...})/set/list/tuple 字面量中的全部字符串元素。"""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set")
    ):
        if len(node.args) != 1:
            return []
        return _collection_string_items(node.args[0])
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return [
            e.value for e in node.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
    return []


def _assign_target_label(node: ast.stmt) -> str:
    """赋值语句的目标变量名标签（违规信息用）。"""
    if isinstance(node, ast.Assign) and node.targets:
        return ",".join(_iter_target_names(node.targets[0]))
    if isinstance(node, ast.AnnAssign):
        return ",".join(_iter_target_names(node.target))
    return "?"


def _is_exempted(node: ast.stmt, source_lines: list[str]) -> bool:
    """该语句行范围内是否带豁免标记（注释不进 AST，按源码行扫描）。"""
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    for i in range(node.lineno, min(end, len(source_lines)) + 1):
        if "# tools-ok:" in source_lines[i - 1]:
            return True
    return False


def _violations_in(tree: ast.Module, rel: str, source: str) -> list[str]:
    """单棵 AST 内的写工具名字面量违规清单。"""
    source_lines = source.splitlines()
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if node.value is None:
            continue
        items = _collection_string_items(node.value)
        overlap = WRITE_TOOL_NAMES.intersection(items)
        if len(overlap) >= 2 and not _is_exempted(node, source_lines):
            out.append(
                f"{rel}:{node.lineno}: 「{_assign_target_label(node)}」"
                f"持有写工具名字面量 ({sorted(overlap)})——语义 owner="
                f"{OWNER}，import 常量或加行内豁免 # tools-ok: <理由>"
            )
    return out


def _display_rel(path: Path) -> str:
    """仓库内文件用相对路径标识，仓库外（如测试探针）用绝对路径。"""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def check_file(path: Path) -> list[str]:
    """单个文件扫描；解析失败返回空（语法门禁另管，不重复执法）。"""
    rel = _display_rel(path)
    if rel == OWNER:
        return []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    return _violations_in(tree, rel, source)


def run_scan() -> int:
    """全库扫描；rc=0 通过 / rc=1 违规。"""
    violations: list[str] = []
    for root in SCAN_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            violations.extend(check_file(path))
    if violations:
        print(f"FAIL 工具写语义漂移（唯一 owner={OWNER}）：")
        for v in violations:
            print("  " + v)
        return 1
    print(f"OK 工具写语义零漂移（owner={OWNER}）")
    return 0


def self_test() -> int:
    """负例自测：注入漂移必须被抓住，豁免标记必须生效，树必须还原。"""
    probe = REPO_ROOT / "plugins" / "_drift_probe_tmp.py"
    probe.write_text(
        "_DRIFT = frozenset({'write_file', 'patch', 'execute_code'})\n",
        encoding="utf-8",
    )
    try:
        caught = check_file(probe)
        assert any("持有写工具名字面量" in c for c in caught), (
            f"负例未被门禁抓住——门禁失效: {caught!r}"  # 期望: 抓住注入漂移
        )
        exempt = REPO_ROOT / "plugins" / "_exempt_probe_tmp.py"
        exempt.write_text(
            "_OK = frozenset({'write_file', 'patch'})  # tools-ok: 自测豁免\n",
            encoding="utf-8",
        )
        try:
            clean = check_file(exempt)
            assert not clean, f"豁免标记失效: {clean!r}"  # 期望: 豁免行放行
        finally:
            exempt.unlink(missing_ok=True)
    finally:
        probe.unlink(missing_ok=True)
    print("OK self-test：注入漂移被抓 + 豁免标记生效 + 树还原干净")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    return run_scan()


if __name__ == "__main__":
    sys.exit(main())
