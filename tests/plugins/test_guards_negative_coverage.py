"""guard 负例覆盖契约——每个 guard 资产必须配套「违规必红」的负例测试。

机制移植自 OntoX check_gate_self_test（门禁自证覆盖率棘轮）：guard 本身可能永远
返回放行（装饰性防线），其有效性从未被机器验证。本契约把「有多少 guard 缺负例」
变成可对账的数字，并用基线棘轮阻止新增无负例 guard。

判定两级：
  - 配套测试存在：tests/plugins/test_*.py 文件名含 guard 资产名（连字符归一为下划线）
  - 负例存在：配套测试 AST 中存在负例用例——测试函数名含负例语义词，或 assert
    语句的字符串常量含拦截语义（block/拦/拦截）

存量缺口记录在 NEGATIVE_COVERAGE_BASELINE（只准降：缺口消除后须从基线删除条目，
基线外新增缺口直接 FAIL——新 guard 必须带负例落地）。
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUARDS_DIR = REPO / "plugins" / "guards"
PLUGINS_DIR = REPO / "plugins"
TESTS_DIR = Path(__file__).resolve().parent

# assert 字符串常量中的拦截语义证据（block=硬拦, hint=软拦 R5, 拦/拦截=中文形态）
BLOCK_WORDS = frozenset({"block", "blocked", "hint", "拦", "拦截"})
# 测试函数名中的负例语义证据
NEGATIVE_NAME_WORDS = ("blocked", "escape", "negative", "defer_pushout", "fail_open", "fail_closed")

# 存量缺口基线（棘轮只准降）。新 guard 落地必须自带负例测试，不进此清单。
# 第一轮实测（2026-10-25）：33 项 guard 资产中 21 项缺负例覆盖。
# 真缺口=无配套测试（read_think_gate/scientific-programming-guard/yinyang-restate-guard 等）；
# 形态盲区=测试存在但断言为计数型/形态型（block_escalation 的 streak 计数、
# skill_diff_echo 的回显断言），判定器不识别——两类都记录在此，逐项清偿时删除。
NEGATIVE_COVERAGE_BASELINE: frozenset[str] = frozenset({
    "block_escalation",
    "dao-recurring-error-guard",
    "declare_act_guard",
    "deep-test-gate",
    "discipline",
    "engine-invariants",
    "failure_preflight",
    "guard_sentinel",
    "guards/coding_standards",
    "guards/duplicate_check",
    "guards/four_axis",
    "guards/preflight",
    "guards/rules_selfcheck",
    "guards/tool_blacklist",
    "jspace-guard",
    "negative-conclusion-guard",
    "pipeline-guard",
    "read_think_gate",
    "scientific-programming-guard",
    "skill_diff_echo",
    "yinyang-restate-guard",
})


def _guard_submodule_assets() -> list[str]:
    """枚举 guards 插件子模块资产名。

    Contract:
        Preconditions: plugins/guards/ 目录存在
        Postconditions: 返回非 __ 开头的 .py 文件名集合（含 guards/ 前缀）
    """
    return [f"guards/{f.stem}" for f in sorted(GUARDS_DIR.glob("*.py")) if not f.name.startswith("__")]


def _guard_plugin_assets() -> list[str]:
    """枚举独立 guard 类插件目录名（plugin.yaml 的 tags 含 guard 或目录名含 guard）。

    Contract:
        Preconditions: plugins/ 目录存在
        Postconditions: 返回目录名列表
    """
    names: list[str] = []
    for pd in sorted(PLUGINS_DIR.iterdir()):
        if not pd.is_dir() or pd.name.startswith("__"):
            continue
        manifest = pd / "plugin.yaml"
        is_guard = (manifest.exists() and "guard" in manifest.read_text(encoding="utf-8")) or "guard" in pd.name
        if is_guard:
            names.append(pd.name)
    return names


def _matching_tests(asset: str) -> list[Path]:
    """找配套测试文件：文件名含资产名（连字符归一为下划线）。

    Contract:
        Preconditions: asset 形如 guards/casebook_gate 或 finish_guard
        Postconditions: 返回匹配的 test_*.py 列表（可为空）
    """
    stem = asset.split("/")[-1].replace("-", "_")
    return [t for t in TESTS_DIR.glob("test_*.py") if stem in t.name]


def _count_block_asserts(func: ast.FunctionDef) -> int:
    """统计函数体内 assert 语句含拦截语义字符串常量的数量。

    Contract:
        Preconditions: func 为测试函数 AST 节点
        Postconditions: 返回 [0, ∞)
    """
    hits = 0
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Assert):
            continue
        consts = {
            c.value
            for c in ast.walk(sub)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        if consts & BLOCK_WORDS:
            hits += 1
    return hits


def _negative_case_count(test_path: Path) -> int:
    """统计测试文件中的负例用例数。

    Contract:
        Preconditions: test_path 为 Python 测试文件
        Postconditions: 返回 [0, ∞)；解析失败返回 -1（显式暴露语法损坏，不静默）
    """
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return -1
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if any(w in node.name for w in NEGATIVE_NAME_WORDS):
            total += 1
            continue
        total += _count_block_asserts(node)
    return total


def _coverage_gaps() -> list[str]:
    """测出全部缺口资产（无配套测试或配套测试零负例）。

    Contract:
        Preconditions: guard 资产枚举可用
        Postconditions: 返回缺口资产名列表；配套测试语法损坏按缺口计
    """
    gaps: list[str] = []
    for asset in _guard_submodule_assets() + _guard_plugin_assets():
        matches = _matching_tests(asset)
        if not matches:
            gaps.append(asset)
            continue
        negatives = sum(_negative_case_count(t) for t in matches)
        if negatives <= 0:
            gaps.append(asset)
    return sorted(gaps)


def test_every_guard_has_negative_case_or_is_baselined() -> None:
    """主契约：缺口必须在基线内（棘轮——基线外新增缺口即 FAIL）。"""
    gaps = _coverage_gaps()
    new_gaps = [g for g in gaps if g not in NEGATIVE_COVERAGE_BASELINE]
    # 期望: 新缺口为零——规则要求新 guard 落地必须配套「违规必红」负例测试
    assert not new_gaps, (
        f"基线外新增无负例 guard: {new_gaps}——每个 guard 必须配套「违规必红」的负例测试；"
        f"补测试或确认存量后进基线（只准降）"
    )
    stale = [b for b in NEGATIVE_COVERAGE_BASELINE if b not in gaps]
    # 期望: 已消除缺口为零——负例补齐后基线必须收缩，棘轮只准降
    assert not stale, f"基线含已消除缺口（负例已补，应从 NEGATIVE_COVERAGE_BASELINE 删除）: {stale}"


def test_negative_counter_catches_absence(tmp_path: Path) -> None:
    """本契约的负例自测：判定器对零负例文件必须返回 0（违规必红的可红性证明）。"""
    clean = tmp_path / "test_fake_guard.py"
    clean.write_text(
        "def test_allowed():\n    assert True\n", encoding="utf-8"
    )
    # 期望: 0——纯放行断言不含拦截语义，判定器必须能识别「没有负例」
    assert _negative_case_count(clean) == 0  # 期望: 0=无拦截断言无负例函数名
    dirty = tmp_path / "test_fake_guard_blocked.py"
    dirty.write_text(
        'def test_write_blocked():\n    v = {"action": "block"}\n    assert v["action"] == "block"\n',
        encoding="utf-8",
    )
    # 期望: >=1——函数名 blocked 与 assert 常量 block 双证据都应命中
    assert _negative_case_count(dirty) >= 1  # 期望: 1=函数名+断言常量双命中
