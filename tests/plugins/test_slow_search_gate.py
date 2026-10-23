"""slow_search 子护栏判据契约测试。

Regression: 委派转录实测(2026-10-24) 4/25 单反方子代理各白等满 420s 工具超时，
均为无保护递归 grep；本护栏拦该形态，豁免通道=--exclude-dir / timeout 包裹 /
# slowgrep-ok 行内注释。
"""

from plugins.guards.slow_search import detect_doomed_grep, on_pre_tool_call

# 正例：应拦截（含三个实测事故形态）
SHOULD_BLOCK = [
    # 事故形态1: 目标 `.` 无 exclude-dir 无 timeout（deleg_3451ac23, 420s 超时）
    "grep -rnE --include='*' -l -E 'auth-backend|oms-backend' .",
    # 事故形态3: 具体目录列表但无 exclude-dir（deleg_53fb779d, 420s 超时）
    "grep -rn -E '交付工作台|诊断助手' /Users/x/ontox/tools /Users/x/ontox/tests",
    # 事故形态4: pipe 后半段过滤 node_modules 但前段仍全仓扫（deleg_d082d0b5, 420s 超时）
    'grep -rn "data-empty-region" --include=*.ts . 2>/dev/null | grep -v node_modules',
    # cwd 递归（无路径参数）
    "grep -rn pattern",
    # 组合命令中一段命中即拦
    "ls -la && grep -rn foo .",
]

# 负例：应放行（豁免通道与替代命令形态）
SHOULD_ALLOW = [
    # 豁免1: timeout 包裹（段头命令名=timeout 非 grep）
    "timeout 60 grep -rn pattern .",
    # 豁免2: --exclude-dir
    "grep -rn --exclude-dir=node_modules pattern apps/",
    # 豁免3: 行内注释豁免
    "grep -rn pattern . # slowgrep-ok: 目标目录仅 3 个文件",
    # 替代1: git grep（段头=git）
    "git grep -n pattern",
    # 替代2: rg
    "rg pattern .",
    # 非递归 grep
    "grep -n pattern file.py",
    # 临时目录目标
    "grep -rn pattern /tmp/probe/",
]


def test_blocks_doomed_forms() -> None:
    for cmd in SHOULD_BLOCK:
        hit = detect_doomed_grep(cmd)
        assert hit is not None, f"应拦截: {cmd}"  # 期望: 全部命中


def test_allows_protected_forms() -> None:
    for cmd in SHOULD_ALLOW:
        assert detect_doomed_grep(cmd) is None, f"应放行: {cmd}"  # 期望: 全部放行


def test_hook_only_for_terminal() -> None:
    blocked = on_pre_tool_call(tool_name="terminal", args={"command": "grep -rn foo ."})
    assert blocked is not None and blocked["action"] == "block"  # 期望: terminal 命中被拦
    assert on_pre_tool_call(tool_name="execute_code", args={"code": "grep -rn foo ."}) is None  # 期望: 非terminal工具放行
    assert on_pre_tool_call(tool_name="terminal", args={}) is None  # 期望: 无命令放行
