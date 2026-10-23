"""guards.source_write 的 in-place 层误报回归测试。

实测缺陷（2026-10-11）：`git commit -m "... sed -i ..." -- tests/x.py` 被拦成
「in-place 编辑源码文件 (tests/x.py)」——模式出现在引号内的消息文本里，却被当成
真的 sed 命令，且 in-place 目标被错误归属到命令行里另一处的源码路径。
根因：`_RE_INPLACE_EDIT` 直接在整条命令文本上匹配。修复：只在「会被 shell 执行」
的文本上匹配（`_executable_text`：引号外 + shell 求值标志后的引号段）。
"""
from __future__ import annotations

from plugins.guards import source_write as sw

_SRC = "/Users/stan/code/apps/foo/services.py"


def test_hermes_cache_tmpdir_is_safe_path():
    import os
    scratch = os.path.expanduser("~/.hermes/cache/scratch/pr_payload.json")
    assert sw._is_safe_path(scratch) is True  # 期望: True——TMPDIR 指向的官方临时区,与 /tmp 同语义
    assert sw._is_safe_path(_SRC) is False    # 期望: False——仓库内源码路径仍拦


def test_commit_message_mentioning_inplace_pattern_allowed():
    """提交信息里提到 in-place 字样（数据非命令）→ 放行。"""
    cmd = 'git commit -m "只覆盖 open(...,\'w\')/重定向/sed -i 三类直写形态" -- tests/plugins/test_scope_gate.py'
    blocked, reason = sw._detect_source_write(cmd)
    assert blocked is False  # 期望: 模式在引号内=数据，不构成 in-place 命令
    assert reason == ""  # 期望: 未命中任何层


def test_grep_pattern_allowed():
    """grep 模式串里含 in-place 字样 → 放行。"""
    blocked, _ = sw._detect_source_write('grep -n "sed -i" plugins/guards/source_write.py')
    assert blocked is False  # 期望: 引号内的模式串是数据


def test_echo_with_safe_redirect_allowed():
    """echo 该字样 + 重定向到 /tmp → 放行。"""
    blocked, _ = sw._detect_source_write('echo "以后别用 sed -i 批量改" > /tmp/probe_sw_note.txt')
    assert blocked is False  # 期望: 引号内是数据，且重定向目标是安全路径


def test_real_inplace_on_source_blocked():
    """真 sed in-place 改源码 → 拦截。"""
    blocked, reason = sw._detect_source_write(f"sed -i 's/a/b/' {_SRC}")
    assert blocked is True  # 期望: sed 是命令名，目标为源码文件
    assert "in-place" in reason  # 期望: 命中 in-place 层


def test_real_inplace_on_tmp_allowed():
    """sed in-place 目标在 /tmp → 放行。"""
    blocked, _ = sw._detect_source_write("sed -i 's/a/b/' /tmp/probe_sw/x.py")
    assert blocked is False  # 期望: 安全路径放行


def test_find_exec_inplace_blocked():
    """find -exec 形态 → 仍拦截。"""
    blocked, _ = sw._detect_source_write("find . -name '*.py' -exec sed -i 's/a/b/' {} +")
    assert blocked is True  # 期望: -exec 后是命令，无明确目标时保守拦截


def test_shell_c_inplace_blocked():
    """bash -c 引号段内是命令 → 仍拦截。"""
    cmd = f"bash -c \"sed -i 's/a/b/' {_SRC}\""
    blocked, reason = sw._detect_source_write(cmd)
    assert blocked is True  # 期望: -c 后的引号段会被 shell 执行
    assert "in-place" in reason  # 期望: 命中 in-place 层
