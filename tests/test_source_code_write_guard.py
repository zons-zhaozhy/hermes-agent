"""source-code-write-guard escape-hatch regression tests.

2026-08-26 漏洞回归：旧 _is_escape_hatch 只查命令包含关键词——
`ls plugins/` 出现在任意位置即整条命令放行，重定向写无关源码文件也被放走。
修复后：仅当所有可检测写入目标都是护栏自身文件才放行。
"""
import importlib.util
from pathlib import Path

_PLUG = Path(__file__).resolve().parent.parent / "plugins" / "source_code_write_guard" / "__init__.py"


def _load():
    spec = importlib.util.spec_from_file_location("scg_under_test", _PLUG)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_escape_hatch_no_longer_leaks_unrelated_writes():
    """漏洞场景:命令含 'plugins/' 字样但重定向写无关源码 → 必须拦截。"""
    m = _load()
    r = m.on_pre_tool_call(tool_name="terminal", args={
        "command": "cd ~/code/ai/cnb/ontox && echo x > /Users/stan/code/ai/cnb/ontox/oms/backend/src/Foo.java && ls plugins/"
    })
    assert r is not None and r.get("action") == "block"


def test_plain_source_write_still_blocked():
    """对照:无任何关键词的源码重定向写 → 拦截。"""
    m = _load()
    r = m.on_pre_tool_call(tool_name="terminal", args={
        "command": "echo x > /Users/stan/code/ai/cnb/ontox/oms/backend/src/Foo.java"
    })
    assert r is not None and r.get("action") == "block"


def test_guard_owned_write_still_allowed():
    """护栏自指:heredoc 写 plugins/ 下文件 → 放行(修护栏的合法通道)。"""
    m = _load()
    r = m.on_pre_tool_call(tool_name="terminal", args={
        "command": "cat > plugins/no-guessing/fix.py <<'EOF'\nx = 1\nEOF"
    })
    assert r is None


def test_readonly_command_with_keyword_allowed():
    """只读命令含 plugins/ → 放行(只读无写入目标,退回关键词匹配)。"""
    m = _load()
    r = m.on_pre_tool_call(tool_name="terminal", args={"command": "ls plugins/ && cat /etc/hosts"})
    assert r is None


def test_mixed_targets_blocked():
    """混合目标:一条命令同时写护栏文件与无关源码 → 拦截(部分目标非护栏)。"""
    m = _load()
    r = m.on_pre_tool_call(tool_name="terminal", args={
        "command": "echo a > plugins/g.py && echo b > /Users/x/src/Main.java"
    })
    assert r is not None and r.get("action") == "block"
