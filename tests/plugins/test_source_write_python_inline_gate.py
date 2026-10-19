"""source_write Layer 1（python 内联写）误拦回归——红绿测试。

规则来源（独立期望，来自 plugin 头部契约）：
- Layer 1 拦截「python 内联脚本真实写源码文件」；
- 只读命令 / 非源码目标（/tmp 等安全路径）恒放行；
- 字符串数据里的写形态字样（测试载荷、echo 参数）不是写动作。

2026-10-19 实锤误拦形态：守卫验证命令（外层 python heredoc，载荷内
bypass = "python3 -c \\"open('x.py','w')\\"" 是字符串数据）被整条拦截——
旧实现全文子串匹配 python3 -c / open(...,'w')，无引号与角色感知。
期望值全部从契约文本推导，不从实现反推。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import source_write as sw


def _v(command: str) -> Optional[dict]:
    return sw.on_pre_tool_call(tool_name="terminal", args={"command": command})


def test_heredoc_payload_with_write_shaped_string_data_allowed() -> None:
    """外层 python heredoc 载荷把写形态当字符串数据赋值 → 非写动作 → 放行。"""
    cmd = (
        "venv/bin/python - <<'PYEOF2'\n"
        "bypass = \"python3 -c \\\"open('/Users/x/repo/apps/x.py','w')\\\"\"\n"
        "print('BLOCKED' if bypass else 'ALLOWED')\n"
        "PYEOF2"
    )
    assert _v(cmd) is None  # 期望: open(...) 位于字符串字面量内=数据非写动作


def test_shell_string_data_with_python_c_mention_allowed() -> None:
    """非 python 头命令（echo）的引号串里出现 python3 -c 写形态字样 → 放行。"""
    cmd = "echo 'run python3 -c \"open(\\'x.py\\',\\'w\\')\" sometime'"
    assert _v(cmd) is None  # 期望: 引号内数据非命令文本→Layer 1 零命中


def test_inline_c_write_source_blocked() -> None:
    """python3 -c 真写源码文件 → 拦截（防旁路语义不回退）。"""
    cmd = "python3 -c \"open('/Users/x/repo/apps/x.py','w').write('y')\""
    v = _v(cmd)
    assert v is not None  # 期望: 真内联写源码→block
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block


def test_heredoc_real_write_source_blocked() -> None:
    """python heredoc 真写源码文件（with open 形态）→ 拦截。"""
    cmd = (
        "python3 <<'EOF'\n"
        "with open('/Users/x/repo/a.py','w') as f:\n"
        "    f.write('y')\n"
        "EOF"
    )
    v = _v(cmd)
    assert v is not None  # 期望: heredoc 真写→block
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block


def test_inline_c_write_tmp_allowed() -> None:
    """python3 -c 写 /tmp（安全路径）→ 放行。"""
    cmd = "python3 -c \"open('/tmp/out.txt','w').write('y')\""
    assert _v(cmd) is None  # 期望: 安全路径写非门禁对象


def test_inline_c_dynamic_path_write_conservatively_blocked() -> None:
    """open(p,'w') 动态路径 → 保守拦截（旧保守语义保持，防绕过）。"""
    cmd = "python3 -c \"p='a.py'; open(p,'w')\""
    v = _v(cmd)
    assert v is not None  # 期望: 写形态存在但路径非字面量→保守 block
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block
