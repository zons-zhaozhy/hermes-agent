"""只读豁免对 shell 命令替换形态的覆盖（block_escalation 插件）。

现场（2026-11-02 03:21 deputy-patrol 杀班）：按 remote-multi-session-ops
SKILL.md 标准形态组合的验活探针含 ``ID=$(sqlite3 -readonly ...)``
赋值段——段首词是 ``ID=$(sqlite3``，不在只读白名单，7883500d91 的
只读豁免对该形态失效，error+blocked 各一次即达 _STREAK_LIMIT=2 升级
杀班。期望（独立于实现推导）：
- 命令替换赋值段（VAR=$(只读命令)）= 赋值本身无副作用，内部只读则段只读；
- 直接替换段（$(date)）同理；
- 解包裹不放松判定：VAR=$(写命令) 必须仍按写意图计数。
"""

import importlib.util
import pathlib
from typing import Any

_PLUGIN = pathlib.Path(__file__).resolve().parents[2] / "plugins" / "block_escalation" / "__init__.py"


def _be() -> Any:
    spec = importlib.util.spec_from_file_location("be_substitution_under_test", _PLUGIN)
    assert spec is not None and spec.loader is not None  # 期望: 插件可加载
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


be = _be()


def test_var_assignment_of_readonly_substitution_is_exempt() -> None:
    cmd = ("ID=$(sqlite3 -readonly ~/.hermes/state.db \"SELECT id FROM sessions "
           "WHERE id LIKE 'abc123%'\"); date '+%F %T'; echo \"=== A ===\" $ID")
    assert be._command_is_readonly(cmd)  # 期望: True — SKILL.md:18 标准形态,赋值包裹的只读命令应豁免


def test_bare_substitution_segment_is_exempt() -> None:
    cmd = "echo now=$(date '+%F %T') && stat -f %z ~/.hermes/state.db"
    assert be._command_is_readonly(cmd)  # 期望: True — $(date) 直接替换段首词非白名单但内部只读


def test_unwrap_does_not_relax_write_detection() -> None:
    cmd = "X=$(rm -rf /tmp/probe) && echo $X"
    assert not be._command_is_readonly(cmd)  # 期望: False — 解包裹后 rm 仍按写意图计数


def test_sed_write_semantics_is_not_readonly_exempt() -> None:
    assert be._command_is_readonly("sed -n '1p' state.md")  # 期望: quiet 打印只读
    assert not be._command_is_readonly("sed -i 's/a/b/' state.md")  # 期望: owner 判写，禁豁免


def test_chained_substitution_readonly_probe_is_exempt() -> None:
    cmd = ("date '+%F %T'; echo '=== A ==='; N=$(sqlite3 -readonly "
           "~/.hermes/state.db \"SELECT COUNT(*) FROM messages\"); echo rows=$N")
    assert be._command_is_readonly(cmd)  # 期望: True — 03:21 杀班探针同形态,全段只读
