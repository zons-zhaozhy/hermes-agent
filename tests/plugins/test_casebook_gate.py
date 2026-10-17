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


def test_readonly_sql_select_not_blocked(env: tuple[Path, Path]) -> None:
    """负例：只读 sqlite SELECT（含 > 比较符/变量赋值/裸词）零打扰。

    2026-10-10 实锤误报根修回归测试：旧判据 " > " 子串把 SQL 的
    WHERE ts > '...' 比较符当重定向，叠加裸词拼根恒真 → 只读查询被拦 2 次。
    """
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": 'DB=~/.hermes/outcomes.db; sqlite3 "$DB" '
                         "\"select rule, count(*) from violations "
                         "where timestamp > '2026-10-09T21:00' group by rule\""},
        session_id="s1",
    )
    assert v is None  # 期望: 只读 SELECT 无重定向文件目标→非写命令→放行（非写零打扰不变量）


def test_terminal_true_redirect_outside_not_blocked(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    """真重定向但目标在 OntoX 根外（临时目录）→ 写目标非 OntoX → 零打扰。"""
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"grep ERROR /tmp/x.log > {tmp_path}/out.txt"},
        session_id="s1",
    )
    assert v is None  # 期望: 重定向目标不在 ontox 根内→_target_from_terminal 返回空→放行


def test_terminal_redirect_ontox_relative_blocked(env: tuple[Path, Path]) -> None:
    """真重定向到 OntoX 相对路径（路径形态含 /）→ 仍受门禁（防漏报）。"""
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"echo data > {ontox}/apps/aml/out.txt"},
        session_id="s1",
    )
    assert v is not None  # 期望: 重定向目标落在 ontox 根内→真写→拦截（漏报面不扩大）
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


def test_config_fallback_when_module_attrs_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 期望: env-at-import 为空时路径从 config.yaml 读——「默认死护栏」缺陷的回归测试
    ontox = tmp_path / "ontox"
    cb = ontox / "education" / "casebook"
    cb.mkdir(parents=True)
    (cb / "INDEX.md").write_text("# 病例索引\n", encoding="utf-8")
    monkeypatch.setattr(casebook_gate, "_ONTOX_ROOT", "")
    monkeypatch.setattr(casebook_gate, "_CASEBOOK_DIR", "")
    monkeypatch.setattr(
        casebook_gate,
        "_config_paths",
        lambda: (str(ontox), str(cb)),
    )
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="patch", args={"path": f"{ontox}/apps/x.py"}, session_id="s1"
    )
    assert v is not None  # 期望: config 提供路径→护栏激活拦截（非死护栏）
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）
    assert str(cb) in v["message"]  # 期望: 修复指引来自 config 路径


# ── 只读误拦第三轮根修回归（2026-10-17）──────────────────────────────
# 规则语义（独立于实现）：门禁对象是「写目标落在 OntoX 根内」的命令。
# token 在命令中的角色是源/脚本路径/表达式/输入文件时，不构成写目标。
# 依据：dispatcher 账本 cp、晚省 evening_retro_facts.py 执行连拦 2 次（实测复现）。
# 期望值全部从「写目标角色」语义推导。


def test_cp_from_ontox_to_scratch_not_blocked(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    """cp 源在 OntoX、目标在外部 scratch → 源是只读用途 → 零打扰。

    2026-10-15~16 dispatcher 账本回看误拦形态：cp 只读复制被当写。
    """
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"cp {ontox}/apps/aml/main.py {tmp_path}/main_ro.py"},
        session_id="s1",
    )
    assert v is None  # 期望: 写目标=最后操作数(外部 scratch)→源不是写目标→放行


def test_cp_into_ontox_still_blocked(env: tuple[Path, Path]) -> None:
    """cp 目标落在 OntoX 根内 → 真写 → 拦截（防漏报扩大）。"""
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"cp /tmp/x.py {ontox}/apps/aml/x.py"},
        session_id="s1",
    )
    assert v is not None  # 期望: 写目标=OntoX 内→拦截
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）


def test_home_script_with_external_redirect_not_blocked(
    env: tuple[Path, Path],
) -> None:
    """执行家目录脚本 + 重定向到 /tmp → 脚本路径是读角色 → 零打扰。

    2026-10-16 晚省 evening_retro_facts.py 连拦 2 次形态：~ 路径被
    非绝对拼根逻辑误判为 OntoX 目标。
    """
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={
            "command": "python3 ~/.hermes/cron/scripts/evening_retro_facts.py"
            " > /tmp/evening_facts.json 2>&1"
        },
        session_id="s1",
    )
    assert v is None  # 期望: 唯一写目标=/tmp/evening_facts.json(根外)→放行


def test_home_jsonl_append_not_blocked(
    env: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """家目录 jsonl 追加（metrics 台账）→ 写目标在 OntoX 外 → 零打扰。

    2026-10-15 三省 metrics 追加被拦形态：~ 重定向目标拼根恒真。
    HOME 隔离到 tmp：~ 展开落在测试沙箱，不触发 home_io_guard。
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={
            "command": 'python3 -c "import json,time;print(json.dumps({1:2}))"'
            " >> ~/.hermes/state/evolution_metrics.jsonl"
        },
        session_id="s1",
    )
    assert v is None  # 期望: 重定向目标 expanduser 后在 OntoX 根外→放行


def test_sed_expr_with_external_file_not_blocked(
    env: tuple[Path, Path], tmp_path: Path
) -> None:
    """sed -i 作用于根外文件 → 表达式 token 非写目标 → 零打扰。"""
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"sed -i 's/a/b/' {tmp_path}/t.txt"},
        session_id="s1",
    )
    assert v is None  # 期望: sed 段写目标=脚本表达式之后的文件操作数(根外)→放行


def test_sed_ontox_file_still_blocked(env: tuple[Path, Path]) -> None:
    """sed -i 文件操作数落在 OntoX 根内 → 真写 → 拦截。"""
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"sed -i 's/a/b/' {ontox}/apps/aml/main.py"},
        session_id="s1",
    )
    assert v is not None  # 期望: 文件操作数=OntoX 内→拦截
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）


def test_interpreter_c_literal_write_blocked(env: tuple[Path, Path]) -> None:
    """terminal 解释器 -c 代码内字面量写 OntoX → 与 execute_code 通道等价防旁路。

    旧代码漏报面（实测）：terminal python3 -c 内部 open(...,'w') 写
    OntoX 零覆盖，而 execute_code 通道有字面量提取——通道间不对齐。
    """
    ontox, _cb = env
    _clear_probes()
    v = casebook_gate.on_pre_tool_call(
        tool_name="terminal",
        args={
            "command": (
                "python3 -c "
                f'"open(\'{ontox}/apps/aml/x.py\',\'w\').write(\'y\')"'
            )
        },
        session_id="s1",
    )
    assert v is not None  # 期望: 解释器代码内 OntoX 路径字面量=旁路写→拦截
    assert v["action"] == "block"  # 期望: 拦截动作字面量=block（规则语义）
