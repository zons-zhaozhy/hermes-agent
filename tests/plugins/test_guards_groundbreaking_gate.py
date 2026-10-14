"""guards.groundbreaking_gate 回归测试。

覆盖设计文档 §6 可验问题清单 9 问 + 反方审查 R1-R8 修正项
（期望值独立推导自修正后设计，非实现反推）：

1. 三无新应用写首代码文件 → block（消息含三源三态+skill 指针）
2. 三证据源任一存在（E1 内容判据/E2 DDL/E3 token 整词）→ 放行
3. 豁免通道：非 apps 路径/文档扩展/仓根豁免清单 → 放行；行内自豁免 → 仍拦（R2）
4. 存量应用零打扰；集合相等（tests/*.py 与 __init__.py 属代码文件，
   先写 tests 也触发门禁——审查 R5 两集合强制相等；其负例「仅 tests 仍拦」
   与集合相等主张互斥，按集合相等语义实现并在此锁定）
5. 空文档/短文档/无标题文档不构成证据（R1）；E3 子串与噪音词不命中（R4）
6. 缓存存证据路径，删除证据后失效重拦（R6）
7. execute_code 覆盖 .py 与 .ts（R3 变体）；守卫异常 fail-open；env 禁用放行

Contract:
  Preconditions: 全部在 tmp_path 伪造目录树，零真实仓依赖；每用例唯一 sid 防缓存互串
  Postconditions: 全部断言通过 = 上述契约成立
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from plugins.guards import groundbreaking_gate as gg
from plugins.guards.duplicate_check import (
    _execute_code_py_targets,
    _execute_code_write_targets,
)

_sid = itertools.count()

GOOD_DOC = "# 架构功课\n" + "\n".join(f"内容行{i}" for i in range(30)) + "\n## 数据结构\n表族说明\n"


def _call(tool: str, path: Path | str, content: str = "x") -> dict[str, Any] | None:
    """以唯一 sid 驱动 write_file 形态（防会话态缓存跨用例互串）。"""
    return gg.on_pre_tool_call(
        session_id=f"test-{next(_sid)}",
        tool_name=tool,
        args={"path": str(path), "content": content},
    )


def _call_code(code: str) -> dict[str, Any] | None:
    """以唯一 sid 驱动 execute_code 形态。"""
    return gg.on_pre_tool_call(
        session_id=f"test-{next(_sid)}", tool_name="execute_code", args={"code": code}
    )


def _mk_repo(tmp_path: Path) -> Path:
    (tmp_path / "apps").mkdir()
    (tmp_path / "docs" / "blueprint").mkdir(parents=True)
    return tmp_path


def test_new_app_without_evidence_blocks(tmp_path: Path) -> None:
    # 期望: 三无新应用写首代码文件 → block(可验①)
    repo = _mk_repo(tmp_path)
    v = _call("write_file", repo / "apps" / "demo_app" / "main.py")
    assert v is not None and v.get("action") == "block"  # 期望: 三证据全无 → 拦
    msg = v.get("message", "")
    assert "[groundbreaking-gate]" in msg  # 期望: 标明守卫来源
    assert "E1" in msg and "E2" in msg and "E3" in msg  # 期望: 三源三态列出
    assert "app-groundbreaking-seven-steps" in msg  # 期望: 指路 skill


def test_e1_groundwork_doc_passes(tmp_path: Path) -> None:
    # 期望: 应用内 30+行含「数据结构」标题的功课 md → 放行(可验②E1)
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "demo_app"
    (app / "docs").mkdir(parents=True)
    (app / "docs" / "ARCHITECTURE.md").write_text(GOOD_DOC, encoding="utf-8")
    assert _call("write_file", app / "main.py") is None  # 期望: E1 命中 → 放行


def test_e1_empty_or_short_doc_still_blocks(tmp_path: Path) -> None:
    # 期望: 空文档/短文档/无标题文档不构成证据(R1, 可验⑦)
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "demo_app"
    (app / "docs").mkdir(parents=True)
    (app / "docs" / "A.md").write_text("", encoding="utf-8")
    (app / "docs" / "B.md").write_text("短\n" * 10, encoding="utf-8")
    (app / "docs" / "C.md").write_text("行\n" * 40, encoding="utf-8")
    v = _call("write_file", app / "main.py")
    assert v is not None and v.get("action") == "block"  # 期望: 三份皆不构成证据 → 拦


def test_e1_readme_not_evidence(tmp_path: Path) -> None:
    # 期望: README 不构成证据
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "demo_app"
    app.mkdir(parents=True)
    (app / "README.md").write_text(GOOD_DOC, encoding="utf-8")
    v = _call("write_file", app / "main.py")
    assert v is not None and v.get("action") == "block"  # 期望: README 属豁免文档 → 仍拦


def test_e2_sql_and_migrations_pass(tmp_path: Path) -> None:
    # 期望: 应用内 DDL .sql 或 migrations/ 目录 → 放行(可验②E2)
    repo = _mk_repo(tmp_path)
    a1 = repo / "apps" / "app_sql"
    a1.mkdir(parents=True)
    (a1 / "ddl_v1.sql").write_text("CREATE TABLE t(a int);\n", encoding="utf-8")
    assert _call("write_file", a1 / "main.py") is None  # 期望: .sql 命中 → 放行
    a2 = repo / "apps" / "app_mig"
    (a2 / "migrations").mkdir(parents=True)
    assert _call("write_file", a2 / "main.py") is None  # 期望: migrations/ 命中 → 放行


def test_e3_token_match_passes_substring_blocked(tmp_path: Path) -> None:
    # 期望: E3 token 整词匹配放行；子串不命中(R4)
    repo = _mk_repo(tmp_path)
    (repo / "docs" / "blueprint" / "DEMO_APP_ARCHITECTURE.md").write_text(
        GOOD_DOC, encoding="utf-8"
    )
    assert _call("write_file", repo / "apps" / "demo-app" / "main.py") is None  # 期望: token 整词(demo_app) → 放行
    (repo / "docs" / "blueprint" / "NEXTGEN_AML_BLUEPRINT.md").write_text(
        GOOD_DOC, encoding="utf-8"
    )
    v = _call("write_file", repo / "apps" / "aml2" / "main.py")
    assert v is not None and v.get("action") == "block"  # 期望: aml2 与 aml token 不等 → 子串不命中 → 拦


def test_e3_noise_token_degraded(tmp_path: Path) -> None:
    # 期望: 应用名全为噪音词(如 api)时 E3 不构成证据(R4 降级)
    repo = _mk_repo(tmp_path)
    (repo / "docs" / "blueprint" / "API_REVIEW.md").write_text(GOOD_DOC, encoding="utf-8")
    v = _call("write_file", repo / "apps" / "api" / "main.py")
    assert v is not None and v.get("action") == "block"  # 期望: signal 为空 → E3 无效 → 拦


def test_legacy_app_zero_disturb(tmp_path: Path) -> None:
    # 期望: 存量应用(已有其他代码文件)零打扰(可验③)
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "legacy"
    app.mkdir(parents=True)
    (app / "old.py").write_text("print()\n", encoding="utf-8")
    assert _call("write_file", app / "new.py") is None  # 期望: 存量 → 放行


def test_tests_file_counts_as_code_set_equality(tmp_path: Path) -> None:
    # 期望: 集合相等(R5)——tests/*.py 属代码文件: 无证据时写 tests 被拦；
    # 有证据放行后目录含 tests → 写主码按存量放行
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "tdd_app"
    v = _call("write_file", app / "tests" / "test_x.py")
    assert v is not None and v.get("action") == "block"  # 期望: 首个代码文件=tests 也拦
    app.mkdir(parents=True, exist_ok=True)
    (app / "tests").mkdir(parents=True, exist_ok=True)
    (app / "tests" / "test_x.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    assert _call("write_file", app / "main.py") is None  # 期望: 目录已有代码文件(tests) → 存量放行


def test_exempt_list_passes_inline_marker_blocked(tmp_path: Path) -> None:
    # 期望: 仓根豁免清单放行；被写文件内行内标记不生效(R2, 可验⑤)
    repo = _mk_repo(tmp_path)
    exempt = repo / "docs" / "blueprint" / "GROUNDBREAKING_EXEMPT.md"
    exempt.write_text(
        "# 豁免清单(应用名|理由|日期)\ndemo_app | 纯适配层无独立表 | 2026-10-14\n",
        encoding="utf-8",
    )
    assert _call("write_file", repo / "apps" / "demo_app" / "adapter.py") is None  # 期望: 清单首列匹配 → 放行
    other = repo / "apps" / "no_list" / "adapter.py"
    v = _call("write_file", other, "# groundbreaking-ok: 纯适配层无独立表\nprint()\n")
    assert v is not None and v.get("action") == "block"  # 期望: 行内自豁免=后门 → 仍拦


def test_cache_invalidated_when_evidence_deleted(tmp_path: Path) -> None:
    # 期望: 缓存存证据路径，证据文件删除后失效重拦(R6, 可验⑨)
    repo = _mk_repo(tmp_path)
    app = repo / "apps" / "cache_app"
    (app / "docs").mkdir(parents=True)
    doc = app / "docs" / "A.md"
    doc.write_text(GOOD_DOC, encoding="utf-8")
    sid = f"test-{next(_sid)}"
    first = gg.on_pre_tool_call(
        session_id=sid, tool_name="write_file",
        args={"path": str(app / "main.py"), "content": "x"},
    )
    assert first is None  # 期望: E1 命中 → 放行+写缓存
    doc.unlink()
    second = gg.on_pre_tool_call(
        session_id=sid, tool_name="write_file",
        args={"path": str(app / "main.py"), "content": "x"},
    )
    assert second is not None and second.get("action") == "block"  # 期望: 证据已删+缓存复验失败 → 重拦


def test_non_apps_and_doc_ext_pass(tmp_path: Path) -> None:
    # 期望: 非 apps 路径/文档扩展零打扰
    repo = _mk_repo(tmp_path)
    assert _call("write_file", repo / "tools" / "x.py") is None  # 期望: 无 apps 段 → 放行
    assert _call("write_file", repo / "apps" / "demo" / "README.md") is None  # 期望: .md 非代码扩展 → 放行


def test_execute_code_py_and_ts_targets_blocked(tmp_path: Path) -> None:
    # 期望: execute_code 写形态 .py 与 .ts 均被拦(R3, 可验⑥)
    repo = _mk_repo(tmp_path)
    v = _call_code(
        f"open('{repo}/apps/ec_app/main.py', 'w').write('x')"
    )
    assert v is not None and v.get("action") == "block"  # 期望: .py 经 execute_code → 拦
    v = _call_code(
        f"from pathlib import Path; Path('{repo}/apps/ec_app2/index.ts').write_text('x')"
    )
    assert v is not None and v.get("action") == "block"  # 期望: .ts 经 execute_code(变体) → 拦
    assert _call_code(f"open('{repo}/tools/x.py', 'w').write('x')") is None  # 期望: 非 apps 路径 → 放行


def test_execute_code_write_targets_variant() -> None:
    # 期望: 变体提取 .ts/.vue 目标；.py 专用版委托等价(单一事实源)
    code = (
        "from pathlib import Path\n"
        "Path('/a/x.ts').write_text('t')\n"
        "Path('/a/y.vue').write_text('v')\n"
        "open('/a/z.py', 'w').write('p')\n"
    )
    hits = _execute_code_write_targets(code, (".ts", ".vue", ".py"))
    assert sorted(p for p, _ in hits) == ["/a/x.ts", "/a/y.vue", "/a/z.py"]  # 期望: 三目标全提取
    py_only = _execute_code_py_targets(code)
    assert [p for p, _ in py_only] == ["/a/z.py"]  # 期望: .py 专用版仅 z.py(委托等价)


def test_guard_exception_fail_open(tmp_path: Path, monkeypatch: Any) -> None:
    # 期望: 守卫自身异常 → None 放行不 raise(可验④)
    monkeypatch.setattr(gg, "_check_target", lambda sid, p: 1 / 0)
    v = gg.on_pre_tool_call(
        session_id="t", tool_name="write_file",
        args={"path": "/x/apps/y/z.py", "content": "x"},
    )
    assert v is None  # 期望: fail-open


def test_env_disable(tmp_path: Path, monkeypatch: Any) -> None:
    # 期望: HERMES_GROUNDBREAKING_GATE_DISABLE=1 → 完全放行
    monkeypatch.setenv("HERMES_GROUNDBREAKING_GATE_DISABLE", "1")
    repo = _mk_repo(tmp_path)
    assert _call("write_file", repo / "apps" / "demo" / "main.py") is None  # 期望: env 禁用 → 放行
