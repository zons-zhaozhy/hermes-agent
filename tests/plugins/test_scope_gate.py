"""guards.scope_gate 行为契约测试。

红线复现（2026-10-11 越权实录）：用户要求「考虑技术架构与关键实现逻辑」，
助手在架构文档刚落盘、方案未经用户确认时，于仓库内新建 apps/<新应用>/ 一整棵
目录树（19 个文件）。本套件断言该形态被 block，并断言各类放行边界不误拦。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from plugins._shared_state import clear_session
from plugins.guards import scope_gate


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """以 tmp_path 作为仓库根，并在非豁免前提下做判定。

    pytest tmp_path 落在 macOS 系统临时目录（/private/var/folders），属生产
    豁免区——block 语义测试须清空豁免前缀表才能构造。仓库根判定走
    _repo_root_of（.git 实测），fixture 须造 .git 目录。
    """
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(scope_gate, "_EXEMPT_PREFIXES", ())
    monkeypatch.chdir(tmp_path)
    clear_session("s-test")
    return tmp_path


def _authorize(monkeypatch: pytest.MonkeyPatch, verdict: bool | None) -> None:
    """把授权判定替换为固定结论（不打真实 LLM）。"""
    monkeypatch.setattr(scope_gate, "_judge_authorized", lambda _text: verdict)


def _user_says(text: str) -> None:
    """记录当轮用户原话（走真实 pre_llm_call 入口）。"""
    scope_gate._on_pre_llm_call(
        session_id="s-test",
        conversation_history=[{"role": "user", "content": text}],
    )


def _write(path: Path | str) -> dict | None:
    """走真实 pre_tool_call 入口发起一次 write_file。"""
    return scope_gate._on_pre_tool_call(
        tool_name="write_file", args={"path": str(path), "content": "x"}, session_id="s-test",
    )


def test_new_app_tree_blocked_without_authorization(repo, monkeypatch):
    """未授权时在仓库内新建目录树 → block（事故形态）。"""
    _authorize(monkeypatch, False)
    _user_says("你现在最重要的是考虑技术架构和关键实现逻辑")
    verdict = _write(repo / "apps" / "humanmarket" / "app" / "main.py")
    assert verdict is not None and verdict["action"] == "block"  # 期望: 新目录树 apps/humanmarket 且当轮未授权
    assert "apps/humanmarket" in verdict["message"]  # 期望: 命中证据写明新目录树根


def test_second_file_same_tree_blocked(repo, monkeypatch):
    """同一未授权目录树的第二个文件同样 block（判定按原话缓存）。"""
    _authorize(monkeypatch, False)
    _user_says("请评估这个方向")
    first = _write(repo / "apps" / "x" / "a.py")
    second = _write(repo / "apps" / "x" / "b.py")
    assert first is not None and first["action"] == "block"  # 期望: 首文件被拦
    assert second is not None and second["action"] == "block"  # 期望: 次文件仍在未授权目录树内


def test_file_in_existing_dir_allowed(repo, monkeypatch):
    """在已存在的目录里新建文件 → 放行（非新目录树）。"""
    (repo / "apps" / "existing").mkdir(parents=True)
    _authorize(monkeypatch, False)
    _user_says("请分析这个模块")
    verdict = _write(repo / "apps" / "existing" / "new_mod.py")
    assert verdict is None  # 期望: 父目录已存在=非新目录树


def test_authorized_new_tree_allowed(repo, monkeypatch):
    """用户原话授权构造时新建目录树 → 放行。"""
    _authorize(monkeypatch, True)
    _user_says("按方案动手吧，把新应用建起来")
    verdict = _write(repo / "apps" / "brandnew" / "app" / "main.py")
    assert verdict is None  # 期望: authorized=True 直接放行


def test_doc_extension_allowed(repo, monkeypatch):
    """文档扩展名在新目录树内 → 放行。"""
    _authorize(monkeypatch, False)
    _user_says("帮我看看架构")
    verdict = _write(repo / "docs" / "newtopic" / "note.md")
    assert verdict is None  # 期望: .md 命中文档豁免


def test_tests_dir_allowed(repo, monkeypatch):
    """路径含 tests 段 → 放行。"""
    _authorize(monkeypatch, False)
    _user_says("看看测试")
    verdict = _write(repo / "someapp" / "tests" / "test_x.py")
    assert verdict is None  # 期望: tests 段豁免


def test_temp_dir_allowed(repo, monkeypatch):
    """/tmp 前缀 → 放行（豁免前缀表保持生产值）。"""
    real_tmp = os.path.realpath("/tmp")
    monkeypatch.chdir(real_tmp)
    verdict = scope_gate._on_pre_tool_call(
        tool_name="write_file",
        args={"path": os.path.join(real_tmp, "newtree", "x", "main.py"), "content": "x"},
        session_id="s-test",
    )
    assert verdict is None  # 期望: /tmp 命中临时目录豁免


def test_boundary_depth_allowed(repo, monkeypatch):
    """新目录树根深度超过 2 → 放行。"""
    (repo / "apps" / "existing").mkdir(parents=True)
    _authorize(monkeypatch, False)
    _user_says("请分析")
    verdict = _write(repo / "apps" / "existing" / "sub" / "deeper" / "a.py")
    assert verdict is None  # 期望: 树根 apps/existing/sub 深度 3 超过上限


def test_judge_unavailable_fail_open(repo, monkeypatch):
    """授权判定不可用（None）→ 放行留痕（fail-open）。"""
    _authorize(monkeypatch, None)
    _user_says("请评估")
    verdict = _write(repo / "apps" / "x" / "a.py")
    assert verdict is None  # 期望: 判定 None=资源不可用，不误死锁


def test_cron_session_allowed(repo, monkeypatch):
    """cron 会话豁免。"""
    _authorize(monkeypatch, False)
    verdict = scope_gate._on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(repo / "apps" / "x" / "a.py"), "content": "x"},
        session_id="cron_night",
    )
    assert verdict is None  # 期望: cron_ 前缀放行


def test_non_write_tool_ignored(repo, monkeypatch):
    """读工具不进射程。"""
    _authorize(monkeypatch, False)
    _user_says("看看")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="read_file",
        args={"path": str(repo / "apps" / "x" / "a.py")},
        session_id="s-test",
    )
    assert verdict is None  # 期望: 射程仅写通道


def test_warn_only_allows(repo, monkeypatch):
    """观察模式：命中但只告警不阻断。"""
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    monkeypatch.setenv("HERMES_SCOPE_GATE_WARN_ONLY", "1")
    verdict = _write(repo / "apps" / "x" / "a.py")
    assert verdict is None  # 期望: WARN_ONLY 命中不阻断


def test_disabled_allows(repo, monkeypatch):
    """禁用开关：完全不判。"""
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    monkeypatch.setenv("HERMES_SCOPE_GATE_DISABLE", "1")
    verdict = _write(repo / "apps" / "x" / "a.py")
    assert verdict is None  # 期望: DISABLE=1 直接放行


def test_execute_code_channel_blocked(repo, monkeypatch):
    """execute_code 通道同射程：字面量写新建目录树 → block。"""
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    target = repo / "apps" / "newapp" / "mod.py"
    verdict = scope_gate._on_pre_tool_call(
        tool_name="execute_code",
        args={"code": f'open("{target}", "w").write("x")\n'},
        session_id="s-test",
    )
    assert verdict is not None and verdict["action"] == "block"  # 期望: execute_code 字面量写新目录树被拦


def test_terminal_mkdir_new_tree_blocked(repo, monkeypatch):
    """terminal 用 mkdir 建新目录树 → block（同射程）。"""
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "mkdir -p apps/newapp/app"},
        session_id="s-test",
    )
    assert verdict is not None and verdict["action"] == "block"  # 期望: mkdir 建新目录树被拦


def test_terminal_chained_mkdir_blocked(repo, monkeypatch):
    """命令链中的 mkdir 同样被拦（逐段解析；cd 段更新基准后 sudo 前缀仍识别）。"""
    (repo / "apps").mkdir()
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"cd {repo}/apps && sudo mkdir -p newapp2"},
        session_id="s-test",
    )
    assert verdict is not None and verdict["action"] == "block"  # 期望: cd 到 apps 下建浅层新树 → 拦


def test_terminal_cd_prefix_deep_subdir_allowed(repo, monkeypatch):
    """cd 到仓库内既有目录后建深层子目录 → 放行（cd 段基准修复，2026-10-14 实录）。"""
    (repo / "apps" / "existing").mkdir(parents=True)
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"cd {repo}/apps/existing && mkdir -p docs"},
        session_id="s-test",
    )
    assert verdict is None  # 期望: 目标=apps/existing/docs depth=3>2 → 放行（修复前误解析到进程 cwd 报 rel_root='apps'）


def test_terminal_workdir_param_deep_subdir_allowed(repo, monkeypatch):
    """terminal workdir 参数作为相对 mkdir 目标的基准 → 深层子目录放行。"""
    (repo / "apps" / "existing").mkdir(parents=True)
    _authorize(monkeypatch, False)
    _user_says("考虑架构")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "mkdir -p apps/existing/docs", "workdir": str(repo)},
        session_id="s-test",
    )
    assert verdict is None  # 期望: workdir 基准下 apps/existing/docs depth=3>2 → 放行


def test_terminal_touch_in_existing_dir_allowed(repo, monkeypatch):
    """touch 落在已有目录 → 放行。"""
    (repo / "apps" / "existing").mkdir(parents=True)
    _authorize(monkeypatch, False)
    _user_says("看看")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "touch apps/existing/new.txt"},
        session_id="s-test",
    )
    assert verdict is None  # 期望: 父目录已存在=非新目录树


def test_terminal_mkdir_temp_allowed(repo, monkeypatch):
    """mkdir 到临时目录 → 放行（前缀豁免）。"""
    _authorize(monkeypatch, False)
    _user_says("看看")
    real_tmp = os.path.realpath("/tmp")
    verdict = scope_gate._on_pre_tool_call(
        tool_name="terminal",
        args={"command": f"mkdir -p {real_tmp}/newtree/app"},
        session_id="s-test",
    )
    assert verdict is None  # 期望: /tmp 前缀豁免
