"""Tests for discipline.test_discipline（禁裸 pytest 动作拦截）.

行为契约（从「一律 scripts/run_tests.sh 禁裸 pytest」独立推导）：
- 裸 `pytest ...` / `python -m pytest ...` terminal 命令 → block，消息含正门指引
- scripts/run_tests.sh 前缀（含 ./ 相对形式、带 -k 透传）→ 放行
- run_tests_parallel.py → 放行（包装器体系内层）
- 纯探针（--version/--help/--collect-only）→ 放行
- 非 terminal 工具 → 放行
- 坏 shell 串（shlex 失败）→ 放行（fail-open）
- TEST_DISCIPLINE_DISABLE=1 → register 后仍放行
"""
from typing import Any, Dict, Optional

import plugins.discipline.test_discipline as td


def _run(command: str) -> Optional[Dict[str, Any]]:
    return td.on_pre_tool_call(
        tool_name="terminal", args={"command": command})


def test_bare_pytest_blocked():
    out = _run("pytest tests/foo/ -x")
    assert out is not None and out["action"] == "block"
    assert "run_tests.sh" in out["message"]


def test_python_m_pytest_blocked():
    out = _run("python3 -m pytest tests/foo/ -q")
    assert out is not None and out["action"] == "block"


def test_wrapper_allowed():
    assert _run("scripts/run_tests.sh tests/plugins/") is None
    assert _run("./scripts/run_tests.sh tests/foo/test_x.py -k test_y") is None


def test_parallel_runner_allowed():
    assert _run("python3 scripts/run_tests_parallel.py tests/") is None


def test_probe_flags_allowed():
    assert _run("pytest --version") is None
    assert _run("pytest --collect-only tests/") is None


def test_non_terminal_tool_allowed():
    out = td.on_pre_tool_call(tool_name="execute_code", args={"code": "pytest"})
    assert out is None


def test_broken_shell_string_fail_open():
    assert _run('pytest "unclosed') is None


def test_empty_command_allowed():
    assert _run("") is None  # 期望: 空命令无 pytest 令牌, 放行


def test_repo_without_wrapper_allowed(tmp_path, monkeypatch):
    """仓库感知：目标仓库无 scripts/run_tests.sh → 放行（如 ontox，CI 即裸 pytest）。"""
    repo = tmp_path / "repo_no_wrapper"
    repo.mkdir()
    monkeypatch.chdir(repo)
    assert _run(f"cd {repo} && python3 -m pytest tests/ -q") is None  # 期望: 无包装器仓库放行
    assert _run("python3 -m pytest tests/ -q") is None  # 期望: 无 cd 时 cwd=无包装器仓库, 同样放行


def test_repo_with_wrapper_blocked(tmp_path, monkeypatch):
    """仓库感知：目标仓库有 scripts/run_tests.sh → 仍拦（如 hermes-agent）。"""
    repo = tmp_path / "repo_with_wrapper"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_tests.sh").write_text("#!/bin/bash\n")
    monkeypatch.chdir(repo)
    out = _run(f"cd {repo} && python3 -m pytest tests/ -q")
    assert out is not None and out["action"] == "block"  # 期望: 有包装器仓库仍拦截
    out = _run("pytest tests/foo/ -x")
    assert out is not None and out["action"] == "block"  # 期望: 无 cd 时 cwd=有包装器仓库, 仍拦截


def test_registered_gated_hook(monkeypatch):
    """register 注册的是带环境开关的包装版；DISABLE=1 时放行。"""
    registered: Dict[str, Any] = {}

    class _Ctx:
        @staticmethod
        def register_hook(name: str, fn: Any) -> None:
            registered[name] = fn

    td.register(_Ctx)
    fn = registered["pre_tool_call"]
    monkeypatch.setenv("TEST_DISCIPLINE_DISABLE", "1")
    assert fn(tool_name="terminal", args={"command": "pytest tests/"}) is None
    monkeypatch.delenv("TEST_DISCIPLINE_DISABLE")
    out = fn(tool_name="terminal", args={"command": "pytest tests/"})
    assert out is not None and out["action"] == "block"
