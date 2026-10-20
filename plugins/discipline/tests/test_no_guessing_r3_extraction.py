"""R3 服务名提取（_extract_service_names）回归测试。

背景：旧实现对整条复合命令全令牌做白名单排除式收集，管道/&& 另一端的
验证命令令牌（sleep/ssh/curl）被当服务名去注册表对账，注册表查不到即拦
——部署后接验证命令的组合被连续误拦两次（实锤）。本文件锁定定位式提取
修复后的行为：服务名只来自 build.sh/deploy.sh 调用位之后的参数区。

Contract:
  Preconditions: 本文件所在目录的父目录含 no_guessing.py（即插件包内）
  Postconditions:
    - 分隔符之后的命令令牌（ssh/sleep/curl 等）不再被提取为服务名；
    - 调用参数区内的未知服务名仍被提取（负例，防过滤过宽）；
    - 多个调用各自成区，全部提取。
"""

import importlib.util
import pathlib
import types

import pytest


def _load_module() -> types.ModuleType:
    """Contract: 返回被加载的 no_guessing 模块（不依赖 hermes 运行时）。"""
    path = pathlib.Path(__file__).resolve().parents[1] / "no_guessing.py"
    spec = importlib.util.spec_from_file_location("no_guessing_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载插件模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ng() -> types.ModuleType:
    """Contract: 模块级共享的 no_guessing 模块实例。"""
    return _load_module()


def test_compound_verify_suffix_not_extracted(ng: types.ModuleType) -> None:
    """部署命令后接验证命令（sleep/ssh/curl）→ 只提取真服务名。"""
    cmd = (
        "bash deploy/deploy.sh cloud portal 2>&1 | tail -4 "
        "&& sleep 5 && curl -sk https://localhost/privacy "
        "&& ssh -i deploy/key.pem ubuntu@host 'ls'"
    )
    assert ng._extract_service_names(cmd) == ["portal"]  # 期望: ssh/sleep/curl 不入列


def test_separator_closes_arg_region(ng: types.ModuleType) -> None:
    """分隔符之后直到下一个调用令牌之前，任何令牌都不收集。"""
    cmd = "bash deploy/deploy.sh cloud portal; echo done; ls -la /opt/ontox/deploy/logs"
    assert ng._extract_service_names(cmd) == ["portal"]  # 期望: echo/ls 不入列


def test_unknown_name_still_extracted(ng: types.ModuleType) -> None:
    """负例：参数区内的未知服务名必须仍被提取（R3 拦截前提）。"""
    cmd = "bash deploy/deploy.sh cloud loom-frontend"
    assert ng._extract_service_names(cmd) == ["loom-frontend"]  # 期望: 错名照常暴露


def test_multiple_invocations_each_extracted(ng: types.ModuleType) -> None:
    """复合命令含多个调用 → 各参数区分别提取。"""
    cmd = "bash deploy/build.sh auth cortex && bash deploy/deploy.sh cloud portal"
    assert ng._extract_service_names(cmd) == ["auth", "cortex", "portal"]  # 期望: 三名全收


def test_no_invocation_returns_empty(ng: types.ModuleType) -> None:
    """无脚本调用令牌 → 空列表（纯验证/巡检命令不触发 R3）。"""
    cmd = "ssh -i key.pem ubuntu@host 'docker ps --format {{.Names}}'"
    assert ng._extract_service_names(cmd) == []  # 期望: 零提取
