"""R7（脚本间接执行）误报回归测试。

背景：no_guessing 的 R7 会读取被调用脚本全文再扫描服务名。它曾把 build.sh
用法注释里的散文令牌（`#   bash deploy/build.sh auth cortex  # 构建多个`）
当成服务名，导致 `bash deploy/build.sh --list` 本身被拦——违反文件头
Contract 里的不变式「never blocks --list itself」。本测试锁定修复后的行为。

Contract:
  Preconditions: 本文件所在目录的父目录含 no_guessing.py（即插件包内）
  Postconditions:
    - --list 调用永不拦；
    - 脚本内只有注释提及 build.sh/deploy.sh 时放行；
    - 脚本内真实调用且服务名未验证时仍拦（负例，防过滤过宽）。
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


def test_registry_list_call_is_recognised(ng: types.ModuleType) -> None:
    """--list 必须被识别为注册表采集调用（放行的前提）。"""
    assert ng._is_registry_list_call("bash deploy/build.sh --list") is True


def test_list_call_never_blocked(ng: types.ModuleType) -> None:
    """不变式：--list 是采集注册表的唯一通道，其本身永不拦。"""
    assert ng._check_script_indirection("bash deploy/build.sh --list") is None


def test_comment_only_mention_passes(ng: types.ModuleType, tmp_path: pathlib.Path) -> None:
    """脚本里只在注释中提及 build.sh（用法文档）→ 不构成间接执行 → 放行。"""
    script = tmp_path / "comment_only.sh"
    script.write_text(
        "#!/bin/bash\n"
        "#   bash deploy/build.sh auth cortex    # 构建多个\n"
        "#   bash deploy/deploy.sh cloud auth\n",
        encoding="utf-8",
    )
    assert ng._check_script_indirection(f"bash {script}") is None


def test_unverified_service_name_still_blocked(
    ng: types.ModuleType, tmp_path: pathlib.Path
) -> None:
    """负例：脚本内真实调用且服务名未经验证 → 必须仍然拦截。"""
    script = tmp_path / "bogus.sh"
    script.write_text(
        "#!/bin/bash\nbash /repo/deploy/build.sh totally-bogus-service\n",
        encoding="utf-8",
    )
    message = ng._check_script_indirection(f"bash {script}")
    assert message is not None
    assert "totally-bogus-service" in message
