"""R5 组合式 sleep 处置回归测试（回退治理 2026-09-26）。

背景：R5 判回退（密度 2.39→2.78/千 7 天不降）。violations 实测主力形态
是 `sleep 90; grep -c OK log` 类组合式——轮询退化姿势非纯干等，旧逻辑
一并 block 导致误报面过宽、密度不降。处置=组合式降级提醒放行，纯 sleep
仍 block。

Contract:
  Postconditions: 组合式返回 "HINT"；纯 sleep 返回 _BLOCK_SLEEP_LOOP；
    短 sleep/background 返回 None
"""

import importlib.util
import pathlib
import types

import pytest


def _load_module() -> types.ModuleType:
    """Contract: 返回被加载的 no_guessing 模块（不依赖 hermes 运行时）。"""
    path = pathlib.Path(__file__).resolve().parents[1] / "no_guessing.py"
    spec = importlib.util.spec_from_file_location("no_guessing_r5_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载插件模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ng() -> types.ModuleType:
    return _load_module()


def test_combo_sleep_returns_hint(ng):
    """组合式 sleep+检查 → HINT（提醒放行，不 block）。"""
    # 期望: sleep 90 后接 grep 检查=轮询退化非干等，旧逻辑此处 block（红），新逻辑 HINT
    assert ng._check_sleep_wait("sleep 90; grep -c ' OK ' /tmp/x.log") == "HINT"  # 期望: 组合式→HINT


def test_pure_sleep_still_blocks(ng):
    """纯 sleep（无后续动作）→ 仍然 block。"""
    # 期望: sleep 90 无任何后续=真干等，新旧逻辑一致返回 block 文案
    assert ng._check_sleep_wait("sleep 90") == ng._BLOCK_SLEEP_LOOP  # 期望: 纯干等→block


def test_pure_sleep_with_only_semicolon_blocks(ng):
    """sleep 后只有分号无实质命令 → 仍 block（防绕过）。"""
    # 期望: `sleep 90 ;` 尾部 split 后无实质 token=纯干等→block（防 `; ` 绕过）
    assert ng._check_sleep_wait("sleep 90 ;") == ng._BLOCK_SLEEP_LOOP  # 期望: 尾无实 token→block


def test_short_sleep_passes(ng):
    """短 sleep（≤limit）→ None 放行不变。"""
    # 期望: sleep 3 ≤ limit 10=页面渲染等待合法姿势→None（不变量保持）
    assert ng._check_sleep_wait("sleep 3; curl -s http://x") is None  # 期望: 短 sleep→None


def test_background_never_blocks(ng):
    """background=true → None 永不拦不变。"""
    # 期望: background 长任务正解姿势→None（不变量保持，与秒数无关）
    assert ng._check_sleep_wait("sleep 300", is_background=True) is None  # 期望: background→None
