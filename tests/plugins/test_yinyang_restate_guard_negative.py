"""yinyang_restate_guard 负例契约测试——豁免开关与 fail-open 边界（违规必红）。

Contract:
    Preconditions: plugins/yinyang-restate-guard/__init__.py 存在
    Postconditions: 环境开关生效；空消息走 fail-open 不误拦
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "yinyang-restate-guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("yinyang_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_env_disable_switch(monkeypatch):
    guard = _load_guard()
    monkeypatch.setenv("YINYANG_RESTATE_GUARD_DISABLE", "1")
    assert guard._plugin_disabled() is True  # 期望: True=显式豁免开关生效（拦截面可显式关闭）
    monkeypatch.setenv("YINYANG_RESTATE_GUARD_DISABLE", "")
    assert guard._plugin_disabled() is False  # 期望: False=默认启用


def test_negative_empty_message_fail_open():
    guard = _load_guard()
    assert guard.is_challenge("") is None  # 期望: None=空消息 fail-open 不误拦
