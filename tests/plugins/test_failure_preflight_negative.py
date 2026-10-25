"""failure_preflight 负例契约测试——动手类任务必须被注入失败预演要求（违规必红）。

Contract:
    Preconditions: plugins/failure_preflight/__init__.py 存在
    Postconditions: 动词开头任务消息触发 context 注入；讨论句式不触发
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "failure_preflight" / "__init__.py"
    spec = importlib.util.spec_from_file_location("failure_preflight_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_action_task_gets_preflight_injection():
    guard = _load_guard()
    out = guard._on_pre_llm_call(user_message="部署这个服务到云端")
    assert "context" in out  # 期望: 动手任务被注入失败预演要求（拦截裸动手）


def test_negative_action_detection():
    guard = _load_guard()
    assert guard._is_action_task("写一个脚本处理数据") is True  # 期望: True=动词开头识别
    assert guard._is_action_task("为什么这个设计是这样") is False  # 期望: False=讨论句式豁免
    assert guard._is_action_task("") is False  # 期望: False=空消息豁免


def test_negative_discussion_not_injected():
    guard = _load_guard()
    out = guard._on_pre_llm_call(user_message="能不能实现这个功能")
    assert out == {}  # 期望: {}=疑问句式不注入
