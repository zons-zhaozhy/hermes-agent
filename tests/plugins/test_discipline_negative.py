"""discipline 负例契约测试——瞎猜纪律检测必须真拦违规命令（违规必红）。

Contract:
    Preconditions: plugins/discipline/no_guessing.py 存在
    Postconditions: 跨架构构建/裸日志/超限 sleep 均被检测判定
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "discipline" / "no_guessing.py"
    spec = importlib.util.spec_from_file_location("discipline_no_guessing_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    sys.modules["discipline_no_guessing_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_negative_cross_arch_build_detected():
    guard = _load_guard()
    assert guard._check_cross_arch_build("docker build -t myapp .") is True  # 期望: True=无 --platform 隐式架构构建必拦


def test_negative_explicit_platform_passes():
    guard = _load_guard()
    assert guard._check_cross_arch_build("docker build --platform linux/amd64 -t x .") is False  # 期望: False=显式声明平台放行


def test_negative_raw_log_reading_detected():
    guard = _load_guard()
    out = guard._check_raw_logs("docker logs app")
    assert out is not None  # 期望: 非空=裸读全量日志被拦（须先 grep 提取）


def test_negative_sleep_over_limit_detected():
    guard = _load_guard()
    out = guard._check_sleep_wait("sleep 60", sleep_limit=10)
    assert out is not None  # 期望: 非空=超限 sleep 必拦


def test_negative_sleep_within_limit_passes():
    guard = _load_guard()
    assert guard._check_sleep_wait("sleep 5", sleep_limit=10) is None  # 期望: None=限内 sleep 放行
