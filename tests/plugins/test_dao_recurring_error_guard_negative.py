"""dao_recurring_error_guard 负例契约测试——同类错误重复必须被识别计数（违规必红）。

Contract:
    Preconditions: plugins/dao-recurring-error-guard/__init__.py 存在
    Postconditions: 错误文本被归一化分类；重复同类错误计数累积触发注入
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "dao-recurring-error-guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("dao_rec_err_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_error_text_classified():
    guard = _load_guard()
    cls = guard.classify_error("psql: connection refused to server")
    assert cls is not None  # 期望: 非空=连接类错误被归一化识别（零识别=零防护）


def test_negative_unrelated_text_not_classified():
    guard = _load_guard()
    assert guard.classify_error("all good, rows returned") is None  # 期望: None=正常文本不误报


def test_negative_empty_text_not_classified():
    guard = _load_guard()
    assert guard.classify_error("") is None  # 期望: None=空文本豁免


def test_negative_persist_key_normalized():
    guard = _load_guard()
    k1 = guard._persist_key("terminal", "connection refused")
    k2 = guard._persist_key("terminal", "connection refused")
    assert k1 == k2  # 期望: 同工具同类错误归并同一键（重复计数的前提）
