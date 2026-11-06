"""coding_standards 负例契约测试——编码规范 AST 检查必须真抓违规（违规必红）。

Contract:
    Preconditions: plugins/guards/coding_standards.py 存在
    Postconditions: 吞异常/裸 except/静默降级均产生对应 rule_id 违规
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "coding_standards.py"
    spec = importlib.util.spec_from_file_location("coding_standards_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rule_ids(guard: ModuleType, code: str) -> set:
    return {v.rule_id for v in guard._run_all_checks(code)}


def test_negative_env_key_guess_detected():  # R025
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        'env.update({"ATOMATCH_DB_HOST": h, "ATOMATCH_PORT": p})\n')
    assert "R025" in rules  # 期望: 无 idsrc 标记的 env 键写入被拦（1105 事故形态）


def test_negative_env_subscript_guess_detected():  # R025
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        'os.environ["ATOMATCH_JDBC_URL"] = url\n')
    assert "R025" in rules  # 期望: 下标赋值形态的猜键名同样被拦


def test_env_key_with_idsrc_annotation_passes():  # R025 豁免通道
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        'env.update({"ATOMATCH_JDBC_URL": u})  # idsrc: db/__init__.py:1769\n')
    assert "R025" not in rules  # 期望: 行内来源标记即放行


def test_env_key_plain_name_not_flagged():  # R025 收窄边界
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        'env.update({"PATH": p, "TZ": tz})\n')
    assert "R025" not in rules  # 期望: 非 UPPER_SNAKE（无下划线）键不拦（PATH 类）


def test_negative_bare_except_pass_detected():
    guard = _load_guard()
    rules = _rule_ids(guard, "try:\n    x = 1\nexcept:\n    pass\n")
    assert "R002" in rules  # 期望: 裸 except:pass 被拦（吞异常检测）


def test_negative_silent_downgrade_detected():
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        "def f():\n"
        "    try:\n"
        "        return compute()\n"
        "    except Exception:\n"
        "        return []\n",
    )
    assert len(rules) > 0  # 期望: 静默降级（异常吞掉返回默认值）被拦


def test_negative_clean_code_passes():
    guard = _load_guard()
    rules = _rule_ids(
        guard,
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )
    assert rules == set()  # 期望: 空集=干净代码零违规
