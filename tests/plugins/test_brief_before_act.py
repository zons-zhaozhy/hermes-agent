"""brief_before_act 契约测试——先亮牌后动手守卫的检测语义正负例。

Contract:
    Preconditions: plugins/brief_before_act/__init__.py 存在
    Postconditions: 裸写(无亮牌标记)检出违规；亮牌后写/纯读/无user窗口均放行
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "brief_before_act" / "__init__.py"
    spec = importlib.util.spec_from_file_location("brief_before_act_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hist(*msgs: dict) -> list[dict]:
    return list(msgs)


def _user(text: str = "把文案改了") -> dict:
    return {"role": "user", "content": text}


def _asst(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _asst_write(tool: str = "patch") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": tool, "arguments": "{}"}}
    ]}


def test_bare_write_detected():
    guard = _load_guard()
    h = _hist(_user(), _asst("我直接改了。"), _asst_write())
    assert guard._find_violation(h) is True  # 期望: 写前无四要素标记=违规


def test_briefed_write_allowed():
    guard = _load_guard()
    brief = "理解：改文案。方案：三处替换。决策：用词跟品牌。动作：patch后跑测试。"
    h = _hist(_user(), _asst(brief), _asst_write())
    assert guard._find_violation(h) is False  # 期望: 四要素齐全后写=放行


def test_read_only_turn_allowed():
    guard = _load_guard()
    h = _hist(_user(), _asst("我先读一下文件。"), {
        "role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}}
        ]
    })
    assert guard._find_violation(h) is False  # 期望: 纯读操作永不触发


def test_no_user_message_window_skipped():
    guard = _load_guard()
    h = _hist(_asst("历史会话残片"), _asst_write())
    assert guard._find_violation(h) is False  # 期望: 无user消息无从界定新任务=放行


def test_liangpai_word_counts_as_marker():
    guard = _load_guard()
    h = _hist(_user(), _asst("先亮牌：我要改这一处。"), _asst_write())
    assert guard._find_violation(h) is False  # 期望: 明示亮牌=标记命中


def test_two_markers_only_still_violation():
    guard = _load_guard()
    h = _hist(_user(), _asst("方案是替换。决策是保留原文。"), _asst_write())
    assert guard._find_violation(h) is True  # 期望: 仅两要素命中(<3)=仍算裸写


def test_injection_carries_contract_and_window():
    guard = _load_guard()
    out = guard._build_injection()
    ctx = out.get("context", "")
    assert "先亮牌" in ctx  # 期望: 注入文本点名契约
    assert "3-5 分钟" in ctx  # 期望: 注入文本含自动推进窗口


def test_hook_returns_none_on_clean_history():
    guard = _load_guard()
    out = guard._on_pre_llm_call(
        session_id="test_brief_clean",
        conversation_history=_hist(_user(), _asst("只想问问情况")),
    )
    assert out is None  # 期望: 无写操作回合不注入
