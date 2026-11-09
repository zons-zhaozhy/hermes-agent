"""blame-shield 注入门控回归——无问号祈使句/续作指令漏注入修复。

旧判定=任务前缀白名单（_TASK_PREFIXES），任务指令是开放集，枚举必然漏：
「给这个表加个字段」「继续」类编码会话最高频指令形态曾零注入，自检纪律
（含第 4 检理想化自检）对编码会话事实性失效。新判定=默认注入+闲聊封闭集豁免。

本文件锁两层：
  G1 门控方向：祈使句/续作指令/提问全部注入；问候/应答/致谢整句豁免。
  G2 注入内容：注入文本含第 4 检（理想化自检）。
"""
import sys
from pathlib import Path

from plugins import discipline
from plugins.discipline import blame_shield

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

mod = blame_shield


def test_imperative_coding_command_injects():
    # 期望: 无问号祈使句（编码会话最高频形态）注入
    assert mod._is_task_message("给这个表加个字段") is True  # 期望: True
    assert mod._is_task_message("把登录页按钮改成蓝色") is True  # 期望: True


def test_continuation_command_injects():
    assert mod._is_task_message("继续") is True  # 期望: 续作指令注入
    assert mod._is_task_message("接着改") is True  # 期望: 续作指令注入


def test_technical_question_injects():
    assert mod._is_task_message("这个接口为什么报错？") is True  # 期望: 工程提问保留自检
    assert mod._is_task_message("how does this build fail?") is True  # 期望: 英文工程提问注入


def test_ordinary_conversation_does_not_inject():
    assert mod._is_task_message("你还在工作吗") is False  # 期望: 状态确认不注入
    assert mod._is_task_message("我今天有点累") is False  # 期望: 情绪表达不注入
    assert mod._is_task_message("为什么慢？") is False  # 期望: 无工程对象的泛化提问不注入


def test_legacy_prefix_still_injects():
    assert mod._is_task_message("帮我看看这个报错") is True  # 期望: 旧白名单覆盖面保持


def test_chat_closed_set_exempt():
    for m in ("你好", "早上好呀~", "好的", "收到", "谢谢哈", "嗯", "ok", "OK", "辛苦了", "拜拜"):
        assert mod._is_task_message(m) is False  # 期望: 闲聊封闭集（含语气词/标点剥离后）豁免


def test_empty_no_inject():
    assert mod._is_task_message("") is False  # 期望: 空消息不注入
    assert mod._is_task_message("   ") is False  # 期望: 纯空白不注入


def test_hook_delivers_idealization_check():
    out = mod._on_pre_llm_call(user_message="给这个表加个字段")
    ctx = out.get("context") or ""
    assert "理想化" in ctx  # 期望: 注入文本含第 4 检
    assert "分支" in ctx  # 期望: 第 4 检的五类分支面在场


def test_self_check_rule_stays_bounded():
    assert len(mod._SELF_CHECK_RULE) <= 500  # 期望: 事前提示不超过 500 字符


class _OwnerCtx:
    def __init__(self) -> None:
        self.hooks: dict[str, list] = {}

    def register_hook(self, name: str, callback) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_system_prompt_section(self, section_id: str, renderer) -> None:
        return None


def test_discipline_owner_registers_blame_shield_self_check() -> None:
    ctx = _OwnerCtx()
    discipline.register(ctx)

    contexts = [
        result.get("context", "")
        for callback in ctx.hooks["pre_llm_call"]
        if (result := callback(user_message="给这个表加个字段", session_id="blame-owner"))
    ]

    assert any("回复前自检" in context and "理想化" in context for context in contexts)
