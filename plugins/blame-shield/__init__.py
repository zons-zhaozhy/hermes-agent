"""
推责/免责话术拦截插件
======================

机制：pre_llm_call hook，每次 LLM 生成前注入自检规则到 user message。

设计原则：
1. 不做事后剥离/遮掩——那是在掩盖问题
2. 在 LLM 开始生成之前就植入自检要求，从源头消灭推责输出
3. 注入的规则要求 LLM 对自己的回复做"偷懒/推诿/过度免责自检"
4. 如果自检发现推责话术，LLM 必须自动重写回复，而不是简单删除

为什么是 pre_llm_call 而不是 transform_llm_output：
- transform_llm_output 只能返回字符串替换，无法触发重做
- pre_llm_call 注入规则到 user message，让 LLM 在生成时就遵守
- 这是从源头改变行为，不是事后修补
"""

import logging

logger = logging.getLogger(__name__)

# 注入到 user message 的自检规则
# 用中文，因为用户用中文交流
_SELF_CHECK_RULE = """\
[回复前自检——必检项，违反即自动重写]
在输出回复之前，对回复内容执行以下自检：

1. 推责自检：回复中是否包含"不是我改的"/"不是我的代码"/"不是我的问题"/"不是我写的"/"这个不是我负责的"/"不是我配置的"/"不是我引入的"等推卸诊断责任的话术？
   → 如果有：删除推责句子，直接从问题现象→根因追踪→修复方案→验证证据开始写。不需要解释代码归属。代码在视野内就是你的责任。

2. 免责自检：回复中是否包含"可能"/"应该"/"大概率"/"一般来说"等不基于实测验证的模糊断言，且该断言涉及技术事实（而非社交用语）？
   → 如果有：要么替换为实测证据（"[实测] xxx"），要么改为"未查证"。社交用语（"你可能需要..."）不拦截。

3. 偷懒自检：是否在解释"为什么不做"而不是直接动手做？
   → 如果有：停止解释，直接执行。

自检不通过时，不是简单剥离违规划句，而是用正确的认知重写整段内容。
"""

# 判断用户消息是否是任务/提问（而非纯闲聊）
# 只匹配明确的任务指令和提问，避免误触发
_TASK_PREFIXES = (
    "怎么", "为什么", "如何", "帮我", "检查", "修复", "查看",
    "分析", "测试", "验证", "跑一下", "帮我看看", "帮我查",
    "你为什么", "你怎么", "你能不能", "你到底",
)


def _is_task_message(user_message: str) -> bool:
    """判断用户消息是否是任务/提问（而非纯闲聊）。

    匹配规则：
    - 包含中文或英文问号
    - 以明确的任务/提问前缀开头（排除"你好"/"你觉得"等闲聊）
    """
    if not user_message:
        return False
    msg = user_message.strip()
    # 包含问号（中英文）
    if "?" in msg or "？" in msg:
        return True
    # 以任务指令开头
    for prefix in _TASK_PREFIXES:
        if msg.startswith(prefix):
            return True
    return False


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：在任务/提问场景下注入自检规则。"""
    user_message = kwargs.get("user_message", "")
    if not _is_task_message(user_message):
        return {}  # 纯闲聊不注入，避免干扰

    logger.debug("blame-shield: 注入自检规则 (user_message前50字: %s)", user_message[:50])
    return {"context": _SELF_CHECK_RULE}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("blame-shield 插件已注册——推责/免责话术拦截就绪")
