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

4. 理想化自检：本次回复若交付代码实现，是否基于理想情况假设（输入恒合法/环境恒正常/依赖恒可用/执行恒成功）而未附分支处置清单？
   → 如果是：停止交付，先逐类排查非理想分支（输入面：非法/缺失/空值/量级；环境面：平台/版本/时区/编码；时序面：并发/幂等/乱序；对端面：网络失败/超时/异常返回；资源面：不足/耗尽），每条给出处置——显式处理、报错阻断、或判定本场景不可达并记录理由——且测试覆盖非理想路径而非只测理想路径，再输出。

自检不通过时，不是简单剥离违规划句，而是用正确的认知重写整段内容。
"""

# 闲聊豁免集（封闭集）：整句剥离语气词/标点后命中的短应答不注入
_CHAT_TOKENS = frozenset({
    "你好", "您好", "早", "早上好", "上午好", "中午好", "下午好", "晚上好",
    "好", "好的", "嗯", "嗯嗯", "哦", "噢", "哈", "哈哈", "呵呵", "行", "OK", "ok",
    "收到", "明白", "了解", "知道啦", "知道", "懂了", "可以", "没问题",
    "谢谢", "谢谢哈", "多谢", "感谢", "辛苦了", "辛苦啦", "麻烦你了",
    "拜拜", "再见", "晚安", "加油", "牛", "牛逼", "厉害", "赞",
})


def _is_task_message(user_message: str) -> bool:
    """任务消息判定——默认注入，仅整句闲聊（封闭集）豁免。

    任务指令是开放集（祈使句/续作/提问形态不可枚举），白名单枚举必然漏——
    「给这个表加个字段」类无问号祈使句曾零注入。故方向反转：默认注入，
    只有整句剥离标点/语气词后落在闲聊封闭集内的短消息才豁免。
    """
    if not user_message:
        return False
    msg = str(user_message).strip()
    if not msg:
        return False
    # 包含问号（中英文）→ 提问，注入
    if "?" in msg or "？" in msg:
        return True
    # 整句闲聊豁免：剥离标点/空白/常见语气词后仍是闲聊词才豁免
    stripped = msg.strip("。，！!？?~～ \t\n")
    for tail in ("哈", "呀", "啦", "呢", "哦", "噢", "喽", "嘛", "吧", "嘿", "诶"):
        if stripped.endswith(tail) and len(stripped) > 1:
            stripped = stripped[: -len(tail)]
    stripped = stripped.strip("。，！!？?~～ \t\n")
    return stripped not in _CHAT_TOKENS


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
