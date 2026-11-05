"""Generation-time self-check for blame, hedging, laziness, and idealization."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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

_CHAT_TOKENS = frozenset({
    "你好", "您好", "早", "早上好", "上午好", "中午好", "下午好", "晚上好",
    "好", "好的", "嗯", "嗯嗯", "哦", "噢", "哈", "哈哈", "呵呵", "行", "OK", "ok",
    "收到", "明白", "了解", "知道啦", "知道", "懂了", "可以", "没问题",
    "谢谢", "谢谢哈", "多谢", "感谢", "辛苦了", "辛苦啦", "麻烦你了",
    "拜拜", "再见", "晚安", "加油", "牛", "牛逼", "厉害", "赞",
})


def _is_task_message(user_message: str) -> bool:
    """Return whether a message gets the generation-time self-check.

    Task requests are open-ended, so the default is to inject. Only a whole
    short message in the closed chat-acknowledgement set is exempt.
    """
    if not user_message:
        return False
    msg = str(user_message).strip()
    if not msg:
        return False
    if "?" in msg or "？" in msg:
        return True

    stripped = msg.strip("。，！!？?~～ \t\n")
    for tail in ("哈", "呀", "啦", "呢", "哦", "噢", "喽", "嘛", "吧", "嘿", "诶"):
        if stripped.endswith(tail) and len(stripped) > 1:
            stripped = stripped[: -len(tail)]
    stripped = stripped.strip("。，！!？?~～ \t\n")
    return stripped not in _CHAT_TOKENS


def _on_pre_llm_call(**kwargs: object) -> dict[str, str]:
    """Inject the self-check before a task response is generated."""
    user_message = kwargs.get("user_message", "")
    if not _is_task_message(user_message):
        return {}

    logger.debug(
        "discipline.blame_shield: inject self-check (first 50 chars=%s)",
        str(user_message)[:50],
    )
    return {"context": _SELF_CHECK_RULE}


def register(ctx: Any) -> None:
    """Register the blame-shield pre-LLM hook."""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("discipline.blame_shield registered——推责/免责自检注入就绪")
