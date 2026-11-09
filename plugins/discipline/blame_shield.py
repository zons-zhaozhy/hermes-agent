"""Generation-time self-check for blame, hedging, laziness, and idealization."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


_SELF_CHECK_RULE = """\
[回复前自检]
1. 不写代码归属推责；直接给现象、根因、修复和验证证据。
2. 技术事实不用「可能/应该/大概」替代验证；无证据写「未查证」。
3. 能执行的不解释为什么不执行。
4. 交付代码时先做理想化自检，再排查非理想分支：非法或缺失输入、环境失败、并发重复、
对端失败、资源不足；每项显式处理、报错阻断或说明不可达理由。"""

_CHAT_TOKENS = frozenset({
    "你好", "您好", "早", "早上好", "上午好", "中午好", "下午好", "晚上好",
    "好", "好的", "嗯", "嗯嗯", "哦", "噢", "哈", "哈哈", "呵呵", "行", "OK", "ok",
    "收到", "明白", "了解", "知道啦", "知道", "懂了", "可以", "没问题",
    "谢谢", "谢谢哈", "多谢", "感谢", "辛苦了", "辛苦啦", "麻烦你了",
    "拜拜", "再见", "晚安", "加油", "牛", "牛逼", "厉害", "赞",
})
_TASK_HINT_RE = re.compile(
    r"代码|文件|报错|错误|bug|测试|接口|函数|数据库|字段|配置|部署|构建|编译|"
    r"服务|页面|样式|日志|性能|分支|提交|仓库|build|git|api|http|python|"
    r"typescript|javascript|java|vue|react|pytest|ci|pr",
    re.IGNORECASE,
)
_TASK_ACTION_RE = re.compile(
    r"给|把|帮我|继续|接着|开始|写|建|创建|改|修|重构|实现|开发|迁移|删除|"
    r"清理|配置|安装|执行|测试|部署|提交|生成|画|做"
)


def _is_task_message(user_message: str) -> bool:
    """Return whether a message gets the generation-time self-check.

    Engineering requests contain a technical object or an editing action. Casual
    status, mood, and generic questions do not need this self-check.
    """
    if not user_message:
        return False
    msg = str(user_message).strip()
    if not msg:
        return False
    stripped = msg.strip("。，！!？?~～ \t\n")
    for tail in ("哈", "呀", "啦", "呢", "哦", "噢", "喽", "嘛", "吧", "嘿", "诶"):
        if stripped.endswith(tail) and len(stripped) > 1:
            stripped = stripped[: -len(tail)]
    stripped = stripped.strip("。，！!？?~～ \t\n")
    if stripped in _CHAT_TOKENS:
        return False
    return bool(_TASK_HINT_RE.search(msg) or _TASK_ACTION_RE.search(msg))


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
