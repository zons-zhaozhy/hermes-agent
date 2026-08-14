"""
回复确定性检查插件
==================
transform_llm_output 钩子：
每次 LLM 生成回复后，扫描文本中的不确定词汇（可能、也许、大概、猜测、假设等）。
如果检测到，在回复末尾追加自检警告；否则透传。

安装后需在 config.yaml plugins.enabled 中添加 "reply-certainty-checker"。
"""

import re

# 不确定词汇黑名单（中英文）
UNCERTAIN_PATTERNS = [
    # 中文
    r"可能",
    r"也许",
    r"或许",
    r"大概",
    r"大约",
    r"猜测",
    r"假设",
    r"假定",
    r"按理说",
    r"应该是",
    r"说不定",
    r"没准",
    r"多半",
    r"兴许",
    r"不见得",
    r"说不准",
    # 英文（大小写不敏感）
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\bpresumably\b",
    r"\bsupposedly\b",
    r"\bconceivably\b",
    r"\bmight\b",
    r"\bcould\s+be\b",
    r"\bI\s+(think|believe|assume|guess|suppose|suspect)\b",
    r"\bmy\s+(best\s+)?guess\b",
    r"\bit\s+(seems|appears)\b",
]

# 编译正则（忽略大小写）
_CMPD = re.compile("|".join(UNCERTAIN_PATTERNS), re.IGNORECASE)


# 安全的通行词上下文——这些场景下允许 "可能"
# 例如：技术方案中的"可能的原因"、错误分析中的"可能是由于"
SAFE_CONTEXTS = [
    "可能的原因",
    "可能是由于",
    "可能的原因包括",
    "可能的根因",
    "可能的问题",
]


def _is_safe_context(text: str, match_start: int) -> bool:
    """检测匹配位置是否在安全上下文内"""
    for ctx in SAFE_CONTEXTS:
        idx = text.find(ctx)
        if idx >= 0 and abs(idx - match_start) <= 10:
            return True
    return False


def _has_uncertainty(text: str) -> list:
    """扫描文本，返回所有匹配的不确定词汇位置"""
    matches = []
    for m in _CMPD.finditer(text):
        if not _is_safe_context(text, m.start()):
            matches.append((m.start(), m.group()))
    return matches


def register(ctx):
    """注册 transform_llm_output 钩子"""

    def check_certainty(response_text, session_id=None, model=None, platform=None, **kwargs):
        if not response_text:
            return None

        matches = _has_uncertainty(response_text)
        if not matches:
            return None  # 透传，不做任何修改

        # 收集匹配词（去重）
        words_found = sorted(set(m[1] for m in matches))

        # 在回复末尾追加自检红牌
        warning = (
            f"\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ 自检失败：检测到不确定词汇 "
            f"({'/'.join(words_found)})\n"
            f"请重新确认事实后再发送。\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return response_text + warning

    ctx.register_hook("transform_llm_output", check_certainty)
