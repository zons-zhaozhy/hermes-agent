"""命中词的否定/引用语境判定——守卫负例面的单一事实源。

no_bypass / no_pushback 的 transform 与 tool 拦截都在回答同一个问题：命中的
词组是模型自己的断言，还是模型在**否定或引用**它？2026-11-10 CLI 事故里
「这次没有绕过这个问题，直接修了根因」被当成违规，于是往用户正文追加
一段训话——用户看到的就是「不说人话」。

判定只用窗口内否定词/引用标记，不做句法分析：窗口 8 字足以覆盖
「没有」「不存在」「禁止」「不允许」「我不会说「…」」这些实际措辞。
"""
from __future__ import annotations

import re

_PREFIX_WINDOW = 8
_SUFFIX_WINDOW = 12

# 前置否定/禁止/引用动词：命中词前 8 字内出现即为边界陈述
_PREFIX_NEGATION_RE = re.compile(
    r"(未|没|没有|不存在|禁止|不允许|不得|避免|不是|并非|不算|无|零|不)"
    r"|(宣称|主张|声称|所谓)"
)  # re-ok: 中文否定/引用词无 str 等价判定

# 后置否定：命中词后 12 字内出现「…的做法被否掉/被禁止」
_SUFFIX_NEGATION_RE = re.compile(
    r"(否掉|否定|推翻|驳回|不允许|禁止|不得|避免)"
)  # re-ok: 中文后置否定动词无 str 等价判定

# 紧邻的引号：被引用的是话术本身，不是断言
_QUOTE_CHARS = "「『“‘\"'"


def is_negated_or_quoted(text: str, match: re.Match[str]) -> bool:
    """Contract: Preconditions: text 为 str，match 为 text 上的匹配对象；
    Postconditions: 命中词处于否定/禁止/引用语境时返回 True（不是断言），
    否则 False。"""
    start, end = match.span()
    if start > 0 and text[start - 1] in _QUOTE_CHARS:
        return True
    prefix = text[max(0, start - _PREFIX_WINDOW):start]
    if _PREFIX_NEGATION_RE.search(prefix):
        return True
    suffix = text[end:end + _SUFFIX_WINDOW]
    return bool(_SUFFIX_NEGATION_RE.search(suffix))
