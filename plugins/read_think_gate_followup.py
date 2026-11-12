"""Status-followup detection for the read-think gate's intent channels.

Split out of ``plugins/read_think_gate/gate.py`` (capped at 2,000 lines by the
code-health ratchet) because this is one self-contained topic: a pure,
model-free reading of a user message that answers "is this a follow-up asking
why something expected was not handled?".

Nothing here imports the gate. Status follow-ups read like questions (so the
semantic classifier calls them read_only), but refusing writes on them
dead-locks the whole turn — the cheapest input to type earns the most
expensive refusal. Both the LLM-channel downgrade and the session-authorization
retention path consume the same verdict, so it must stay the single
implementation.

Contract:
    Preconditions: the function takes a ``str`` (may be empty or None).
    Postconditions: pure string analysis; no I/O, no exceptions, no state;
      returns True iff the lowercased text contains at least one phrase in
      ``STATUS_FOLLOWUP_TERMS``.
"""

from __future__ import annotations

# 状态追问短语——语义上是问句，但语境指向「一件事本该被处理而未被处理」。
# 刻意不收「怎么回事/为什么」这类纯信息问句，它们保持 read_only。
STATUS_FOLLOWUP_TERMS = (
    "咋回事", "咋了", "咋整", "咋办",
    "为啥没", "为啥不", "为啥还", "为啥一直",
    "为什么没", "为什么不", "为什么还", "为什么一直",
    "怎么没", "怎么不", "怎么还", "怎么一直",
    "没处理", "没人管", "没人处理", "没人理",
    "还没好", "还没弄", "还没搞定", "还没回复", "怎么还没", "为啥还没",
    "卡住了", "卡着", "还在跑吗",
    "why hasn't", "why hasn`t", "why isn't", "why isn`t",
    "still not", "not handled", "no one handled",
)


def is_status_followup(text: str | None) -> bool:
    """判定消息是否为状态追问（追问处理进度/为何未处理/是否卡住）。"""
    lowered = (text or "").lower()
    return any(term in lowered for term in STATUS_FOLLOWUP_TERMS)
