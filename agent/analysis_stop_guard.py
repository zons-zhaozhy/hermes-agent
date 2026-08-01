"""Analysis-then-stop guard.

Detects when the agent outputs analysis/conclusions as text-only without
following up with tool calls, and nudges it to continue executing.

Two-layer detection:
1. Heuristic pre-filter (cheap regex + structural analysis)
2. LLM-as-judge (via auxiliary_client) for semantic confirmation

Mirrors the pattern in read_think_gate._judge_investigation and
verification_stop.build_verify_on_stop_nudge.
"""

from __future__ import annotations

import logging
import random
import re  # noqa: required for multi-pattern keyword matching (Chinese/English unfinished heuristics)
from typing import Any

logger = logging.getLogger(__name__)

_MAX_NUDGES = 2

# Long analysis threshold: if text is longer than this AND follows tool
# results, it's likely an analysis dump rather than a direct answer.
_LONG_ANALYSIS_THRESHOLD = 120

# Heuristic patterns: if ANY of these match, the response looks unfinished.
_UNFINISHED_PATTERNS: list[str] = [
    # ── Chinese: forward-looking action language ──
    r"现在可以继续",
    r"下一步",
    r"接下来",
    r"然后我",
    r"需要.*做",
    r"我应该",
    r"我将",
    r"我会",
    r"让我",
    r"可以继续",
    r"继续执行",
    r"接着",
    r"首先.*然后",
    r"建议",        # "建议修复" = analysis without action
    r"方案[是为]",   # "方案是..." = described but not executed
    # ── English ──
    r"\bnow\b.*\b(continue|proceed)",
    r"\bnext\b.*\bstep",
    r"\bI will\b",
    r"\bLet me\b",
    r"\bgoing to\b",
    r"\bshould\b.*\b(fix|update|change|modify|create)",
    r"\bwe need to\b",
    r"\bto fix this\b",
    r"\bthe fix (is|would be)\b",
]

# Patterns that indicate a genuine final response — skip everything.
_FINAL_RESPONSE_PATTERNS: list[str] = [
    r"^完成[。！]?$",
    r"^done[。！.]?$",
    r"已修复.*[。！]?$",
    r"已完成.*[。！]?$",
    r"验证通过.*[。！]?$",
    r"全部通过.*[。！]?$",
    r"测试通过.*[。！]?$",
    r"你问的是",
]

# Structural markers: markdown headers, numbered lists, bullet lists,
# code blocks — these suggest an analysis report, not a direct answer.
_STRUCTURE_MARKERS = [
    r"^#{1,4}\s",           # markdown headers # ## ### ####
    r"^\d+[\.\)]\s",        # numbered list 1. 2) 3.
    r"^[-*]\s",             # bullet list - *
    r"^```",                # code block
    r"^\|.*\|",             # table row
    r"^>\s",                # blockquote
]
_STRUCTURE_PATTERN = re.compile(
    r"|".join(_STRUCTURE_MARKERS), re.MULTILINE
)

# Progressive nudge messages — escalate urgency.
_NUDGE_MESSAGES = [
    "[System: Your last response contained analysis or findings but no "
    "tool calls to act on them. Continue executing — issue the next "
    "tool call now. Do not repeat the analysis.]",

    "[System: This is the FINAL nudge. You described analysis or a plan "
    "but did not execute it. You MUST issue tool calls NOW to act on "
    "your analysis. A text-only response after this will be treated as "
    "task failure.]",
]


def _count_structure_markers(text: str) -> int:
    """Count structural markdown elements — headers, lists, code blocks."""
    return len(_STRUCTURE_PATTERN.findall(text))


def _has_unfinished_heuristics(text: str) -> bool:
    """Cheap pre-filter: determine if the response looks unfinished.

    Contract:
      Preconditions: text is a non-None string (may be empty)
      Postconditions: returns True iff layers 1-3 collectively indicate
        the response is analysis-without-action; False means genuinely final.

    Layer 1: Final-response patterns → False (genuinely done)
    Layer 2: Direct unfinished keyword match → True
    Layer 3: Structural analysis (long text with markdown structure
             following tool results) → True
    """
    stripped = text.strip()

    # Layer 1: explicit final patterns
    for pat in _FINAL_RESPONSE_PATTERNS:
        if re.search(pat, stripped):
            return False

    # Layer 2: direct keyword match
    for pat in _UNFINISHED_PATTERNS:
        if re.search(pat, stripped, re.IGNORECASE):
            return True

    # Layer 3: long structured analysis — if the text is long AND contains
    # markdown structure (headers/lists/code blocks), it's likely an
    # analysis report rather than a direct conversational answer.
    if len(stripped) > _LONG_ANALYSIS_THRESHOLD:
        structure_count = _count_structure_markers(stripped)
        if structure_count >= 3:
            return True

    return False


def _prev_turn_has_tool_results(messages: list[dict]) -> bool:
    """Check if there are tool results in the recent conversation context.

    Contract:
      Preconditions: messages is a list of message dicts (may be empty)
      Postconditions: returns True iff a tool result exists within 5
        messages before the last assistant message.

    Walk backwards from the end to find the last assistant message,
    then check if tool results precede it — indicating the assistant
    just consumed tool output but responded with text only.
    """
    if len(messages) < 2:
        return False
    idx = len(messages) - 1
    # Find last assistant message
    while idx >= 0:
        if messages[idx].get("role") == "assistant":
            break
        idx -= 1
    else:
        return False
    if idx < 1:
        return False
    # Check: any tool message in the 5 messages before the assistant turn
    look_back = max(0, idx - 5)
    for i in range(look_back, idx):
        if messages[i].get("role") == "tool":
            return True
    return False


def _llm_judge(task: str, response: str) -> bool:
    """Run LLM judge. Returns True if UNFINISHED (should nudge), False if FINISHED."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client("analysis_stop_judge")
        if client is None or not model:
            logger.debug("analysis-stop guard: no auxiliary client — heuristic only")
            # No client: trust heuristics (already matched unfinished patterns)
            return True

        prompt = _JUDGE_PROMPT.format(
            task=(task or "")[:500],
            response=response[:2000],
        )
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
            timeout=10,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = (result.choices[0].message.content or "").strip().upper()
        logger.info("analysis-stop guard: judge verdict = %s", raw[:20])
        return "UNFINISHED" in raw

    except Exception:
        logger.warning("analysis-stop guard: LLM judge failed — heuristic only", exc_info=True)
        return True  # trust heuristics


def _build_nudge(nudge_count: int) -> str:
    """Build a progressive nudge message. Escalates urgency with each attempt.

    Contract:
      Preconditions: nudge_count >= 0
      Postconditions: returns a non-empty string from _NUDGE_MESSAGES,
        clamped to the last message for count beyond array bounds.
    """
    assert nudge_count >= 0, f"nudge_count must be non-negative, got {nudge_count}"
    idx = min(nudge_count, len(_NUDGE_MESSAGES) - 1)
    return _NUDGE_MESSAGES[idx]


def check_analysis_stop(
    messages: list[dict[str, Any]],
    assistant_content: str,
    finish_reason: str | None,
    user_message: str,
    *,
    nudge_count: int = 0,
) -> str | None:
    """Check if the agent stopped after analysis without acting.

    Contract:
      Preconditions:
        - messages is a list of message dicts (may be empty)
        - assistant_content is the text of the final assistant response
        - finish_reason is the provider's finish_reason or None
        - nudge_count >= 0 (reset per turn by turn_context)
      Postconditions:
        - returns None if the response is final or nudge limit reached
        - returns a non-empty nudge string if the response looks unfinished
          AND the nudge limit (_MAX_NUDGES) has not been reached

    Returns a synthetic nudge string if the agent should continue, None otherwise.

    Integration point: call this in conversation_loop.py right before
    the turn exit (after verify-on-stop and kanban-stop guards),
    following the same pattern as those nudge mechanisms.
    """
    if nudge_count >= _MAX_NUDGES:
        return None

    # Only check text-only responses (no tool calls)
    if finish_reason == "tool_calls":
        return None

    if not assistant_content or not assistant_content.strip():
        return None

    # Must have tool results in recent context
    if not _prev_turn_has_tool_results(messages):
        return None

    # Heuristic pre-filter — skip LLM judge if no unfinished signals
    if not _has_unfinished_heuristics(assistant_content):
        return None

    # LLM-as-judge
    is_unfinished = _llm_judge(user_message, assistant_content)
    if not is_unfinished:
        return None

    logger.info(
        "analysis-stop guard: nudging agent to continue (nudge %d/%d, content=%d chars)",
        nudge_count + 1, _MAX_NUDGES, len(assistant_content),
    )
    return _build_nudge(nudge_count)


# Judge prompt — include structural hints for better accuracy.
_JUDGE_PROMPT = """\
You are a brief judge. Decide if the assistant's response is FINISHED or UNFINISHED.

UNFINISHED means: the assistant did analysis/research but should continue with \
tool calls (reading files, editing code, running commands) — the task is not complete.

FINISHED means: the assistant directly answered the user's question, reported \
a completed action with results, or asked a clarifying question.

Key signals of UNFINISHED:
- Contains analysis, findings, or a plan but no tool calls to act on them
- Ends with forward-looking language ("next step", "I will check", etc.)
- Describes what needs to be done but hasn't done it yet
- Outputs a structured report (headers, lists, tables) after tool results \
without following up with execution

Key signals of FINISHED:
- Directly answers a user's question with factual information
- Reports a completed action and its verified result
- Presents a final conclusion with evidence
- Asks the user a clarifying question

User's original request: {task}

Assistant's last response (text-only, no tool calls):
---
{response}
---

Reply with exactly one word: FINISHED or UNFINISHED"""
