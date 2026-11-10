"""ReadThink Gate — turn 级结构性分离：推理阶段 → 执行阶段。

不是事后纠错（"你没思考，回去想"），而是结构引导（"先调查，再分析，执行权限自动解锁"）。

两阶段门控：
  推理阶段（turn 开始，gate 未解锁）：
    - 只读工具（read/search/web）→ 放行（调查本身就是思考的体现）
    - 执行类工具（write/patch/terminal）→ 拦截，引导先调查
  执行阶段（gate 解锁后）：
    - 全部工具放行

解锁条件（满足任一）：
  1. 直接输出充分分析文本（content >= min_reasoning_chars）
  2. 用户明确授权修改
  3. 完成目标文件、调用关系、测试/配置、历史意图和验证方案的全面调查

轮数、等待时间和“目标文件已读或不存在”都不是修改授权或调查完成证明。

任务分级（复杂度自适应）：
  硬证据意图（优先）：明确禁止/执行/确认由交互状态机判定。
  LLM 语义分类（兜底）：仅处理缺少硬证据的复杂语境。
  关键词 fallback（降级）：LLM 不可用时保守判定为 ambiguous/read-only/modify。
  三级：simple / normal / complex

与 tool_guardrails 的区别：
  - guardrails 检测"循环失败"（重复调用同一工具失败）
  - read_think_gate 执行"推理期/执行期分离"（先调查再动手）

生命周期：调查证据 per-turn，由 build_turn_context 重置；同一 session 的用户修改授权
跨 turn 持久，直到用户给出新的只读意图或显式撤销修改。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# 四轴闸门 marker 文件路径——ReadThinkGate 写入，pre_tool_call 插件读取。
# 四轴全部验证通过时，写入此文件作为双防线同步信号。
# Path resolved lazily per access: get_hermes_home() is profile-aware and can
# be redirected by tests mid-process (HERMES_HOME); a module-level constant
# would freeze the first resolution and break profile isolation (PR #3575
# class of bugs).


def _marker_session_slug(session_id: str) -> str:
    """会话桶 slug——写读两端唯一实现（plugins/guards/four_axis.py import 本函数）。

    文件名安全：仅保留字母数字与 -_.，其余字符替换为 _；空值归桶 ``nosession``。
    """
    slug = "".join(c if c.isalnum() or c in "-_." else "_" for c in session_id)
    return slug or "nosession"


def _four_axis_marker_path(session_id: str = "") -> Path:
    """解析四轴 marker 文件路径（profile-aware，每次调用现算）。

    Returns:
        Path: hermes_home/cache/four_axis_gate_{pid}_{session}.json
    """
    from hermes_constants import get_hermes_home

    # 文件名带 PID + session_id：PID 隔离跨进程；session_id 隔离同进程多会话——
    # 网关单进程服务多会话（含 cron），此前按 PID 共享单文件时，任一会话的
    # turn-start（reset_for_turn → _clear_four_axis_marker）会清掉其他会话已
    # 凑齐的四轴 marker，导致证据齐备仍被副防线反复拦截（与 diff_debt 按
    # session 分桶同类）。写读两端（本模块 + plugins/guards/four_axis.py）经
    # 同一 slug 函数解析路径，同会话内必然一致。
    return (
        get_hermes_home() / "cache"
        / f"four_axis_gate_{os.getpid()}_{_marker_session_slug(session_id)}.json"
    )

# 四轴关键词表（2026-08-26 恢复：e12e46fd3 删除后 use_llm_judge=false 时四轴无解锁路径）。
# 子串匹配 O(n) 无回溯（历史上 re.search 曾致灾难性回溯 CPU 100%，禁用正则）。
_FOUR_AXIS_KEYWORDS: dict[str, list[str]] = {
    "影响面": [
        "影响面",  # 2026-09-26: 轴名本身入表——拦截消息要求「影响面清单」但模型常写
        # 「影响面=…」等自然变体，裸轴名缺席致 judge 判 3/4 假拦截五轮（实测 0926）。
        # 轴名出现即证明模型在输出该轴，语义零误报。
        "[源码确认]", "[搜索推断]",
        "caller", "consumer", "importer",
        "调用方", "依赖此接口", "调用点", "受影响",
        "影响面清单", "下游",
    ],
    "原意图": [
        "原意图",  # 2026-09-26: 同上，轴名入表
        "git log",
        "前置条件", "后置条件", "副作用约定", "不变量",
        "原意图溯源",
    ],
    "根因": [
        "症状位置", "根因位置",
        "根因", "上游", "数据源头", "配置源",
        "阻断方案", "入口校验", "配置强制", "启动报错",
        "根因定位",
    ],
    "风险": [
        "风险",  # 2026-09-26: 同上，轴名入表（「根因」轴本就含轴名，此为补齐另三轴）
        "静默数据损坏", "向后不兼容", "缓存失效",
        "并发竞争", "异常路径被吞", "第三方依赖超时",
        "触发条件", "影响范围", "可恢复",
        "风险矩阵",
    ],
}

# 四轴检测支持 LLM 语义判断（_judge_investigation 的 A/B/C/D 四维度
# 完全覆盖 影响面/原意图/根因/风险 的语义空间）与关键词内容扫描双路径。
# 保留标签列表供 _missing_axes 和 _build_block_message 展示信息。
_FOUR_AXIS_LABELS = ["影响面", "原意图", "根因", "风险"]


def _clear_four_axis_marker(session_id: str = "") -> None:
    """清除四轴 marker 文件——每个 turn 开始时调用（只清本会话的桶）。

    同时清扫陈旧的跨进程/跨会话 marker（four_axis_gate_*.json）：进程退出或
    会话结束后其命名的 marker 不再被任何 turn-start 清除，避免长期运行后
    cache 目录堆积。
    """
    try:
        _marker = _four_axis_marker_path(session_id)
        if _marker.exists():
            _marker.unlink()
    except Exception:
        logger.warning("failed to clear four-axis marker file", exc_info=True)
    try:
        _cache_dir = _four_axis_marker_path().parent
        _cutoff = time.time() - 86400
        for _stale in _cache_dir.glob("four_axis_gate_*.json"):
            try:
                if _stale.stat().st_mtime < _cutoff:
                    _stale.unlink()
            except OSError:
                continue
    except Exception:
        logger.warning("failed to sweep stale four-axis marker files", exc_info=True)

# 默认门控的执行类工具——只覆盖代码编辑工具。
# terminal/browser/delegate_task/cronjob/process 等运维交互工具不在默认门控范围——
# 它们不是代码编辑，拦截它们会把 MySQL 运维、docker、git、测试等操作误伤。
# 用户可通过 config.yaml → read_think_gate.gated_tools 扩展列表。
# 写工具语义 owner=plugins/guards/tool_classification（C2 收敛）。
from plugins.guards.tool_classification import (
    STANDARD_WRITE_TOOLS as GATED_TOOL_NAMES,
)
from plugins._shared_state import get_session_state

# Shared with guards.pre_write: the batch gate publishes this turn's verdict,
# and the per-tool guard re-checks it before dispatching a write.
_STATE_NAMESPACE = "read_think_gate"
INTENT_READ_ONLY = "read_only"
INTENT_MODIFY = "modify"
INTENT_AMBIGUOUS = "ambiguous"
_INTENT_VALUES = frozenset({INTENT_READ_ONLY, INTENT_MODIFY, INTENT_AMBIGUOUS})

# 只读调查工具白名单——只有这些工具的调用才算真正的"调查"。
# 其他非 gated 工具（如 memory、process、cronjob 等）不算调查行为。
# browser_exec 于 2026-10-27 替换旧 browser_* 分体工具名（旧名已不存在，
# 锈化=四轴证据凑齐更慢→更多误拦）；session_search/vision_analyze 同属
# 纯读取证。
READ_ONLY_INVESTIGATION_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "browser_exec",
        "vision_analyze",
        "session_search",
        "skill_view",
    }
)

# terminal 写语义的唯一实现（2026-11-10 拆出：gate.py 已到 2000 行上限）。
# 命令行写检测与只读取证判定都取自同一模块，勿在此另写一份。
from plugins.read_think_gate_terminal import (
    _terminal_is_investigation,
    _terminal_writes_file,
)


# ── 任务复杂度检测 ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ComplexityProfile:
    """单个复杂度级别的门控参数。"""

    min_read_only_calls: int
    min_reasoning_chars: int
    min_reflection_chars: int
    max_reasoning_rounds: int


# 默认复杂度对应关系
COMPLEXITY_PROFILES: dict[str, ComplexityProfile] = {
    "simple": ComplexityProfile(
        min_read_only_calls=1, min_reasoning_chars=100,
        min_reflection_chars=30, max_reasoning_rounds=3,
    ),
    "normal": ComplexityProfile(
        min_read_only_calls=2, min_reasoning_chars=200,
        min_reflection_chars=50, max_reasoning_rounds=6,
    ),
    "complex": ComplexityProfile(
        min_read_only_calls=3, min_reasoning_chars=300,
        min_reflection_chars=100, max_reasoning_rounds=8,
    ),
}

# ── 关键词 fallback（LLM 不可用时降级用） ──────────────────────────

_FALLBACK_COMPLEX: frozenset[str] = frozenset(
    {
        "重构", "refactor", "restructure", "overhaul", "rearchitecture",
        "架构设计", "系统设计", "整体方案", "技术选型",
        "从零", "从0", "from scratch",
        "多服务", "跨服务",
        "架构改造", "微服务",
        "全面审计", "全面重构", "全面改造",
        "architect", "comprehensive",
    }
)

_FALLBACK_SIMPLE: frozenset[str] = frozenset(
    {
        "修typo", "typo", "改一行", "换个", "改个名", "改个",
        "加个注释", "删一行", "拼写", "加个空格", "改一个字",
        "fix typo", "one line", "trivial", "nit",
        "quick fix", "cosmetic",
    }
)


def _fallback_detect(text: str) -> str:
    """关键词 fallback——只在 LLM 不可用时使用。"""
    for trigger in _FALLBACK_COMPLEX:
        if trigger in text:
            return "complex"
    for trigger in _FALLBACK_SIMPLE:
        if trigger in text:
            return "simple"
    return "normal"


# ── 用户意图分类 ───────────────────────────────────────────────────

_INTENT_CACHE_MAX = 256
_intent_cache: dict[str, str] = {}

_INTENT_CLASSIFY_PROMPT = """判断用户最新消息要求的操作类型。只回答一个 JSON 对象：
{"read_only": true/false, "modify": true/false}

定义：
- read_only=true：用户在询问、解释、分析、诊断、审查、评估、比较或制定方案；不要修改文件/代码/配置。
- modify=true：用户明确要求修复、修改、实现、增加、删除、重构、创建或更新代码/配置/文件。
- 两者都为 false：意图不明确、纯闲聊、只想确认事实，或无法确定用户是否授权修改。

判定规则：
- 只有 user 消息可以授权修改；assistant 消息只用于理解指代和用户对具体方案的确认。
- 用户说“同意/可以/继续/按这个来”且明确承接一个具体的修改方案时，判 modify=true。
- 用户只是要求看、查、解释、排查原因或给出建议时，read_only=true；即使内容涉及“怎么改”。
- 用户明确说“不要修改/只看/只分析/别动代码”时，read_only=true。
- 不完整修改授权（例如“先看看怎么改”）不能判 modify。
- 不确定时两个键都回答 false，禁止猜成 modify。

补充三条实测误判（都是"没写修改动词、但确实在要活儿/或在把活儿交出来"）：
- 用户下达执行指令时判 modify=true，即使没有"修改/修复"这类动词：
  例如"现在开始"、"按这个顺序推进"、"开工"、"继续做"、"把剩下的补完"。
- 用户把处置权交出来时不能判 read_only=true——例如"你决定"、"你看看能怎么办"、
  "随便你"、"这个测试挂了，你看看"：这是把问题交给你处理，两个键都回答 false。
- 反问式提议（"或许应该拆成两个文件？"、"能改成异步吗"）问的是可行性，不是授权，
  两个键都回答 false。但问"怎么做"的（怎么/如何/需要哪些步骤）仍是 read_only=true。

只回答 JSON。"""

# 词表是有方向的：_MODIFY_TERMS 只能在「没有疑问形态」时**授予**（放行后仍要过调查
# 契约），_READ_ONLY_TERMS 只在 _is_confirmation 里做**否决**（把提问排除在确认之外），
# 永远不再凭它拒绝写入——拒绝只认显式禁止。见 _fallback_detect_intent 的说明。
_MODIFY_TERMS = (
    "修复", "修好", "修改", "改成", "改掉", "改一下", "修一下", "调整",
    "实现", "增加", "新增", "删除", "移除", "重构", "创建", "新建",
    "补上", "补测试", "更新", "替换", "重写", "生成", "保存到", "导出",
    "写到", "写入", "输出到", "整理成",
    "fix ", "modify", "implement", "add ", "remove ", "delete ",
    "refactor", "create ", "update ", "replace ", "rewrite", "write ",
    "generate ", "save ", "export ",
)
# 只用于否决：一句话含这些词时不能被当成对方案的确认（「看看能不能改」不是授权）。
_READ_ONLY_TERMS = (
    "看看", "看一下", "查看", "检查", "确认", "解释", "分析", "审查", "评估", "为什么",
    "怎么回事", "排查", "调查", "定位原因", "给出建议", "给个方案",
    "如何", "怎么", "是否", "能不能", "可以吗",
    "explain", "analy", "review", "investigate", "diagnose",
    "why ", "how ", "what ", "whether ",
)
_EXPLICIT_NO_MODIFY_TERMS = (
    "不要修改", "不要改", "不要修", "不要动", "别修改", "别改", "别修", "别动",
    "不许改", "不准改", "不用改", "不必改",
    "不要创建", "不要新建", "别创建", "别新建",
    "只看", "只分析", "仅分析", "只调查", "仅调查",
    "只汇报", "仅汇报", "只报告", "仅报告",
    "do not modify", "don't modify", "do not create", "don't create",
    "read-only", "read only", "report only",
)

_STRONG_MODIFY_TERMS = (
    "请修复", "帮我修", "帮忙修", "修复一下", "修复吧", "修好", "改掉",
    "请修改", "帮我改", "帮忙改", "改一下", "修一下", "直接修", "直接改",
    "立即修复", "马上修复", "开始修改", "开始修复",
    "please fix", "fix this", "fix it", "make the change",
)

_CONFIRMATION_TERMS = (
    "同意", "可以", "好", "行", "没问题", "允许", "按这个来", "按这个做", "照这个做",
    "就这么办", "就这么改", "动手吧", "开始吧", "执行吧", "实施吧",
    "批准", "继续",
    "go ahead", "proceed", "do it", "approved", "sounds good",
)
_EXECUTION_CONFIRMATION_TERMS = (
    "确认执行", "执行确认", "按方案执行", "按这个方案执行",
)
_EXACT_CONFIRMATION_TERMS = frozenset({"确认", "执行", "好", "好的", "行", "行的"})
_CONFIRMATION_CHECK_TERMS = frozenset({
    "确认一下", "请确认一下", "我确认一下", "确认下", "请确认下", "我确认下",
})
_OPERATIONAL_FEEDBACK_TERMS = (
    "啥情况", "进度", "进展", "怎么样", "跑批", "汇报", "反馈", "闷头",
    "定时说明",
    "status", "progress", "update me",
)

_PROPOSAL_TERMS = (
    *_MODIFY_TERMS,
    "方案", "建议", "计划", "准备", "下一步", "我将", "我会",
    "plan", "proposal", "next step", "i will", "i'll",
)


_AUTH_REQUEST_FRAME = (
    "[GATE: assistant 上一轮因等待用户授权而拦截，以下为该请求的结尾原文，"
    "用户最新消息是对此请求的回应]\n{}"
)


def _message_text(content: object) -> str:
    """Extract textual content from a string or OpenAI-style content parts."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ).strip()


def _is_confirmation_check(message: str) -> bool:
    """Recognize a request to verify details rather than approve execution."""
    normalized = "".join(char for char in (message or "").strip().lower() if char.isalnum())
    return normalized in _CONFIRMATION_CHECK_TERMS


def _last_assistant_proposal(conversation_history: list[dict] | None) -> bool:
    """Return whether the latest assistant text contains a concrete proposal.

    Assistant text cannot grant permission, but it supplies the referent for a
    terse user confirmation such as "同意" or "按这个做".
    """
    for msg in reversed(conversation_history or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text = _message_text(msg.get("content")).lower()
        return bool(text) and any(term in text for term in _PROPOSAL_TERMS)
    return False


def _is_confirmation(message: str, conversation_history: list[dict] | None) -> bool:
    """Recognize terse approval only when it refers to an actual proposal.

    句级判定：一条消息里授权句与追问句可以共存。切句时保留句末标点，
    问号分隔出的句子整体视为疑问句；单字确认词只允许整句匹配，避免
    「不行」因为包含「行」而被误判为授权。
    """
    text = (message or "").strip().lower()
    if not text:
        return False
    parts = _SENTENCE_SPLIT_RE.split(text)
    for index, sentence in enumerate(parts):
        sentence = sentence.strip()
        if not sentence:
            continue
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        if delimiter in {"?", "？"}:
            continue
        normalized = "".join(char for char in sentence if char.isalnum())
        if not normalized or len(normalized) > 40:
            continue
        if normalized.endswith(("吗", "么", "呢")):
            continue
        if any(term in sentence for term in _EXPLICIT_NO_MODIFY_TERMS):
            continue
        if normalized in _EXACT_CONFIRMATION_TERMS:
            return _last_assistant_proposal(conversation_history)
        if any(term in normalized for term in _EXECUTION_CONFIRMATION_TERMS):
            return _last_assistant_proposal(conversation_history)
        if any(term in sentence for term in _READ_ONLY_TERMS):
            continue
        if any(term in normalized for term in _CONFIRMATION_TERMS if len(term) > 1):
            return _last_assistant_proposal(conversation_history)
    return False


# 句级切分并保留分隔符：问号本身决定该句是疑问句，不能在丢弃标点后再猜测。
_SENTENCE_SPLIT_RE = re.compile(r"([\n。！？?;；])")  # re-ok: 句末标点字符类切句，无回溯风险

_OPTION_ASSIGN_RE = re.compile(r"^\s*([A-Za-z0-9]{1,3})\s*(?:[=＝:：]|表示|代表|即|是)\s*(.+)$")
_OPTION_REPLY_RE = re.compile(r"^\s*(?:回复|回|选|按)\s*([A-Za-z0-9]{1,3})\s*(?:[=＝:：]|表示|代表|即|是)?\s*(.+)$")
_QUOTED_TOKEN_RE = re.compile(r'[「『"“]([^」』"”]{1,4})[」』"”]')
_APPROVAL_HINT_TERMS = ("授权", "允许", "同意", "批准", "放行", "执行", "修改", "继续", "动手")
_DENIAL_HINT_TERMS = ("只读", "仅", "不修改", "不改", "不动", "不要", "别", "拒绝")


def _pending_option_definitions(pending: str, token: str) -> tuple[str, ...]:
    """The definitions a pending request gives for *token*."""
    definitions: list[str] = []
    for segment in re.split(r"[；;。，,\n]", pending):
        for pattern in (_OPTION_ASSIGN_RE, _OPTION_REPLY_RE):
            mapped = pattern.match(segment)
            if mapped is not None and mapped.group(1).lower() == token:
                definitions.append(mapped.group(2).lower())
    return tuple(definitions)


def _definition_authorizes(definition: str) -> bool:
    """Whether one option definition unambiguously means "go ahead"."""
    return (
        not any(term in definition for term in _DENIAL_HINT_TERMS)
        and any(term in definition for term in _APPROVAL_HINT_TERMS)
    )


def _pending_request_authorizes_reply(pending_request: str, message: str) -> bool:
    """Whether the pending request itself defines *message* as "go ahead".

    A gate that blocked a turn asked a direct question, so the answer is often a
    bare option label ("B", "1", 「修」). That label authorizes only when the
    request it answers says so — "「修」字授权修改", "B=授权修改", "回复 y 授权修改".
    A label the request maps to a read-only outcome ("A=只读继续"), or one it never
    mentions, must not unlock writes; and with no pending request neither branch is
    reached, so a stray "b" in open conversation authorizes nothing.
    """
    pending = str(pending_request or "")
    token = "".join(char for char in str(message or "").strip().lower() if char.isalnum())
    if not pending or not token or len(token) > 4:
        return False
    definitions = [
        re.split(r"[；;。，,\n]", pending[quoted.end():], maxsplit=1)[0][:12]
        for quoted in _QUOTED_TOKEN_RE.finditer(pending)
        if "".join(char for char in quoted.group(1).lower() if char.isalnum()) == token
    ]
    # 只看这一句里引号之后的那段: 「B」不要继续 是否决, 「修」字授权修改 才是放行
    definitions.extend(_pending_option_definitions(pending, token))
    return bool(definitions) and all(_definition_authorizes(d) for d in definitions)


# 疑问形态是**句法**事实，不是语义猜测：句末问号与疑问助词都能照着字面判。
# 它只用来阻止「动词表把提问当成指令」，不作为拒绝写入的依据——拒绝只认显式禁止。
_INTERROGATIVE_MARKERS = ("怎么", "如何", "为什么", "是否", "能不能", "可不可以", "吗？", "吗?")


def _is_interrogative(text: str) -> bool:
    """Whether the message asks rather than tells.

    Contract:
        Preconditions: text is the lowercased, stripped user message.
        Postconditions: True on a trailing question mark or an interrogative
                        particle; never raises.
    """
    stripped = (text or "").strip()
    if stripped.endswith(("?", "？")):
        return True
    return any(marker in stripped for marker in _INTERROGATIVE_MARKERS)


def _fallback_detect_intent(
    user_message: str,
    conversation_history: list[dict] | None = None,
    *,
    pending_auth_request: str = "",
) -> str:
    """Conservative intent fallback for an unavailable semantic classifier.

    The two directions are deliberately asymmetric, because the costs are: a
    refusal the user did not deserve deadlocks the whole turn, while a grant
    they did not ask for only costs an investigation they can still see and
    stop. So the fallback may *grant* on a verb it reads as an imperative, but
    may only *refuse* on an explicit prohibition — a phrase that literally says
    "do not". Everything else is ambiguous, and an ambiguous turn runs the
    investigation contract instead of being refused.

    Scanning read-only verbs for the refusal made the verdict depend on which
    verbs happened to be enumerated: 「看看这个 bug 然后修掉」 matched a
    read-only verb and no listed modify verb, so the fallback refused a real fix
    request and the turn deadlocked.
    """
    text = (user_message or "").strip().lower()
    if not text:
        return INTENT_AMBIGUOUS
    if any(term in text for term in _EXPLICIT_NO_MODIFY_TERMS):
        return INTENT_READ_ONLY
    if _is_confirmation_check(text):
        return INTENT_READ_ONLY
    if any(term in text for term in _STRONG_MODIFY_TERMS):
        return INTENT_MODIFY
    if _pending_request_authorizes_reply(pending_auth_request, user_message):
        return INTENT_MODIFY
    if not _is_interrogative(text) and any(term in text for term in _MODIFY_TERMS):
        return INTENT_MODIFY
    if _is_confirmation(text, conversation_history):
        return INTENT_MODIFY
    return INTENT_AMBIGUOUS


def _hard_intent(
    user_message: str,
    conversation_history: list[dict] | None,
    *,
    pending_auth_request: str = "",
) -> str | None:
    """Resolve intent when conversation evidence is stronger than semantics.

    Explicit prohibitions, explicit execution requests and a terse confirmation
    of a concrete proposal are protocol facts; so is a terse request merely to
    check details, and a label reply the pending authorization request itself
    defines as "go ahead". A semantic classifier may handle ambiguous phrasing,
    but must not rewrite these verdicts.
    """
    text = (user_message or "").strip().lower()
    if any(term in text for term in _EXPLICIT_NO_MODIFY_TERMS):
        return INTENT_READ_ONLY
    if any(term in text for term in _STRONG_MODIFY_TERMS):
        return INTENT_MODIFY
    if _is_confirmation_check(text):
        return INTENT_READ_ONLY
    if _pending_request_authorizes_reply(pending_auth_request, user_message):
        return INTENT_MODIFY
    if _is_confirmation(text, conversation_history):
        return INTENT_MODIFY
    return None


def _build_intent_context(
    user_message: str,
    conversation_history: list[dict] | None,
) -> str:
    """Build a bounded transcript containing only user/assistant text.

    Tool/system output is excluded: it can explain context but cannot grant a
    modification. The latest user message is repeated after history so a
    truncation boundary cannot hide the request being classified.
    """
    lines: list[str] = []
    for msg in (conversation_history or [])[-12:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = _message_text(msg.get("content"))
        if not content:
            continue
        lines.append(f"{role.upper()}: {content[:600]}")
    latest = f"USER: {(user_message or '').strip()[:2000]}"
    if not lines or lines[-1] != latest:
        lines.append(latest)
    return "\n".join(lines)[-4000:]


def _intent_from_labels(
    read_only: bool | None, modify: bool | None, confident: bool | None = None,
) -> str | None:
    """Map the classifier's labels onto one intent.

    Both false means the classifier declined to decide, which is the ambiguous
    class; both true is a contradiction the prompt forbids, so it is treated as
    "no usable answer" rather than silently picking a side.

    ``confident`` is accepted but deliberately not consulted: asking the model
    to self-report its certainty on the read-only verdict measured *zero*
    effect on the corpus (the same 2 refusals were wrong either way, and it
    cost four proposal confirmations) — self-reported confidence is not
    calibrated. A real tier needs the logprobs channel, which this auxiliary
    endpoint does not serve. The parameter stays so a calibrated signal can be
    threaded through here without moving the mapping.

    Contract:
        Preconditions: each argument is True, False or None.
        Postconditions: returns one of INTENT_*; None only when the answer is
                        unusable (missing label or contradictory labels).
    """
    if read_only is True and modify is False:
        return INTENT_READ_ONLY
    if read_only is False and modify is True:
        return INTENT_MODIFY
    if read_only is False and modify is False:
        return INTENT_AMBIGUOUS
    return None


def _classify_intent_via_llm(
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> str | None:
    """Classify the latest user message as read_only/modify/ambiguous.

    Returns None only when the semantic classifier is unavailable or returns
    no usable labels; callers then use the conservative deterministic fallback.
    """
    context = _build_intent_context(user_message, conversation_history)
    key = _cache_key(context)
    cached = _intent_cache.get(key)
    if cached is not None:
        logger.info("read-think gate: intent=%s channel=llm-cached", cached)
        return cached
    try:
        from plugins._llm_judge import llm_judge_multi

        verdict = llm_judge_multi(
            task="read_think_intent",
            system=_INTENT_CLASSIFY_PROMPT,
            text=context,
            keys=["read_only", "modify"],
        )
    except Exception:
        logger.warning("read-think gate: intent classifier failed", exc_info=True)
        return None

    result = _intent_from_labels(
        verdict.get("read_only"), verdict.get("modify"), verdict.get("confident"),
    )
    if result is None:
        logger.info(
            "read-think gate: intent=unusable channel=llm labels=(%r,%r)",
            verdict.get("read_only"), verdict.get("modify"),
        )
        return None

    if len(_intent_cache) >= _INTENT_CACHE_MAX:
        _intent_cache.pop(next(iter(_intent_cache)))
    _intent_cache[key] = result
    logger.info("read-think gate: intent=%s channel=llm", result)
    return result


def _detect_user_intent(
    user_message: str | None,
    conversation_history: list[dict] | None,
    *,
    use_llm: bool,
    pending_auth_request: str = "",
) -> str:
    """Resolve intent from hard conversation evidence, then semantics."""
    message = user_message or ""
    if not message.strip():
        return INTENT_AMBIGUOUS
    hard_intent = _hard_intent(
        message, conversation_history, pending_auth_request=pending_auth_request
    )
    if hard_intent is not None:
        logger.info("read-think gate: intent=%s channel=conversation", hard_intent)
        return hard_intent
    if use_llm:
        detected = _classify_intent_via_llm(message, conversation_history)
        if detected in _INTENT_VALUES:
            return detected
    fallback = _fallback_detect_intent(
        message, conversation_history, pending_auth_request=pending_auth_request
    )
    logger.info("read-think gate: intent=%s channel=fallback", fallback)
    return fallback


# ── LLM 复杂度分类 ─────────────────────────────────────────────────

# 进程内缓存：相同消息不重复调 API。上限 256 条 LRU。
_CACHE_MAX = 256
_complexity_cache: dict[str, str] = {}

_CLASSIFY_PROMPT = """任务复杂度分类。只回答一个词：simple / normal / complex

参考标准：
simple: 改一行、修typo、换变量名、加注释、格式调整
normal: 修bug、写函数、写测试、更新API、加参数校验
complex: 重构架构、系统设计、从零搭建、多服务联调、大规模改造

只回答一个词。"""

_CLASSIFY_MAX_TOKENS = 50


# ── LLM-as-Judge: 调查质量评估 ──────────────────────────────────────

_JUDGE_PROMPT = """你是一个代码调查质量评审员。评估 agent 的调查分析是否充分——不是数次数，而是评估理解深度。

任务描述：
{task}

agent 要执行的工具：{tool}

调查行为记录（包括代码读取、搜索、网页访问、技能加载等）：
{investigation_evidence}

agent 的分析内容：
---
{analysis}
---

{history_context}
逐条评估（每条 PASS 或 FAIL）：

A. 目标理解：是否说明了目标代码/配置当前的逻辑，并读取了全部既有写入目标？（引用具体函数/行号/数据流）
B. 关系分析：是否识别了调用方、被依赖方、相关兄弟实现和影响面？
C. 既有约束：是否检查了相关测试、配置、项目约定或等价实现？
D. 历史意图：如果会改变既有行为，是否检查了历史记录/提交/文档并说明原意图？
E. 修改准备：是否给出验收标准、验证命令/证据、风险、以及合理方案对比？

判定规则：
- 5 条全 PASS → APPROVED
- 有任何 FAIL → NEEDS_MORE_WORK
- 分析内容为空或只有意图陈述 → NEEDS_MORE_WORK
- 重要：如果前几轮已反馈过某个不足，而 agent 本轮分析中已针对该不足做了补充，
  则该项应判 PASS。不要反复找新角度——收敛判定。
- 如果 5 个维度已经基本覆盖（即使深度不够完美），也应该 APPROVED。
  "完美"不是标准，"充分理解了代码逻辑和方案"才是标准。

只回答 APPROVED 或 NEEDS_MORE_WORK。在 NEEDS_MORE_WORK 后用一句话说明缺了什么。"""

_JUDGE_MAX_TOKENS = 200


def _judge_investigation(
    task: str,
    tool: str,
    files_read: set[str],
    analysis: str,
    investigation_evidence: list[str] | None = None,
    judge_history: list[str] | None = None,
    fail_count: int = 0,
) -> tuple[bool, str, bool]:
    """用 LLM 评估调查质量。

    Args:
        investigation_evidence: 所有调查工具的调用摘要（search_files/web_search 等）
        judge_history: 前几轮 judge 的反馈列表（最近 3 条），用于让 judge 知道之前已反馈了什么
        fail_count: judge 连续失败次数（辅助模型不可用/异常时累计），达到阈值后 fail-open

    Returns:
        (approved, feedback, was_infra_failure)
        - approved=True 时 feedback 为空，was_infra_failure=False
        - was_infra_failure=True 表示是基础设施问题（client 不可用/异常），
          调用方应递增 fail_count 而非重置；两种失败都保持 block。
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client, build_judge_thinking_extra_body

        client, model = get_text_auxiliary_client("investigation_judge")
        if client is None or not model:
            logger.warning(
                "read-think gate: no auxiliary client for judge → block (fail_count=%d)",
                fail_count,
            )
            return False, "调查质量评审服务暂不可用；修改保持锁定，请稍后重试。", True

        # 构建调查证据文本：文件路径 + 所有调查工具调用记录
        evidence_parts = []
        if files_read:
            evidence_parts.extend(f"  文件: {f}" for f in sorted(files_read))
        if investigation_evidence:
            evidence_parts.extend(f"  {e}" for e in investigation_evidence)
        evidence_str = "\n".join(evidence_parts) if evidence_parts else "  (无)"

        # 漏洞 2 修复：注入历史反馈，让 judge 知道之前已要求过什么
        history_context = ""
        if judge_history:
            recent = judge_history[-3:]  # 最近 3 条反馈
            history_lines = []
            for i, fb in enumerate(recent, 1):
                history_lines.append(f"  第{i}轮反馈：{fb[:200]}")
            history_context = "前几轮评审反馈（注意：如果 agent 已针对这些反馈做了补充，不要再要求同样的内容）：\n" + "\n".join(history_lines) + "\n"

        prompt = _JUDGE_PROMPT.format(
            task=task[:500],
            tool=tool,
            investigation_evidence=evidence_str,
            analysis=analysis[:2000],
            history_context=history_context,
        )

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_JUDGE_MAX_TOKENS,
            temperature=0,
            timeout=15,
            extra_body=build_judge_thinking_extra_body(model),
        )
        raw = (response.choices[0].message.content or "").strip()

        if "APPROVED" in raw.upper():
            logger.info("read-think gate: judge APPROVED investigation")
            return True, "", False

        feedback = raw.replace("NEEDS_MORE_WORK", "").strip()
        logger.info("read-think gate: judge NEEDS_MORE_WORK: %s", feedback)
        return False, feedback, False

    except Exception:
        logger.warning(
            "read-think gate: judge failed → block (fail_count=%d)",
            fail_count,
            exc_info=True,
        )
        return False, "调查质量评审服务异常；修改保持锁定，请稍后重试。", True


def _build_history_summary(conversation_history: list[dict] | None) -> str:
    """从历史中提取用户消息摘要。

    取最近 15 条用户消息，覆盖整段对话上下文（含会话开始时的目标导向消息）。
    拼接成 ≤600 字简短上下文，用于给复杂度分类提供语义背景。
    """
    if not conversation_history:
        return ""

    user_msgs = []
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_msgs.append(content.strip())
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                if parts:
                    user_msgs.append(" ".join(parts).strip())
            if len(user_msgs) >= 15:
                break

    if not user_msgs:
        return ""

    summary = " ".join(reversed(user_msgs))[:600]
    return summary


def _cache_key(message: str) -> str:
    """生成缓存键。"""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:32]


def _classify_via_llm(user_message: str, history_summary: str = "") -> str | None:
    """用 auxiliary client 调 flash 模型做复杂度分类。

    Args:
        user_message: 当前用户消息
        history_summary: 最近用户消息摘要（≤200字），用于提供语义背景

    Returns:
        "simple" / "normal" / "complex"，或 None（调用失败时）
    """
    key = _cache_key(user_message + history_summary)
    cached = _complexity_cache.get(key)
    if cached is not None:
        logger.debug("read-think gate: complexity cache hit → %s", cached)
        return cached

    try:
        from agent.auxiliary_client import get_text_auxiliary_client, build_judge_thinking_extra_body

        client, model = get_text_auxiliary_client("complexity_classify")
        if client is None or not model:
            logger.debug("read-think gate: no auxiliary client → fallback")
            return None

        # thinking mode 下 GLM/DeepSeek 可能将答案放在 reasoning_content。
        # 禁用推理强制直接输出；GLM-5.3 不支持禁用 → 最低档 effort。
        extra_body = build_judge_thinking_extra_body(model)

        # 构建上下文（当前消息 + 历史摘要）
        context = user_message[:500]
        if history_summary:
            context = f"历史对话摘要：{history_summary}\\n\\n当前消息：{user_message[:500]}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_PROMPT},
                {"role": "user", "content": context},
            ],
            max_tokens=_CLASSIFY_MAX_TOKENS,
            temperature=0,
            timeout=10,
            extra_body=extra_body,
        )
        choice = response.choices[0]
        msg = choice.message
        raw = (msg.content or "").strip().lower()

        # thinking mode fallback：content 为空但 reasoning_content 有值
        if not raw:
            reasoning = getattr(msg, "reasoning_content", None)
            if not reasoning and msg.model_extra:
                reasoning = msg.model_extra.get("reasoning_content")
            if reasoning:
                raw = reasoning.strip().lower()

        for level in ("simple", "normal", "complex"):
            if level in raw:
                if len(_complexity_cache) >= _CACHE_MAX:
                    _complexity_cache.pop(next(iter(_complexity_cache)))
                _complexity_cache[key] = level
                logger.info(
                    "read-think gate: LLM classified complexity=%s (raw=%r, model=%s)",
                    level, raw, model,
                )
                return level

        logger.warning("read-think gate: LLM returned unparseable result %r → fallback", raw)
        return None

    except Exception:
        logger.warning("read-think gate: LLM classify failed → fallback", exc_info=True)
        return None


def detect_complexity(user_message: str | None) -> str:
    """从用户消息中检测任务复杂度。

    优先用 LLM 语义分类。LLM 不可用时降级到关键词匹配。

    Args:
        user_message: 用户消息原文

    Returns:
        "simple" / "normal" / "complex"
    """
    if not user_message:
        return "normal"

    # LLM 分类（优先）
    result = _classify_via_llm(user_message)
    if result is not None:
        return result

    # 关键词 fallback
    text = user_message.lower()
    return _fallback_detect(text)


# ── Config ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReadThinkGateConfig:
    """config.yaml → read_think_gate 段配置。"""

    enabled: bool = True
    max_reasoning_rounds: int = 5
    min_reasoning_chars: int = 80
    min_reflection_chars: int = 20
    min_read_only_calls: int = 1
    complexity_adaptive: bool = False
    complexity_profiles: Mapping[str, Mapping[str, int]] | None = None
    # 是否用 LLM 做复杂度分类（默认 True，需 auxiliary client 可用）
    use_llm_classifier: bool = True
    # Semantic classifier for phrases without hard conversation evidence. Hard
    # intent is resolved first, so only an ambiguous turn spends this call.
    use_llm_intent_classifier: bool = True
    # Optional semantic investigation judge. The deterministic evidence gate is
    # the default so a write does not silently add another paid model call.
    use_llm_judge: bool = False
    # 用户自定义额外门控工具（合并到 GATED_TOOL_NAMES 之上）。
    # 默认只门控 write_file/patch/execute_code；如需门控 terminal 等，在此追加。
    gated_tools: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ReadThinkGateConfig:
        if not isinstance(data, Mapping):
            return cls()
        profiles_raw = data.get("complexity_profiles")
        return cls(
            enabled=_as_bool(data.get("enabled"), True),
            max_reasoning_rounds=_positive_int(data.get("max_reasoning_rounds"), 5),
            min_reasoning_chars=_positive_int(data.get("min_reasoning_chars"), 80),
            min_reflection_chars=_positive_int(data.get("min_reflection_chars"), 20),
            min_read_only_calls=_positive_int(data.get("min_read_only_calls"), 1),
            complexity_adaptive=_as_bool(data.get("complexity_adaptive"), False),
            complexity_profiles=profiles_raw if isinstance(profiles_raw, Mapping) else None,
            use_llm_classifier=_as_bool(data.get("use_llm_classifier"), True),
            use_llm_intent_classifier=_as_bool(
                data.get("use_llm_intent_classifier"), True
            ),
            use_llm_judge=_as_bool(data.get("use_llm_judge"), False),
            gated_tools=tuple(
                str(x) for x in (data.get("gated_tools") or [])
                if isinstance(x, str)
            ),
        )

    def get_profile(self, complexity: str) -> ComplexityProfile:
        """获取指定复杂度级别的门控参数。

        优先使用 config.yaml 中的自定义配置，回退到默认 COMPLEXITY_PROFILES。
        """
        if self.complexity_profiles and complexity in self.complexity_profiles:
            raw = self.complexity_profiles[complexity]
            return ComplexityProfile(
                min_read_only_calls=_positive_int(raw.get("min_read_only_calls"), COMPLEXITY_PROFILES[complexity].min_read_only_calls),
                min_reasoning_chars=_positive_int(raw.get("min_reasoning_chars"), COMPLEXITY_PROFILES[complexity].min_reasoning_chars),
                min_reflection_chars=_positive_int(raw.get("min_reflection_chars"), COMPLEXITY_PROFILES[complexity].min_reflection_chars),
                max_reasoning_rounds=_positive_int(raw.get("max_reasoning_rounds"), COMPLEXITY_PROFILES[complexity].max_reasoning_rounds),
            )
        return COMPLEXITY_PROFILES.get(complexity, COMPLEXITY_PROFILES["normal"])


class ReadThinkGate:
    """Per-turn 两阶段审议门控制器。

    推理阶段：只读工具放行，执行工具引导调查。
    执行阶段：全部放行。

    任务分级：通过 reset_for_turn(user_message) 传入用户消息，
    自动检测复杂度（simple/normal/complex）并调整门控参数。

    生命周期：
      1. build_turn_context → reset_for_turn()   （进入推理阶段）
      2. 每轮 LLM 响应 → check_batch(content, tool_names) → 放行/拦截
      3. 解锁后 → 本 turn 后续全部放行
    """

    def __init__(self, config: ReadThinkGateConfig | None = None, session_id: str = ""):
        self.config = config or ReadThinkGateConfig()
        # marker 桶维度——写/清 marker 均以此会话身份解析路径（见
        # _four_axis_marker_path：同进程多会话隔离）。
        self._session_id = session_id
        # 合并默认门控工具 + 用户自定义扩展，形成实例级集合。
        # check_batch 用这个 set 而不是直接引用全局 GATED_TOOL_NAMES。
        self._gated_tools: set[str] = set(GATED_TOOL_NAMES)
        if self.config.gated_tools:
            self._gated_tools.update(self.config.gated_tools)
        self._four_axis_found: set[str] = set()  # 本 turn 四轴证据集合
        self._session_modification_authorized: bool = False
        self._authorized_user_message: str = ""
        self._awaiting_user_authorization: bool = False
        self._pending_auth_request: str = ""  # 拦截请求的 assistant 原文（授权回应分类用）
        self.reset_for_turn()

    # ── Per-turn state ──────────────────────────────────────────────

    def reset_for_turn(
        self,
        user_message: str | None = None,
        conversation_history: list[dict] | None = None,
        internal: bool = False,
    ) -> None:
        """每轮开始时重置——回到推理阶段。

        Args:
            user_message: 当前用户消息
            conversation_history: 对话历史（用于提取历史摘要辅助复杂度检测）
        """
        self._satisfied: bool = False
        self._reasoning_rounds: int = 0
        self._read_only_count: int = 0
        self._active_complexity: str = "normal"
        self._files_read: set[str] = set()  # 本 turn 读取的文件路径
        self._investigation_evidence: list[str] = []  # 所有调查工具的证据摘要
        self._user_message: str = user_message or ""  # judge 用：当前任务描述
        self._clarify_context: str = ""  # clarify 批次的 assistant 文本（回答确认指代用）
        self._last_judge_feedback: str = ""  # judge 最近一次反馈
        # 漏洞 2 修复：judge 历史反馈列表——让 judge 知道前几轮已要求过什么
        self._judge_feedback_history: list[str] = []
        # 漏洞 4 修复：judge 连续失败计数——达到阈值后 fail-open
        self._judge_fail_count: int = 0
        self._intent: str = INTENT_AMBIGUOUS
        self._modification_authorized: bool = False
        self._investigation_complete: bool = False
        # 捕获上一 turn 结束时的等待授权状态（下面的非 internal 分支会清零）。
        prior_awaiting = self._awaiting_user_authorization
        if not internal:
            self._awaiting_user_authorization = False
        # 四轴闸门——每 turn 重新计数
        self._four_axis_found: set[str] = set()
        # 清除本会话上一 turn 的四轴 marker 文件（不触碰其他会话的桶）
        _clear_four_axis_marker(self._session_id)
        # 清理上一 turn 动态加入 _gated_tools 的 terminal（漏洞 3 修复副作用）。
        # 但保留用户通过 config 显式配置的 gated_tools。
        if "terminal" not in self.config.gated_tools:
            self._gated_tools.discard("terminal")

        if internal:
            self._intent = (
                INTENT_MODIFY
                if self._session_modification_authorized
                else INTENT_AMBIGUOUS
            )
            self._modification_authorized = self._session_modification_authorized
            if self._session_modification_authorized:
                self._user_message = self._authorized_user_message
        else:
            # 上一 turn 门禁因等待授权而拦截时，本条用户消息是对该请求的回应：
            # 请求尾段注入分类上下文供指代消解（同 clarify 先例），短回复
            # （"B"、"1"、裸「修」）也才有锚点可依，否则只能判成 ambiguous。
            pending_auth = bool(
                prior_awaiting
                and self._pending_auth_request
                and not any(
                    term in (user_message or "").lower()
                    for term in _EXPLICIT_NO_MODIFY_TERMS
                )
            )
            history = conversation_history
            if pending_auth:
                history = [
                    *(conversation_history or []),
                    {
                        "role": "assistant",
                        "content": _AUTH_REQUEST_FRAME.format(
                            self._pending_auth_request[-600:]
                        ),
                    },
                ]
            self._intent = _detect_user_intent(
                user_message,
                history,
                use_llm=self.config.use_llm_intent_classifier,
                pending_auth_request=self._pending_auth_request if pending_auth else "",
            )
            self._pending_auth_request = ""
            if self._intent == INTENT_MODIFY:
                self._session_modification_authorized = True
                self._authorized_user_message = user_message or ""
            elif (
                self._session_modification_authorized
                and self._is_operational_feedback(user_message)
                and not any(
                    term in (user_message or "").lower()
                    for term in _EXPLICIT_NO_MODIFY_TERMS
                )
            ):
                self._intent = INTENT_MODIFY
                self._user_message = (
                    f"{self._authorized_user_message}\n[follow-up] {user_message or ''}"
                ).strip()
            else:
                self._session_modification_authorized = False
                self._authorized_user_message = ""
            self._modification_authorized = self._intent == INTENT_MODIFY
        self._publish_state()

        if self.config.complexity_adaptive:
            history_summary = ""
            if conversation_history:
                history_summary = _build_history_summary(conversation_history)

            classification_message = (
                self._authorized_user_message
                if internal and self._session_modification_authorized
                else user_message
            )
            if self.config.use_llm_classifier and classification_message:
                detected = (
                    _classify_via_llm(classification_message, history_summary) or "normal"
                )
            else:
                # 2026-08-18 修复：use_llm_classifier=False 时必须真正绕过 LLM。
                # 此前 else 分支调 detect_complexity()，其内部无条件优先
                # _classify_via_llm()，导致 config.yaml 的 false 被架空——
                # 每 turn 仍烧一次同步 LLM 调用（timeout=10s，阻塞回合启动），
                # 实测日志 110 次 "LLM classified" 均发生在 false 配置下。
                detected = (
                    _fallback_detect(str(classification_message or "").lower())
                    if classification_message
                    else "normal"
                )
            self._active_complexity = detected
            self._active_profile = self.config.get_profile(detected)
            if detected != "normal":
                logger.info(
                    "read-think gate: detected complexity=%s (reads=%d reason_chars=%d reflect_chars=%d rounds=%d)",
                    detected,
                    self._active_profile.min_read_only_calls,
                    self._active_profile.min_reasoning_chars,
                    self._active_profile.min_reflection_chars,
                    self._active_profile.max_reasoning_rounds,
                )
        else:
            self._active_profile = ComplexityProfile(
                min_read_only_calls=self.config.min_read_only_calls,
                min_reasoning_chars=self.config.min_reasoning_chars,
                min_reflection_chars=self.config.min_reflection_chars,
                max_reasoning_rounds=self.config.max_reasoning_rounds,
            )

    def _publish_state(self) -> None:
        """Publish this turn's verdict for ``guards.pre_write``.

        The batch hook and per-tool hook run in the same profile scope. This
        state is an enforcement handoff, not an authorization source: only
        ``reset_for_turn`` derives intent from user conversation.
        """
        if not self._session_id:
            return
        state = get_session_state(self._session_id, _STATE_NAMESPACE)
        state.clear()
        state.update(
            {
                "active": bool(self.config.enabled),
                "intent": self._intent,
                "modification_authorized": self._modification_authorized,
                "investigation_complete": self._investigation_complete,
                "awaiting_user_authorization": self._awaiting_user_authorization,
            }
        )

    @staticmethod
    def _is_operational_feedback(user_message: str | None) -> bool:
        """Whether a human message steers an active task without replacing it."""
        text = (user_message or "").strip().lower()
        return bool(text) and any(term in text for term in _OPERATIONAL_FEEDBACK_TERMS)

    @property
    def user_intent(self) -> str:
        """This turn's immutable user-intent classification."""
        return self._intent

    @staticmethod
    def _normalized_path(path: str) -> str:
        """Canonical path form for read-target comparisons."""
        return os.path.realpath(os.path.abspath(os.path.expanduser(path)))

    def _write_targets(
        self,
        tool_names: list[str],
        tool_args: list[dict] | None,
    ) -> list[str]:
        """Return concrete existing file targets in this write batch."""
        if not tool_args:
            return []
        targets: list[str] = []
        for name, args in zip(tool_names, tool_args):
            if name not in ("write_file", "patch") or not isinstance(args, dict):
                continue
            raw = str(args.get("path") or "").strip()
            if raw and os.path.exists(self._normalized_path(raw)):
                targets.append(self._normalized_path(raw))
        return targets

    def _unread_write_targets(
        self,
        tool_names: list[str],
        tool_args: list[dict] | None,
    ) -> list[str]:
        """Existing write targets not read in full this turn."""
        read = {self._normalized_path(path) for path in self._files_read}
        return [path for path in self._write_targets(tool_names, tool_args) if path not in read]

    def _mark_satisfied(self) -> None:
        """Unlock writes and publish the evidence verdict to pre_write."""
        self._satisfied = True
        self._investigation_complete = True
        self._awaiting_user_authorization = False
        self._publish_state()

    def _scan_four_axis(self, content: str | None) -> None:
        """扫描 assistant 回复文本，检测四轴证据并累积到 self._four_axis_found。

        每轴独立判定：该轴的任一关键词命中即标记为 found（O(n) 子串匹配，无正则回溯）。
        四轴全部 found 时写入 marker 文件，供 pre_tool_call 插件读取。

        Contract:
          Preconditions: content 为本 turn assistant 回复文本（可能为 None）
          Postconditions: _four_axis_found 仅增不减；四轴齐时 marker 文件存在且 verified=True
        2026-08-26 修复：e12e46fd3 将此方法改为空操作并赌 use_llm_judge 恒开，
        但用户配置 use_llm_judge=false（config.yaml:911）→ 四轴无任何解锁路径，
        patch 永远被拦（误报）。恢复关键词扫描 = 恢复「judge 显式通过或内容检测
        自然累积」双路径不变量（check_batch :1029 注释）。judge 开启时行为不变。
        """
        if not content:
            return
        # 全文扫描：四轴证据可能出现在正文任何位置（此前 8KB 尾窗截断会把
        # 前段写好的四轴证据顶出窗口造成误拦——子串匹配 O(n) 无回溯，
        # 30 关键词 × 100KB 亚毫秒级，无 CPU 顾虑）。
        scan_text = content
        progressed = False
        for axis, keywords in _FOUR_AXIS_KEYWORDS.items():
            if axis in self._four_axis_found:
                continue
            for kw in keywords:
                if kw in scan_text:
                    self._four_axis_found.add(axis)
                    progressed = True
                    logger.info(
                        "read-think gate: four-axis '%s' detected — %d/4 axes found",
                        axis, len(self._four_axis_found),
                    )
                    break
        # 有新轴命中 → 落盘进度 marker（verified=是否已齐 4 轮）。
        # verified=False 的记录只供副防线拦截消息诊断缺失轴（four_axis.py
        # _read_marker 的放行判定仍只认 verified=True+4 轮，本写入不改变
        # 放行语义）。
        # 写盘条件=新增命中 或 四轴已齐：仅新增命中时写会在长写链(>30min TTL)
        # 且四轴早已齐备的场景结构性死锁——_four_axis_found 满后 progressed
        # 恒 False，marker 永不续期，副防线判过期拦截且重贴证据无效
        # （2026-10-27 实测：verified=true 四轴齐、timestamp 36 分钟前、
        # patch 连续 4 次被拦）。未齐状态仍仅新增命中才写（防中间进度刷 TTL）。
        # 跨任务沿用防线不变：turn-start 清本会话 marker + 会话分桶文件名。
        if progressed or len(self._four_axis_found) == 4:
            try:
                _marker = _four_axis_marker_path(self._session_id)
                _marker.parent.mkdir(parents=True, exist_ok=True)
                _marker.write_text(json.dumps({
                    "verified": len(self._four_axis_found) == 4,
                    "source": "content_scan",
                    "timestamp": time.time(),
                    "axes": sorted(self._four_axis_found),
                }))
                if len(self._four_axis_found) == 4:
                    logger.info(
                        "read-think gate: four-axis complete via content scan — "
                        "marker written (session=%s)", self._session_id,
                    )
            except Exception:
                logger.warning("read-think gate: failed to write four-axis marker", exc_info=True)

    def mark_four_axis_complete(self) -> None:
        """LLM judge 通过后调用——标记四轴完成并写入 marker 文件。

        Contract:
          Preconditions: gate 处于活跃状态，LLM judge 已判定 APPROVED
          Postconditions: _four_axis_complete() == True, marker 文件存在且 verified=True
        """
        self._four_axis_found = set(_FOUR_AXIS_LABELS)
        assert self._four_axis_complete(), "mark_four_axis_complete failed to set all axes"
        try:
            _marker = _four_axis_marker_path(self._session_id)
            _marker.parent.mkdir(parents=True, exist_ok=True)
            _marker.write_text(json.dumps({
                "verified": True,
                "source": "llm_judge",
                "timestamp": time.time(),
                "axes": sorted(self._four_axis_found),
            }))
            logger.info(
                "read-think gate: four-axis complete via LLM judge — "
                "marker written (session=%s)", self._session_id,
            )
        except Exception:
            logger.warning("read-think gate: failed to write four-axis marker", exc_info=True)

    def _four_axis_complete(self) -> bool:
        """四轴是否全部检测到。"""
        return len(self._four_axis_found) == 4

    def _missing_axes(self) -> list[str]:
        """返回尚未检测到的轴名称列表。"""
        return [a for a in _FOUR_AXIS_LABELS if a not in self._four_axis_found]

    def _strict_gap_items(
        self,
        profile: ComplexityProfile,
        content_len: int,
        tool_names: list[str],
    ) -> list[str]:
        """严格模式下当前未达标的全部缺口，一次列全。

        与 _try_unlock 的四关一一对应：①调查量 ②搜索类多样性 ③分析
        字数 ④四轴。四轴对所有修改通道都强制，terminal 直写也不能绕过。
        调查量/字数未达标时 _try_unlock 不评估后续关——本方法打破该
        顺序依赖，把后续关缺口一并披露，被拦方一次补齐而非逐层试探。

        Contract:
          Preconditions: 修改请求的 _try_unlock 返回 False 后由
            _build_block_message 调用
          Postconditions: 返回零或多条缺口描述，每条含当前值/目标值/
            可执行动作；空列表 = 四关全过（被拦属状态异常，调用方兜底）
        """
        items: list[str] = []
        done = self._read_only_count
        needed = profile.min_read_only_calls
        if done < needed:
            items.append(
                f"① 只读调查 {done}/{needed}：还差 {needed - done} 次。计数工具："
                "read_file / search_files / web_search / web_extract / "
                "browser_exec / vision_analyze / session_search / skill_view，"
                "以及只读 terminal 命令（ls/git log/sqlite3 查询等；写文件/"
                "解释器/脚本形态不计数；read_file 失败调用不计数）。"
            )
        if not self._has_diverse_investigation(profile):
            items.append(
                "② 搜索类多样性未达标：还没用 search_files / web_search / "
                "web_extract / browser_exec 任一——read_file 堆数量不算，"
                "至少 1 次搜索类调用（查调用方/同类实现）。"
            )
        if content_len < profile.min_reasoning_chars:
            items.append(
                f"③ 分析字数 {content_len}/{profile.min_reasoning_chars}：还差 "
                f"{profile.min_reasoning_chars - content_len} 字。正文回答：目标"
                "代码当前逻辑（引行号）/相关既有程序与既有做法/方案对比与最优性。"
            )
        if not self._four_axis_complete():
            missing = "、".join(self._missing_axes())
            items.append(
                f"④ 四轴缺「{missing}」：正文逐轴输出并紧贴写调用——影响面清单"
                "（受影响调用方 绝对路径+函数名+行号）/原意图溯源（git log 最初"
                " commit + 不变量逐条标注保持/修改/破坏）/根因定位（症状位置 vs "
                "根因位置 文件:行号）/风险矩阵（最坏场景+触发条件+影响范围+"
                "可恢复性）。"
            )
        return items

    def _has_diverse_investigation(self, profile: ComplexityProfile) -> bool:
        """调查是否包含搜索类工具，而非只有 read_file/skill_view 堆数量。

        搜索类工具：search_files（代码搜索）、web_search（网络搜索）、
        web_extract（网页内容提取）、browser_*（浏览器交互）。

        当 use_llm_judge 关闭时，此方法替代语义评估——
        防止 agent 用 3 次 read_file 在无关文件上凑数过关。
        """
        has_search = any(
            e.startswith(("search_files", "web_search", "web_extract", "browser_"))
            for e in self._investigation_evidence
        )
        return has_search

    @property
    def is_satisfied(self) -> bool:
        """门是否已解锁（进入执行阶段）。"""
        return self._satisfied

    @property
    def phase(self) -> str:
        """当前阶段标识。"""
        return "execution" if self._satisfied else "reasoning"

    # ── 核心判定方法 ────────────────────────────────────────────────

    def check_batch(
        self,
        assistant_content: str | None,
        tool_names: list[str],
        tool_args: list[dict] | None = None,
    ) -> str | None:
        """批量门控检查——同一 assistant_message 只调一次。

        Args:
            assistant_content: 当前 assistant_message 的 content 文本
            tool_names: 当前批次所有工具名
            tool_args: 每个工具的 args dict（用于 write-target 覆盖率验证 + terminal 写入检测）

        Returns:
            拦截消息（纯文本），放行时返回 None
        """
        # ── 调查证据追踪（必须在所有 early return 之前）──
        # 即使 gate 已解锁，调查证据仍需记录。
        # 追踪所有 READ_ONLY_INVESTIGATION_TOOLS 的调用，不只 read_file 路径。
        if tool_args:
            for tn, ta in zip(tool_names, tool_args):
                if tn == "terminal":
                    # terminal 不在 READ_ONLY_INVESTIGATION_TOOLS——只读命令
                    # 单独按 _terminal_is_investigation 计入（2026-11-03 根修）。
                    if not isinstance(ta, dict):
                        continue
                    cmd = str(ta.get("command", ""))
                    if cmd and _terminal_is_investigation(cmd):
                        self._investigation_evidence.append(f"terminal(readonly): {cmd[:80]}")
                    continue
                if tn not in READ_ONLY_INVESTIGATION_TOOLS or not isinstance(ta, dict):
                    continue
                if tn == "read_file":
                    rp = ta.get("path", "")
                    if rp:
                        self._files_read.add(rp)
                        self._investigation_evidence.append(f"read_file: {rp}")
                elif tn == "search_files":
                    pat = ta.get("pattern", "")
                    tgt = ta.get("target", "")
                    self._investigation_evidence.append(f"search_files({tgt}): {pat}")
                elif tn in ("web_search", "web_extract"):
                    urls = ta.get("urls", [])
                    q = ta.get("query", "")
                    if q:
                        self._investigation_evidence.append(f"{tn}: {q}")
                    elif urls:
                        self._investigation_evidence.append(f"{tn}: {urls[0]}")
                elif tn.startswith("browser_"):
                    url = ta.get("url", "")
                    q = ta.get("question", "")
                    self._investigation_evidence.append(f"{tn}: {url or q[:80]}")
                elif tn == "skill_view":
                    name = ta.get("name", "")
                    self._investigation_evidence.append(f"skill_view: {name}")

        # ── 四轴证据扫描（每次 assistant 回复都扫，累积到 self._four_axis_found）──
        self._scan_four_axis(assistant_content)

        # clarify 发起批次：暂存该 assistant 文本——后续 clarify 回答是「修/按方案执行」
        # 类短句时，确认指代的就是这段提案（note_user_clarification 消费）。
        if assistant_content and "clarify" in tool_names:
            self._clarify_context = str(assistant_content)[:2000]

        if not self.config.enabled:
            return None

        # ── 漏洞 3 修复：检测 terminal 中的文件写入操作 ──
        # terminal 默认不在 gated_tools 中（运维工具），但如果命令包含
        # 文件写入操作（> >> tee sed -i cp mv dd 等），则视为代码编辑，需要门控。
        # 检测到写入操作时把 terminal 动态加入 _gated_tools，
        # 这样 tool_executor 的 `function_name in _gated_set` 检查才能匹配到。
        terminal_writing_files = False
        if tool_args:
            for tn, ta in zip(tool_names, tool_args):
                if tn == "terminal" and isinstance(ta, dict):
                    cmd = ta.get("command", "")
                    if cmd and _terminal_writes_file(cmd):
                        terminal_writing_files = True
                        # 动态加入 _gated_tools，让 tool_executor 分发路径也能拦截
                        self._gated_tools.add("terminal")
                        logger.info(
                            "read-think gate: terminal command writes file → treated as gated: %s", cmd,
                        )
                        break

        has_mutating = any(t in self._gated_tools for t in tool_names) or terminal_writing_files
        # ── 漏洞 5 修复：只对白名单中的调查工具计数 ──
        # 2026-11-03 扩展：只读 terminal 命令同样计入（枚举白名单缺口根修）。
        has_read_only_investigation = any(
            t in READ_ONLY_INVESTIGATION_TOOLS for t in tool_names
        ) or bool(tool_args) and any(
            tn == "terminal"
            and isinstance(ta, dict)
            and _terminal_is_investigation(str(ta.get("command", "")))
            for tn, ta in zip(tool_names, tool_args)
        )
        content_text = assistant_content or ""
        content_len = len(content_text)

        if has_read_only_investigation:
            self._read_only_count += 1

        # Investigation may complete before the eventual write. It unlocks
        # only after the full strict evidence contract, never after a single
        # read, a timer, or a round-budget exhaust.
        if (
            not self._satisfied
            and self._intent != INTENT_READ_ONLY
            and self._try_unlock(content_len, content_text, tool_names)
        ):
            self._mark_satisfied()

        if not has_mutating:
            logger.debug(
                "read-think gate: read-only batch — continuing (investigation %d complexity=%s)",
                self._read_only_count,
                self._active_complexity,
            )
            return None

        if self._intent == INTENT_READ_ONLY:
            return self._refuse_read_only_intent(tool_names, content_text, content_len)

        unread_targets = self._unread_write_targets(tool_names, tool_args)
        if unread_targets:
            self._reasoning_rounds += 1
            first_gated = next(
                (t for t in tool_names if t in self._gated_tools),
                "terminal",
            )
            target_list = "\n".join(f"  - {path}" for path in unread_targets)
            return _make_synthetic_result(
                first_gated,
                (
                    f"[ReadThink Gate — 用户已要求修改，但调查未完成] 工具 "
                    f"'{first_gated}' 暂时不可用。\n\n"
                    "以下既有目标文件尚未在本 turn 完整读取：\n"
                    f"{target_list}\n\n"
                    "先 read_file 读取目标现状，再检查调用方、相关实现、测试、配置"
                    "和必要的历史意图。新建文件也不能跳过这次上下文调查。"
                ),
                content_len,
            )

        if self._satisfied:
            return None

        first_gated = next(t for t in tool_names if t in self._gated_tools) if any(t in self._gated_tools for t in tool_names) else "terminal"
        if self._reasoning_rounds >= self._active_profile.max_reasoning_rounds:
            self._awaiting_user_authorization = True
            self._pending_auth_request = content_text[-2000:]
            self._publish_state()
            return _make_synthetic_result(
                first_gated,
                self._build_handoff_message(first_gated),
                content_len,
            )

        # ── 推理轮数递增 ──
        self._reasoning_rounds += 1

        block_msg = self._build_block_message(first_gated, content_len, tool_names)
        logger.info(
            "read-think gate: blocking %s (round %d/%d, content=%d, investigated=%d, complexity=%s)",
            first_gated,
            self._reasoning_rounds,
            self._active_profile.max_reasoning_rounds,
            content_len,
            self._read_only_count,
            self._active_complexity,
        )
        return _make_synthetic_result(first_gated, block_msg, content_len)

    def _try_unlock(
        self, content_len: int, content: str = "", tool_names: list[str] | None = None,
    ) -> bool:
        """Return True only after the complete modification investigation.

        There is deliberately no round-count, wall-clock, or read-target
        escape hatch: those used to turn "the agent kept trying" into
        "the agent understood the code".
        """
        profile = self._active_profile

        if not content or content_len < profile.min_reasoning_chars:
            return False
        if self._read_only_count < profile.min_read_only_calls:
            return False

        if self.config.use_llm_judge:
            tool_name = ""
            if tool_names:
                tool_name = next(
                    (t for t in tool_names if t in self._gated_tools),
                    tool_names[0],
                )
            approved, feedback, was_infra_failure = _judge_investigation(
                self._user_message,
                tool_name,
                self._files_read,
                content,
                investigation_evidence=self._investigation_evidence,
                judge_history=self._judge_feedback_history,
                fail_count=self._judge_fail_count,
            )
            if not approved:
                if was_infra_failure:
                    self._judge_fail_count += 1
                else:
                    self._judge_fail_count = 0
                    self._judge_feedback_history.append(feedback)
                self._last_judge_feedback = feedback
                logger.info(
                    "read-think gate: judge rejected (reads=%d, complexity=%s): %s",
                    self._read_only_count,
                    self._active_complexity,
                    feedback,
                )
                return False
            self._judge_fail_count = 0
            self.mark_four_axis_complete()
        elif not self._has_diverse_investigation(profile):
            logger.info(
                "read-think gate: investigation not diverse (reads=%d, complexity=%s)",
                self._read_only_count,
                self._active_complexity,
            )
            return False

        if not self._four_axis_complete():
            logger.info(
                "read-think gate: four-axis incomplete — missing %s",
                self._missing_axes(),
            )
            return False

        logger.info(
            "read-think gate: unlocked — complete investigation "
            "(reads=%d>=%d, reasoning=%d>=%d, complexity=%s)",
            self._read_only_count,
            profile.min_read_only_calls,
            content_len,
            profile.min_reasoning_chars,
            self._active_complexity,
        )
        return True

    def note_user_clarification(self, answer_text: str) -> None:
        """接收 clarify 表单回传的用户回答并重判修改授权。

        clarify 结果携带的是用户本人的选择（tools/clarify_tool.py 的 answers
        来自平台回调的用户表单），属用户输入而非模型可造的工具输出；回合
        开头的意图快照不得无视它——否则用户在表单里明确选了「修」仍被拦到
        下一条聊天消息才放行（2026-11-08 实测）。

        Contract:
            Preconditions: answer_text 为已 answered 的 user_response 拼接文本
              （调用方已过滤 skipped/unanswered）；空文本直接忽略。
            Postconditions: 恒不抛异常；以捕获的 clarify 批次 assistant 文本
              作为确认指代上下文重跑意图判定，仅当判定为 modify 才升级授权
              ——read_only/ambiguous 回答永不降级既有授权。
        """
        text = (answer_text or "").strip()
        if not text:
            return
        authorized_message = (
            self._user_message + "\n[clarify] " + text[:500]
        ).strip()
        history = (
            [{"role": "assistant", "content": self._clarify_context}]
            if self._clarify_context
            else None
        )
        intent = _detect_user_intent(
            text, history, use_llm=self.config.use_llm_intent_classifier,
        )
        if intent != INTENT_MODIFY:
            logger.info(
                "read-think gate: clarify answer does not authorize (intent=%s)", intent
            )
            return
        self._intent = INTENT_MODIFY
        self._modification_authorized = True
        self._session_modification_authorized = True
        self._authorized_user_message = authorized_message
        self._awaiting_user_authorization = False
        self._user_message = authorized_message
        self._publish_state()
        logger.info("read-think gate: clarify answer authorizes modification")

    def _refuse_read_only_intent(
        self, tool_names: list[str], content_text: str, content_len: int,
    ) -> str:
        """Refuse a write the user explicitly asked not to make.

        Only the read-only verdict blocks. An ambiguous verdict is *not* a
        refusal: refusing on it cost the whole turn whenever the classifier
        could not read an interjection such as 「加油」, and repeating the
        refusal was unbounded — the round budget was only checked on the
        investigation path, so an agent that retried the same write looped
        until the user typed something else. A repeated read-only attempt now
        escalates to the handoff message instead of re-blocking in silence.
        """
        self._reasoning_rounds += 1
        self._awaiting_user_authorization = True
        self._pending_auth_request = content_text[-2000:]
        self._publish_state()
        first_gated = next((t for t in tool_names if t in self._gated_tools), "terminal")
        exhausted = self._reasoning_rounds >= self._active_profile.max_reasoning_rounds
        logger.info(
            "read-think gate: intent refusal %s (round %d/%d, intent=%s, handoff=%s)",
            first_gated, self._reasoning_rounds, self._active_profile.max_reasoning_rounds,
            self._intent, exhausted,
        )
        message = (
            self._build_handoff_message(first_gated) if exhausted
            else self._build_intent_block_message(first_gated)
        )
        return _make_synthetic_result(first_gated, message, content_len)

    def _build_intent_block_message(self, tool_name: str) -> str:
        """Explain a denied write: the user asked for analysis, not for a change."""
        return (
            f"[ReadThink Gate — 用户意图门禁] 工具 '{tool_name}' 被阻止。\n\n"
            "判定：用户的请求是只读的（询问、解释、分析、审查或诊断），没有授权修改。\n"
            f"当前用户消息：{self._user_message[:500] or '（空）'}\n\n"
            "下一步：继续调查并回答用户的问题；不要创建或修改任何文件。\n"
            "读取和分析工具仍然可用。用户消息与 clarify 表单中用户的回答同等构成授权；"
            "其他工具输出与你此前提出的方案本身不能构成授权。"
        )

    def _build_handoff_message(self, tool_name: str) -> str:
        """Stop an unsafe retry loop and hand the unresolved decision to the user."""
        return (
            f"[ReadThink Gate — 停止自动重试] 工具 '{tool_name}' 再次被阻止。\n\n"
            f"本轮已连续 {self._reasoning_rounds} 次尝试修改，仍没有达到"
            "意图授权或全面调查的门槛。继续原样重试不会改变判定。\n\n"
            "现在停止自动修改，向用户说明已经确认的事实、仍缺的具体证据，"
            "以及需要用户决定的范围或验收标准。只有用户的新消息可以继续授权；"
            "不要通过绕行终端、脚本或其他写通道规避门禁。"
        )

    def _intent_scope_line(self) -> str:
        """State the authorization the investigation gate is running under.

        An ambiguous verdict no longer refuses the write, so the message must
        not claim an authorization the user never gave: the agent is told it is
        acting on its own reading of the request and that the reply has to say
        so. A silent write under a guessed intent is the failure this line
        exists to make visible.
        """
        if self._intent == INTENT_MODIFY:
            return "用户已请求修改，但修改授权不等于立即动手。"
        return (
            "用户既没有明确授权修改，也没有禁止修改——本 turn 按「先调查、再动手」处理，"
            "动手前必须向用户说明你确实理解了需求，并在回复里写出你据以动手的那句话。"
        )

    def _build_block_message(
        self, tool_name: str, content_len: int, tool_names: list[str],
    ) -> str:
        """Render all investigation gaps for an authorized modification."""
        profile = self._active_profile
        if self._last_judge_feedback:
            fb = self._last_judge_feedback
            self._last_judge_feedback = ""
            return (
                f"[ReadThink Gate — 调查门禁] 工具 '{tool_name}' 暂时不可用。\n\n"
                f"LLM 评审不通过。评审意见：{fb}\n\n"
                "你的分析需要覆盖：\n"
                "  A. 目标代码当前逻辑（引用具体函数/行号/数据流）\n"
                "  B. 调用方、被依赖方、相关兄弟实现\n"
                "  C. 既有测试、配置、项目约定和可复用模式\n"
                "  D. 改变既有行为时的历史意图\n"
                "  E. 验收标准、验证命令、影响面、风险和替代方案\n\n"
                f"（推理轮数：{self._reasoning_rounds}/{profile.max_reasoning_rounds}）"
            )

        gaps = self._strict_gap_items(profile, content_len, tool_names)
        header = (
            f"[ReadThink Gate — 调查门禁] 工具 '{tool_name}' 暂时不可用。\n\n"
            f"{self._intent_scope_line()}"
            "先全面搞清楚：\n"
            "  1. 目标代码现在如何工作？读过完整目标文件和关键行号吗？\n"
            "  2. 调用方、被依赖方、兄弟实现和相关配置在哪里？\n"
            "  3. 现有测试和项目约定是什么？改行为前查过历史意图吗？\n"
            "  4. 验收标准、验证命令、影响面、风险和替代方案是什么？\n\n"
            "新建文件同样需要这次上下文调查。轮数、等待时间和文件已存在与否"
            "都不能替代证据。\n\n"
            "以下为本次拦截的全部缺口清单：\n"
        )
        body = "\n".join(gaps) if gaps else (
            "（全部门槛已达标，但内部状态仍拒绝写入；这说明门禁状态异常，不能绕过。）"
        )
        return (
            f"{header}{body}\n"
            f"（推理轮数：{self._reasoning_rounds}/{profile.max_reasoning_rounds}）"
        )


def _make_synthetic_result(
    tool_name: str, block_message: str, content_len: int
) -> str:
    """返回纯文本错误消息——调用方会自行 JSON 包装为 {\"error\": msg}。"""
    return block_message


# ── 工具函数 ──────────────────────────────────────────────────────


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default
