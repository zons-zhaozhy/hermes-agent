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
  2. 完成调查（调用了只读工具）+ 至少简要反思（content >= min_reflection_chars）
  3. 调查后无条件解锁（unlock_after_investigation=True 时）
  4. 推理轮数达到上限（max_reasoning_rounds）→ 防死循环

任务分级（复杂度自适应）：
  LLM 语义分类（优先）：用 auxiliary client 调 flash 模型做一次分类。
  关键词 fallback（降级）：LLM 不可用时用规则匹配。
  三级：simple / normal / complex

与 tool_guardrails 的区别：
  - guardrails 检测"循环失败"（重复调用同一工具失败）
  - read_think_gate 执行"推理期/执行期分离"（先调查再动手）

生命周期：per-turn，由 build_turn_context 重置。
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# 需要门控的执行类工具——能造成不可逆变更或副作用。
# read-only 工具（search_files/read_file/web_search 等）不受门控——调查即思考。
GATED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "terminal",
        "write_file",
        "patch",
        "execute_code",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_dialog",
        "delegate_task",
        "cronjob",
        "process",
    }
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
        min_read_only_calls=3, min_reasoning_chars=300,
        min_reflection_chars=80, max_reasoning_rounds=8,
    ),
    "complex": ComplexityProfile(
        min_read_only_calls=5, min_reasoning_chars=500,
        min_reflection_chars=150, max_reasoning_rounds=12,
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
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client("complexity_classify")
        if client is None or not model:
            logger.debug("read-think gate: no auxiliary client → fallback")
            return None

        # thinking mode 下 GLM/DeepSeek 可能将答案放在 reasoning_content。
        # 用 extra_body.thinking=disabled 禁用推理，强制直接输出。
        extra_body = {"thinking": {"type": "disabled"}}

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
                    level, raw[:50], model,
                )
                return level

        logger.warning("read-think gate: LLM returned unparseable result %r → fallback", raw[:100])
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
    unlock_after_investigation: bool = True
    min_read_only_calls: int = 1
    complexity_adaptive: bool = False
    complexity_profiles: Mapping[str, Mapping[str, int]] | None = None
    # 是否用 LLM 做复杂度分类（默认 True，需 auxiliary client 可用）
    use_llm_classifier: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ReadThinkGateConfig":
        if not isinstance(data, Mapping):
            return cls()
        profiles_raw = data.get("complexity_profiles")
        return cls(
            enabled=_as_bool(data.get("enabled"), True),
            max_reasoning_rounds=_positive_int(data.get("max_reasoning_rounds"), 5),
            min_reasoning_chars=_positive_int(data.get("min_reasoning_chars"), 80),
            min_reflection_chars=_positive_int(data.get("min_reflection_chars"), 20),
            unlock_after_investigation=_as_bool(data.get("unlock_after_investigation"), True),
            min_read_only_calls=_positive_int(data.get("min_read_only_calls"), 1),
            complexity_adaptive=_as_bool(data.get("complexity_adaptive"), False),
            complexity_profiles=profiles_raw if isinstance(profiles_raw, Mapping) else None,
            use_llm_classifier=_as_bool(data.get("use_llm_classifier"), True),
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

    def __init__(self, config: ReadThinkGateConfig | None = None):
        self.config = config or ReadThinkGateConfig()
        self.reset_for_turn()

    # ── Per-turn state ──────────────────────────────────────────────

    def reset_for_turn(
        self,
        user_message: str | None = None,
        conversation_history: list[dict] | None = None,
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

        if self.config.complexity_adaptive:
            history_summary = ""
            if conversation_history:
                history_summary = _build_history_summary(conversation_history)

            if self.config.use_llm_classifier and user_message:
                detected = _classify_via_llm(user_message, history_summary) or "normal"
            else:
                detected = detect_complexity(user_message) if user_message else "normal"
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

    @property
    def _investigation_done(self) -> bool:
        """调查是否达标——只读调用次数 >= 当前复杂度要求的次数。"""
        return self._read_only_count >= self._active_profile.min_read_only_calls

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
    ) -> str | None:
        """批量门控检查——同一 assistant_message 只调一次。

        Args:
            assistant_content: 当前 assistant_message 的 content 文本
            tool_names: 当前批次所有工具名

        Returns:
            拦截消息（JSON error string），放行时返回 None
        """
        if not self.config.enabled or self._satisfied:
            return None

        has_mutating = any(t in GATED_TOOL_NAMES for t in tool_names)
        has_read_only = any(t not in GATED_TOOL_NAMES for t in tool_names)
        content_len = len(assistant_content or "")

        if has_read_only:
            self._read_only_count += 1

        if self._try_unlock(content_len):
            return None

        if not has_mutating:
            logger.debug(
                "read-think gate: read-only batch — continuing (investigation %d/%d complexity=%s)",
                self._read_only_count,
                self._active_profile.min_read_only_calls,
                self._active_complexity,
            )
            return None

        self._reasoning_rounds += 1

        first_gated = next(t for t in tool_names if t in GATED_TOOL_NAMES)
        block_msg = self._build_block_message(first_gated, content_len)
        logger.info(
            "read-think gate: blocking %s (round %d/%d, content=%d, investigated=%d/%d, complexity=%s)",
            first_gated,
            self._reasoning_rounds,
            self._active_profile.max_reasoning_rounds,
            content_len,
            self._read_only_count,
            self._active_profile.min_read_only_calls,
            self._active_complexity,
        )
        return _make_synthetic_result(first_gated, block_msg, content_len)

    def _try_unlock(self, content_len: int) -> bool:
        """尝试解锁。返回 True 如果状态已变为 satisfied。

        解锁优先级（任一满足即解锁）：
        1. 直接充分推理：content >= 当前复杂度 min_reasoning_chars
        2. 调查后反射：调查完成 + content >= 当前复杂度 min_reflection_chars
        3. 调查后无条件解锁（unlock_after_investigation=True 时）
        4. 推理轮数耗尽：强制解锁防死循环
        """
        profile = self._active_profile

        # 条件1：直接充分推理
        if content_len >= profile.min_reasoning_chars:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — direct reasoning %d chars >= %d (complexity=%s)",
                content_len, profile.min_reasoning_chars, self._active_complexity,
            )
            return True

        # 条件2：调查完成 + 简要反思
        if self._investigation_done and content_len >= profile.min_reflection_chars:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — investigation done (%d/%d reads) + reflection %d chars >= %d (complexity=%s)",
                self._read_only_count, profile.min_read_only_calls,
                content_len, profile.min_reflection_chars, self._active_complexity,
            )
            return True

        # 条件3：调查后无条件解锁——避免"做了调查但回复简短"被反复拦截
        if self._investigation_done and self.config.unlock_after_investigation:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — investigation done (unconditional, content=%d complexity=%s)",
                content_len, self._active_complexity,
            )
            return True

        # 条件4：推理轮数耗尽 → 强制解锁
        if self._reasoning_rounds >= profile.max_reasoning_rounds:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — max reasoning rounds reached (%d, complexity=%s)",
                profile.max_reasoning_rounds, self._active_complexity,
            )
            return True

        return False

    def _build_block_message(self, tool_name: str, content_len: int) -> str:
        """生成引导性拦截消息——告诉 agent 要做什么，不只是禁什么。"""
        profile = self._active_profile
        remaining = profile.min_read_only_calls - self._read_only_count
        if not self._investigation_done:
            if self._read_only_count == 0:
                guide = (
                    "你还没有做任何调查。请先用只读工具收集相关信息：\n"
                    "  · search_files — 搜索代码库\n"
                    "  · read_file — 阅读相关文件\n"
                    "  · web_search — 搜索技术文档\n"
                    f"至少 {profile.min_read_only_calls} 次只读调查后，执行工具将自动解锁。"
                )
            else:
                guide = (
                    f"调查次数不足（{self._read_only_count}/{profile.min_read_only_calls}）。\n"
                    f"还需要 {remaining} 次只读调查：\n"
                    "  · search_files — 搜索代码库\n"
                    "  · read_file — 阅读相关文件\n"
                    "  · web_search — 搜索技术文档"
                )
        else:
            guide = (
                "调查已完成，但分析不够深入。请逐条回答以下问题：\n"
                "  1. 现状全貌：读了哪些文件？关键发现是什么？有没有同类问题？\n"
                "  2. 方案对比：有几条可行路径？各自优劣？为什么选这条？\n"
                "  3. 架构影响：上下游谁受影响？有没有更优雅的方式？\n"
                "  4. 验收标准：改完怎么验证？对齐了哪些用户铁律？\n"
                f"至少 {profile.min_reflection_chars} 字符。敷衍的空洞回答会被再次拦截。"
            )

        complexity_label = {"simple": "简单", "normal": "标准", "complex": "复杂"}.get(
            self._active_complexity, "标准"
        )
        return (
            f"[ReadThink Gate — 推理阶段 · {complexity_label}任务] 工具 '{tool_name}' 暂时不可用。\n\n"
            f"{guide}\n\n"
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
