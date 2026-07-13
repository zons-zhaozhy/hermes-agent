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
  3. 推理轮数达到上限（max_reasoning_rounds）→ 防死循环

与 tool_guardrails 的区别：
  - guardrails 检测"循环失败"（重复调用同一工具失败）
  - read_think_gate 执行"推理期/执行期分离"（先调查再动手）

生命周期：per-turn，由 build_turn_context 重置。
"""

from __future__ import annotations

import logging
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


@dataclass(frozen=True)
class ReadThinkGateConfig:
    """config.yaml → read_think_gate 段配置。"""

    enabled: bool = True
    # 推理阶段最大 API 调用轮数。超过后自动解锁，防死循环。
    max_reasoning_rounds: int = 5
    # 直接解锁：content 达到此字符数即视为"充分推理"。
    min_reasoning_chars: int = 80
    # 调查后解锁：完成至少一次只读工具调用 + content 达到此值即可。
    min_reflection_chars: int = 20

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ReadThinkGateConfig":
        if not isinstance(data, Mapping):
            return cls()
        return cls(
            enabled=_as_bool(data.get("enabled"), True),
            max_reasoning_rounds=_positive_int(data.get("max_reasoning_rounds"), 5),
            min_reasoning_chars=_positive_int(data.get("min_reasoning_chars"), 80),
            min_reflection_chars=_positive_int(data.get("min_reflection_chars"), 20),
        )


class ReadThinkGate:
    """Per-turn 两阶段审议门控制器。

    推理阶段：只读工具放行，执行工具引导调查。
    执行阶段：全部放行。

    生命周期：
      1. build_turn_context → reset_for_turn()   （进入推理阶段）
      2. 每轮 LLM 响应 → check_batch(content, tool_names) → 放行/拦截
      3. 解锁后 → 本 turn 后续全部放行
    """

    def __init__(self, config: ReadThinkGateConfig | None = None):
        self.config = config or ReadThinkGateConfig()
        self.reset_for_turn()

    # ── Per-turn state ──────────────────────────────────────────────

    def reset_for_turn(self) -> None:
        """每轮开始时重置——回到推理阶段。"""
        self._satisfied: bool = False
        self._reasoning_rounds: int = 0
        self._investigation_done: bool = False

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
        # 门关闭 or 已解锁 → 放行
        if not self.config.enabled or self._satisfied:
            return None

        has_mutating = any(t in GATED_TOOL_NAMES for t in tool_names)
        has_read_only = any(t not in GATED_TOOL_NAMES for t in tool_names)
        content_len = len(assistant_content or "")

        # 标记调查进度——只要调用了只读工具就算
        if has_read_only:
            self._investigation_done = True

        # ── 解锁判定 ────────────────────────────────────────────────

        if self._try_unlock(content_len):
            return None

        # ── 推理阶段：无门控工具 → 放行（让调查继续）───────────────
        # 只读批次不消耗 reasoning_rounds——调查就是调查，不是"推理轮数"
        if not has_mutating:
            logger.debug(
                "read-think gate: read-only batch — continuing (investigation done=%s)",
                self._investigation_done,
            )
            return None

        # ── 推理阶段：有门控工具且未解锁 → 拦截 ─────────────────────
        # 只有被拦截的执行尝试才计数，防止只读调查白白消耗轮数
        self._reasoning_rounds += 1

        first_gated = next(t for t in tool_names if t in GATED_TOOL_NAMES)
        block_msg = self._build_block_message(first_gated, content_len)
        logger.info(
            "read-think gate: blocking %s (round %d/%d, content=%d, investigated=%s)",
            first_gated,
            self._reasoning_rounds,
            self.config.max_reasoning_rounds,
            content_len,
            self._investigation_done,
        )
        return _make_synthetic_result(first_gated, block_msg, content_len)

    def _try_unlock(self, content_len: int) -> bool:
        """尝试解锁。返回 True 如果状态已变为 satisfied。"""
        # 条件1：直接充分推理
        if content_len >= self.config.min_reasoning_chars:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — direct reasoning %d chars >= %d",
                content_len,
                self.config.min_reasoning_chars,
            )
            return True

        # 条件2：调查完成 + 简要反思
        if self._investigation_done and content_len >= self.config.min_reflection_chars:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — investigation done + reflection %d chars >= %d",
                content_len,
                self.config.min_reflection_chars,
            )
            return True

        # 条件3：推理轮数耗尽 → 强制解锁
        if self._reasoning_rounds >= self.config.max_reasoning_rounds:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — max reasoning rounds reached (%d)",
                self.config.max_reasoning_rounds,
            )
            return True

        return False

    def _build_block_message(self, tool_name: str, content_len: int) -> str:
        """生成引导性拦截消息——告诉 agent 要做什么，不只是禁什么。"""
        if not self._investigation_done:
            guide = (
                "你还没有做任何调查。请先用只读工具收集相关信息：\n"
                "  · search_files — 搜索代码库\n"
                "  · read_file — 阅读相关文件\n"
                "  · web_search — 搜索技术文档\n"
                "完成调查后，总结你的发现和分析，执行工具将自动解锁。"
            )
        else:
            guide = (
                "你已经完成了调查，但还没有给出分析。\n"
                "请总结你的发现，说明：\n"
                "  1. 问题本质是什么？\n"
                "  2. 你的分析和方案是什么？\n"
                "  3. 为什么选这个路径？\n"
                f"至少 {self.config.min_reflection_chars} 字符的分析即可解锁执行权限。"
            )

        return (
            f"[ReadThink Gate — 推理阶段] 工具 '{tool_name}' 暂时不可用。\n\n"
            f"{guide}\n\n"
            f"（推理轮数：{self._reasoning_rounds}/{self.config.max_reasoning_rounds}）"
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
