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
import json
import logging
import os
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


def _four_axis_marker_path() -> Path:
    """解析四轴 marker 文件路径（profile-aware，每次调用现算）。

    Returns:
        Path: hermes_home/cache/four_axis_gate.json
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "four_axis_gate.json"

# 四轴关键词表（2026-08-26 恢复：e12e46fd3 删除后 use_llm_judge=false 时四轴无解锁路径）。
# 子串匹配 O(n) 无回溯（历史上 re.search 曾致灾难性回溯 CPU 100%，禁用正则）。
_FOUR_AXIS_KEYWORDS: dict[str, list[str]] = {
    "影响面": [
        "[源码确认]", "[搜索推断]",
        "caller", "consumer", "importer",
        "调用方", "依赖此接口",
        "影响面清单",
    ],
    "原意图": [
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


# 四轴闸门只对直接代码编辑工具强制要求。
# execute_code 不在此列——它可以是纯只读（审计/查询/分析），
# 也可以通过 write_file/patch 间接写文件（后者已被四轴覆盖）。
# terminal（即使动态 gated）不需要四轴——运维操作不产出"风险矩阵"。
_FOUR_AXIS_REQUIRED_TOOLS: frozenset[str] = frozenset(
    {
        "write_file",
        "patch",
    }
)


def _clear_four_axis_marker() -> None:
    """清除四轴 marker 文件——每个 turn 开始时调用。"""
    try:
        _marker = _four_axis_marker_path()
        if _marker.exists():
            _marker.unlink()
    except Exception:
        logger.warning("failed to clear four-axis marker file", exc_info=True)

# 默认门控的执行类工具——只覆盖代码编辑工具。
# terminal/browser/delegate_task/cronjob/process 等运维交互工具不在默认门控范围——
# 它们不是代码编辑，拦截它们会把 MySQL 运维、docker、git、测试等操作误伤。
# 用户可通过 config.yaml → read_think_gate.gated_tools 扩展列表。
GATED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "write_file",
        "patch",
        "execute_code",
    }
)

# 只读调查工具白名单——只有这些工具的调用才算真正的"调查"。
# 其他非 gated 工具（如 memory、process、cronjob 等）不算调查行为。
READ_ONLY_INVESTIGATION_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "browser_navigate",
        "browser_snapshot",
        "browser_console",
        "browser_vision",
        "browser_get_images",
        "skill_view",
    }
)

# terminal 中可能写入文件的 shell 操作模式——用于检测绕过 gate 的文件写入。
# 用 _terminal_writes_file() 函数检测，而非简单子串匹配。
# 优化原则：只匹配确实表示写文件的模式，避免 "grep 'cp '" / "git commit -m 'mv'" 误报。

# 高置信度写操作——几乎不会出现在非写入上下文
_TERMINAL_WRITE_PATTERNS_HIGH: tuple[str, ...] = (
    "sed -i",      # sed 原地编辑——唯一用途就是改文件
    "dd of=",      # dd 输出到文件——唯一用途就是写
    "rsync ",      # rsync——同步文件到目标
    "tee /",       # tee 写绝对路径
    "tee ~",       # tee 写 home 目录
    "tee ./",      # tee 写相对路径文件
    "tee \"",       # tee 写引号包裹的路径
    "tee '",       # tee 写引号包裹的路径
)


def _terminal_writes_file(command: str) -> bool:
    """检测 terminal 命令是否包含文件写入操作。

    策略：
    1. 高置信度模式（sed -i/dd of=/rsync/tee+路径）直接匹配
    2. shell 重定向（> >>）后跟文件路径特征——排除 /dev/null/2>&1
    3. cp/mv 目标是文件路径——要求前导空格 + 路径特征（排除 grep/docker/git 中的字符串）
    """
    if not command:
        return False

    # 0. heredoc 模式不是直接的文件写入——命令只是启动解释器
    #    python3 << 'PYEOF' / cat > "file" << 'EOF' 是 heredoc 输入，不拦截
    if "<<" in command:
        return False

    # 1. 高置信度模式
    for pat in _TERMINAL_WRITE_PATTERNS_HIGH:
        if pat in command:
            return True

    # 2. shell 重定向——检测 > 或 >> 后面跟文件路径
    #    排除：> /dev/null, > /dev/stdout, 2>&1, >&2, > &-
    _REDIRECT_SINKS = ("/dev/null", "/dev/stdout", "/dev/stderr")
    idx = 0
    while True:
        gt_pos = command.find(">", idx)
        if gt_pos == -1:
            break
        # 跳过 >& 和 2>& 等操作符
        after = command[gt_pos + 1:]
        if after.startswith("&"):
            idx = gt_pos + 1
            continue
        # 取 > 后面的目标
        target = after.lstrip("> ").strip()
        # 如果目标以文件路径特征开头，且不是已知 sink
        if target and not any(target.startswith(s) for s in _REDIRECT_SINKS):
            # 检查是否像文件路径
            first_token = target.split()[0] if target.split() else target
            if (first_token.startswith(("/", "~", "\"", "'", "./", "../"))
                    or "." in first_token  # file.py, config.yaml
                    or first_token.endswith((".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".sh", ".md", ".txt", ".sql"))):
                return True
        idx = gt_pos + 1

    # 3. cp/mv——要求前面有管道或空格（排除 grep 'cp '），且目标像文件路径
    for cmd_prefix in (" cp ", " mv "):
        pos = command.find(cmd_prefix)
        if pos == -1:
            continue
        # 确保不是在引号内（grep 'cp ...'）
        before = command[:pos + 1]  # 包含 "cp " 的 "cp" 部分
        # 简单检查：如果 cp/mv 前面是引号或不是命令边界，跳过
        # 实际 cp/mv 作为命令时前面通常是行首、管道、分号、&& 或 ||
        preceding = command[:pos].rstrip()
        if preceding and not preceding.endswith(("|", ";", "&&", "||")):
            # 检查 preceding 的最后一个 token 是否是 cp/mv 的合法前导
            last_word = preceding.split()[-1] if preceding.split() else ""
            # 如果 last_word 不像是命令结束符，可能是 grep/docker 等参数内的 cp
            if last_word and not last_word.endswith(("cp", "mv", "&&", "||", ";")):
                continue
        return True

    return False


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

A. 代码理解：是否说明了目标代码当前的逻辑？（引用了具体函数/行号/数据流）
B. 关系分析：是否识别了与目标代码有关系的既有程序？（调用方、被依赖方、同类实现）
C. 既有模式：是否检查了项目中是否已有等价实现？是否说明了既有程序是怎么做类似事情的？
D. 方案评估：是否论证了即将采取的做法是最优的？是否考虑过替代方案？

判定规则：
- 4 条全 PASS → APPROVED
- 有任何 FAIL → NEEDS_MORE_WORK
- 分析内容为空或只有意图陈述 → NEEDS_MORE_WORK
- 重要：如果前几轮已反馈过某个不足，而 agent 本轮分析中已针对该不足做了补充，
  则该项应判 PASS。不要反复找新角度——收敛判定。
- 如果 4 个维度已经基本覆盖（即使深度不够完美），也应该 APPROVED。
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
          调用方应递增 fail_count 而非重置。
    """
    try:
        from agent.auxiliary_client import get_text_auxiliary_client, build_judge_thinking_extra_body

        client, model = get_text_auxiliary_client("investigation_judge")
        if client is None or not model:
            # 漏洞 4 修复：fail-closed 前 2 次（不允许），fail-open 第 3 次（防死锁）
            if fail_count < 2:
                logger.warning(
                    "read-think gate: no auxiliary client for judge → block (fail_count=%d)", fail_count,
                )
                return False, "调查质量评审服务暂不可用，请输出完整分析后重试。", True
            logger.warning(
                "read-think gate: no auxiliary client, fail_count=%d → allow (fail-open)", fail_count,
            )
            return True, "", True

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
        logger.info("read-think gate: judge NEEDS_MORE_WORK: %s", feedback[:100])
        return False, feedback, False

    except Exception:
        # 漏洞 4 修复：fail-closed 前 2 次，fail-open 第 3 次
        if fail_count < 2:
            logger.warning(
                "read-think gate: judge failed → block (fail_count=%d)", fail_count, exc_info=True,
            )
            return False, "调查质量评审服务异常，请输出完整分析后重试。", True
        logger.warning(
            "read-think gate: judge failed, fail_count=%d → allow (fail-open)", fail_count, exc_info=True,
        )
        return True, "", True


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
    # 是否用 LLM-as-judge 评估调查质量（严格模式下生效）
    use_llm_judge: bool = True
    # 用户自定义额外门控工具（合并到 GATED_TOOL_NAMES 之上）。
    # 默认只门控 write_file/patch/execute_code；如需门控 terminal 等，在此追加。
    gated_tools: tuple[str, ...] = ()

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
            use_llm_judge=_as_bool(data.get("use_llm_judge"), True),
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

    def __init__(self, config: ReadThinkGateConfig | None = None):
        self.config = config or ReadThinkGateConfig()
        # 合并默认门控工具 + 用户自定义扩展，形成实例级集合。
        # check_batch 用这个 set 而不是直接引用全局 GATED_TOOL_NAMES。
        self._gated_tools: set[str] = set(GATED_TOOL_NAMES)
        if self.config.gated_tools:
            self._gated_tools.update(self.config.gated_tools)
        self._four_axis_found: set[str] = set()  # 本 turn 四轴证据集合
        self._turn_start_time: float = 0.0  # 墙钟逃逸阀
        self._total_tool_calls: int = 0  # 总工具调用次数（含read-only），防膨胀
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
        self._files_read: set[str] = set()  # 本 turn 读取的文件路径
        self._investigation_evidence: list[str] = []  # 所有调查工具的证据摘要
        self._user_message: str = user_message or ""  # judge 用：当前任务描述
        self._last_judge_feedback: str = ""  # judge 最近一次反馈
        # 漏洞 2 修复：judge 历史反馈列表——让 judge 知道前几轮已要求过什么
        self._judge_feedback_history: list[str] = []
        # 漏洞 4 修复：judge 连续失败计数——达到阈值后 fail-open
        self._judge_fail_count: int = 0
        # 四轴闸门——每 turn 重新计数
        self._four_axis_found: set[str] = set()
        # 清除上一 turn 的四轴 marker 文件
        _clear_four_axis_marker()
        # 漏洞修复：记录 turn 开始时间——墙钟逃逸阀防止用户感知的"卡死"
        self._turn_start_time = time.time()
        self._total_tool_calls = 0
        # 清理上一 turn 动态加入 _gated_tools 的 terminal（漏洞 3 修复副作用）。
        # 但保留用户通过 config 显式配置的 gated_tools。
        if "terminal" not in self.config.gated_tools:
            self._gated_tools.discard("terminal")

        if self.config.complexity_adaptive:
            history_summary = ""
            if conversation_history:
                history_summary = _build_history_summary(conversation_history)

            if self.config.use_llm_classifier and user_message:
                detected = _classify_via_llm(user_message, history_summary) or "normal"
            else:
                # 2026-08-18 修复：use_llm_classifier=False 时必须真正绕过 LLM。
                # 此前 else 分支调 detect_complexity()，其内部无条件优先
                # _classify_via_llm()，导致 config.yaml 的 false 被架空——
                # 每 turn 仍烧一次同步 LLM 调用（timeout=10s，阻塞回合启动），
                # 实测日志 110 次 "LLM classified" 均发生在 false 配置下。
                detected = (
                    _fallback_detect(user_message.lower())
                    if user_message
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

    @property
    def _investigation_done(self) -> bool:
        """调查是否达标——至少做了 1 次只读调查。"""
        return self._read_only_count >= 1

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
        # 四轴关键词通常在结论部分。对超长文本只扫描尾部 8KB，
        # 避免超大 content 做全文扫描的 CPU 开销。
        scan_text = content if len(content) <= 8192 else content[-8192:]
        for axis, keywords in _FOUR_AXIS_KEYWORDS.items():
            if axis in self._four_axis_found:
                continue
            for kw in keywords:
                if kw in scan_text:
                    self._four_axis_found.add(axis)
                    logger.info(
                        "read-think gate: four-axis '%s' detected — %d/4 axes found",
                        axis, len(self._four_axis_found),
                    )
                    break
        # 四轴全部到位 → 写入 marker 文件
        if len(self._four_axis_found) == 4:
            try:
                _marker = _four_axis_marker_path()
                _marker.parent.mkdir(parents=True, exist_ok=True)
                _marker.write_text(json.dumps({
                    "verified": True,
                    "source": "content_scan",
                    "timestamp": time.time(),
                    "axes": sorted(self._four_axis_found),
                }))
                logger.info("read-think gate: four-axis complete via content scan — marker written")
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
            _marker = _four_axis_marker_path()
            _marker.parent.mkdir(parents=True, exist_ok=True)
            _marker.write_text(json.dumps({
                "verified": True,
                "source": "llm_judge",
                "timestamp": time.time(),
                "axes": sorted(self._four_axis_found),
            }))
            logger.info("read-think gate: four-axis complete via LLM judge — marker written")
        except Exception:
            logger.warning("read-think gate: failed to write four-axis marker", exc_info=True)

    def _four_axis_complete(self) -> bool:
        """四轴是否全部检测到。"""
        return len(self._four_axis_found) == 4

    def _missing_axes(self) -> list[str]:
        """返回尚未检测到的轴名称列表。"""
        return [a for a in _FOUR_AXIS_LABELS if a not in self._four_axis_found]

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

        if not self.config.enabled or self._satisfied:
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
                            "read-think gate: terminal command writes file → treated as gated: %s", cmd[:80],
                        )
                        break

        has_mutating = any(t in self._gated_tools for t in tool_names) or terminal_writing_files
        # ── 漏洞 5 修复：只对白名单中的调查工具计数 ──
        has_read_only_investigation = any(
            t in READ_ONLY_INVESTIGATION_TOOLS for t in tool_names
        )
        content_text = assistant_content or ""
        content_len = len(content_text)

        if has_read_only_investigation:
            self._read_only_count += 1

        # 总工具调用计数——防止 read-only 和 gated 交替膨胀
        self._total_tool_calls += len(tool_names)

        # ── write-target 覆盖率检查（优先于 _try_unlock）──
        # 如果要改的文件已经读过，或文件不存在（新建），直接放行。
        # 不需要再走 diversity/推理门槛——目的性调查已经完成。
        if tool_args and not self.config.unlock_after_investigation:
            target_satisfied = False
            for tn, ta in zip(tool_names, tool_args):
                if tn in ("write_file", "patch") and isinstance(ta, dict):
                    wp = ta.get("path", "")
                    if wp:
                        if not os.path.exists(wp):
                            logger.debug(
                                "read-think gate: write target %s is new file → skip read check",
                                wp,
                            )
                            target_satisfied = True
                            break
                        if wp in self._files_read:
                            self._satisfied = True
                            logger.info(
                                "read-think gate: unlocked — write target %s was read before edit",
                                wp,
                            )
                            return None
            if target_satisfied:
                # 所有 write/patch target 都满足 → 放行
                return None

        if self._try_unlock(content_len, content_text, tool_names):
            return None

        if not has_mutating:
            logger.debug(
                "read-think gate: read-only batch — continuing (investigation %d complexity=%s)",
                self._read_only_count,
                self._active_complexity,
            )
            return None

        # ── 推理轮数递增 ──
        # _try_unlock 的 max_reasoning_rounds 兜底在递增之前检查，
        # 所以当 _reasoning_rounds 达到 max 时直接放行，不会多一轮。
        # 这里递增后，拦截消息显示的轮次与实际一致（第1次拦截显示 1/N）。
        self._reasoning_rounds += 1

        first_gated = next(t for t in tool_names if t in self._gated_tools) if any(t in self._gated_tools for t in tool_names) else "terminal"
        block_msg = self._build_block_message(first_gated, content_len)
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
        """尝试解锁。返回 True 如果状态已变为 satisfied。

        严格模式四层门控：
          1. 机械门槛：调查次数 + 推理量达标
          2. LLM-as-judge：语义评估调查质量（代码理解/关系分析/既有模式/方案评估）
          3. 四轴闸门：影响面/原意图/根因/风险 四轴证据全部在回复中出现
          4. max_reasoning_rounds 兜底防死循环（跳过四轴检查）
        """
        profile = self._active_profile

        # ── 漏洞 7 修复：兜底检查移到这里，在调查/推理量检查之前 ──
        # 这样推理轮数达到上限时直接放行，不会多等一轮。
        if self._reasoning_rounds >= profile.max_reasoning_rounds:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — max reasoning rounds reached (%d, complexity=%s)",
                profile.max_reasoning_rounds, self._active_complexity,
            )
            return True

        # ── 墙钟逃逸阀：turn 开始后超过 3 分钟直接解锁 ──
        # 防止 LLM 模型每轮调用耗时长（reasoning model 1-2 min/轮）
        # 导致 8 轮 × 2 分钟 = 16 分钟用户感知"卡死"。
        # 3 分钟是用户耐心的合理上限——超过即认为调查已足够。
        if self._turn_start_time > 0:
            elapsed = time.time() - self._turn_start_time
            if elapsed > 180:
                self._satisfied = True
                logger.info(
                    "read-think gate: unlocked — wall-clock escape valve (%.0fs > 180s, rounds=%d/%d, complexity=%s)",
                    elapsed, self._reasoning_rounds, profile.max_reasoning_rounds,
                    self._active_complexity,
                )
                return True

        # ── 总调用次数逃逸阀：防止 read-only/gated 交替膨胀 ──
        # 场景：agent 做 search → terminal 被 block → search → terminal 被 block ...
        # read-only 轮不计入 _reasoning_rounds，导致实际 LLM 调用数远超 max。
        # 上限 = max_reasoning_rounds × 3（每次 block 最多允许 2 次 read-only 穿插）。
        total_call_cap = profile.max_reasoning_rounds * 3
        if self._total_tool_calls >= total_call_cap:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — total tool call cap reached (%d >= %d, rounds=%d/%d, complexity=%s)",
                self._total_tool_calls, total_call_cap,
                self._reasoning_rounds, profile.max_reasoning_rounds,
                self._active_complexity,
            )
            return True

        if not self.config.unlock_after_investigation:
            # 严格模式第一关：调查量 + 推理量 双达标
            if (self._read_only_count >= profile.min_read_only_calls
                    and content_len >= profile.min_reasoning_chars):
                # 严格模式第二关：LLM-as-judge 语义评估
                if self.config.use_llm_judge and content:
                    tool_name = ""
                    if tool_names:
                        tool_name = next(
                            (t for t in tool_names if t in self._gated_tools), tool_names[0] if tool_names else ""
                        )
                    # ── 漏洞 2 修复：传入 judge 历史 ──
                    # ── 漏洞 4 修复：传入 fail_count ──
                    approved, feedback, was_infra_failure = _judge_investigation(
                        self._user_message, tool_name, self._files_read, content,
                        investigation_evidence=self._investigation_evidence,
                        judge_history=self._judge_feedback_history,
                        fail_count=self._judge_fail_count,
                    )
                    if not approved:
                        # judge 不通过——不解锁，反馈给 agent
                        # 区分：基础设施失败 → 递增 fail_count；judge 合法拒绝 → 重置 fail_count
                        if was_infra_failure:
                            self._judge_fail_count += 1
                        else:
                            self._judge_fail_count = 0
                            self._judge_feedback_history.append(feedback)
                        self._last_judge_feedback = feedback
                        logger.info(
                            "read-think gate: judge rejected (reads=%d, complexity=%s, history=%d): %s",
                            self._read_only_count, self._active_complexity,
                            len(self._judge_feedback_history), feedback[:80],
                        )
                        return False
                    else:
                        # judge 通过——重置状态 + 标记四轴完成
                        self._judge_fail_count = 0
                        self.mark_four_axis_complete()
                elif not self._has_diverse_investigation(profile):
                    # 无 LLM judge 时：调查必须包含搜索类工具（search_files/web_search/web_extract），
                    # 不能只是 read_file 堆数量。防止三连 read_file 凑数过关。
                    logger.info(
                        "read-think gate: investigation not diverse — need search_files/web_search/web_extract "
                        "(reads=%d, complexity=%s)",
                        self._read_only_count, self._active_complexity,
                    )
                    return False
                else:
                    # 无 LLM judge 但调查多样化达标 → 不自动标记四轴完成。
                    # 四轴由内容检测（_four_axis_found）自然累积，或由 LLM judge 显式通过。
                    # 之前这里错误地调用 mark_four_axis_complete() 导致 diversity 替代四轴，
                    # 严格模式下 write_file 被四轴形同虚设。
                    pass
                # 四轴闸门——严格模式下四轴不齐不解锁。
                # 但只对核心代码编辑工具（write_file/patch/execute_code）强制要求。
                # terminal 即使被动态 gated（含文件写入），也不要求四轴——
                # 运维操作（pytest/grep/tail）不应被"风险矩阵"卡住。
                requesting_core_edit = bool(
                    tool_names and any(t in _FOUR_AXIS_REQUIRED_TOOLS for t in tool_names)
                )
                if requesting_core_edit and not self._four_axis_complete():
                    missing = self._missing_axes()
                    logger.info(
                        "read-think gate: four-axis incomplete — missing %s (judge path, complexity=%s)",
                        missing, self._active_complexity,
                    )
                    return False
                self._satisfied = True
                judge_label = "judge=pass" if self.config.use_llm_judge else "judge=off(diverse)"
                logger.info(
                    "read-think gate: unlocked — strict mode (reads=%d>=%d, reasoning=%d>=%d, %s, complexity=%s)",
                    self._read_only_count, profile.min_read_only_calls,
                    content_len, profile.min_reasoning_chars, judge_label, self._active_complexity,
                )
                return True
        else:
            # 宽松模式：推理文字够长 → 放行（调查量不强制）
            if content_len >= profile.min_reasoning_chars:
                self._satisfied = True
                logger.info(
                    "read-think gate: unlocked — direct reasoning %d chars >= %d (complexity=%s)",
                    content_len, profile.min_reasoning_chars, self._active_complexity,
                )
                return True

        # 宽松模式：做过调查就放行
        if self._investigation_done and self.config.unlock_after_investigation:
            self._satisfied = True
            logger.info(
                "read-think gate: unlocked — investigation done (%d reads, complexity=%s)",
                self._read_only_count, self._active_complexity,
            )
            return True

        return False

    def _build_block_message(self, tool_name: str, content_len: int) -> str:
        """生成拦截消息——注入调查框架 + judge 反馈。"""
        profile = self._active_profile
        done = self._read_only_count
        needed = profile.min_read_only_calls

        if not self.config.unlock_after_investigation:
            # judge 有反馈 → 优先展示
            if self._last_judge_feedback:
                fb = self._last_judge_feedback
                self._last_judge_feedback = ""  # 消费后清空
                return (
                    "[ReadThink Gate — 推理阶段 · 标准任务] 工具 '%s' 暂时不可用。\n\n"
                    "LLM 评审不通过。评审意见：%s\n\n"
                    "你的分析需要覆盖：\n"
                    "  A. 目标代码当前逻辑（引用具体函数/行号/数据流）\n"
                    "  B. 与它有关系的既有程序（调用方、被依赖方、同类实现）\n"
                    "  C. 既有程序是怎么做这件事的（等价实现/可复用的模式）\n"
                    "  D. 你的方案为什么是最优的（对比过的替代方案）\n\n"
                    "（推理轮数：%d/%d）"
                    % (tool_name, fb,
                       self._reasoning_rounds, profile.max_reasoning_rounds)
                )
            if done < needed:
                # ── 漏洞 6 修复：推理轮数用 self._reasoning_rounds 而非硬编码 1 ──
                remaining_reads = needed - done
                return (
                    "[ReadThink Gate — 推理阶段 · 标准任务] 工具 '%s' 暂时不可用。\n\n"
                    "动手前必须搞清楚：\n"
                    "  1. 要改的代码当前怎么写的？逻辑是什么？（读源码，引用行号）\n"
                    "  2. 跟它有关系的既有程序有哪些？调用方在哪？被谁依赖？\n"
                    "  3. 既有程序是怎么做这件事的？是否已有等价实现可以复用或扩展？\n"
                    "  4. 你打算怎么改？这是最优方案吗？有没有更稳妥的做法？\n\n"
                    "用 search_files 搜调用方和同类实现，用 read_file 读目标文件全貌，\n"
                    "用 codegraph/gitnexus 追依赖链。搞清楚再动手。\n\n"
                    "【通过诊断】还需要 %d 次只读调查（当前 %d/%d）+ 至少 %d 字分析。\n"
                    "（推理轮数：%d/%d）"
                    % (tool_name, remaining_reads, done, needed,
                       profile.min_reasoning_chars,
                       self._reasoning_rounds, profile.max_reasoning_rounds)
                )
            # 调查次数达标但缺少搜索类工具——不是 read_file 堆数量就够的
            if not self._has_diverse_investigation(profile):
                return (
                    "[ReadThink Gate — 推理阶段 · 标准任务] 工具 '%s' 暂时不可用。\n\n"
                    "已做 %d 次只读调查（read_file），但还没用搜索类工具。\n"
                    "用 search_files 搜调用方和同类实现，或用 web_search 查外部文档——\n"
                    "搞清楚全局关系，不能只盯着单个文件。\n\n"
                    "【通过诊断】再做 1 次搜索类工具调用（search_files/web_search）即可通过此关。\n"
                    "（调查次数：%d/%d，推理轮数：%d/%d）"
                    % (tool_name, done, done, needed,
                       self._reasoning_rounds, profile.max_reasoning_rounds)
                )
            # 四轴闸门未通过——调查做了但四轴输出不齐
            if not self._four_axis_complete():
                missing = self._missing_axes()
                return (
                    "[ReadThink Gate — 推理阶段 · 四轴闸门] 工具 '%s' 暂时不可用。\n\n"
                    "调查已完成，但四轴闸门未通过。缺少：%s\n\n"
                    "在前面的回复中逐项输出以下四轴内容，缺一不可：\n"
                    "  1. 影响面清单：列出每一个受影响调用方（绝对路径+函数名+行号）\n"
                    "  2. 原意图溯源：git log -p 追溯到最初 commit + 不变量清单（逐条标注保持/修改/破坏）\n"
                    "  3. 根因定位：区分症状位置与根因位置（文件+行号），修复目标必须是根因\n"
                    "  4. 风险矩阵：枚举最坏场景（触发条件+影响范围+可恢复性），覆盖六类风险\n\n"
                    "四轴的输出就是 commit message 的 body。即写即用。\n\n"
                    "【通过诊断】补齐缺轴「%s」的输出即可通过——在接下来的回复中直接写出对应段落。\n"
                    "（推理轮数：%d/%d）"
                    % (tool_name, "、".join(missing), "、".join(missing),
                       self._reasoning_rounds, profile.max_reasoning_rounds)
                )
            return (
                "[ReadThink Gate — 推理阶段 · 标准任务] 工具 '%s' 暂时不可用。\n\n"
                "已做 %d 次只读调查，但还没有输出分析结论。在回复中明确回答：\n\n"
                "  ▸ 要改的代码当前逻辑是什么？（引用具体行号和代码）\n"
                "  ▸ 哪些既有程序与它有关系？它们是怎么做的？\n"
                "  ▹ 你的改动方案是什么？为什么是最优的？（对比过的替代方案）\n\n"
                "用充分的分析填满回复——这不是走流程，是确保改对。\n\n"
                "【通过诊断】调查次数已达标（%d/%d），但推理量不足（当前 %d 字 < 需要 %d 字）。\n"
                "在回复中写出上述分析，达到 %d 字即可通过。\n"
                "（推理轮数：%d/%d）"
                % (tool_name, done, done, needed, content_len,
                   profile.min_reasoning_chars, profile.min_reasoning_chars,
                   self._reasoning_rounds, profile.max_reasoning_rounds)
            )

        if done == 0:
            return "[ReadThink] 先用 search_files/read_file 调查再动手。（轮 %d/%d）" % (
                self._reasoning_rounds, profile.max_reasoning_rounds,
            )
        return "[ReadThink] 读过了但你要改的文件还没读。先 read_file 目标文件。（轮 %d/%d）" % (
            self._reasoning_rounds, profile.max_reasoning_rounds,
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
