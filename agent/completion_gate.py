"""Completion gate — 代码级门控，防止LLM过早停止或跳步。

三个门控:
1. 完成度检查: LLM说"完了"时，代码验证是否真的做完了
2. 工具前置约束: 写文件前必须先读过
3. 迭代下限: 代码修改类任务至少跑N轮

全部默认关闭，通过 agent._completion_gate_enabled / _tool_prereq_enabled /
_iteration_floor_enabled 开启。

设计原则:
- 不调LLM，全是确定性Python布尔判断
- 不改核心循环结构，只做插入
- 不改函数签名
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# 常量
# =============================================================================

# 需要读前检查的写操作工具
_WRITE_TOOLS: Set[str] = {"write_file", "patch"}

# 满足"已读取"条件的工具
_READ_TOOLS: Set[str] = {"read_file", "search_files", "browser_snapshot", "browser_vision"}

# 需要验证的修改类工具
_MUTATE_TOOLS: Set[str] = {"write_file", "patch", "terminal"}

# 验证类工具
_VERIFY_TOOLS: Set[str] = {"terminal"}

# 代码修改类任务的最少迭代轮数
_ITERATION_FLOOR_MUTATE = 3  # 读→改→验证

# 泛话模式 — 回答全是空洞套话，没有实质性内容
_VAGUE_PATTERNS = [
    r"^(好的|明白|没问题|收到|了解)",
    r"^(我会|我将|我马上|让我来|我来)",
    r"(我将帮你|我来帮你|让我来帮你|我会帮你)",
]

# 最大完成度检查重试次数（防止无限循环）
_MAX_COMPLETION_RETRIES = 3


# =============================================================================
# 门控1：完成度检查
# =============================================================================

def check_completion(
    turn_tool_names: List[str],
    turn_failed_mutations: Dict[str, Dict[str, Any]],
    original_user_message: str,
    final_response: str,
    completion_retry_count: int,
) -> tuple[bool, Optional[str]]:
    """检查LLM的"最终回答"是否真的完成了任务。

    Args:
        turn_tool_names: 本轮调用过的工具名称列表
        turn_failed_mutations: 本轮失败的文件写入 {path: {error: ...}}
        original_user_message: 用户原始消息
        final_response: LLM的最终回答文本
        completion_retry_count: 之前已完成度检查重试的次数

    Returns:
        (should_continue, nudge_message)
        should_continue=True 表示没做完，推回去继续
        should_continue=False 表示通过检查，可以结束
    """
    # 防止无限循环
    if completion_retry_count >= _MAX_COMPLETION_RETRIES:
        return False, None

    failed_checks: List[str] = []

    # 检查1：失败的文件写入
    if turn_failed_mutations:
        failed_files = list(turn_failed_mutations.keys())
        failed_checks.append(
            f"文件写入失败({len(failed_files)}个): {', '.join(failed_files[:3])}"
        )

    # 检查2：修改了文件但没有验证（读回来确认或跑测试）
    mutate_tools_used = set(turn_tool_names) & _MUTATE_TOOLS
    verify_tools_used = set(turn_tool_names) & _VERIFY_TOOLS
    if mutate_tools_used and not verify_tools_used:
        failed_checks.append("修改了代码/文件但没有验证结果（需要回读文件或运行测试）")

    # 检查3：直接写文件但没有先读取（跳步检查）
    write_used = set(turn_tool_names) & _WRITE_TOOLS
    read_used = set(turn_tool_names) & _READ_TOOLS
    if write_used and not read_used:
        failed_checks.append("直接修改文件但没有先读取内容（跳步）")

    # 检查4：回答是纯泛话（说了要做但没做）
    if _is_vague_response(final_response) and not turn_tool_names:
        failed_checks.append("只表达了意图但没有执行任何操作")

    if not failed_checks:
        return False, None

    # 构建推回消息
    nudge = "任务尚未完成，以下检查未通过：\n" + "\n".join(
        f"  - {c}" for c in failed_checks
    )
    nudge += "\n请继续执行，直到所有检查通过。"
    return True, nudge


def _is_vague_response(text: str) -> bool:
    """判断回答是否是纯泛话（没有实质内容）。"""
    if not text or not text.strip():
        return True
    stripped = text.strip()
    # 短回答 + 匹配泛话模式 = 纯泛话
    if len(stripped) < 100:
        return any(re.search(p, stripped) for p in _VAGUE_PATTERNS)
    return False


# =============================================================================
# 门控2：工具前置约束
# =============================================================================

def check_tool_prerequisite(
    tool_name: str,
    turn_tool_log: Set[str],
) -> Optional[str]:
    """检查调用某个工具是否满足了前置条件。

    Args:
        tool_name: 即将调用的工具名
        turn_tool_log: 本轮已经执行过的工具名集合

    Returns:
        None 表示可以执行
        str 表示阻止消息
    """
    if tool_name not in _WRITE_TOOLS:
        return None

    # 已有读取记录，放行
    if turn_tool_log & _READ_TOOLS:
        return None

    # 没读过任何文件就尝试写
    return (
        f"工具 {tool_name} 要求先读取相关文件。"
        f"请先使用 read_file 或 search_files 了解现有内容，"
        f"再进行修改。"
    )


# =============================================================================
# 门控3：迭代下限
# =============================================================================

def compute_iteration_floor(
    turn_tool_names: List[str],
    api_call_count: int,
) -> int:
    """根据本轮工具使用情况计算最低迭代轮数。

    Args:
        turn_tool_names: 本轮调用过的工具名称列表
        api_call_count: 当前已执行的API调用轮数

    Returns:
        最低要求的迭代轮数（0表示不限制）
    """
    mutate_used = bool(set(turn_tool_names) & _MUTATE_TOOLS)
    if not mutate_used:
        return 0  # 非修改类任务不限制

    return _ITERATION_FLOOR_MUTATE


def check_iteration_floor(
    api_call_count: int,
    min_iterations: int,
) -> tuple[bool, Optional[str]]:
    """检查当前迭代次数是否达到下限。

    Args:
        api_call_count: 当前API调用轮数
        min_iterations: 最低要求轮数

    Returns:
        (should_continue, nudge_message)
    """
    if min_iterations <= 0 or api_call_count >= min_iterations:
        return False, None

    return True, (
        f"任务涉及代码修改，至少需要 {min_iterations} 轮操作"
        f"（读取→修改→验证）。当前只有 {api_call_count} 轮，请继续执行。"
    )
