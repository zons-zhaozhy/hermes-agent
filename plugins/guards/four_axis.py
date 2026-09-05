"""four-axis-guard plugin — pre_tool_call 四轴闸门副防线。

与 ReadThinkGate 协同：Gate 扫描 assistant_content 检测四轴证据，
写入 marker 文件。本插件读取 marker 文件，四轴未齐即阻断写工具。

双防线逻辑：
  - Gate（主）：每 turn 累积四轴证据 → 四轴齐备后写入 marker + 解锁
  - 插件（副）：每次 write_file/patch/execute_code 前读 marker → 不齐即阻断

marker 文件路径：~/.hermes/cache/four_axis_gate.json
格式：{"verified": true, "timestamp": 1234567890.0, "axes": ["影响面","原意图","根因","风险"]}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# marker 文件路径由 ReadThinkGate 经 get_hermes_home() 解析（profile-aware），
# 本插件必须用同一解析方式，否则 profile 隔离下写读两端路径不一致（PR #3575
# 同类 bug）。延迟解析：每次读取时现算，不冻结进程首次解析结果。
# 与主闸门 agent/read_think_gate.py::_FOUR_AXIS_REQUIRED_TOOLS 对齐——
# 主闸门明确裁定 execute_code 不强制四轴(可为纯只读分析,写路径已被
# write_file/patch 覆盖)。副防线曾额外拦 execute_code,与主防线裁定
# 冲突,实测产生 474 次误拦。以主防线为唯一裁定源。
_WRITE_TOOLS = frozenset({"write_file", "patch"})
_MARKER_MAX_AGE_SECONDS = 600  # marker 有效期 10 分钟，超时视为无效


def _marker_file() -> Path:
    """解析 marker 文件路径（与 ReadThinkGate 写入端同一解析方式）。

    Returns:
        Path: hermes_home/cache/four_axis_gate.json
    """
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "four_axis_gate.json"

# 2026-08-15 修复：agent 自维护产物（日报/指标/state/cron 输出/缓存/记忆）不是
# "核心代码编辑"，无调用方无编译影响，属闸门设计范围外（escape-valve-runbook
# 第79行设计意图是 core code edit）。此前无路径过滤导致三省 cron 写日报被拦，
# 与 ErrorDiscipline 级联形成互锁死锁（拦截计数>=2 -> terminal 封锁）。
# 前缀同样经 get_hermes_home() 延迟解析（profile-aware，与 marker 路径一致）。


def _agent_owned_prefixes() -> tuple[Path, ...]:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return (
        home / "retrospectives",
        home / "state",
        home / "cron" / "output",
        home / "cache",
        home / "memories",
        home / "skill_suggestions",  # 每日审计报告等 agent 自产文档（同 2026-08-15 设计意图）
    )


# hermes home 之外的 agent 自维护产物目录。aml-bid 是 AML 投标管线的巡检/
# 状态日志（PIPELINE-STATE.md、MONITOR-LOG.md），由夜间调度 cron 每 15 分钟
# 追加——与 retrospectives 同类（agent 自产文档，非核心代码编辑）。不豁免则
# 与 patch-first（改文件唯一合法通道=patch）互斥，巡检日志被迫绕行终端写入。
_EXTRA_AGENT_OWNED_PREFIXES: tuple[str, ...] = (
    "~/Documents/aml-bid",
)


def _extra_agent_owned_prefixes() -> tuple[Path, ...]:
    """解析 hermes home 之外的豁免前缀（expanduser，每次现算，不冻结）。"""
    return tuple(Path(s).expanduser() for s in _EXTRA_AGENT_OWNED_PREFIXES)


def _is_agent_owned_path(path_str: str) -> bool:
    """判断目标路径是否属于 agent 自维护产物（四轴豁免范围）。

    Contract:
      Preconditions: path_str 为字符串（可为空，空返回 False）
      Postconditions: 返回 True 当且仅当 resolve 后的路径等于某个
        agent-owned 前缀或位于其下；解析失败返回 False（fail-closed）
    """
    if not path_str:
        return False
    try:
        p = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "agent-owned path check failed (fail-closed, guard stays active): "
            "path=%r error=%s", path_str, exc, exc_info=True,
        )
        return False
    prefixes = _agent_owned_prefixes() + _extra_agent_owned_prefixes()
    result = any(p == prefix or prefix in p.parents for prefix in prefixes)
    if result and any(
        p == prefix or prefix in p.parents for prefix in _agent_owned_prefixes()
    ):
        from hermes_constants import get_hermes_home

        hermes_root = str(get_hermes_home())
        assert str(p).startswith(hermes_root), (
            f"exempted path escaped hermes home scope: {p}"
        )
    return result


def _read_marker() -> Optional[dict]:
    """读取 marker 文件，验证时效性。"""
    try:
        _marker = _marker_file()
        if not _marker.exists():
            return None
        data = json.loads(_marker.read_text())
        age = time.time() - data.get("timestamp", 0)
        if age > _MARKER_MAX_AGE_SECONDS:
            logger.debug("four-axis marker expired (age=%.0fs)", age)
            return None
        if data.get("verified") and len(data.get("axes", [])) == 4:
            return data
    except Exception:
        logger.warning("four-axis marker read failed (treated as no marker)", exc_info=True)
    return None


def _block_message(tool_name: str) -> Optional[Dict[str, Any]]:
    """生成阻断消息。"""
    return {
        "action": "block",
        "message": (
            f"[四轴闸门 · 副防线] 工具 '{tool_name}' 被阻断。\n\n"
            "四轴闸门未通过——在动手修改代码前，必须在回复中逐项输出：\n\n"
            "  1. 影响面清单：每个受影响调用方（绝对路径+函数名+行号）\n"
            "  2. 原意图溯源：git log -p 最初 commit + 不变量清单\n"
            "  3. 根因定位：症状位置 vs 根因位置（文件+行号）\n"
            "  4. 风险矩阵：最坏场景（触发条件+影响范围+可恢复性）\n\n"
            "四轴就是 commit message 的 body。不用另写。"
        ),
    }


def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """pre_tool_call hook：写工具执行前检查四轴 marker。"""
    tool_name = kwargs.get("tool_name", "")
    if tool_name not in _WRITE_TOOLS:
        return None

    args = kwargs.get("args") or {}
    target = str(args.get("path") or args.get("file_path") or "")
    if _is_agent_owned_path(target):
        logger.debug(
            "four-axis guard: skipping %s — agent-owned path %s", tool_name, target,
        )
        return None

    marker = _read_marker()
    if marker is None:
        logger.info(
            "four-axis guard: blocking %s — marker missing or expired", tool_name,
        )
        return _block_message(tool_name)

    logger.debug(
        "four-axis guard: allowing %s — marker valid (axes=%s)",
        tool_name, marker.get("axes"),
    )
    return None


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("four-axis-guard plugin registered")
