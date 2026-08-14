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

_MARKER_FILE = Path.home() / ".hermes" / "cache" / "four_axis_gate.json"
_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})
_MARKER_MAX_AGE_SECONDS = 600  # marker 有效期 10 分钟，超时视为无效


def _read_marker() -> Optional[dict]:
    """读取 marker 文件，验证时效性。"""
    try:
        if not _MARKER_FILE.exists():
            return None
        data = json.loads(_MARKER_FILE.read_text())
        age = time.time() - data.get("timestamp", 0)
        if age > _MARKER_MAX_AGE_SECONDS:
            logger.debug("four-axis marker expired (age=%.0fs)", age)
            return None
        if data.get("verified") and len(data.get("axes", [])) == 4:
            return data
    except Exception:
        logger.debug("four-axis marker read failed", exc_info=True)
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
