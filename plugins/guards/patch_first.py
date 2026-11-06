"""Patch-first enforcement for terminal and execute-code writes.

The sub-guard keeps the standalone plugin's contract: sed classification is
delegated to :mod:`plugins.guards.terminal_semantics`, while execute-code
Python ``open(..., "w")`` forms remain a narrow mechanical block. It exists
alongside ``source_write`` because its scope is the modification channel, not
whether the target happens to be a source file.
"""

from __future__ import annotations

import logging

from plugins.guards.terminal_semantics import sed_writes_file

logger = logging.getLogger(__name__)


def _is_sed_write(tool_name: str, args: dict[str, object]) -> bool:
    """Return whether a tool call uses a prohibited bulk-write channel."""
    if tool_name == "terminal":
        command = args.get("command", "")
        if not isinstance(command, str):
            return False
        return sed_writes_file(command)

    if tool_name == "execute_code":
        code = args.get("code", "")
        if not isinstance(code, str):
            return False
        if sed_writes_file(code):
            return True
        # Python write-file channel: sed semantics does not own open rules.
        return "open(" in code and (", 'w'" in code or ', "w"' in code)

    return False


_BLOCK_MESSAGE = """\
[patch-first] terminal 中检测到 sed/脚本写入源文件，已拦截（硬闸门，非提醒）。

文件修改唯一合法通道：
  - 修改已有文件 → patch 工具（fuzzy matching，返回 unified diff）
  - 新建文件 → write_file 工具

sed 仅允许只读查找（sed -n 'N,Mp' file / sed -n '/pattern/p' file）。
若此拦截为误报：在 patch 工具中完成该修改，不要换通道绕过。\
"""


def _on_pre_tool_call(**kwargs: object) -> dict[str, object] | None:
    """Block sed/script writes before the tool is executed."""
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})
    if not isinstance(args, dict) or not _is_sed_write(tool_name, args):
        return None

    command = str(args.get("command", args.get("code", "")))
    logger.info(
        "guards.patch_first: blocked terminal sed/script write [%s] cmd=%s",
        tool_name,
        command,
    )
    return {"action": "block", "message": _BLOCK_MESSAGE}


def register(ctx) -> None:
    """Register the patch-first pre-tool hook."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("guards.patch_first registered——sed/脚本写入硬拦截就绪")
