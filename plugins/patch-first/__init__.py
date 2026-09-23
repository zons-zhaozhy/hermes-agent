"""
代码/文档修改纪律拦截插件
==========================

机制：pre_tool_call 事前拦截——检测到 sed/脚本批量替换时写入状态文件，
      pre_llm_call 注入「patch 优先」提醒。

预防目标：禁止用 sed 和脚本批量替换修改代码/文档文件。
  正确方式：patch 工具（fuzzy matching，安全替换）
  允许方式：sed 仅用于只读查找（grep/行号确认）
  禁止方式：sed -i、sed 替换写入文件、脚本循环替换多文件

检测层：判断命令是否为 sed 写入操作。
  排除 sed 只读用法（sed -n/p 不修改文件）。
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_STATE_FILE = "/tmp/.hermes_patch_first_pending"

_REMINDER = """\
[修改方式安全提醒]
检测到 sed/脚本批量替换操作。代码和文档文件修改一律使用 patch 工具。

patch 工具优势：
- fuzzy matching：小空格/缩进差异不会导致替换失败
- 自动语法检查：替换后自动 lint
- 原子操作：要么全部成功，要么全部回滚
- 可追溯：返回 unified diff

sed 仅允许用于只读查找：
- sed -n '10,20p' file（查看行）
- sed -n '/pattern/p' file（查看匹配行）
- 配合 grep 做行号确认

禁止：
- sed -i（原地修改）
- sed 's/old/new/' file > file（覆盖写入）
- 脚本中循环 find + sed 批量替换多文件
- execute_code 中用 open(path, 'w') 写文件（不落盘）
"""

# sed 写入操作标记（有限集合）
_SED_WRITE_MARKERS = (
    "sed -i",         # sed 原地修改
    "sed 's/",        # sed 替换（非只读 -n -p）
    'sed "s/',        # sed 替换双引号形式
    "sed s/",         # sed 替换无引号
)


def _is_sed_write(tool_name: str, args: dict) -> bool:
    """判断是否为 sed 写入操作（排除只读用法）。"""
    if tool_name == "terminal":
        command = args.get("command", "")
        if not isinstance(command, str):
            return False

        # 排除只读模式：sed -n 不修改文件
        if "-n" in command and " -p" in command:
            return False
        # 排除纯 grep/查找
        if command.strip().startswith("grep"):
            return False

        for marker in _SED_WRITE_MARKERS:
            if marker in command:
                return True
        return False

    if tool_name == "execute_code":
        code = args.get("code", "")
        if not isinstance(code, str):
            return False
        # execute_code 中 sed 替换
        for marker in _SED_WRITE_MARKERS:
            if marker in code:
                return True
        # execute_code 中 open(path, 'w') 提醒
        if "open(" in code and ", 'w'" in code:
            return True
        if 'open(' in code and ', "w"' in code:
            return True
        return False

    return False


_BLOCK_MESSAGE = """\
[patch-first] terminal 中检测到 sed/脚本写入源文件，已拦截（硬闸门，非提醒）。

文件修改唯一合法通道：
  - 修改已有文件 → patch 工具（fuzzy matching，返回 unified diff）
  - 新建文件 → write_file 工具

sed 仅允许只读查找（sed -n 'N,Mp' file / sed -n '/pattern/p' file）。
若此拦截为误报：在 patch 工具中完成该修改，不要换通道绕过。\
"""


def _on_pre_tool_call(**kwargs):
    """pre_tool_call 回调：检测 sed 写入操作，硬拦截（返回 block）。

    Contract:
      Preconditions: kwargs 含 tool_name(str)、args(dict)
      Postconditions: 检测到 sed 写入时返回
        {"action": "block", "message": _BLOCK_MESSAGE}（工具不执行）；
        否则返回 None（放行）
    """
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})

    if _is_sed_write(tool_name, args):
        command = str(args.get("command", args.get("code", "")))
        logger.info(
            "patch-first: blocked terminal sed/script write [%s] cmd=%s",
            tool_name,
            command,
        )
        return {"action": "block", "message": _BLOCK_MESSAGE}
    return None


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("patch-first 插件已注册——sed/脚本写入硬拦截就绪")
