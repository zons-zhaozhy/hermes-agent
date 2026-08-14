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


def _on_pre_tool_call(**kwargs) -> None:
    """pre_tool_call 回调：检测 sed 写入操作，写入状态文件。"""
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})

    if _is_sed_write(tool_name, args):
        try:
            payload = {
                "tool_name": tool_name,
                "command": str(args.get("command", args.get("code", "")))[:200],
                "timestamp": time.time(),
            }
            with open(_STATE_FILE, "w") as f:
                json.dump(payload, f)
            logger.debug("patch-first: 检测到 sed 写入操作 [%s]", tool_name)
        except OSError as e:
            logger.warning("patch-first: 写入状态文件失败: %s", e)


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：如果有 sed 写入待提醒，注入 patch 优先提醒。"""
    if not os.path.exists(_STATE_FILE):
        return {}

    try:
        with open(_STATE_FILE) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        try:
            os.remove(_STATE_FILE)
        except OSError:
            logger.warning("patch-first: 状态文件清理失败")
        logger.warning("patch-first: 状态文件读取失败: %s", e)
        return {}

    # 超过 60 秒视为过期
    if time.time() - payload.get("timestamp", 0) > 60:
        try:
            os.remove(_STATE_FILE)
        except OSError:
            logger.debug("patch-first: 过期状态文件清理失败")
        return {}

    try:
        os.remove(_STATE_FILE)
    except OSError:
        logger.warning("patch-first: 状态文件清理失败")

    logger.info(
        "patch-first: 注入 patch 优先提醒 (工具=%s)",
        payload.get("tool_name", "?"),
    )
    return {"context": _REMINDER}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("patch-first 插件已注册——修改方式纪律拦截就绪")
