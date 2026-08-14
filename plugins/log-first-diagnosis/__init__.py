"""
错误诊断强制拦截插件
====================

机制：双 hook 联动——post_tool_call 检测错误信号，pre_llm_call 注入提醒。

检测层：纯结构化字段判断，零信号词枚举，零正则。
  四个字段任一命中即为错误：
  1. status == "error"（框架推导）
  2. result["error"] 非空
  3. result["exit_code"] != 0
  4. result["success"] is False

提醒层：一条统一提醒覆盖所有错误场景（DB/测试/HTTP/进程）。
  错误信息本身往往就是答案——提醒只是防止跳过诊断步骤。
"""

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_STATE_FILE = "/tmp/.hermes_log_first_pending"

_REMINDER = """\
[错误诊断强制提醒]
上一个工具返回了错误信号。按以下流程自检，禁止跳步：

1. 读错误信息本身——它往往就是答案
   (no such host=DNS失败, ReadTimeout=对端慢, 401=认证缺失, exit_code≠0=执行失败)

2. 如果是 DB 操作出错（表/列不存在）：
   禁猜表名！先查 schema: \dt 看全部表名, \d table_name 看全部列名
   查询返回空不等于没有数据——先确认表名/列名拼写正确

3. 如果是测试 FAIL：
   先验证测试脚本自身正确性（路径/参数名/解析逻辑），禁止直接当真实问题修
   期望值是独立推导的还是凑绿灯的？

4. 如果是 HTTP/进程报错：
   提取文件名:行号 → 读代码 → 沿调用链追踪根因

5. 通用铁律：
   禁穷举curl/猜配置/绕弯路。连续2次相同失败→切换策略。
"""


def _dict_has_error(d: dict) -> bool:
    """检查 dict 的结构化错误字段。纯字段值判断，零信号词。"""
    err_val = d.get("error")
    if err_val is not None:
        if isinstance(err_val, str) and err_val.strip():
            return True
        if isinstance(err_val, dict) and err_val:
            return True
        if isinstance(err_val, (int, float)) and err_val != 0:
            return True
    exit_code = d.get("exit_code")
    if isinstance(exit_code, (int, float)) and exit_code != 0:
        return True
    if d.get("success") is False:
        return True
    status_val = d.get("status")
    if isinstance(status_val, str) and status_val.lower() == "error":
        return True
    return False


def _detect_error(result: Any, status: str = "") -> bool:
    """结构化字段判断工具结果是否为错误。零信号词枚举，零正则。"""
    if status == "error":
        return True
    if result is None:
        return False
    if isinstance(result, dict):
        return _dict_has_error(result)
    if isinstance(result, str):
        stripped = result.strip()
        if not stripped:
            return False
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return _dict_has_error(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return False


def _on_post_tool_call(**kwargs) -> None:
    """post_tool_call 回调：检测错误信号，写入状态文件。"""
    tool_name = kwargs.get("tool_name", "")
    status = kwargs.get("status", "")
    result = kwargs.get("result")
    error_message = kwargs.get("error_message", "")

    if _detect_error(result, status):
        detail = error_message or ""
        if not detail and isinstance(result, dict):
            detail = str(result.get("output", ""))[:200]
        elif not detail and isinstance(result, str):
            detail = result[:200]

        try:
            payload = {
                "tool_name": tool_name,
                "detail": detail[:200],
                "timestamp": time.time(),
            }
            with open(_STATE_FILE, "w") as f:
                json.dump(payload, f)
            logger.debug(
                "log-first-diagnosis: 检测到错误 [%s] %s",
                tool_name, detail[:80],
            )
        except OSError as e:
            logger.warning("log-first-diagnosis: 写入状态文件失败: %s", e)


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：如果有未处理的错误，注入诊断提醒。"""
    if not os.path.exists(_STATE_FILE):
        return {}

    try:
        with open(_STATE_FILE) as f:
            payload = json.load(f)
        os.remove(_STATE_FILE)
    except (json.JSONDecodeError, KeyError, OSError):
        try:
            os.remove(_STATE_FILE)
        except OSError:
            pass
        return {}

    if time.time() - payload.get("timestamp", 0) > 60:
        return {}

    logger.info(
        "log-first-diagnosis: 注入诊断提醒 (工具=%s, 错误=%s)",
        payload.get("tool_name", "?"),
        payload.get("detail", "?"),
    )
    return {"context": _REMINDER}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("log-first-diagnosis 插件已注册——错误诊断强制拦截就绪")
