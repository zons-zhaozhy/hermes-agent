"""
HTTP 请求安全拦截插件
====================

机制：pre_tool_call 事前拦截——检测到 curl/httpx/requests 调用时写入状态文件，
      pre_llm_call 注入「先查 API 签名」提醒。

预防目标：禁止瞎猜 API 路径和参数。
  正确流程：读源码路由定义 / OpenAPI 文档 → 确认路径+方法+参数名+请求体结构 → 再发请求
  错误流程：凭记忆猜路径 → 404/422 → 猜参数名 → 又 422 → 穷举 curl

检测层：判断命令/代码是否包含 HTTP 客户端调用。
  检测标记是 HTTP 协议标准工具和 Python 标准库（有限集合）。
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

_STATE_FILE = "/tmp/.hermes_curl_safety_pending"

_REMINDER = """\
[HTTP 请求安全提醒]
即将发送 HTTP 请求。API 签名是固化的设计——禁止凭记忆猜测。

发送请求前确认：
1. 路径正确吗？→ 先读源码路由定义 或 OpenAPI 文档 (/docs, /openapi.json)
2. HTTP 方法对吗？→ GET/POST/PUT/DELETE 必须与路由定义一致
3. 参数名对吗？→ 从 Pydantic Request Model 读取实际字段名，不猜
4. 请求体结构对吗？→ 确认 Content-Type 和 JSON 结构
5. 认证带了吗？→ Token/Authorization 是否正确

422 响应？→ 读 detail.loc 字段，FastAPI 会告诉你缺哪个字段/字段名错误
404 响应？→ 先确认路由是否注册，再确认路径拼写
"""

# HTTP 客户端工具/库（有限集合）
_HTTP_CLIENT_MARKERS = (
    "curl ",         # curl CLI
    "curl\"",        # curl 在脚本中
    "httpx.",        # Python httpx
    "httpx ",        # Python httpx CLI
    "requests.",     # Python requests
    "requests.get(",
    "requests.post(",
    "requests.put(",
    "requests.delete(",
    "http.get(",
    "http.post(",
    "http.put(",
    "http.delete(",
    "urllib.request",
    "aiohttp",
    "fetch(",        # JS fetch
)


def _is_http_request(tool_name: str, args: dict) -> bool:
    """判断工具调用是否为 HTTP 客户端请求。"""
    if tool_name == "terminal":
        command = args.get("command", "")
        if not isinstance(command, str):
            return False
        for marker in _HTTP_CLIENT_MARKERS:
            if marker in command:
                return True
        return False

    if tool_name == "execute_code":
        code = args.get("code", "")
        if not isinstance(code, str):
            return False
        for marker in _HTTP_CLIENT_MARKERS:
            if marker in code:
                return True
        return False

    return False


def _on_pre_tool_call(**kwargs) -> None:
    """pre_tool_call 回调：检测 HTTP 请求，写入状态文件。"""
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})

    if _is_http_request(tool_name, args):
        try:
            payload = {
                "tool_name": tool_name,
                "command": str(args.get("command", args.get("code", "")))[:200],
                "timestamp": time.time(),
            }
            with open(_STATE_FILE, "w") as f:
                json.dump(payload, f)
            logger.debug("curl-safety: 检测到 HTTP 请求 [%s]", tool_name)
        except OSError as e:
            logger.warning("curl-safety: 写入状态文件失败: %s", e)


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：如果有 HTTP 请求待提醒，注入安全提醒。"""
    if not os.path.exists(_STATE_FILE):
        return {}

    try:
        with open(_STATE_FILE) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, KeyError, OSError) as e:
        try:
            os.remove(_STATE_FILE)
        except OSError:
            logger.warning("curl-safety: 状态文件清理失败")
        logger.warning("curl-safety: 状态文件读取失败: %s", e)
        return {}

    # 超过 60 秒视为过期
    if time.time() - payload.get("timestamp", 0) > 60:
        try:
            os.remove(_STATE_FILE)
        except OSError:
            logger.debug("curl-safety: 过期状态文件清理失败")
        return {}

    try:
        os.remove(_STATE_FILE)
    except OSError:
        logger.warning("curl-safety: 状态文件清理失败")

    logger.info(
        "curl-safety: 注入 HTTP 请求安全提醒 (工具=%s)",
        payload.get("tool_name", "?"),
    )
    return {"context": _REMINDER}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("curl-safety 插件已注册——HTTP 请求安全拦截就绪")
