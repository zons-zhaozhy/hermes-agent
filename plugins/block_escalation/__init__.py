"""block-escalation plugin.

拦截置信度升级：拦截器只认「单次调用」，不认「意图」——被拦后换一条
通道重试同一意图时，各拦截器互不知情，每次都只看到第 1 次。本插件在
post_tool_call 观察点收敛「意图指纹」：对 status="blocked" 的调用按
被改写目标（路径/命令文本）计算指纹，同一指纹在时间窗内第 2 次被拦
即升级——通过 pre_llm_call 提醒 + transform_llm_output 用户可见
警示双通道强制停下，禁止第三次尝试。

指纹设计（第一性：意图 = 对什么目标做什么写操作）：
  - terminal: 被拦命令中出现的仓库内文件路径（改写目标）
  - write_file/patch: 目标 path
  - execute_code 写操作: code 中出现的文件路径
  - 无路径可提取时退化为命令文本的 token 集合（排序去重），忽略通道差异

状态进程内存放（subprocess-per-file 隔离不影响单会话内升级判定）。
"""

import logging
import time

logger = logging.getLogger("hermes.plugin.block-escalation")

_WINDOW_SECS = 1800          # 同意图重复被拦的判定时间窗
_STREAK_LIMIT = 2            # 窗口内第 2 次即升级（用户拍板：连续 2 次同因=停）
_STATE_CAP = 128             # 进程级防膨胀上限

# fingerprint -> {"count": int, "last_ts": float, "tools": set, "sample": str}
_streaks: dict = {}
_escalated: set = set()      # 已升级指纹（用于 transform_llm_output 层提示）


def _extract_paths(text: str) -> tuple:
    """Contract: Preconditions: text 为 str；Postconditions: 返回 text 中
    疑似文件路径的元组（含扩展名或斜杠的 token，排序）。纯 str 方法，零正则。"""
    if not text:
        return ()
    hits, loose = [], []
    for tok in text.replace('"', " ").replace("'", " ").split():
        tok = tok.strip("`,;()")
        if tok.endswith((".py", ".md", ".yaml", ".yml", ".json", ".ts", ".js",
                ".sh", ".sql", ".toml", ".txt", ".java", ".vue", ".xml")):
            hits.append(tok)
        elif "/" in tok and len(tok) > 3:
            loose.append(tok)
    pool = hits if hits else loose
    return tuple(sorted(set(pool))[:8])


def _intent_fingerprint(tool_name: str, args: dict) -> tuple:
    """Contract: Preconditions: args 为 dict；Postconditions: 返回意图指纹
    （目标路径元组，或退化 token 元组）；无内容时返回 ()（不参与计数）。"""
    if tool_name == "terminal":
        cmd = str(args.get("command") or "")
        return _extract_paths(cmd) or tuple(sorted(set(cmd.split()))[:12])
    if tool_name in ("write_file", "patch", "edit"):
        return (str(args.get("path") or ""),)
    if tool_name == "execute_code":
        return _extract_paths(str(args.get("code") or ""))
    return ()


def _note_blocked(tool_name: str, args: dict) -> None:
    """记录一次被拦；超限升级。进程内状态，cap 驱逐最旧。"""
    fp = _intent_fingerprint(tool_name, args)
    if not fp:
        return
    now = time.time()
    # 窗口过期与超限驱逐
    for k in [k for k, v in _streaks.items() if now - v["last_ts"] > _WINDOW_SECS]:
        _streaks.pop(k, None)
        _escalated.discard(k)
    while len(_streaks) >= _STATE_CAP:
        _streaks.pop(min(_streaks, key=lambda k: _streaks[k]["last_ts"]), None)
    rec = _streaks.get(fp)
    if rec is None:
        _streaks[fp] = {"count": 1, "last_ts": now, "tools": {tool_name}, "sample": str(args)[:200]}
        return
    rec["count"] += 1
    rec["last_ts"] = now
    rec["tools"].add(tool_name)
    if rec["count"] >= _STREAK_LIMIT:
        _escalated.add(fp)
        logger.warning(
            "block_escalation: 同一意图第 %d 次被拦（通道=%s），升级终止",
            rec["count"], sorted(rec["tools"]))


def _on_post_tool_call(tool_name: str = "", status: str = "", error_type: str = "",
                       args: dict | None = None, **kwargs) -> None:
    """Contract: Preconditions: 核心以 status/error_type 复播工具结果；
    Postconditions: status=blocked 时记录意图指纹；纯观察无返回值。"""
    if status != "blocked":
        return
    _note_blocked(tool_name, args or {})


def register(ctx):
    """Contract: Preconditions: ctx 暴露 register_hook(hook_name, callback)；
    Postconditions: post_tool_call/transform_llm_output 两钩子注册成功并落日志。"""
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    logger.info("block-escalation plugin registered (post_tool_call + transform_llm_output)")


def _on_transform_llm_output(response_text: str = "", **kwargs) -> str:
    """Contract: Preconditions: response_text 为本回合 LLM 输出（核心键名）；
    Postconditions: 存在已升级指纹（说明被拦后仍在尝试或继续输出）时追加
    用户可见系统提示，否则原样返回。"""
    output = response_text or ""
    if not _escalated:
        return output
    logger.warning("block_escalation: 升级指纹仍活跃，追加用户可见终止提示")
    _escalated.clear()  # 单次提示，避免重复堆叠
    return output + (

        "\n\n[拦截升级] 同一写操作意图已连续 2 次被拦且发生通道切换。本轮必须立即停止第 3 次"
        "尝试：①逐字重读两次拦截信息原文②按拦截信息指明的合法通道补齐证据原路重试；若指引"
        "不可行，停下向用户呈报拦截原文与已尝试通道清单，等待人工裁决。禁止再换任何新通道。")
