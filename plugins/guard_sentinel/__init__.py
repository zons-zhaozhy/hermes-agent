"""guard-sentinel plugin — 守卫面完整性哨兵。

守卫插件是 opt-in（config.yaml plugins.enabled 逐个勾选）。required 守卫
缺席，或已启用 advisory 守卫加载失败/空注册时，核心照常运行——防线静默
消失，零报错零痕迹。刻意关闭的 advisory 守卫只保留在诊断报告中。
本插件在 on_session_start / on_session_reset 时扫描全部插件清单。
守卫由 ``tags: [guard]`` 显式标识；Gate Plane 报告中的 missing 队列
（required 缺席或已启用 guard 破损）以 WARNING 留痕，并在本会话第一条
最终回复时再记录一次「门禁缺席」诊断；它不进入用户答案正文。

看护名单零登记：新守卫打 tags: [guard] 即被自动看护。泛用 pre_tool_call /
pre_llm_call 的观察型插件不会被误当成守卫。

Contract:
  Precondition: 无——扫描全部 manifest, 不依赖任何登记表。
  Postconditions: never raises; 缺席守卫只告警不阻断。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("plugins.guard-sentinel")

# 守卫身份和门禁注册健康都来自 PluginManager 的共享 Gate Plane 报告；
# 这里只负责把 unknown 状态转成日志诊断。
GUARD_TAG = "guard"

_reported: set[str] = set()
_pending: dict[str, str] = {}


def _check_guards(surface: str, session_id: str = "") -> None:
    try:
        from hermes_cli.plugins import get_plugin_manager
        mgr = get_plugin_manager()
        absent: list[str] = []
        report = mgr.gate_plane_report(session_id=session_id)
        if report["overall_status"] == "healthy":
            return
        for suite in report["suites"]:
            for item in suite["missing"]:
                gate_hooks = item["hooks"]
                name = item["name"]
                issue = item["reason"]
                declaration = (
                    f"hooks: {', '.join(gate_hooks)}" if gate_hooks
                    else f"tags: [{GUARD_TAG}]"
                )
                absent.append(f"{name} ({declaration}; {issue})")
                dedupe = f"{name}:{issue}"
                if dedupe in _reported:
                    continue
                _reported.add(dedupe)
                logger.warning(
                    "GUARD ABSENT [%s]: guard plugin %r (self-declared %s) is NOT healthy "
                    "— its protection is missing or unregistered this session (%s). "
                    "Fix: `hermes plugins enable %s` or investigate the load error.",
                    surface, name, declaration, issue, name,
                )
        for item in report.get("manual_guards", ()):
            if item.get("policy") != "required" or item.get("status") == "pass":
                continue
            name = str(item.get("name", "unknown"))
            issue = str(item.get("reason", "unknown"))
            absent.append(f"{name} (manual audit; {issue})")
            dedupe = f"{name}:{issue}"
            if dedupe in _reported:
                continue
            _reported.add(dedupe)
            logger.warning(
                "GUARD ABSENT [%s]: manual guard %r has no fresh audit evidence "
                "this session (%s). Run its audit or investigate the load error.",
                surface, name, issue,
            )
        pending_key = session_id or "_global"
        if absent:
            _pending[pending_key] = (
                "\n\n[门禁缺席 / UNKNOWN] 本会话有守卫未激活："
                + "; ".join(absent)
                + "。对应保护当前未生效，不能视为通过；请启用或检查加载错误。"
            )
        else:
            _pending.pop(pending_key, None)
    except Exception:
        logger.warning("guard-sentinel: check failed", exc_info=True)
        _pending[session_id or "_global"] = (
            "\n\n[门禁状态不可用 / UNKNOWN] guard-sentinel 无法读取插件注册表，"
            "本会话无法确认门禁是否完整；不能视为通过。请检查插件加载日志。"
        )


def on_session_start(session_id: str = "", **_kwargs: Any) -> None:
    _check_guards("session_start", session_id)


def on_session_reset(session_id: str = "", **_kwargs: Any) -> None:
    _pending.pop(session_id or "_global", None)
    _check_guards("session_reset", session_id)


def on_session_end(session_id: str = "", **_kwargs: Any) -> None:
    """Drop any notice the turn did not surface; session ids are not process-lifetime keys."""
    _pending.pop(session_id or "_global", None)


def transform_llm_output(
    response_text: str = "", session_id: str = "", **_kwargs: Any,
) -> str | None:
    """Record an absent-guard notice once without rewriting the user's answer."""
    notice = _pending.pop(session_id or "_global", None)
    if not notice:
        return None
    logger.warning("guard-sentinel: %s", notice.strip())
    return None


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("transform_llm_output", transform_llm_output)
    logger.info(
        "guard-sentinel registered (diagnostics only, discovering guards via the explicit %r tag)",
        GUARD_TAG,
    )
