"""guard-sentinel plugin — 守卫面完整性哨兵。

守卫插件是 opt-in（config.yaml plugins.enabled 逐个勾选）。任何守卫
缺席（未启用/加载失败）时核心照常运行——防线静默消失，零报错零痕迹。
本插件在 on_session_start / on_session_reset 时核对守卫面名单，
缺席者以 WARNING 级响亮报错（每会话一次），不阻断不修复。

Contract:
  Precondition: 名单 GUARD_PLUGINS 里的目录名存在于 plugins/。
  Postconditions: never raises; 缺席守卫只告警不阻断。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("plugins.guard-sentinel")

# 守卫面 SSOT 名单：目录名 -> config enabled 键（缺省同目录名）。
# 新守卫插件落地时在此登记。
GUARD_PLUGINS: dict[str, str] = {
    "read_think_gate": "read-think-gate",
    "finish_guard": "finish_guard",
    "block_escalation": "block_escalation",
    "guards": "guards",
    "discipline": "discipline",
    "engine-invariants": "engine-invariants",
    "failure_preflight": "failure_preflight",
}

_reported: set[str] = set()


def _check_guards(surface: str) -> None:
    try:
        from hermes_cli.plugins import get_plugin_manager
        mgr = get_plugin_manager()
        active = {
            key for key, plugin in mgr._plugins.items()
            if getattr(plugin, "enabled", False)
        }
        for directory, key in GUARD_PLUGINS.items():
            if directory in active or key in active:
                continue
            tag = directory
            if tag in _reported:
                continue
            _reported.add(tag)
            logger.warning(
                "GUARD ABSENT [%s]: guard plugin %r (config key %r) is NOT loaded — "
                "its protection is silently OFF this session. "
                "Fix: `hermes plugins enable %s` or investigate the load error.",
                surface, directory, key, key,
            )
    except Exception:
        logger.warning("guard-sentinel: check failed", exc_info=True)


def on_session_start(**_kwargs: Any) -> None:
    _check_guards("session_start")


def on_session_reset(**_kwargs: Any) -> None:
    _check_guards("session_reset")


def register(ctx: Any) -> None:
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_reset", on_session_reset)
    logger.info("guard-sentinel registered (2 hooks, watching %d guards)", len(GUARD_PLUGINS))
