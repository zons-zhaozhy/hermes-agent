"""guard-sentinel plugin — 守卫面完整性哨兵。

守卫插件是 opt-in（config.yaml plugins.enabled 逐个勾选）。任何守卫
缺席（未启用/加载失败）时核心照常运行——防线静默消失，零报错零痕迹。
本插件在 on_session_start / on_session_reset 时扫描全部插件清单，
凡 plugin.yaml 自我声明 tags: [guard] 的守卫未处于激活态，即以
WARNING 级响亮报错（每会话每守卫一次），不阻断不修复。

看护名单零登记：新守卫在 plugin.yaml 打 tags: [guard] 即被自动看护。

Contract:
  Precondition: 无——扫描全部 manifest, 不依赖任何登记表。
  Postconditions: never raises; 缺席守卫只告警不阻断。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("plugins.guard-sentinel")

# 守卫自我声明标记：plugin.yaml 打 tags: [guard] 即被哨兵看护。
GUARD_TAG = "guard"

_reported: set[str] = set()


def _check_guards(surface: str) -> None:
    try:
        from hermes_cli.plugins import get_plugin_manager
        mgr = get_plugin_manager()
        for plugin in list(mgr._plugins.values()):
            manifest = getattr(plugin, "manifest", None)
            tags = getattr(manifest, "tags", None) or []
            if GUARD_TAG not in tags:
                continue
            if getattr(plugin, "enabled", False) and not getattr(plugin, "error", None):
                continue  # active guard — nothing to report
            key = getattr(manifest, "key", "") or getattr(manifest, "name", "")
            name = getattr(manifest, "name", key)
            error = getattr(plugin, "error", None) or "not enabled"
            dedupe = f"{key}:{error}"
            if dedupe in _reported:
                continue
            _reported.add(dedupe)
            logger.warning(
                "GUARD ABSENT [%s]: guard plugin %r (self-declared tags: [%s]) is NOT active "
                "— its protection is silently OFF this session (%s). "
                "Fix: `hermes plugins enable %s` or investigate the load error.",
                surface, name, GUARD_TAG, error, key,
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
    logger.info("guard-sentinel registered (2 hooks, auto-discovering guards via manifest tag %r)", GUARD_TAG)
