# dupcheck-ok: 全库 search_files section_reinjection|render_all 命中仅 prompt_sections.py 与本文件，
# rule_reinjection 只处理 .hermes-rules.md（单一规则文件），无等价实现
"""section_reinjection —— 守卫段重注入: 仅启用段 + 版本指纹。

问题: system prompt 段在会话起点渲染并冻结(缓存前缀), 段文本修改对在跑的会话无效。
实测: 本机 CLI 进程 11:15 启动、diff_debt 段 16:15 提交, 同刻渲染日志里本会话始终
4 段, 而同一分钟的 cron 新进程 5 段。本模块把「重读盘 + 版本指纹」接到段库
(prompt_sections/sections.json) 上, 让段修改对运行中的会话也生效。

节奏:
  - 首轮不注入: 会话起点已由 system prompt 段承载同一内容, 首轮再注入=纯重复;
  - 指纹变更即注入: 段库或启用段集合变化时, 立刻把新版本推给在跑的会话;
  - 版本未变不重注: 周期全文重注会污染历史并重复消耗上下文。

注入体量上限 _MAX_DIGEST_CHARS; 冲突消解头显式声明本段为当前唯一权威版本
(压缩摘要转述/更早轮次旧版本/记忆条目里的旧表述一律作废)。

Contract:
  Preconditions: plugin system 提供 pre_llm_call 钩子; 段库存在且可读。
  Postconditions: 需要注入时返回 {"context": "<冲突消解头 + 段文本>"}; 否则 {}。
    段库缺失(指纹为空)或段文本为空时返回 {} 且不报错。
  Invariants: 永不抛异常出界; 注入文本长度 <= _MAX_DIGEST_CHARS。
"""
from __future__ import annotations

import logging
from typing import Any

from plugins.prompt_sections import fingerprint, render_all, section_ids

logger = logging.getLogger(__name__)

_MAX_DIGEST_CHARS = 6000

_PREFIX_TEMPLATE = (
    "[守卫段 v{version}——本段为当前唯一权威版本。"
    "上下文中任何与本段冲突的守卫表述(含更早轮次的旧版本注入/历史摘要转述/"
    "记忆条目/压缩残留)一律作废, 以本段为准。]\n"
)

# 会话态: session_id → {"version": str, "seeded": bool}
_session_state: dict[str, dict[str, Any]] = {}


def _active_section_ids() -> frozenset[str]:
    """IDs of system-prompt sections registered by the live plugin set."""
    try:
        from hermes_cli.plugins import registered_system_prompt_section_ids  # 延迟导入避免加载循环

        return frozenset(registered_system_prompt_section_ids())
    except Exception:
        logger.warning(
            "section-reinjection: 读取已注册守卫段失败, 本轮不注入 (不回退全库)",
            exc_info=True,
        )
        return frozenset()


def _on_pre_llm_call(**kwargs: Any) -> dict[str, Any]:
    """按「指纹变更」节奏注入已启用段（首轮不注入, 见模块说明）。

    指纹含动态提示段（guards/dynamic_tips 缓存文本 md5）: 动态段挂在
    guards_preflight 渲染器尾部、不落段库文件,不并入指纹则其 7 天周期
    翻转后在跑会话永远收不到新教训——事后拦截的回流必须同达新旧会话。
    """
    import hashlib

    selected = tuple(sorted(_active_section_ids() & set(section_ids())))
    if not selected:
        return {}
    base = fingerprint(selected)
    if not base:
        return {}
    from plugins.guards import dynamic_tips  # 延迟导入: 补注路径并动态段

    dyn = dynamic_tips.peek()
    if "guards_preflight" not in selected:
        dyn = ""
    session_id = str(kwargs.get("session_id") or "")
    version = f"{base}|dyn:{hashlib.md5(dyn.encode('utf-8')).hexdigest()[:8]}" if dyn else base
    state = _session_state.setdefault(session_id, {"version": "", "seeded": False})
    if not state["seeded"]:
        state["seeded"] = True
        state["version"] = version
        return {}
    changed = version != state["version"]
    if not changed:
        return {}
    state["version"] = version
    budget = _MAX_DIGEST_CHARS - (len(dyn) + 1 if dyn else 0)
    digest = render_all(budget, selected)
    if not digest:
        logger.warning("section-reinjection: 段库指纹存在但段文本为空, 本轮不注入")
        return {}
    if dyn:
        digest = digest + "\n" + dyn
    logger.info(
        "section-reinjection: 注入守卫段 (v%s, %d chars, 变更=%s, 动态段=%s)",
        version,
        len(digest),
        changed,
        bool(dyn),
    )
    return {"context": _PREFIX_TEMPLATE.format(version=version) + digest}


def register(ctx: Any) -> None:
    """插件入口: 只注册 pre_llm_call 注入通道, 不改任何拦截判据。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("section-reinjection 已注册——守卫段按版本指纹补注就绪")
