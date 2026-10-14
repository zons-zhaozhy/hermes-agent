# dupcheck-ok: 全库 search_files section_reinjection|render_all 命中仅 prompt_sections.py 与本文件，
# rule_reinjection 只处理 .hermes-rules.md（单一规则文件），无等价实现
"""section_reinjection —— 守卫段重注入: 版本指纹 + 周期补注。

问题: system prompt 段在会话起点渲染并冻结(缓存前缀), 段文本修改对在跑的会话无效。
实测: 本机 CLI 进程 11:15 启动、diff_debt 段 16:15 提交, 同刻渲染日志里本会话始终
4 段, 而同一分钟的 cron 新进程 5 段。rule_reinjection 已用「重读盘 + 版本指纹 +
每 N 轮补注」解决常驻规则衰减; 本模块把同一机制接到段库
(prompt_sections/sections.json) 上, 让段修改对运行中的会话也生效。

节奏(与 rule_reinjection 同口径, 一处例外):
  - 首轮不注入: 会话起点已由 system prompt 段承载同一内容, 首轮再注入=纯重复;
  - 指纹变更即注入: 段库被修改(版本指纹变化)时, 立刻把新版本推给在跑的会话;
  - 每 _REINJECT_INTERVAL 轮补注: 对抗长会话纪律衰减与上下文压缩后的弱化转述。

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

from plugins.prompt_sections import fingerprint, render_all

logger = logging.getLogger(__name__)

_MAX_DIGEST_CHARS = 6000
_REINJECT_INTERVAL = 10

_PREFIX_TEMPLATE = (
    "[守卫段 v{version}——本段为当前唯一权威版本。"
    "上下文中任何与本段冲突的守卫表述(含更早轮次的旧版本注入/历史摘要转述/"
    "记忆条目/压缩残留)一律作废, 以本段为准。]\n"
)

# 会话态: session_id → {"version": str, "calls": int, "seeded": bool}
_session_state: dict[str, dict[str, Any]] = {}


def _should_inject(state: dict[str, Any], version: str) -> tuple[bool, bool, bool]:
    """判定本轮是否注入。

    Args:
        state: 会话态（calls 已自增, seeded 标记本会话是否已登记首次指纹）。
        version: 当前段库版本指纹。

    Returns:
        (是否注入, 是否指纹变更, 是否周期补注) —— 首轮只登记指纹不注入。
    """
    if not state["seeded"]:
        state["seeded"] = True
        state["version"] = version
        return False, False, False
    changed = version != state["version"]
    periodic = int(state["calls"]) % _REINJECT_INTERVAL == 0
    if changed:
        state["version"] = version
    return (changed or periodic), changed, periodic and not changed


def _on_pre_llm_call(**kwargs: Any) -> dict[str, Any]:
    """按「指纹变更 + 周期补注」节奏注入段库全文（首轮不注入, 见模块说明）。"""
    version = fingerprint()
    if not version:
        return {}
    session_id = str(kwargs.get("session_id") or "")
    state = _session_state.setdefault(session_id, {"version": "", "calls": 0, "seeded": False})
    state["calls"] = int(state["calls"]) + 1
    inject, changed, periodic = _should_inject(state, version)
    if not inject:
        return {}
    digest = render_all(_MAX_DIGEST_CHARS)
    if not digest:
        logger.warning("section-reinjection: 段库指纹存在但段文本为空, 本轮不注入")
        return {}
    logger.info(
        "section-reinjection: 注入守卫段 (v%s, %d chars, 变更=%s, 周期=%s)",
        version,
        len(digest),
        changed,
        periodic,
    )
    return {"context": _PREFIX_TEMPLATE.format(version=version) + digest}


def register(ctx: Any) -> None:
    """插件入口: 只注册 pre_llm_call 注入通道, 不改任何拦截判据。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("section-reinjection 已注册——守卫段按版本指纹补注就绪")
