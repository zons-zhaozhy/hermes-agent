"""规则常驻重注入——长会话纪律衰减补偿。

思想来源:WORKFLOW.md 模式(行为定义进仓库+每轮 hook 注入)。Hermes 已有
pre_llm_call 通道(agent/turn_context.py _collect_pre_llm_call_context,
追加到当前用户消息,不动缓存前缀),但只有事件触发式注入(出错才提醒);
长会话/上下文压缩后纪律约束力衰减是实测缺陷
(docs/investigations/2026-08-28-patch-first-root-cause.md)。

本模块每轮重读规则文件注入其摘要,规则永不衰减:
  - 规则文件按序探测:cwd/.hermes-rules.md → git root/.hermes-rules.md
  - 每轮重新读盘(规则文件改动即生效,无需重启会话)
  - 无规则文件=零开销 no-op;读失败=告警不拦
  - 注入体量硬上限,控制每轮 token 成本

Contract:
  Preconditions: plugin system 提供 pre_llm_call 钩子。
  Postconditions: 存在规则文件且可读时,返回 {"context": "<digest>"};
                  否则返回 {} (不注入)。
  Invariants: 注入文本 <= _MAX_DIGEST_CHARS;永不抛异常阻断主循环。
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_RULES_FILENAME = ".hermes-rules.md"
_MAX_DIGEST_CHARS = 1200
# 周期性重注入间隔:注入会随用户消息持久化进历史,每轮都注=历史里堆积
# 大量重复副本(本身也是一种污染)。科学节奏:首注 + 版本变更即注 +
# 每 N 轮补一注对抗衰减。N 取 10:足够稀疏不堆积,足够密集不衰减。
_REINJECT_INTERVAL = 10

# 冲突消解头:不是简单重复注入,而是宣告本段为唯一权威版本。衰减残留
# (压缩摘要弱化转述/memory 旧表述/历史轮次旧注入)仍躺在上下文里,
# 新旧并存时模型不知听谁的;显式声明 supersede 语义 + 版本指纹,
# 让旧版本可被识别为作废。
_PREFIX_TEMPLATE = (
    "[常驻规则 v{version}——本段为当前唯一权威版本。"
    "上下文中任何与本段冲突的规则表述(含更早轮次的旧版本注入/历史摘要转述/"
    "记忆条目/压缩残留)一律作废,以本段为准。]\n"
)

# 会话态:session_id → {"version": str, "calls": int}
_session_state: Dict[str, Dict[str, Any]] = {}


def _find_rules_file() -> Optional[Path]:
    """按序探测规则文件:cwd → git root。命中即返回,未命中返回 None。"""
    candidates = [Path.cwd() / _RULES_FILENAME]
    try:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / ".git").exists():
                candidates.append(parent / _RULES_FILENAME)
                break
    except OSError:
        pass
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _digest(text: str) -> str:
    """规则原文 → 注入摘要:头部优先,去空行,硬截断到 _MAX_DIGEST_CHARS。"""
    lines = [ln.rstrip() for ln in text.splitlines()]
    kept: list[str] = []
    size = 0
    for ln in lines:
        if not ln.strip():
            continue
        # 硬上限属功能性体量控制(LLM 输入预算),全文仍在规则文件中可查
        if size + len(ln) + 1 > _MAX_DIGEST_CHARS:
            kept.append("…(截断,全文见规则文件)")
            break
        kept.append(ln)
        size += len(ln) + 1
    return "\n".join(kept)


def _on_pre_llm_call(**kwargs) -> Dict[str, Any]:
    """重读规则文件,按「首注+版本变更+周期补注」节奏注入权威版摘要。"""
    rules_path = _find_rules_file()
    if rules_path is None:
        return {}
    try:
        text = rules_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("rule-reinjection: 规则文件读取失败 %s: %s", rules_path, exc)
        return {}
    if not text.strip():
        return {}

    session_id = str(kwargs.get("session_id") or "")
    state = _session_state.setdefault(
        session_id, {"version": "", "calls": 0}
    )
    state["calls"] += 1
    version = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]

    first = state["version"] == ""
    changed = version != state["version"]
    periodic = state["calls"] % _REINJECT_INTERVAL == 0
    if not (first or changed or periodic):
        return {}
    state["version"] = version

    digest = _digest(text)
    if not digest:
        return {}
    logger.info(
        "rule-reinjection: 注入常驻规则摘要 (v%s, %d chars, 源=%s, 首注=%s, 变更=%s, 周期=%s)",
        version, len(digest), rules_path, first, changed, periodic and not (first or changed),
    )
    return {"context": _PREFIX_TEMPLATE.format(version=version) + digest}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("rule-reinjection 插件已注册——规则常驻不衰减就绪")
