"""dynamic_tips —— 拦截账本到事前提示段的数据回流（双保险闭环）。

事后拦截的每类高频形态,其拦截消息里都写着修复指引;但下个新会话的提示
段仍是静态文本,同一形态反复拦（outcomes.db 实测: 全期 56,977 次拦截中
42,429 次为同 turn 第 2 次及以上重复拦截）。本模块把近 7 天高频拦截签名
聚合成一句话正解,追加进 guards_preflight 段尾——新会话第一次就做对,
拦截发生率下降,提示层与拦截层形成数据驱动闭环。

设计约束:
  - 会话内恒定: 聚合结果缓存 7 天（周期内指纹不变,不破坏缓存前缀）。
  - fail-open: 库不可用/查询失败返回旧缓存或空串,段渲染绝不因数据面
    故障报错。
  - 宁缺毋滥: 只输出已知护栏的短正解；未知护栏与原始拦截文案不进段。
  - 零核心改动: 挂在 guards_preflight 渲染器尾部,复用既有段注册通道。

Contract:
  Preconditions: session_info 为 core 冻结的只读映射,含 platform 键。
  Postconditions: 排除平台/无数据/查询失败 → 空串; 正常返回 ≤3 条、
    ≤_BUDGET_CHARS 字符的短正解文本。
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_BUDGET_CHARS = 300
_TIP_LIMIT = 3
_CACHE_VERSION = 2
_FRESH_SECONDS = 7 * 86400
_TIP_TEXT: dict[str, str] = {
    "SlowSearchGuard": "用 search_files；terminal grep 加 timeout 和排除目录。",
    "DBSafety": "先查表结构，再执行 SQL。",
    "PreWriteGuard": "写前先读目标文件。",
    "patch-first": "改源码用 patch，新建文件用 write_file。",
    "SourceCodeWriteGuard": "改源码用 patch，新建文件用 write_file。",
    "diff 欠账闸门": "每个写调用前出示上一步 diff。",
    "护栏聚合": "同轮修完全部拦截项再重试。",
    "CodingStandardsGuard": "先修正拦截给出的规范违规。",
    "scientific-programming-guard": "复杂逻辑拆小函数并补类型与断言。",
    "CasebookGate": "写 OntoX 前先查病例库索引。",
    "TOOL-BLACKLIST BLOCK": "按拦截说明改用正解工具。",
}


def _guard_label(msg: str) -> str:
    """从拦截消息提取护栏标识（``[XxxGuard] …`` 前缀形态）。

    动态段不复制拦截原文；只有能转成短正解的已知标识才进段。
    无 ``[标识]`` 前缀的消息（回调超时等基础设施噪音）跳过,不进段。

    Contract:
      Postconditions: 有前缀返回去括号标识; 无前缀返回空串; 永不抛异常。
    """
    s = msg.strip()
    if not s.startswith("["):
        return ""
    end = s.find("]")
    if end <= 1:
        return ""
    return s[1:end]


def _cache_path() -> Path:
    """动态段缓存路径（hermes home 感知,禁硬编码 ~/.hermes）。"""
    from hermes_constants import get_hermes_home

    cache_dir = Path(get_hermes_home()) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "guard_dynamic_tips.json"


def _db_path() -> Path:
    """outcomes.db 路径（复用既有库,不新建;与 no_guessing 同源定位方式）。"""
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "outcomes.db"


def _aggregate_tips() -> str:
    """聚合近 7 天高频拦截签名 → 提示段文本。

    Contract:
      Postconditions: 返回 ≤_BUDGET_CHARS 文本（可能为空串）; 库缺失/
        连接失败/表不存在 → raise（由调用方 fail-open 处理,禁吞）。
    """
    conn = sqlite3.connect(f"file:{_db_path()}?mode=ro", uri=True, timeout=3)
    try:
        rows = conn.execute(
            "SELECT substr(error_message, 1, 120) AS msg, COUNT(*) AS c "
            "FROM tool_outcomes WHERE status='blocked' "
            "AND timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-7 days') "
            "GROUP BY msg ORDER BY c DESC LIMIT 40"
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            logger.warning("dynamic_tips: conn.close() 失败", exc_info=True)
    used: set[str] = set()
    lines: list[str] = []
    for msg, count in rows:
        label = _guard_label(msg)
        tip = _TIP_TEXT.get(label, "")
        if not tip or label in used:
            continue
        used.add(label)
        lines.append(f"- {label}({count}次): {tip}")
        if len(lines) >= _TIP_LIMIT:
            break
    if not lines:
        return ""
    head = "[近期高频拦截教训]"
    text = head + "\n" + "\n".join(lines)
    while len(text) > _BUDGET_CHARS and len(lines) > 1:
        lines.pop()
        text = head + "\n" + "\n".join(lines)
    return text


def _load_cached(now: float, max_age: float) -> str:
    """读同版本缓存;未过期（now-generated < max_age）返回文本,否则空串。"""
    cache = _cache_path()
    if not cache.exists():
        return ""
    data = json.loads(cache.read_text(encoding="utf-8-sig"))
    if data.get("version") != _CACHE_VERSION:
        return ""
    if now - float(data.get("generated", 0)) >= max_age:
        return ""
    return str(data.get("text", ""))


def _store_cache(text: str, now: float) -> None:
    """原子写缓存（tmp+rename,防并发读到半写文件）。"""
    cache = _cache_path()
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"generated": now, "text": text, "version": _CACHE_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, cache)


def peek() -> str:
    """读动态段缓存文本（不聚合、不触库、不写缓存）——供补注通道并入指纹。

    Contract:
      Postconditions: 缓存缺失/损坏/读取失败 → 空串（fail-open）;
        正常返回缓存内文本（无论新旧——刷新由新会话渲染器触发,
        在跑会话只消费,保证补注路径零查询开销）。
    """
    try:
        return _load_cached(time.time(), float("inf"))
    except Exception:
        logger.warning("dynamic_tips: peek 失败", exc_info=True)
        return ""


def render(session_info: Mapping[str, Any]) -> str:
    """guards_preflight 段尾追加的动态提示段。

    Contract:
      Preconditions: session_info 含 platform 键。
      Postconditions: 排除平台 → 空串; 缓存新鲜 → 直接复用（周期内输出
        恒定,缓存前缀稳定）; 聚合链路故障 → 旧缓存兜底,再失败空串
        （fail-open,段本体不受影响）。
    """
    from plugins.guards import _EXCLUDED_PLATFORMS

    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    now = time.time()
    try:
        fresh = _load_cached(now, _FRESH_SECONDS)
        if fresh:
            return "\n" + fresh
        text = _aggregate_tips()
        _store_cache(text, now)
        return ("\n" + text) if text else ""
    except Exception:
        # 聚合链路故障（库被锁/WAL ro 打开失败等）: 旧缓存兜底,宁旧勿空。
        try:
            old = _load_cached(now, float("inf"))
            if old:
                return "\n" + old
        except Exception:
            logger.warning("dynamic_tips: 旧缓存读取失败", exc_info=True)
        logger.warning("dynamic_tips: 聚合失败,fail-open 空段", exc_info=True)
        return ""
