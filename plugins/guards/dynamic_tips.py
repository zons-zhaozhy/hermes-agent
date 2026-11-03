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
  - 宁缺毋滥: 只输出命中静态映射表的签名（正解来自既有拦截消息的修复
    指引语义）,未映射的原始错误文案一律不进段。
  - 零核心改动: 挂在 guards_preflight 渲染器尾部,复用既有段注册通道。

Contract:
  Preconditions: session_info 为 core 冻结的只读映射,含 platform 键。
  Postconditions: 排除平台/无数据/查询失败 → 空串; 正常返回带日期头+
    ≤5 条映射命中的提示文本（≤_BUDGET_CHARS 字符）。
"""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_BUDGET_CHARS = 600
_TIP_LIMIT = 5
_FRESH_SECONDS = 7 * 86400
# 签名关键词 → (短标签, 一句话正解)。正解内容与对应子护栏拦截消息的
# 修复指引同源,新增子护栏时同步维护本表。
_TIP_MAP: tuple[tuple[str, tuple[str, str]], ...] = (
    ("SlowSearchGuard", ("slow_search", "全仓递归文本扫描改用 search_files 工具;确需 grep 限定子目录并加 --include/--exclude-dir")),
    ("diff", ("diff 对照", "写工具调用前同轮正文先贴 diff;新文件先 git add -N + git diff")),
    ("四轴", ("四轴证据", "写代码前正文逐项列出影响面/原意图/根因/风险（时效 30 分钟）")),
    ("ReadThink", ("写前调查", "写代码前先 read_file 目标文件+search_files 同类实现（terminal 只读命令也计数）")),
    ("DBSafety", ("SQL 前 schema", "执行 SQL 前先 \\dt/\\d 确认表名字段名;写库前 ps eww+lsof 对账实连库")),
    ("casebook", ("病例库", "OntoX 家族写操作前先查 education/casebook/INDEX.md")),
    ("pre_write", ("先读后写", "patch/write_file 前同会话先 read_file 目标文件")),
    ("callback timed out", ("护栏超时", "属护栏故障误拦,原样重试同一调用即可,勿换通道")),
    ("SKILL_PRUNED", ("技能重载", "先 skill_view(name=...) 重载全文再写")),
    ("scope_gate", ("新目录树", "仓库内新开目录树须当轮用户原话授权")),
    ("先查重", ("新建查重", "新建 .py 前 search_files 全库查同名/同功能实现")),
)


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
            "SELECT substr(error_message, 1, 60) AS sig, COUNT(*) AS c "
            "FROM tool_outcomes WHERE status='blocked' "
            "AND timestamp >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-7 days') "
            "GROUP BY sig ORDER BY c DESC LIMIT 40"
        ).fetchall()
    finally:
        try:
            conn.close()
        except Exception:
            logger.warning("dynamic_tips: conn.close() 失败", exc_info=True)
    used: set[str] = set()
    lines: list[str] = []
    for sig, count in rows:
        for key, (label, tip) in _TIP_MAP:
            if key in sig and label not in used:
                used.add(label)
                lines.append(f"- {label}({count}次): {tip}")
                break
        if len(lines) >= _TIP_LIMIT:
            break
    if not lines:
        return ""
    head = f"[高频拦截教训·近7天实测聚合 {time.strftime('%Y-%m-%d')}]"
    text = head + "\n" + "\n".join(lines)
    while len(text) > _BUDGET_CHARS and len(lines) > 1:
        lines.pop()
        text = head + "\n" + "\n".join(lines)
    return text


def _load_cached(now: float, max_age: float) -> str:
    """读缓存;未过期（now-generated < max_age）返回文本,否则空串。"""
    cache = _cache_path()
    if not cache.exists():
        return ""
    data = json.loads(cache.read_text(encoding="utf-8"))
    if now - float(data.get("generated", 0)) >= max_age:
        return ""
    return str(data.get("text", ""))


def _store_cache(text: str, now: float) -> None:
    """原子写缓存（tmp+rename,防并发读到半写文件）。"""
    cache = _cache_path()
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"generated": now, "text": text}, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, cache)


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
