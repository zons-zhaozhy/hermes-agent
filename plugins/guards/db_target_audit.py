"""guards.db_target_audit — block DB writes when multiple live DB instances exist.

Motivation: repeated wrong-database incidents (three occurrences through
2026-10-25): a data-fix UPDATE targeted local 5416 while the consuming OMS
process was actually connected to the cloud tunnel 15416 — the fix silently
did nothing, discovered only when alert counters stayed flat. The agent
validated the DSN string, never the *consumer process's* live connection.
db_safety enforces "confirm schema first"; this guard enforces "confirm the
target instance against live consumer connections".

Blocked shape (ALL must hold for a shell segment):
  1. segment head is ``psql`` (with ``-c`` whose SQL contains a write verb:
     INSERT/UPDATE/DELETE/DROP/TRUNCATE/ALTER) or ``pg_restore`` (with
     ``--clean``)
  2. segment carries an explicit connection flag ``-h`` or ``-p``
     (default single-instance invocations are unambiguous — not blocked)
  3. no inline waiver comment ``# dbtarget-ok: <reason>``
  4. live lsof evidence shows >= 2 distinct active PostgreSQL server
     endpoints (host:port) — single-instance environments are never blocked

Evidence is gathered by the guard itself (``lsof -iTCP -sTCP:ESTABLISHED -P -n``,
3 s timeout); lsof failure or timeout fails OPEN (allow).

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args.
  Postconditions: returns None (allow) or {"action": "block", "message": str};
    never raises; the only subprocess call is lsof with a hard timeout.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Optional

from plugins.guards.source_write import _split_compound

logger = logging.getLogger(__name__)

_WAIVER_MARK = "# dbtarget-ok"

_WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER")


def _segment_tokens(seg: str) -> list[str]:
    """段 token 化（shlex 引号感知；不闭合退化空格切分）。

    Contract:
      Postconditions: 恒返回列表（可能为空表），不抛异常。
    """
    try:
        import shlex

        return shlex.split(seg.splitlines()[0], posix=True)
    except (ValueError, IndexError):
        return seg.split()


def _has_endpoint_flag(tokens: list[str]) -> bool:
    """段是否带显式连接参数（-h/-p 族；默认单实例调用不算）。"""
    return any(
        t in ("-h", "-p") or (t.startswith("-h") and len(t) > 2) or (t.startswith("-p") and len(t) > 2)
        for t in tokens[1:]
    )


def _psql_write(tokens: list[str]) -> bool:
    """psql 段且 -c SQL 含写动词。

    Contract:
      Preconditions: tokens[0] 为 psql。
      Postconditions: 返回 bool；纯判定。
    """
    joined = " ".join(tokens).upper()
    return any(v in joined for v in _WRITE_VERBS)


def _is_db_write_segment(tokens: list[str]) -> bool:
    """段是否为带显式连接参数的 DB 写操作（psql 写动词 / pg_restore --clean）。

    Contract:
      Preconditions: tokens 为段首行 token 化结果（可为空表）。
      Postconditions: 返回 bool；纯判定，无副作用。
    """
    if not tokens or not _has_endpoint_flag(tokens):
        return False
    head = tokens[0].rsplit("/", 1)[-1]
    if head == "psql":
        return _psql_write(tokens)
    if head == "pg_restore":
        return "--clean" in tokens
    return False


def _active_pg_endpoints() -> list[str]:
    """lsof 实测活动 PostgreSQL 服务端点（distinct host:port）；失败返回空表。

    Contract:
      Postconditions: 返回端点列表（可能为空）；lsof 3 秒超时，异常吞掉记日志，
        永不抛出（fail-open 依据）。
    """
    try:
        proc = subprocess.run(
            ["lsof", "-iTCP", "-sTCP:ESTABLISHED", "-P", "-n"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("guards.db_target_audit: lsof unavailable (%s); failing open", exc)
        return []
    endpoints: set[str] = set()
    for line in proc.stdout.splitlines():
        low = line.lower()
        if ("postgres" in low or "psql" in low) and "->" in line:
            peer = line.rsplit("->", 1)[-1].strip()
            if peer:
                endpoints.add(peer.rsplit("(", 1)[0].strip())
    return sorted(endpoints)


_BLOCK_TEMPLATE = (
    "[DBTargetAudit] 检测到多库环境下的 DB 写命令，已拦截——目标库歧义未消除。\n"
    "实测当前活动 PostgreSQL 端点: {endpoints}\n"
    "错库事故实录（三次同款）：数据修复打在本地库而消费进程实连云端隧道库，"
    "修复零效、告警数字纹丝不动，白清两轮。\n"
    "放行前须出示进程连库对账：\n"
    "  1. ps eww <消费进程关键词> | tr ' ' '\\n' | grep -i 'DB\\|DATABASE'   # 看进程环境实连哪个库\n"
    "  2. lsof -p <pid> -a -iTCP   # 看该 PID 的活动 DB 连接端点\n"
    "  3. 确认写命令的 -h/-p 与消费进程实连端点一致后重试；确认无误可加行内注释 # dbtarget-ok: <理由>"
)


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook: block ambiguous-target DB writes in multi-DB setups.

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict with command key).
      Postconditions: returns None (allow) or {"action":"block","message":str}.
    """
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "terminal":
        return None

    args = kwargs.get("args") or {}
    command = str(args.get("command", ""))
    if not command or _WAIVER_MARK in command:
        return None

    hit_seg = None
    for seg in _split_compound(command):
        if _is_db_write_segment(_segment_tokens(seg)):
            hit_seg = seg
            break
    if hit_seg is None:
        return None

    endpoints = _active_pg_endpoints()
    if len(endpoints) < 2:
        return None  # 单实例/取证失败：无歧义或无法判定，fail-open

    logger.warning(
        "guards.db_target_audit: blocked multi-instance DB write (endpoints=%s, seg=%s)",
        endpoints,
        hit_seg,
    )
    return {
        "action": "block",
        "message": _BLOCK_TEMPLATE.format(endpoints=", ".join(endpoints)),
    }


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.db_target_audit registered")
