"""modify-guard — 多会话并发修改串行化门禁。

同一 git 仓的工作区被多个会话共享（群机器人每人独立 session、管理员 CLI、cron），
并发修改存在两类实测竞态：后写覆盖先写、commit 夹带他人在途修改。本门禁两道防线：

1. 仓级写互斥锁：write_file/patch 落盘前按目标文件锚定 git 仓根，抢该仓的修改锁
   （flock 排他 + 原子替换 + TTL 600s 自动过期 + JSON 损坏自愈）。持锁者=另一活跃
   会话时拦截并说明持有者与剩余等待时间；同会话重复写放行（一个修改流程多次落盘）。
2. git 夹带禁令：terminal 命令含 ``git add -A`` / ``git add .`` / ``git commit -a``
   时拦截，要求显式文件列表——同仓多人在途修改只能各提各的文件。

Contract:
  Preconditions: pre_tool_call kwargs 含 tool_name/args/session_id。
  Postconditions: 返回 None（放行）或 {"action": "block", "message": str}；锚定不到
    git 仓根或任何内部异常一律 fail-open 放行并记日志；永不 raise。
  Waiver: 命令行内注释 ``# modify-ok: <reason>`` 跳过本门禁。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes.plugins.modify_guard")

_LOCK_TTL_S = 600  # 持锁会话崩溃后的自动释放窗口
_WAIVER_MARK = "# modify-ok"
_GIT_SWALLOW_MARKS = ("git add -A", "git add --all", "git add .", "git commit -a")


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)


def _repo_root_for(path: Path) -> Optional[Path]:
    """沿祖先链锚定目标所在 git 仓根；仓外/异常 fail-open（返回 None）。"""
    try:
        cur = path.expanduser().resolve()
        for cand in (cur, *cur.parents):
            if (cand / ".git").exists():
                return cand
    except Exception:
        logger.warning("modify-guard: resolve failed for %s", path, exc_info=True)
    return None


def _lock_path(root: Path) -> Path:
    cache = Path.home() / ".hermes" / "cache" / "modify-guard"
    return cache / (str(abs(hash(str(root)))) + ".lock")


def _acquire(root: Path, session_id: str) -> tuple[bool, str]:
    """抢仓级修改锁。返回 (是否获得, 持有者说明)。flock 排他 + 原子替换 + TTL。"""
    lock_file = _lock_path(root)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    import fcntl  # POSIX-only: delayed so the plugin loads on Windows; flock paths are Unix-only by design

    now = time.time()
    holder: dict[str, Any] = {}
    with open(lock_file, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = f.read().strip()
            if raw:
                try:
                    holder = json.loads(raw)
                except ValueError:
                    logger.warning("modify-guard: corrupt lock reset %s", lock_file)
                    holder = {}
            held_at = float(holder.get("acquired_at") or 0)
            held_by_other = (
                holder.get("session_id")
                and holder.get("session_id") != session_id
                and now - held_at < _LOCK_TTL_S
            )
            if held_by_other:
                remain = int(_LOCK_TTL_S - (now - held_at))
                return False, (f"另一会话（{holder.get('session_id')}）正在修改该仓，"
                               f"约 {remain}s 后自动释放；请其完成提交后再试")
            record = json.dumps({"session_id": session_id, "acquired_at": now, "repo": str(root)})
            tmp = lock_file.with_suffix(f".tmp{time.time_ns()}")
            tmp.write_text(record, encoding="utf-8")
            tmp.replace(lock_file)
            return True, ""
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _check_terminal(command: str) -> Optional[dict]:
    """git 夹带禁令：全量暂存/全量提交形态拦截，豁免注释优先。"""
    if _WAIVER_MARK in command:
        return None
    if any(mark in command for mark in _GIT_SWALLOW_MARKS):
        return {"action": "block", "message": (
            "modify-guard: 禁止 git add -A / git add . / git commit -a——同仓可能存在其他成员的在途修改，"
            "全量暂存会夹带他人文件。请改为显式文件列表：git add <本次方案列出的文件>。"
            "确有全量需求在命令行内加注释 # modify-ok: <理由>"
        )}
    return None


def _check_file_write(args: dict, session_id: str) -> Optional[dict]:
    """仓级写互斥：write_file/patch 目标锚定 git 仓根后抢锁。"""
    raw_path = str(args.get("path") or args.get("file_path") or "")
    if not raw_path:
        return None
    root = _repo_root_for(Path(raw_path))
    if root is None:
        return None
    ok, why = _acquire(root, session_id)
    if ok:
        return None
    return {"action": "block", "message": f"modify-guard: {why}；同一时间只允许一个会话修改该仓，避免覆盖他人修改"}


def _on_pre_tool_call(tool_name: str, args: dict, session_id: str = "") -> Optional[dict]:
    try:
        if tool_name == "terminal":
            return _check_terminal(str(args.get("command") or ""))
        if tool_name in ("write_file", "patch"):
            return _check_file_write(args, session_id)
        return None
    except Exception:
        logger.warning("modify-guard: fail-open on %s", tool_name, exc_info=True)
        return None
