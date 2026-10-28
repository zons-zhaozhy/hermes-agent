"""engineering_roadmap —— 既定工程连做执法（opt-in 声明即执法）.

背景（2026-10-28 用户点名「谁让你总是停摆的」）：跨轮次的既定工程授权
（如「其余 6 技能出题」）此前只存在于会话 memory——提示词级约束只作用于
当轮 LLM，对「以待办清单收尾停轮」无机械执法力（首技能收官轮实际发生）。
本模块把工程授权沉为仓库级声明文件 .hermes/engineering_roadmap.json，
声明即执法：收尾停轮 + 声明内仍有 pending 项 → pre_verify 强制续跑。
未声明的仓库零影响（向后兼容，同 ui_hard_assertions.json 模式）。
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ROADMAP_NAME = "engineering_roadmap.json"

# 停轮措辞（收尾尾部命中即视为停轮信号；与 _DEFER_RE 互补——本表针对
# 「工程收尾清单」形态：报完进度后把未完成项留在尾巴上结束回合）
_STALL_RE = re.compile(
    r"(待铺开|待出题|待完成|待按|其余\s*\d+\s*技|剩余\s*\d+|下一步.{0,8}(铺开|出题|继续)"
    r"|按此工序铺开|后续\s*\d+\s*技)"
)  # re-ok: 中文停轮句式模糊匹配无 str 等价
# 完成态否定：全部做完的收尾（全绿/已收官）不拦
_DONE_RE = re.compile(r"(全部完成|已收官|全部\s*\d+\s*(技能|项).{0,6}(完成|done)|全绿收官)")


def _repo_root(path_str: str) -> Path | None:
    """Contract: Preconditions: path_str 为仓库内某路径（可相对/绝对/不存在）;
    Postconditions: 沿祖先链找到含 .git 的目录返回之, 无仓库根返回 None。
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = Path.cwd() / p
    for base in [p, *p.parents]:
        if (base / ".git").exists():
            return base
    return None


def _pending_items(roadmap: dict) -> list[str]:
    """Contract: Preconditions: roadmap 为声明文件顶层 dict;
    Postconditions: 返回全部 status != "done" 的项描述列表（空=工程收官）。
    """
    pending: list[str] = []
    for proj in roadmap.get("projects") or []:
        if not isinstance(proj, dict):
            continue
        name = str(proj.get("name", "?"))
        for item in proj.get("items") or []:
            if isinstance(item, dict) and str(item.get("status", "pending")) != "done":
                pending.append(f"{name}/{item.get('name', '?')}")
    return pending


def _load_pending(changed_paths: list[str]) -> list[str] | None:
    """Contract: Preconditions: changed_paths 为本回合改动路径列表;
    Postconditions: 仓库根存在声明文件且可解析时返回 pending 项列表（可为空）,
    无仓库根/无声明文件/解析失败时返回 None（不执法）。
    """
    root = None
    for p in changed_paths:
        root = _repo_root(p)
        if root is not None:
            break
    if root is None:
        return None
    roadmap_file = root / ".hermes" / _ROADMAP_NAME
    if not roadmap_file.exists():
        return None
    try:
        roadmap = json.loads(roadmap_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[roadmap] %s 解析失败, 本次不执法: %s", _ROADMAP_NAME, exc)
        return None
    if not isinstance(roadmap, dict):
        return None
    return _pending_items(roadmap)


def check(final_response: str, changed_paths: list[str],
          attempt: int, max_nudges: int) -> dict:
    """Contract: Preconditions: final_response 为本回合回复文本, changed_paths
    为本回合改动路径（用于发现仓库根）, attempt 为续跑轮次;
    Postconditions: 声明文件存在且含 pending 项、回复尾部命中停轮措辞且不
    命中完成态、attempt 未达上限时返回 {"action":"continue","message":...}
    强制续跑; 其余情形返回 {} 放行。
    """
    if attempt >= max_nudges or not final_response:
        return {}
    pending = _load_pending(changed_paths)
    if pending is None or not pending:
        return {}
    tail = final_response[-400:] if len(final_response) > 400 else final_response
    if not _STALL_RE.search(tail) or _DONE_RE.search(tail):
        return {}
    logger.warning(
        "finish_guard/roadmap: 既定工程存在 %d 个 pending 项但回复以停轮措辞收尾"
        "（attempt=%d）, 强制续跑", len(pending), attempt,
    )
    return {
        "action": "continue",
        "message": _nudge_message(pending),
    }


def _nudge_message(pending: list[str]) -> str:
    """Contract: Preconditions: pending 为非空未完成项描述列表;
    Postconditions: 返回强制续跑提示文本（含 pending 清单与工序指令）。
    """
    shown = "；".join(pending[:8]) + ("…" if len(pending) > 8 else "")
    return (
        f"[finish_guard/roadmap] 本仓库已声明既定工程（.hermes/"
        f"{_ROADMAP_NAME}），当前仍有 {len(pending)} 个未完成项：{shown}。"
        "既定工程按序连做禁停轮——撤回「待铺开/下一步」式收尾，直接开工"
        "下一个 pending 项（按已验证工序做完一项再收尾汇报）；完成一项后"
        "把声明文件中该项 status 改为 done 再继续。真正的不可逆/凭据/付费/"
        "需拍板事项不受本条约束，走「必须先确认：」通道。"
    )
