"""guards.casebook_gate — 写操作前强制查病例库（AI-for-AI 知识回路硬接线）。

OntoX 家族路径上的 write_file/patch/terminal(写)/execute_code(写) 被拦截，直到本会话
内出现过一次针对 casebook 的检索（read_file/search_files/terminal grep 命中
education/casebook 路径）。拦截一次后放行（引导而非无限门禁），与
ontox-casebook skill 的纪律 1（写代码前先查病例库）同构的机器强制层。

设计参照（AIBuildAI PostTrain Agent 知识系统）：Agent 决策前强制检索知识库
（证据而非参数化记忆），执行后观察写回——本护栏是"决策前检索"的硬接线。

Contract:
  Preconditions: plugin system 提供 pre_tool_call / post_tool_call 钩子;
    CASEBOOK_ONTOX_ROOT / CASEBOOK_DIR 环境变量指定 OntoX 根与病例库 SSOT 路径
    （未设置时按"库不可达"处理，见 fail-open 语义）。
  Postconditions: 病例库不可达时 fail-open（零静默——记 warning 日志后放行）;
    每会话首次 OntoX 写操作最多拦截一次; 非 OntoX 路径零打扰。
  Invariants: 不修改 casebook 内容; 只读 INDEX.md; 不写任何文件。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "casebook_gate"

# OntoX 家族仓库根（写操作目标命中才门控）。仅 env 提供——缺失走 fail-open 分支
# （_casebook_exists 判 False + warning 日志），不在代码里固化机器特定路径。
_ONTOX_ROOT = os.path.expanduser(os.environ.get("CASEBOOK_ONTOX_ROOT", ""))
# 病例库 SSOT 目录（同上，仅 env 提供）
_CASEBOOK_DIR = os.path.expanduser(os.environ.get("CASEBOOK_DIR", ""))

_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})

# terminal 写文件的高置信模式（对齐 read_think_gate/gate.py 的模式表，取子集）
_TERMINAL_WRITE_CMD_PREFIXES = (
    "sed -i", "tee /", "tee ~", "cp ", "mv ", "rsync ", "dd of=",
)


def _config_paths() -> tuple[str, str]:
    """从 config.yaml guards.casebook_gate 段读路径（对齐 read_think_gate 的 config 模式）。

    Contract:
      Postconditions: config 不可读/无该段 → ("", "") 并记 warning（fail-open 语义，
        env-at-import 值与测试 monkeypatch 值优先级更高，见 _paths）。
    """
    try:
        from hermes_cli.config import load_config_readonly

        sec = ((load_config_readonly() or {}).get("guards") or {}).get("casebook_gate") or {}
        return str(sec.get("ontox_root", "")), str(sec.get("casebook_dir", ""))
    except Exception:
        logger.warning("casebook_gate: config read failed → fall back to env", exc_info=True)
        return "", ""


def _paths() -> tuple[str, str]:
    """路径解析：模块属性（env-at-import 或测试 monkeypatch）优先 → config fallback。

    Contract:
      Postconditions: 恒返回二元组，空串=未配置（各消费点 fail-open）。
    """
    if _ONTOX_ROOT or _CASEBOOK_DIR:
        return _ONTOX_ROOT, _CASEBOOK_DIR
    return _config_paths()


def _plugin_disabled() -> bool:
    """Contract: Postconditions: CASEBOOK_GATE_DISABLE∈{1,true,yes,on} → True。"""
    return os.environ.get("CASEBOOK_GATE_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _is_ontox_target(path: str) -> bool:
    """写目标是否落在 OntoX 家族仓库内。

    Contract:
      Postconditions: 相对路径按 OntoX 根解析后判定; 绝对路径直接前缀判定;
        OntoX 根未配置时返回 False（fail-open）;
        无斜杠裸词（echo/cp/变量引用/命令名）不是路径形态 → False
        （2026-10-10 误报根修：裸词拼根恒真导致只读 SELECT 被误拦）。
    """
    if not path or "/" not in path:
        return False
    ontox_root, _cb = _paths()
    if not ontox_root:
        return False
    p = Path(path)
    if not p.is_absolute():
        p = Path(ontox_root) / p
    try:
        p.resolve().relative_to(Path(ontox_root).resolve())
        return True
    except ValueError:
        return False


def _redirect_targets(cmd: str) -> list[str]:
    """提取 > / >> 重定向的目标 token 列表。

    Contract: Postconditions: 只返回 > / >> 之后紧邻的 token（可能为空表）;
      2>&1 合并流不是文件目标（跳过）; 引号不闭合无法 token 化时返回空表。
    """
    targets: list[str] = []
    try:
        import shlex

        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        return targets
    for i, tok in enumerate(tokens):
        if tok in (">", ">>") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if not nxt.startswith("&"):  # 2>&1 形态非文件目标
                targets.append(nxt)
    return targets


def _terminal_writes_file(cmd: str) -> bool:
    """Contract: Postconditions: 命中写命令前缀或存在重定向文件目标 → True。

    判据收紧史（2026-10-10 误报根修）：旧表含 " > " 字面子串——SQL 的
    WHERE ts > 'x' 比较符与其同形，叠加对无斜杠裸词拼根恒真的路径判定，
    导致只读 sqlite SELECT 被误拦（当日亲历 2 次）。重定向改为 token 化
    判定（> / >> 的下一 token 才是目标文件），命令前缀族保留子串语义。
    """
    if any(
        cmd.startswith(p) or f"; {p}" in cmd or f"&& {p}" in cmd
        for p in _TERMINAL_WRITE_CMD_PREFIXES
    ):
        return True
    return bool(_redirect_targets(cmd))


def _extract_first_ontox_path(code: str) -> str:
    """从 execute_code 代码里提取第一个落在 OntoX 根内的路径字面量。

    Contract:
      Postconditions: 找不到返回空串（调用方 fail-open）。
    """
    import re  # re-ok: 字符串字面量提取

    hits = re.findall(r'["\']([^"\']+)["\']', code)  # re-ok: 引号配对提取,str方法无法表达
    for h in hits:
        if "/" in h and _is_ontox_target(h):
            return h
    return ""


def _casebook_exists() -> bool:
    """Contract: Postconditions: 库目录未配置或不存在 → False（调用方 fail-open）。"""
    _root, cb_dir = _paths()
    return bool(cb_dir) and os.path.isdir(cb_dir)


def _get_state(sid: str) -> dict[str, Any]:
    return get_session_state(sid, _NAMESPACE)


def _casebook_probed(sid: str) -> bool:
    """本会话是否已检索过病例库（证据集非空）。"""
    return bool(_get_state(sid).get("probes"))


def _record_probe(sid: str, evidence: str) -> None:
    st = _get_state(sid)
    st.setdefault("probes", set()).add(evidence)


# ── 检索证据记录 ─────────────────────────────────────────────────────


def _probe_from_read(args: dict[str, Any]) -> str:
    """read_file 指向 casebook 时产出证据串，否则空。

    Contract:
      Postconditions: 仅 education/casebook 路径命中才返回非空。
    """
    path = str(args.get("path", ""))
    return f"read_file: {path}" if "education/casebook" in path else ""


def _probe_from_search(args: dict[str, Any]) -> str:
    """search_files 指向 casebook 时产出证据串，否则空。

    Contract:
      Postconditions: path 或 pattern 含 casebook 即命中。
    """
    path = str(args.get("path", ""))
    if "casebook" in path or "casebook" in str(args.get("pattern", "")):
        return f"search_files: {path or '.'}"
    return ""


def _probe_from_terminal(args: dict[str, Any]) -> str:
    """terminal 命令串含 casebook 时产出证据串，否则空。

    Contract:
      Postconditions: 命令含 casebook 关键词即命中（grep/ls 病例库均算）。
    """
    cmd = str(args.get("command", ""))
    return "terminal grep casebook" if "casebook" in cmd else ""


# ── 写目标解析 ───────────────────────────────────────────────────────


def _target_from_write_tool(tool_name: str, args: dict[str, Any]) -> str:
    """从 write_file/patch/execute_code 参数解析写目标路径。

    Contract:
      Postconditions: execute_code 仅当代码含写操作特征且能提取 OntoX 路径
        才返回非空; 其余返回 args.path 原值（可能为空）。
    """
    if tool_name in ("write_file", "patch"):
        return str(args.get("path", ""))
    code = str(args.get("code", ""))
    if "open(" in code or ".write" in code or "write_text" in code:
        return _extract_first_ontox_path(code)
    return ""


def _target_from_terminal(cmd: str) -> str:
    """从 terminal 写命令里提取第一个 OntoX 路径 token。

    Contract:
      Postconditions: 非写命令或无 OntoX 路径 → 空串。
    """
    if not _terminal_writes_file(cmd) or "casebook" in cmd:
        return ""
    for tok in cmd.replace(";", " ").split():
        tok_clean = tok.strip("\"'")
        if _is_ontox_target(tok_clean):
            return tok_clean
    return ""


def _resolve_write_target(tool_name: str, args: dict[str, Any]) -> str:
    """统一入口：按工具类型解析写目标，非写工具返回空。

    Contract:
      Postconditions: 返回空串=非门控对象（调用方放行）。
    """
    if tool_name in _WRITE_TOOLS:
        return _target_from_write_tool(tool_name, args)
    if tool_name == "terminal":
        return _target_from_terminal(str(args.get("command", "")))
    return ""


# ── hooks ────────────────────────────────────────────────────────────


def on_post_tool_call(**kwargs) -> None:
    """记录本会话内对病例库的检索证据（read_file/search_files/terminal grep）。"""
    if _plugin_disabled():
        return
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}
    if tool_name == "read_file":
        probe = _probe_from_read(args)
    elif tool_name == "search_files":
        probe = _probe_from_search(args)
    elif tool_name == "terminal":
        probe = _probe_from_terminal(args)
    else:
        return
    if probe:
        _record_probe(sid, probe)


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """OntoX 写操作前置门：未查病例库即拦截，查过即放行。

    Contract:
      Postconditions: 非 OntoX 目标/已查/库不可达 → None（放行）;
        命中且未查 → {"action": "block", "message": ...}，消息含修复指引。
    """
    if _plugin_disabled():
        return None
    if not _casebook_exists():
        logger.warning("casebook_gate: CASEBOOK_DIR not set or missing → fail-open")
        return None

    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args") or {}

    target = _resolve_write_target(tool_name, args)
    if not target or not _is_ontox_target(target):
        return None

    if _casebook_probed(sid):
        return None
    return _block_message(target)


def _block_message(target: str) -> dict[str, Any]:
    """构造拦截指令（修复指引内嵌）。

    Contract:
      Postconditions: 返回 {"action": "block", "message": ...} 且消息含
        INDEX 路径与三种解锁方式。
    """
    _root, cb_dir = _paths()
    index_path = os.path.join(cb_dir, "INDEX.md")
    return {
        "action": "block",
        "message": (
            "[CasebookGate] OntoX 家族写操作前必须先查病例库（防复发纪律的机器强制层）。\n"
            f"  目标: {target}\n"
            f"  修复（三选一，做一次即解锁本会话）:\n"
            f"    1. read_file('{index_path}') 扫一眼病例索引\n"
            f"    2. search_files(path='{cb_dir}', pattern='<缺陷类别关键词>') 查同类病例\n"
            f"    3. 确认本次改动与任何既有病例无关（新缺陷类别）后直接重试——拦截一次即放行\n"
            f"  病例族谱速查: 静默系/漂移系/链路系（详见 INDEX.md）"
        ),
    }


# ── Registration ──────────────────────────────────────────────────────


def register(ctx: Any) -> None:
    """Contract: Postconditions: 注册 pre/post 两个钩子；不抛异常。"""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("guards.casebook_gate registered")
