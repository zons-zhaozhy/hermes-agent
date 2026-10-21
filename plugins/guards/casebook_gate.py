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

# fail-open 只告警一次——未配置部署里逐 pre_tool_call 刷 warning 属日志噪音
_fail_open_warned = False

_NAMESPACE = "casebook_gate"

# OntoX 家族仓库根（写操作目标命中才门控）。仅 env 提供——缺失走 fail-open 分支
# （_casebook_exists 判 False + warning 日志），不在代码里固化机器特定路径。
_ONTOX_ROOT = os.path.expanduser(os.environ.get("CASEBOOK_ONTOX_ROOT", ""))
# 病例库 SSOT 目录（同上，仅 env 提供）
_CASEBOOK_DIR = os.path.expanduser(os.environ.get("CASEBOOK_DIR", ""))

_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})

# terminal 写目标解析（角色感知）。2026-10-17 第三轮误拦根修：旧实现
# 用「写命令前缀表 + 全 token 扫描」判定，不区分 token 在命令中的角色，
# 导致 cp 只读源（dispatcher 账本回看）、脚本路径（晚省盘点）、sed 表达式
# 全被当写目标。新实现按命令头解析「写目标角色」的 token：
#   cp/mv/rsync → 目标=最后一个操作数（源不判）
#   sed -i      → 目标=脚本表达式之后的文件操作数
#   dd          → 目标=of= 操作数
#   tee         → 目标=非选项操作数
#   其它        → 仅重定向目标（> / >>）
#   解释器 -c   → 载荷含写特征时追加引号内路径字面量提取（与 execute_code
#                 通道同一特征集，2026-10-19 第四轮根修：纯读脚本零提取）
_FILE_COPY_HEADS = frozenset({"cp", "mv", "rsync"})
_INTERPRETER_HEADS = frozenset({"python", "python3", "perl", "ruby", "node"})

# 解释器载荷写特征（terminal -c 与 execute_code 两通道共用的单一事实源）。
# 无写特征的纯读脚本不做字面量提取——2026-10-19 误拦根修：dataless 盘点
# 只读脚本 print(f'...{t/1e6:.1f}MB') 的格式说明符斜杠被当路径分隔符，
# 相对拼根恒真，纯读诊断被拦（目标列显示为 f-string 片段）。
_PY_WRITE_FEATURES = ("open(", ".write", "write_text")


def _split_segments(cmd: str) -> list[str]:
    """引号感知地把命令切分为 ; / && / || 分隔的子命令段。

    Contract:
      Postconditions: 引号内的分隔符不切分; 返回非空段列表（可能为空表）。
    """
    segments: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == ";":
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        if ch in "&|" and cmd.startswith(ch * 2, i):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def _segment_tokens(seg: str) -> list[str]:
    """单段 token 化（shlex，引号不闭合时退化到空格切分）。

    Contract:
      Postconditions: 恒返回列表（可能为空表），不抛异常。
    """
    try:
        import shlex

        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.split()


def _sed_file_operands(toks: list[str]) -> list[str]:
    """提取 sed 文件操作数（跳过选项、-e/-f 携带的脚本参数与主脚本）。

    Contract:
      Postconditions: 仅返回文件操作数 token（可能为空表）; sed 语法
        之外的 token 按「第一个非选项=主脚本，其后=文件」推导。
    """
    operands: list[str] = []
    have_script = False
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in ("-e", "-f") and i + 1 < len(toks):
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        if not have_script:
            have_script = True
            i += 1
            continue
        operands.append(t)
        i += 1
    return operands


def _copy_targets(toks: list[str]) -> list[str]:
    """cp/mv/rsync 的写目标=最后一个非选项操作数（源是只读角色不判）。

    Contract:
      Postconditions: 无操作数返回空表; 恰返回 0 或 1 个目标。
    """
    operands = [t for t in toks[1:] if not t.startswith("-")]
    return [operands[-1]] if operands else []


def _dd_targets(toks: list[str]) -> list[str]:
    """dd 的写目标=of= 操作数。

    Contract:
      Postconditions: 无 of= 返回空表。
    """
    return [t[len("of="):] for t in toks if t.startswith("of=")]


def _tee_targets(toks: list[str]) -> list[str]:
    """tee 的写目标=全部非选项操作数（tee 写每个文件操作数）。

    Contract:
      Postconditions: 仅含非选项操作数（可能为空表）。
    """
    return [t for t in toks[1:] if not t.startswith("-")]


def _sed_targets(toks: list[str]) -> list[str]:
    """sed 的写目标=仅 -i 就地改写时的文件操作数（无 -i 是纯读）。

    Contract:
      Postconditions: 非 -i 形态返回空表。
    """
    if any(t == "-i" or t.startswith("-i") for t in toks[1:]):
        return _sed_file_operands(toks)
    return []


def _interpreter_payload(toks: list[str]) -> str:
    """解释器 -c 之后的代码载荷 token（无 -c 时退化整段拼接）。

    Contract:
      Postconditions: shlex 已剥掉载荷外层引号，字面量提取可直接进行;
        无 -c 时返回整段（脚本路径场景通常无引号字面量→空提取）。
    """
    for i, t in enumerate(toks):
        if t == "-c" and i + 1 < len(toks):
            return toks[i + 1]
    return " ".join(toks)


def _default_targets(seg: str, toks: list[str]) -> list[str]:
    """其它命令头：仅重定向目标是写目标; 解释器 -c 追加字面量提取。

    Contract:
      Postconditions: 解释器载荷含写特征时，代码内 OntoX 路径字面量计入
        目标（与 execute_code 通道同一特征集防旁路）; 无写特征纯读脚本
        零提取; 普通命令只认 > / >> 目标。

    字面量提取须用 -c 载荷 token 而非整段原文——外层包裹引号会使
    引号配对提前闭合，路径落在配对间隙（2026-10-17 实测定性）。
    """
    targets = list(_redirect_targets(seg))
    if toks and toks[0].rsplit("/", 1)[-1] in _INTERPRETER_HEADS:
        payload = _interpreter_payload(toks)
        if _payload_has_write_feature(payload):
            lit = _extract_first_ontox_path(payload)
            if lit:
                targets.append(lit)
    return targets


def _segment_write_targets(seg: str) -> list[str]:
    """按命令头分派单段内「写目标角色」的 token 列表。

    Contract:
      Postconditions: 只返回构成写目标的 token（源/脚本路径/表达式/输入
        文件不进目标集）; 空表=本段无写目标（只读段）。
    """
    toks = _segment_tokens(seg)
    if not toks:
        return []
    head = toks[0]
    if head in _FILE_COPY_HEADS:
        return _copy_targets(toks)
    if head == "dd":
        return _dd_targets(toks)
    if head == "tee":
        return _tee_targets(toks)
    if head == "sed":
        return _sed_targets(toks)
    return _default_targets(seg, toks)


def _terminal_write_targets(cmd: str) -> list[str]:
    """全命令（多段）的写目标合集。

    Contract:
      Postconditions: 恒返回列表; 空表=命令无任何写目标（只读命令）。
    """
    out: list[str] = []
    for seg in _split_segments(cmd):
        out.extend(_segment_write_targets(seg))
    return out


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
    # ~ 前缀是家目录绝对语义，必须先展开再判定——2026-10-17 误拦根修：
    # 旧逻辑把 ~/.hermes/... 当"相对路径"拼到 OntoX 根下恒真，导致晚省
    # 盘点脚本与 metrics 台账追加被拦（实测 2 形态）。
    p = Path(path).expanduser()
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


def _target_from_terminal(cmd: str) -> str:
    """从 terminal 命令里提取第一个落在 OntoX 根内的写目标。

    Contract:
      Postconditions: 非写命令或写目标均不在 OntoX 根内 → 空串;
        命令含 casebook 关键词（检索病例库自身）→ 空串（豁免）。

    判据收紧史：2026-10-10 裸词拼根恒真 + SQL > 比较符两处根修;
    2026-10-17 第三轮——全 token 角色不分（cp 源/脚本路径/sed 表达式
    误当写目标）替换为按命令头的写目标角色解析（见 _segment_write_targets）。
    """
    if "casebook" in cmd:
        return ""
    for t in _terminal_write_targets(cmd):
        if _is_ontox_target(t):
            return t
    return ""


def _payload_has_write_feature(code: str) -> bool:
    """解释器载荷是否含写调用特征（两通道共用的单一事实源）。

    Contract:
      Postconditions: 任一 _PY_WRITE_FEATURES 子串命中 → True;
        纯读脚本（os.walk/lstat/print 等）→ False。
    """
    return any(f in code for f in _PY_WRITE_FEATURES)


def _looks_like_path(s: str) -> bool:
    """字符串字面量是否为路径形态（排除格式串与 URL）。

    Contract:
      Postconditions: 含空白/花括号（f-string·format 占位）/URL 方案
        （://）→ False（不可能构成文件路径字面量）; 其余非空 → True。
    """
    if not s or "://" in s:
        return False
    return not any(ch.isspace() or ch in "{}" for ch in s)


def _extract_first_ontox_path(code: str) -> str:
    """从 execute_code 代码里提取第一个落在 OntoX 根内的路径字面量。

    Contract:
      Postconditions: 找不到返回空串（调用方 fail-open）; 格式串/URL
        字面量不构成候选（2026-10-19 根修：{t/1e6:.1f} 类斜杠占位
        曾被相对拼根恒真升级为写目标）。
    """
    import re  # re-ok: 字符串字面量提取

    hits = re.findall(r'[\"\']([^\"\']+)[\"\']', code)  # re-ok: 引号配对提取,str方法无法表达
    for h in hits:
        if _looks_like_path(h) and _is_ontox_target(h):
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
    if _payload_has_write_feature(code):
        return _extract_first_ontox_path(code)
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
        global _fail_open_warned
        if not _fail_open_warned:
            logger.warning("casebook_gate: CASEBOOK_DIR not set or missing → fail-open（此后静默）")
            _fail_open_warned = True
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
