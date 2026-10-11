# dupcheck-ok: _on_pre_tool_call/_on_pre_llm_call 属插件框架约定回调名（guards 各子模块同形态），非重复实现；写入目标解析复用 duplicate_check 基建
"""guards.scope_gate — 新增产物的授权闸门（L2 子护栏）。

规则来源：2026-10-11 越权实录——用户要求「考虑技术架构与关键实现逻辑」，
助手在架构文档刚落盘、方案未经用户确认时，直接新建 apps/<新应用>/ 一整棵
目录树（19 个文件）。事后按用户指令整棵移出仓库。

为什么必须落在这一层：认知层纪律与 memory 条目反复被证明拦不住动态决策
瞬间的动作（用户原话「刻记忆是完全不够的」）。本模块把「新增产物须先获得
授权」这条纪律机器化，拦在写入路径上。

判定（下列条件同时成立才拦）：
  1. 本次写入落在代码仓库内（cwd 之下，非临时/缓存目录）；
  2. 目标文件当前不存在（新建，不是修改）；
  3. 目标的父目录当前也不存在——即本次写入要在仓库内新开一棵目录树；
  4. 新目录树的根距仓库根 ≤ _MAX_NEW_TREE_DEPTH 级；
  5. 本回合用户原话未被判定为「授权构造」（LLM 语义判定，禁词表/正则）。

条件 3 是关键分辨：在已有目录里加文件属常规实现动作（该目录的边界已被既有
授权覆盖）；在仓库里新开一棵目录树属方向级动作（新应用/新包/新目录平面），
影响面大且不能由既有授权推导。

放行边界（宁纵勿枉）：
  - 文档/配置类扩展名（.md/.txt/.rst/.json/.yaml/.yml/.toml/.ini/.cfg/.conf/.csv）
  - 路径任一段以 . 开头（.github/.hermes 等工具目录）
  - 路径含 tests/test 段（测试树）
  - 临时/缓存目录（/tmp、/var/folders、~/.hermes/cache）
  - 目标路径不可静态解析（猜错会误拦，宁放行不误拦）
  - 无用户原话（cron/子代理会话）
  - 授权判定失败或不可用（None）→ 放行并 WARNING（纪律增强层 fail-open 优于误死锁）

terminal 通道不在本模块射程：terminal 直写源码已由更上游的
guards.source_write 拦截（sed -i / cat > / python -c）。本模块覆盖
write_file / patch / execute_code（后者复用 duplicate_check 的字面量写形态）。

环境变量:
  HERMES_SCOPE_GATE_DISABLE=1    完全禁用
  HERMES_SCOPE_GATE_WARN_ONLY=1  仅提示不阻断（观察模式）

Contract:
  Preconditions: plugin 系统提供 pre_tool_call / pre_llm_call 钩子；
    pre_llm_call kwargs 含 conversation_history 与 session_id（declare_act_guard 同源）。
  Postconditions: 命中「未授权新建目录树」时返回 {"action":"block","message":...}；
    其余一律返回 None 放行；判定/解析任何异常放行并 logger.warning，绝不 raise。
  Invariants: 只判「新开目录树」；已存在目录内的新建与修改一律不参与判定。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from plugins._shared_state import get_session_state
from plugins.guards.duplicate_check import _execute_code_py_targets

logger = logging.getLogger(__name__)

_NAMESPACE = "scope_gate"

# 允许在仓库内自由新开的扩展名（文档/配置类，非代码产物）
_EXEMPT_EXTS = (
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".csv",
)

# 临时/产物目录豁免（绝对路径前缀匹配）
_EXEMPT_PREFIXES = (
    "/tmp",
    "/private/tmp",        # macOS /tmp 的真实路径
    "/var/folders",        # macOS 系统临时目录
    "/var/tmp",
    "/private/var/tmp",    # macOS pytest tmp_path 实际落点
    os.path.expanduser("~/.hermes/cache"),
)

# 路径段豁免：测试树与字节码缓存
_EXEMPT_SEGMENTS = frozenset({"tests", "test", "__pycache__"})

# 新目录树根距仓库根的最大级数——2 即 <容器>/<新名>（如 apps/humanmarket）
_MAX_NEW_TREE_DEPTH = 2

# 进 block 消息的用户原话截断长度（供人工复核判定依据）
_USER_TEXT_PREVIEW = 120


def _disabled() -> bool:
    """禁用开关。

    Contract:
      Postconditions: HERMES_SCOPE_GATE_DISABLE=1 时返回 True，否则 False。
    """
    return os.environ.get("HERMES_SCOPE_GATE_DISABLE", "").strip() == "1"


def _warn_only() -> bool:
    """观察模式开关。

    Contract:
      Postconditions: HERMES_SCOPE_GATE_WARN_ONLY=1 时返回 True，否则 False。
    """
    return os.environ.get("HERMES_SCOPE_GATE_WARN_ONLY", "").strip() == "1"


def _is_exempt(full: str) -> bool:
    """判断绝对路径是否落在豁免区。

    Contract:
      Preconditions: full 为绝对路径字符串。
      Postconditions: 命中临时目录前缀 / 文档配置扩展名 / 点开头路径段 /
                      测试树路径段 之一时返回 True，否则 False。
    """
    if full.startswith(_EXEMPT_PREFIXES):
        return True
    if os.path.splitext(full)[1].lower() in _EXEMPT_EXTS:
        return True
    segments = [seg for seg in full.split(os.sep) if seg]
    if any(seg.startswith(".") for seg in segments):
        return True
    return any(seg in _EXEMPT_SEGMENTS for seg in segments)


def _new_dir_root(full: str, root: str) -> str:
    """返回 full 在仓库内「当前不存在的最浅目录」的相对路径。

    Contract:
      Preconditions: root 为仓库根（realpath），full 为绝对路径。
      Postconditions: 目标父目录不存在且位于 root 之内时，返回新目录树根的
                      相对路径；父目录全部存在、或 full 不在 root 内时返回 ""。
      Invariants: 只遍历目录段（文件名不参与）；逐段 isdir 实测，不猜。
    """
    if not root or not full.startswith(root + os.sep):
        return ""
    parts = os.path.relpath(full, root).split(os.sep)
    accumulated = root
    for part in parts[:-1]:
        accumulated = os.path.join(accumulated, part)
        if not os.path.isdir(accumulated):
            return os.path.relpath(accumulated, root)
    return ""


def _tree_depth(rel_root: str) -> int:
    """目录树根的级数（"apps/humanmarket" → 2）。

    Contract:
      Preconditions: rel_root 为相对路径字符串（可空）。
      Postconditions: 返回非空路径段个数；空串返回 0。
    """
    return len([seg for seg in rel_root.split(os.sep) if seg])


def _candidate_targets(tool_name: str, args: Any) -> list[str]:
    """解析本次调用可能新建的目标路径（跨通道统一射程）。

    Contract:
      Preconditions: args 为 dict 或任意类型。
      Postconditions: 返回候选项的绝对路径列表；无可判定目标 → []。
      Invariants: write_file/patch 取 args.path；execute_code 复用
        duplicate_check 的字面量写形态识别（覆盖 .py 目标）；动态路径不进表。
    """
    if not isinstance(args, dict):
        return []
    if tool_name in ("write_file", "patch"):
        path = str(args.get("path") or "").strip()
        return [os.path.abspath(os.path.expanduser(path))] if path else []
    if tool_name == "execute_code":
        code = str(args.get("code") or "")
        return [os.path.abspath(os.path.expanduser(raw)) for raw, _ in _execute_code_py_targets(code)]
    return []


def _latest_user_text(history: Any) -> str:
    """取会话历史中最后一条 user 消息的文本。

    Contract:
      Preconditions: history 为 list 或 None 或任意类型。
      Postconditions: 返回 str（可为空串）；绝不 raise。
    """
    if not isinstance(history, list):
        return ""
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
    return ""


def _judge_authorized(user_text: str) -> Optional[bool]:
    """LLM 语义判定：用户原话是否授权在仓库内新建代码产物。

    Contract:
      Preconditions: user_text 非空字符串。
      Postconditions: 返回 True/False/None；None 表示判定失败或不可用
                      （调用方据此 fail-open 放行并留痕）。
    """
    from plugins._llm_judge import llm_judge_bool

    system = (
        "你是行为纪律审查器。判断【用户指令】是否授权助手在代码仓库里新建代码产物"
        "（新建应用/新建目录树/新建模块/新建文件这类构造动作）。"
        "只答 JSON：{\"authorized\": true/false}。判定标准："
        "用户要求实现/开发/搭建/新建/创建/落地某个代码产物 → true；"
        "用户要求分析/设计/规划/评估/审查/讨论/汇报/查看/写文档/改现有代码 → false。"
    )
    return llm_judge_bool(
        task=_NAMESPACE, system=system, text=user_text[:2000], true_key="authorized"
    )


def _on_pre_llm_call(**kwargs: Any) -> None:
    """pre_llm_call：按当轮用户原话缓存一次授权判定（不注入任何内容）。

    Contract:
      Preconditions: kwargs 可能含 session_id 与 conversation_history。
      Postconditions: 用户原话变化时刷新 state["authorized"]；同一原话不重复判定；
                      任何异常记日志并返回，绝不 raise。
    """
    try:
        sid = str(kwargs.get("session_id") or "")
        state = get_session_state(sid, _NAMESPACE)
        text = _latest_user_text(kwargs.get("conversation_history"))
        if not text or state.get("user_text") == text:
            return
        state["user_text"] = text
        state["authorized"] = _judge_authorized(text)
        logger.info("scope_gate: 当轮授权判定 authorized=%s", state["authorized"])
    except Exception:
        logger.exception("scope_gate: pre_llm_call 判定失败，本回合按未判定处理（放行）")


def _authorized(sid: str) -> Optional[bool]:
    """读取当轮授权判定结果（不重复判定）。

    Contract:
      Preconditions: sid 为会话标识（可空）。
      Postconditions: 返回 True/False/None（None=尚未判定或判定失败）。
    """
    return get_session_state(sid, _NAMESPACE).get("authorized")


def _block_hit(full: str, root: str, sid: str) -> Optional[str]:
    """单个候选目标的拦截判定。

    Contract:
      Preconditions: full 为绝对路径；root 为仓库根（realpath）；sid 为会话标识。
      Postconditions: 命中「未授权新建目录树」时返回新目录树根相对路径，
                      否则返回 None；判定资源不可用时记 WARNING 并返回 None。
    """
    rel_root = _new_dir_root(full, root)
    if not rel_root or _is_exempt(full):
        return None
    if _tree_depth(rel_root) > _MAX_NEW_TREE_DEPTH or os.path.exists(full):
        return None
    verdict = _authorized(sid)
    if verdict is True:
        return None
    if verdict is None:
        logger.warning("scope_gate: 授权判定不可用，放行（fail-open）: %s", rel_root)
        return None
    return rel_root


def _render(rel_root: str, trigger: str, user_text: str) -> str:
    """渲染三段式拦截消息（命中证据 / 判定结论 / 下一步）。

    Contract:
      Preconditions: rel_root 与 trigger 为非空相对路径；user_text 为 str（可空）。
      Postconditions: 返回含证据、结论、下一步的文本；始终非空。
    """
    preview = user_text.strip().replace("\n", " ")[:_USER_TEXT_PREVIEW] or "(未能读取当轮原话)"
    return (
        "[scope_gate 闸门] 阻断：本次写入要在仓库内新开一棵目录树。\n"
        f"  新建目录树: {rel_root}\n"
        f"  触发文件:   {trigger}\n"
        f"  当轮用户原话（判定依据）: {preview}\n"
        "  判定: 该指令未授权新建代码产物（构造类动作），属方向级变更，不可由既有授权推导。\n"
        "\n"
        "下一步（按推荐顺序，任选其一）:\n"
        "  1. 确需新建：先把「要建什么 + 为什么 + 影响面」呈报用户，拿到明确授权后再写；\n"
        "  2. 本可复用：把写入目标改到已存在的目录树内，用 patch 扩展既有文件；\n"
        "  3. 判定有误（当轮原话确已授权构造）：在回复中引用该原话原文并重试；\n"
        "     临时降级观察模式 HERMES_SCOPE_GATE_WARN_ONLY=1，完全关闭 HERMES_SCOPE_GATE_DISABLE=1。"
    )


def _on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_tool_call：新建仓库内目录树且当轮未授权时 block。

    Contract:
      Preconditions: kwargs 可能含 tool_name/args/session_id。
      Postconditions: 命中时返回 {"action":"block","message":...}；否则 None；
                      任何异常记日志并返回 None，绝不 raise。
    """
    if _disabled():
        return None
    try:
        sid = str(kwargs.get("session_id") or "")
        if sid.startswith("cron_"):
            return None
        root = os.path.realpath(os.getcwd())
        for full in _candidate_targets(str(kwargs.get("tool_name") or ""), kwargs.get("args")):
            hit = _block_hit(full, root, sid)
            if hit is None:
                continue
            message = _render(hit, os.path.relpath(full, root), str(
                get_session_state(sid, _NAMESPACE).get("user_text") or ""
            ))
            if _warn_only():
                logger.warning("scope_gate 命中但观察模式放行:\n%s", message)
                continue
            logger.warning("scope_gate: 阻断未授权新建目录树 %s", hit)
            return {"action": "block", "message": message}
        return None
    except Exception:
        logger.exception("scope_gate: pre_tool_call 判定异常，放行")
        return None


def register(ctx: Any) -> None:
    """注册 pre_tool_call 与 pre_llm_call 两个钩子。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
