# dupcheck-ok: hook 回调 _on_pre_tool_call 为插件框架约定名（diff_debt/tool_blacklist 同形态），非重复实现；写形态识别已 import 复用 duplicate_check 基建
"""guards.batch_write — execute_code 批量修改已有文件拦截。

规则来源：agent-operating-discipline 铁律「批量修改已有文件禁 execute_code，
应手工 patch 逐文件」（用户 2026-08-01 原话 + 2026-10-10 违反实录）。认知层
纪律多次违反证明提醒无效，本模块把纪律机器化：

  pre_tool_call(execute_code) → AST 解析待执行源码，静态提取文件写入调用
  的目标路径；命中「≥2 个磁盘上已存在的文件、且不在临时目录白名单内」→
  block，指引逐文件 patch（patch 自带 diff 回显，天然满足 diff_debt 的
  证据义务）。

写形态识别复用 duplicate_check 的 AST 基建（open(w/a/x) / Path().write_text
/write_bytes / write_file() / f.write + with/赋值绑定表 + _literal_str），
本模块新增：f-string / BinOp(+) 拼接路径解析——2026-10-10 事故形态恰为
循环内拼接路径，纯字面量识别拦不住，必须解析拼接。

放行边界（宁纵勿枉）：
  - 只写 1 个已有文件 → 放行（单文件不属批量，义务由 diff_debt 管）
  - 目标全是新建文件 → 放行（无「改前状态」可丢，查重由 duplicate_check 管）
  - 路径不可静态解析 → 放行（猜错会误拦；只认高置信形态）
  - 临时/缓存目录（scratch、/tmp、macOS /var/folders）→ 豁免（审计产物
    批量落盘是合法用途）

Contract:
  Preconditions: plugin 系统提供 pre_tool_call 钩子，kwargs 携带 args.code。
  Postconditions: 命中批量修改已有文件时返回 {"action":"block",...}；
    其余一律返回 None 放行；cron 会话豁免（与 diff_debt 同判据）；
    AST 解析异常放行并 logger.warning（纪律增强层，fail-open 优于误死锁）。
  Invariants: 只拦「多个已存在文件」的写入；block 消息必须给出改走 patch
    的正路指引；永不阻断读工具。
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Any, Optional

from plugins.guards.duplicate_check import (
    _literal_str,
    _open_path_bindings,
    _py_write_call_target,
)

logger = logging.getLogger(__name__)

# 一次 execute_code 内写入「已有文件」个数达到阈值才拦——批量语义的量化定义。
# 阈值 2：1 个文件 patch 即可，2 个及以上逐个 patch 仍可行且必然更可控。
_BATCH_THRESHOLD = 2

# 临时/产物目录豁免（绝对路径前缀匹配）：审计产物、diff 落盘是合法批量写。
_EXEMPT_PREFIXES = (
    "/tmp",
    "/private/tmp",       # macOS /tmp 的真实路径
    "/var/folders",       # macOS 系统临时目录
    "/var/tmp",
    "/private/var/tmp",   # macOS pytest tmp_path 实际落点
    os.path.expanduser("~/.hermes/cache"),
)


def _is_exempt(path: str) -> bool:
    """临时/缓存目录豁免判定。

    Contract:
      Preconditions: path 为规范化后的绝对路径
      Postconditions: 命中任一豁免前缀返回 True，否则 False
    """
    return any(path.startswith(p) for p in _EXEMPT_PREFIXES)


_MAX_VALUES = 16  # 单表达式解析值数上限，防笛卡尔积爆炸


def _resolve_joined(node: ast.JoinedStr, strings: dict[str, list[str]]) -> list[str]:
    """f-string 解析：各段候选做笛卡尔积（上限截断）。"""
    combos = [""]
    for seg in node.values:
        seg_vals = _resolve_expr(seg, strings)
        if not seg_vals:
            return []
        combos = [c + s for c in combos for s in seg_vals][:_MAX_VALUES]
    return combos


def _resolve_binop(node: ast.BinOp, strings: dict[str, list[str]]) -> list[str]:
    """加法/除法拼接解析：两侧候选做笛卡尔积（上限截断）。"""
    if not isinstance(node.op, (ast.Add, ast.Div)):
        return []
    left = _resolve_expr(node.left, strings)
    right = _resolve_expr(node.right, strings)
    if not left or not right:
        return []
    joiner = "/" if isinstance(node.op, ast.Div) else ""
    return [l + joiner + r for l in left for r in right][:_MAX_VALUES]


def _resolve_path_call(node: ast.Call, strings: dict[str, list[str]]) -> list[str]:
    """Path(x) / pathlib.Path(x) 包裹形态解析（透明传参）。"""
    fn = node.func
    is_path = (
        isinstance(fn, ast.Name) and fn.id == "Path"
    ) or (
        isinstance(fn, ast.Attribute) and fn.attr == "Path"
    )
    if is_path and node.args:
        return _resolve_expr(node.args[0], strings)
    return []


def _resolve_expr(node: ast.AST, strings: dict[str, list[str]]) -> list[str]:
    """尽力把 AST 表达式解析为路径字符串候选列表（变量绑定/f-string/拼接/除法）。

    Contract:
      Preconditions: strings 为「变量名→已知字符串值列表」绑定表
      Postconditions: 可解析返回候选列表（≥1 项，≤_MAX_VALUES）；
        不确定返回 []（调用方放行，不猜）
      Invariants: 只做高置信解析——f-string 可解析段、+ 与 / 拼接的
        笛卡尔积、变量在绑定表内；其余形态一律 []
    """
    if isinstance(node, ast.Constant):
        s = _literal_str(node)
        return [s] if s is not None else []
    if isinstance(node, ast.Name):
        return list(strings.get(node.id, []))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        vals: list[str] = []
        for item in node.elts:
            vals.extend(_resolve_expr(item, strings))
        return vals[:_MAX_VALUES]
    if isinstance(node, ast.JoinedStr):
        return _resolve_joined(node, strings)
    if isinstance(node, ast.Call):
        return _resolve_path_call(node, strings)
    if isinstance(node, ast.BinOp):
        return _resolve_binop(node, strings)
    return []


def _bind_assign_target(
    t: ast.expr, value: ast.expr, strings: dict[str, list[str]],
) -> None:
    """单条 Assign 目标绑定：名字直绑；元组/列表目标与元组值按位配对。"""
    if isinstance(t, ast.Name):
        vals = _resolve_expr(value, strings)
        if vals:
            strings[t.id] = vals
    elif isinstance(t, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        for tgt, item in zip(t.elts, value.elts):
            _bind_assign_target(tgt, item, strings)


def _collect_string_assigns(tree: ast.AST) -> dict[str, list[str]]:
    """收集「变量 = 可解析字符串（含元组多赋值/for 循环字面量遍历）」绑定表。

    Contract:
      Postconditions: 返回 {变量名: 解析出的字符串值列表}；
        不可解析的赋值不进表
    """
    strings: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _bind_assign_target(t, node.value, strings)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            vals = _resolve_expr(node.iter, strings)
            if vals:
                strings[node.target.id] = vals
    return strings


def _open_or_writefile_var_target(
    node: ast.Call, merged: dict[str, list[str]],
) -> list[str]:
    """open(p,"w") / write_file(p,...) 的变量/拼接实参解析（写模式校验）。

    Contract:
      Preconditions: node.func 为名为 open 或 write_file 的调用且 node.args 非空
      Postconditions: 返回可解析的写目标路径列表；非写模式/不可解析返回 []
    """
    fn = node.func
    assert isinstance(fn, ast.Name) and fn.id in ("open", "write_file")
    resolved = _resolve_expr(node.args[0], merged)
    if not resolved:
        return []
    if fn.id == "write_file":
        return resolved
    modes = _resolve_expr(node.args[1], merged) if len(node.args) > 1 else [""]
    if any(m and any(f in m for f in ("w", "a", "x")) for m in modes):
        return resolved
    return []


def _single_call_target(
    node: ast.Call, with_bindings: dict[str, str], merged: dict[str, list[str]],
) -> list[str]:
    """单个写调用取目标：字面量→基建；变量/拼接→增强绑定表解析。

    Contract:
      Preconditions: node 为 ast.Call
      Postconditions: 返回可解析写目标列表；非写形态/不可解析返回 []
    """
    target = _py_write_call_target(node, with_bindings)
    if target is not None:
        return [target[0]]
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr in ("write_text", "write_bytes"):
        return _resolve_expr(fn.value, merged)
    if isinstance(fn, ast.Name) and fn.id in ("open", "write_file") and node.args:
        return _open_or_writefile_var_target(node, merged)
    return []


def _write_targets(tree: ast.AST) -> set[str]:
    """提取全部文件写入调用的目标路径（含拼接与变量绑定解析；不可解析跳过）。

    Contract:
      Postconditions: 返回可解析的目标路径集合；无写调用返回空集
      Invariants: 字面量形态委托 duplicate_check._py_write_call_target；
        变量/拼接形态由 _resolve_expr 在增强绑定表上补判——open(p,"w") 的
        实参与 write_text 的宿主均须解析（事故形态=循环内绑定+拼接）
    """
    strings = _collect_string_assigns(tree)
    with_bindings = _open_path_bindings(tree)
    merged: dict[str, list[str]] = {**strings, **{k: [v] for k, v in with_bindings.items()}}
    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        targets.update(_single_call_target(node, with_bindings, merged))
    return targets


def _existing_targets(code: str) -> list[str]:
    """解析源码中「磁盘上已存在且非豁免目录」的写入目标。

    Contract:
      Preconditions: code 为待执行 Python 源码
      Postconditions: 返回已存在目标的绝对路径列表；语法错误/无目标返回 []
      Invariants: 路径不可解析的写不参与计数——未证明批量即放行
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning("batch_write: 源码解析失败，放行（fail-open）: %s", e)
        return []
    existing: list[str] = []
    for raw in _write_targets(tree):
        full = os.path.abspath(os.path.expanduser(raw))
        if _is_exempt(full):
            continue
        if os.path.exists(full):
            existing.append(full)
    return existing


def _on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_tool_call：execute_code 批量修改已有文件时 block。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name != "execute_code":
        return None
    sid = str(kwargs.get("session_id") or "")
    if sid.startswith("cron_"):
        return None
    args = kwargs.get("args") or {}
    if not isinstance(args, dict):
        return None
    existing = _existing_targets(str(args.get("code") or ""))
    if len(existing) < _BATCH_THRESHOLD:
        return None
    preview = "; ".join(existing[:3])
    more = f"（共 {len(existing)} 个）" if len(existing) > 3 else ""
    return {
        "action": "block",
        "message": (
            f"[batch_write 闸门] execute_code 被阻断：本次将批量写入 {len(existing)} 个"
            f"已存在文件：{preview}{more}\n\n"
            "规则（agent-operating-discipline）：批量修改已有文件禁 execute_code，"
            "应逐文件 patch——patch 自带 diff 回显且可控可审计；批量脚本写入曾造成"
            "替换静默跳过与事后无法重建 diff（2026-08-01/2026-10-10 两次实录）。\n"
            "正路：改用 patch 工具逐文件修改（写入目标 <2 个已存在文件时本闸门放行）；"
            "确为合法批量产物落盘，写入目标放临时/缓存目录（scratch、/tmp）。"
        ),
    }


def register(ctx: Any) -> None:
    """注册 pre_tool_call hook。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
