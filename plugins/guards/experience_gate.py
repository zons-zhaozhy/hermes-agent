"""guards.experience_gate — frontend experience-completeness gate (opt-in).

Motivation: aml_agents 2026-10-26 — the user had to point out, one by one,
one-way tab linkage, mismatched stat-card font sizes, unpaged tables and
bare tables. Root cause: experience completeness had no machine-enforced
criteria, so every release shipped on "better than last version" instead of
an absolute standard. Docs/checklists alone do not hold; this guard makes
the statically-decidable subset mechanical.

Blocked shape (ALL must hold):
  1. tool is ``write_file`` or ``patch``
  2. target extension is frontend render: .html/.vue/.tsx/.jsx
  3. an opt-in config ``.hermes/experience_gate.yaml`` exists on the target
     path's ancestor chain (no file ⇒ fail-open)
  4. AND one of:
     a. pagination: content contains a table-render marker (renderTable /
        el-table / <tbody / <table) but none of the pagination markers
        (pagination / pager / PAGE_SIZE / el-pagination / 分页)
     b. type scale: a ``font-size: Npx`` value in the content is not in the
        project's ``type_scale`` ladder
  5. the content does not carry an inline waiver ``exp-ok:``

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args.
  Postconditions: returns None (allow) or {"action":"block","message":str};
    never raises; config load failures fail open with a log line.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = ".hermes/experience_gate.yaml"
_WAIVER_MARK = "exp-ok"
_FRONTEND_EXT = frozenset({".html", ".vue", ".tsx", ".jsx"})
from plugins.guards.tool_classification import FILE_WRITE_TOOLS as _TOOLS

_DEFAULT_TABLE_MARKERS = ("renderTable", "el-table", "<tbody", "<table")
_DEFAULT_PAGINATION_MARKERS = ("pagination", "pager", "PAGE_SIZE", "el-pagination", "分页")
_DEFAULT_TYPE_SCALE: tuple[float, ...] = (
    10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 20, 22, 24, 30,
)

_FONT_SIZE_RE = re.compile(r"font-size:\s*(\d+(?:\.\d+)?)px")  # re-ok: 数值档位提取,无回溯风险

# 配置缓存: (mtime_ns, size) -> 解析结果。进程内生效,配置变更下次写入可见。
_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _find_config(path: str) -> Optional[str]:
    """沿目标路径祖先链找启用配置;找不到返回 None。

    Contract:
      Postconditions: 返回存在的配置绝对路径或 None;不抛异常。
    """
    probe = os.path.abspath(path)
    while True:
        parent = os.path.dirname(probe)
        if parent == probe:
            return None
        candidate = os.path.join(parent, _CONFIG_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        probe = parent


def _parse_config(data: Any) -> dict[str, Any]:
    """从配置 YAML 对象提取 type_scale 与两组特征表;非法项静默跳过。

    Contract:
      Preconditions: data 为 safe_load 结果(可能为 None/dict)。
      Postconditions: 返回 dict(键可缺省);纯转换无副作用,不抛异常。
    """
    parsed: dict[str, Any] = {}
    if not isinstance(data, dict):
        return parsed
    if isinstance(data.get("type_scale"), list):
        scale = tuple(float(x) for x in data["type_scale"]
                      if isinstance(x, (int, float)) and x > 0)
        if scale:
            parsed["type_scale"] = scale
    for key in ("table_render_markers", "pagination_markers"):
        val = data.get(key)
        if isinstance(val, list):
            markers = tuple(str(x) for x in val if str(x))
            if markers:
                parsed[key] = markers
    return parsed


def _load_config(cfg_path: str) -> dict[str, Any]:
    """按 mtime+size 缓存解析配置;任何失败返回空 dict(fail-open)。

    Contract:
      Preconditions: cfg_path 存在(调用方已 isfile 判定)。
      Postconditions: 返回 dict(可能为空);解析异常吞掉记日志,永不抛出。
    """
    try:
        from hermes_cli.config_backend import config_version, read_config_doc_readonly

        stamp = config_version(cfg_path)  # 期望: raises OSError like os.stat; FileNotFoundError ⇒ missing config
    except OSError:
        return {}
    cached = _cache.get(cfg_path)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        parsed = _parse_config(read_config_doc_readonly(cfg_path))  # 期望: doc dict in, parsed dict out
    except Exception:
        logger.warning(
            "guards.experience_gate: config %s unreadable; failing open", cfg_path, exc_info=True
        )
        return {}
    _cache[cfg_path] = (stamp, parsed)
    return parsed


def _pagination_missing(text: str, cfg: dict[str, Any]) -> bool:
    """表格渲染特征存在而分页特征全缺。

    Contract:
      Preconditions: text 为写入内容字符串。
      Postconditions: 返回 bool;纯扫描无副作用。
    """
    table_markers = cfg.get("table_render_markers") or _DEFAULT_TABLE_MARKERS
    page_markers = cfg.get("pagination_markers") or _DEFAULT_PAGINATION_MARKERS
    has_table = any(m in text for m in table_markers)
    has_paging = any(m in text for m in page_markers)
    return has_table and not has_paging


def _off_scale_sizes(text: str, cfg: dict[str, Any]) -> list[tuple[float, int]]:
    """返回不在档位表内的 (px 值, 行号) 列表(不含豁免行)。

    Contract:
      Preconditions: text 为写入内容字符串。
      Postconditions: 返回列表(可能为空);纯扫描无副作用。
    """
    scale_set = {float(x) for x in (cfg.get("type_scale") or _DEFAULT_TYPE_SCALE)}
    hits: list[tuple[float, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _WAIVER_MARK in line:
            continue
        for m in _FONT_SIZE_RE.finditer(line):
            px = float(m.group(1))
            if px not in scale_set:
                hits.append((px, line_no))
    return hits


def _target_text(args: dict[str, Any]) -> Optional[tuple[str, str]]:
    """提取 (path, 写入内容);不适用形态返回 None。

    Contract:
      Preconditions: args 为 dict。
      Postconditions: 返回非空 path+text 二元组或 None;纯提取无副作用。
    """
    path = str(args.get("path") or "")
    if not path or os.path.splitext(path)[1].lower() not in _FRONTEND_EXT:
        return None
    text = str(args.get("content") or args.get("new_string") or "")
    if not text:
        return None
    return path, text


def _collect_problems(text: str, cfg: dict[str, Any]) -> list[tuple[str, str]]:
    """汇总 (问题, 修法) 对;空列表=放行。

    Contract:
      Preconditions: text 为写入内容字符串,cfg 为已解析配置。
      Postconditions: 返回可空列表;纯扫描无副作用。
    """
    problems: list[tuple[str, str]] = []
    if _pagination_missing(text, cfg) and _WAIVER_MARK not in text:
        problems.append((
            "表格渲染存在但无任何分页特征(pagination/pager/PAGE_SIZE/分页)",
            "表格渲染必须带分页(>20 行分页或容器限高+滚动),或静态小表整段加 exp-ok",
        ))
    off = _off_scale_sizes(text, cfg)
    if off:
        problems.append((
            "字号档位表外的 font-size: " + ", ".join(f"{px:g}px(第{no}行)" for px, no in off[:6]),
            "字号必须取自项目档位表 type_scale;新档位先扩配置再使用",
        ))
    return problems


_BLOCK_TEMPLATE = (
    "[ExperienceGate] 前端文件写入命中体验完备性门禁——体验判据已机器化,不再靠用户逐条指出。\n"
    "配置: {cfg}\n"
    "命中: {hits}\n"
    "处置: {fix}\n"
    "豁免: 确属静态展示小表/既有档位外的合法设计值,在该行加行内豁免标记 exp-ok: <理由>;\n"
    "调整档位表/特征表直接改配置文件(无配置 fail-open 不执法)。"
)


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook: block frontend writes violating the experience gate.

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict).
      Postconditions: returns None (allow) or {"action":"block","message":str}.
    """
    if kwargs.get("tool_name", "") not in _TOOLS:
        return None
    target = _target_text(kwargs.get("args") or {})
    if target is None:
        return None
    path, text = target

    cfg_path = _find_config(path)
    if cfg_path is None:
        return None
    problems = _collect_problems(text, _load_config(cfg_path))
    if not problems:
        return None
    logger.warning(
        "guards.experience_gate: blocked frontend write (%s problems, cfg=%s)",
        len(problems), cfg_path,
    )
    return {
        "action": "block",
        "message": _BLOCK_TEMPLATE.format(
            cfg=cfg_path,
            hits="; ".join(p for p, _ in problems),
            fix="; ".join(f for _, f in problems),
        ),
    }


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.experience_gate registered")
