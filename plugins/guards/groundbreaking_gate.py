"""guards.groundbreaking_gate — 新建应用首代码文件的功课证据门禁（L2 子护栏）。

规则来源：2026-10-14 experiment_lab 事故 + 门禁 v1 推翻重做。用户授权「门禁
必须上」后直接编码的形态（授权通过/功课缺失/格式合规）在既有全防线零拦截——
本模块把「七步功课产物存在」机器化，拦在写入路径上。

设计 SSOT：docs/groundbreaking_gate_design_20261014.md（反方审查
deleg_13eaea95 有条件通过，8 项修正已回灌——本实现按修正后判据）。

判定（全部确定性文件存在性检查，零 LLM 零正则）：
  触发 = write_file/patch/execute_code(写) 目标为 apps/<X>/ 下代码文件
  （.py/.ts/.vue/.go/.java，存量扫描集合=触发集合，含 tests 与 __init__）
  且 apps/<X>/ 内无其他代码文件（新应用首代码文件）
  放行（按序）：
    1. 会话态缓存命中（存证据绝对路径，exists 复验，失效即清）
    2. 仓根豁免清单 GROUNDBREAKING_EXEMPT.md 首列含应用名（放行+WARNING）
    3. 存量应用不回溯
    4. 三证据源任一存在：
       E1 apps/<X>/ 内非 README 的 .md，有效行数≥_MIN_DOC_LINES 且
          标题行含「数据结构/链路/架构」任一（空文档不构成证据）
       E2 apps/<X>/ 内 *.sql 或 migrations/alembic/db/schema/ 目录
       E3 仓根 docs/blueprint/*.md 文件名 token 整词匹配应用名（禁子串：
          实测 'aml' 子串误命中 3 份无关文档）且满足 E1 同款内容判据
  全无 → block（三段式消息：命中证据/三源三态/分档下一步+skill 指针）

Contract:
  Preconditions: plugin system 提供 pre_tool_call 钩子; execute_code 写形态
    识别复用 duplicate_check._execute_code_write_targets（单一事实源）。
  Postconditions: 守卫自身异常 → None 放行（fail-open+WARNING，纪律增强层
    宁放不锁死）；判定语义 fail-closed（三证据全无=拦）。
  Invariants: 只拦新应用首代码文件；存量应用/非 apps 路径/文档扩展零打扰。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from plugins._shared_state import get_session_state
from plugins.guards.duplicate_check import _execute_code_write_targets

logger = logging.getLogger(__name__)

_NAMESPACE = "groundbreaking_gate"

# 代码扩展集合：存量扫描集合=触发集合（含 tests/*.py 与 __init__.py，两集合强制相等）
_CODE_EXTS = (".py", ".ts", ".vue", ".go", ".java")
# E1/E3 最小内容判据：非空有效行数下限 + 必备章节标题
_MIN_DOC_LINES = 30
# 结构章节必备词组（任一命中即可，覆盖域惯用写法）：表结构是重中之重
# （用户定调 2026-10-14「表结构的质量和要素覆盖情况直接影响最终系统的成败和质量」），
# 缺结构章节的功课文档不构成证据——链路/架构章节不能替代表结构本体
_SCHEMA_HEADINGS = ("数据结构", "表族", "表结构")
# 数据库裁决为必备章节（用户指令 2026-10-14「确定数据结构的时候要先确定数据库」）：
# 缺库归属裁决的功课文档不构成证据
_MANDATORY_HEADINGS = ("数据库",)
# 用户层章节必备词组（任一命中即可，2026-10-15 补：七步法第 6 步「产品能力」的
# 机器判据——谁用/什么场景/拿到什么。缺用户层章节=能力规划未做，功课文档不构成证据；
# 对应本会话用户四问暴露的「诉求覆盖无对账载体」缺口）
_USER_HEADINGS = ("用户", "旅程", "诉求")
# 业务层章节必备词组（任一命中即可，2026-10-15 补：功课文档必须回答
# 「要处理哪些业务」——处理域清单/场景枚举。缺业务章节=业务边界未划清，
# 功课文档不构成证据；对应用户四问的「要处理哪些业务」轴）
_BIZ_HEADINGS = ("业务", "场景", "领域")
# E2 数据结构痕迹：DDL 目录名
_SCHEMA_DIR_NAMES = ("migrations", "alembic", "db", "schema")
# E3 token 匹配噪音词表：短结构词整词命中降级（不构成 E3 信号 token）
_NOISE_TOKENS = frozenset({"api", "web", "backend", "data", "app", "core", "lib"})
# 仓根豁免清单（逐应用一行：应用名+理由+日期；禁在被写文件内自豁免）
_EXEMPT_FILENAME = "GROUNDBREAKING_EXEMPT.md"


def _plugin_disabled() -> bool:
    """Contract: Postconditions: env 设 1/true/yes → True（完全禁用）。"""
    return os.environ.get("HERMES_GROUNDBREAKING_GATE_DISABLE", "").lower() in {
        "1", "true", "yes",
    }


def _anchored_candidates(path: Path) -> list[Path]:
    """相对路径锚定候选：cwd 及其各级祖先拼接原始相对路径。

    Contract:
      Postconditions: 返回 [cwd/p, cwd.parent/p, ..., /p]；目录是否真实存在
        由调用方逐个检验（此处只生成候选，不做存在性断言）。
    """
    cwd = Path.cwd()
    return [anc / path for anc in [cwd, *cwd.parents]]


def _locate_app(path: Path) -> Optional[tuple[Path, str]]:
    """定位 /apps/<name>/ 段，返回 (apps 根, 应用名)；无该段返回 None。

    相对路径先沿 cwd 祖先链锚定（插件进程 cwd 可能不是仓库根——resolve 会
    把 apps/<X>/ 解析到不存在位置，导致存量应用被误判为新应用且三证据全查
    错路径）；锚定成功的判据=候选 apps 根目录自身真实存在（仅查父目录会放
    过幽灵根——cwd 下不存在的 apps/<X> 被误判为新应用）。绝对路径与锚定
    全失败时回退纯 resolve（新应用首目录未建时仍可触发门禁语义）。

    Contract:
      Preconditions: path 为绝对或可绝对化路径
      Postconditions: 命中=apps 段后仍有子路径（目标在应用目录内）
    """
    candidates: list[Path] = [path] if path.is_absolute() else _anchored_candidates(path)
    for cand in candidates:
        parts = cand.resolve().parts
        for i, seg in enumerate(parts):
            if seg == "apps" and i + 2 < len(parts):
                root = Path(*parts[: i + 1])
                if root.is_dir():
                    return root, parts[i + 1]
    parts = path.resolve().parts
    for i, seg in enumerate(parts):
        if seg == "apps" and i + 2 < len(parts):
            root = Path(*parts[: i + 1])
            if root.is_dir():
                return root, parts[i + 1]
    return None


def _has_any_heading(lines: list[str], headings: tuple[str, ...]) -> bool:
    """判定标题行是否命中词组集合中任一词（零正则，str 方法）。

    Contract:
      Postconditions: 任一标题行（# 开头）含任一词 → True，否则 False。
    """
    return any(
        ln.strip().startswith("#") and any(h in ln for h in headings)
        for ln in lines
    )


def _heading_flags(lines: list[str]) -> tuple[bool, bool, bool, bool]:
    """扫描标题行，返回 (含必备「数据库」标题, 含用户层标题任一, 含业务层标题任一, 含结构标题任一)。"""
    return (
        _has_any_heading(lines, _MANDATORY_HEADINGS),
        _has_any_heading(lines, _USER_HEADINGS),
        _has_any_heading(lines, _BIZ_HEADINGS),
        _has_any_heading(lines, _SCHEMA_HEADINGS),
    )


def _doc_is_groundwork(md_path: Path) -> bool:
    """E1/E3 内容判据：非空有效行数≥30 且标题行含必备「数据库」+用户层标题任一
    （用户/旅程/诉求——产品能力证据）+业务层标题任一
    （业务/场景/领域——处理域边界证据）+结构标题任一
    （数据结构/表族/表结构——表结构本体证据，链路/架构章节不能替代）。

    Contract:
      Postconditions: 文件不可读/行数不足/无「数据库」标题/无用户层标题/
        无业务层标题/无结构标题任一 → False（空文档与缺库归属裁决、缺产品
        能力、缺业务边界、缺表结构本体的文档均不构成证据）
    """
    try:
        lines = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    if sum(1 for ln in lines if ln.strip()) < _MIN_DOC_LINES:
        return False
    has_mandatory, has_user, has_biz, has_schema = _heading_flags(lines)
    return has_mandatory and has_user and has_biz and has_schema


def _e1_groundwork_doc(app_dir: Path) -> Optional[Path]:
    """E1：返回应用内首份满足内容判据的功课文档路径（非 README），无则 None。

    Contract:
      Postconditions: 返回具体文档路径（缓存载体——目录不可作载体，否则
        删除文档后目录仍在=缓存续命，违反 R6 复验语义）。
    """
    for p in app_dir.rglob("*.md"):
        if not p.name.lower().startswith("readme") and _doc_is_groundwork(p):
            return p
    return None


def _e2_schema_in_app(app_dir: Path) -> bool:
    """E2：应用内存在数据结构痕迹（*.sql 或 DDL 目录）。

    Contract:
      Postconditions: 目录不存在 → False（iterdir 只对存在目录调用，
        否则 FileNotFoundError 被 fail-open 吞掉=守卫静默失效）。
    """
    if not app_dir.is_dir():
        return False
    if any(True for _ in app_dir.rglob("*.sql")):
        return True
    return any(
        p.is_dir() and p.name in _SCHEMA_DIR_NAMES for p in app_dir.iterdir()
    )


def _name_tokens(name: str) -> list[str]:
    """按 -_ 切分标识符 token（str 方法，零正则）。"""
    return [t for t in name.replace("-", "_").replace(".", "_").split("_") if t]


def _e3_groundwork_doc(repo_root: Path, app_name: str) -> Optional[Path]:
    """E3：返回仓根 docs/blueprint/ 中 token 整词匹配应用名的功课文档路径。

    Contract:
      Postconditions: 应用名 token 中的噪音词（api/web 等）不作为匹配信号，
        须由非噪音 token 整词匹配（禁子串）；匹配到的文档还须过内容判据；
        返回具体文档路径（缓存载体），无则 None。
    """
    app_tokens = [t.lower() for t in _name_tokens(app_name)]
    signal = [t for t in app_tokens if t not in _NOISE_TOKENS]
    if not signal:
        return None
    bp = repo_root / "docs" / "blueprint"
    if not bp.is_dir():
        return None
    for p in bp.glob("*.md"):
        file_tokens = {t.lower() for t in _name_tokens(p.stem)}
        if file_tokens.issuperset(signal) and _doc_is_groundwork(p):
            return p
    return None


def _evidence_path(app_dir: Path) -> Optional[str]:
    """三证据源判定，返回命中证据具体文件路径（缓存载体），全无返回 None。

    Contract:
      Postconditions: E1/E2 查应用目录，E3 查仓根；任一命中返回实际存在的
        具体文件路径——目录作载体会在证据文件被删后缓存续命（违反 R6）。
    """
    e1 = _e1_groundwork_doc(app_dir)
    if e1 is not None:
        return str(e1)
    if _e2_schema_in_app(app_dir):
        sql = next(iter(app_dir.rglob("*.sql")), None)
        return str(sql if sql is not None else app_dir / "migrations")
    e3 = _e3_groundwork_doc(app_dir.parent.parent, app_dir.name)
    if e3 is not None:
        return str(e3)
    return None


def _exempt_listed(repo_root: Path, app_name: str) -> bool:
    """仓根豁免清单首列含应用名（归一化整名相等）；命中记 WARNING 留痕。

    Contract:
      Postconditions: 清单不存在/不可读/无该应用行 → False（放行逻辑不触发）。
    """
    exempt = repo_root / "docs" / "blueprint" / _EXEMPT_FILENAME
    if not exempt.is_file():
        return False
    try:
        rows = exempt.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return False
    norm = app_name.replace("-", "_")
    for row in rows:
        cells = row.split("|") if "|" in row else row.split()
        if not cells:
            continue
        first = cells[0].strip()
        if first and first.replace("-", "_") == norm:
            logger.warning("groundbreaking_gate: %s 经豁免清单放行: %s", app_name, row)
            return True
    return False


def _get_state(sid: str) -> dict[str, Any]:
    return get_session_state(sid, _NAMESPACE)


def _cache_hit(sid: str, app_name: str) -> bool:
    """会话态缓存：存命中证据绝对路径，命中时 exists 复验，失效即清。

    Contract:
      Postconditions: 证据文件被删 → 缓存清空并返回 False（重走三证据）。
    """
    ev = str(_get_state(sid).get("apps", {}).get(app_name, "") or "")
    if ev and os.path.exists(ev):
        return True
    if ev:
        _get_state(sid).setdefault("apps", {}).pop(app_name, None)
    return False


def _cache_put(sid: str, app_name: str, evidence: str) -> None:
    _get_state(sid).setdefault("apps", {})[app_name] = evidence


def _call_targets(tool_name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
    """解析本次写调用的 [(路径, 内容)] 目标集。

    Contract:
      Postconditions: write_file/patch 单目标；execute_code 走 duplicate_check
        ._execute_code_write_targets（单一事实源，五扩展全集）；其余工具 []。
    """
    if tool_name in ("write_file", "patch"):
        raw = str(args.get("path", ""))
        text = str(args.get("content") or args.get("new_string") or "")
        return [(raw, text)] if raw else []
    if tool_name == "execute_code":
        code = str(args.get("code", ""))
        return _execute_code_write_targets(code, _CODE_EXTS) if code else []
    return []


def _block_message(raw_path: str, app_name: str, e1: bool, e2: bool, e3: bool) -> dict[str, Any]:
    """三段式 block 反馈（命中证据/三源三态/分档下一步）。"""
    repo_hint = f"apps/{app_name}/docs/ 或仓根 docs/blueprint/（文件名含 {app_name}）"
    return {
        "action": "block",
        "message": (
            "[groundbreaking-gate] 新建应用缺功课证据，禁止直接写代码文件：\n"
            f"命中证据: 目标={raw_path}（新应用 apps/{app_name}/ 首个代码文件）\n"
            f"三证据源: E1 应用内功课文档(非README, ≥{_MIN_DOC_LINES}行含必备「数据库」标题+数据结构/链路/架构任一)"
            f"={'有' if e1 else '无'}; E2 数据结构痕迹(.sql/migrations/alembic/db/schema)"
            f"={'有' if e2 else '无'}; E3 仓根 docs/blueprint/ 文件名含 {app_name} 的功课文档"
            f"={'有' if e3 else '无'}\n"
            "判定: 三源全无=空壳应用形态（代码先于本体）\n"
            "下一步(按 skill:app-groundbreaking-seven-steps 顺序): "
            f"①完成七步功课并保存到 {repo_hint}; "
            "②先建专属数据结构（app 内 DDL/migrations，表绿后才写消费代码）; "
            f"③纯适配层在仓根 docs/blueprint/{_EXEMPT_FILENAME} 登记（应用名+理由+日期）"
        ),
    }


def _existing_code_file(app_dir: Path, target: Path) -> bool:
    """存量判定：应用目录内存在除本次目标外的任意代码文件（含 tests/__init__）。

    Contract:
      Postconditions: 目录不存在/无其他代码文件 → False（触发门禁）。
    """
    if not app_dir.is_dir():
        return False
    for p in app_dir.rglob("*"):
        if p.is_file() and p.suffix in _CODE_EXTS and p.resolve() != target:
            return True
    return False


def _check_target(sid: str, raw_path: str) -> Optional[dict[str, Any]]:
    """单目标判定流：定位 app → 缓存 → 豁免清单 → 存量 → 三证据 → block/None。"""
    path = Path(raw_path)
    located = _locate_app(path)
    if located is None or path.suffix not in _CODE_EXTS:
        return None
    apps_root, app_name = located
    app_dir = apps_root / app_name
    if _cache_hit(sid, app_name):
        return None
    if _exempt_listed(apps_root.parent, app_name):
        return None
    if _existing_code_file(app_dir, path.resolve()):
        _cache_put(sid, app_name, str(app_dir))
        return None
    e1_doc = _e1_groundwork_doc(app_dir) if app_dir.is_dir() else None
    e2 = app_dir.is_dir() and _e2_schema_in_app(app_dir)
    e3_doc = _e3_groundwork_doc(apps_root.parent, app_name)
    evidence = _evidence_path(app_dir)
    if evidence is not None:
        _cache_put(sid, app_name, evidence)
        return None
    return _block_message(raw_path, app_name, e1_doc is not None, e2, e3_doc is not None)


def on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre_tool_call 入口——遍历写目标，任一命中 block 即返回。

    Contract:
      Postconditions: 目标全放行 → None；命中未做功课 → {"action":"block",...}；
        守卫自身异常 → None 放行（fail-open+WARNING，绝不 raise）。
    """
    if _plugin_disabled():
        return None
    try:
        sid = str(kwargs.get("session_id", "") or kwargs.get("task_id", ""))
        tool_name = str(kwargs.get("tool_name", ""))
        args = kwargs.get("args") or {}
        for raw_path, _text in _call_targets(tool_name, args):
            verdict = _check_target(sid, raw_path)
            if verdict is not None:
                return verdict
        return None
    except Exception:
        logger.warning("groundbreaking_gate 检查失败(放行)", exc_info=True)
        return None


def register(ctx: Any) -> None:
    """Contract: Postconditions: 注册 pre_tool_call 钩子；不抛异常。"""
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.groundbreaking_gate registered")
