"""
duplicate-check — 新建 .py 前置查重拦截插件 v4.0。

v4.0 射程补全（旧版 `if tool_name != "write_file": return None` 使 execute_code /
terminal 通道可绕开查重——守卫射程漏洞，非能力边界）:
- 四通道统一射程：write_file / patch / execute_code / terminal 一律判「新建 .py」。
  execute_code 用 AST 静态识别字面量写形态（open(...,"w") / Path().write_text /
  write_file(path, content) / 链式 open().write），terminal 识别写算子（> >> tee cp mv）
  与 heredoc 正文；同目录既有守卫（pre_write / casebook_gate）早已按 _WRITE_TOOLS 多通道。
- 内容不可解析（写调用存在但内容非字面量）→ 提示档明示「符号级比对未生效」，不猜不静默；
  路径非字面量（变量/拼接/动态生成）无法解析 → 不猜（猜错会误拦），射程边界如实标注。

v3.0 根因修复（旧版把「文件名主题词」当等价证据 → 系统性误报；且反馈不可执行）:
- 判定升级为符号级证据：从新文件内容提取将要定义的 def/class 名，与全库同名符号
  比对——同名才是真重复信号。文件名主题词命中降级为提示（主题词在成熟命名空间下
  必然命中，如 connector/api/report，命中不等于等价实现）。
- 反馈可执行：输出命中证据（哪个文件/哪个符号）+ 判定结论 + 分档下一步，
  不再把「全局关闭查重」当唯一出路。新增按文件豁免标记 `# dupcheck-ok: <理由>`。

阻断档（内容级硬证据）:
  策略1 新文件与同目录既有文件共享 ≥4 字符文件名词干 **且** 存在同名 def/class
  策略2 新文件声明的 def/class 名与既有文件同名
提示档（放行 + 留痕）:
  仅文件名主题词命中 / 仅同目录词干命中（无同名符号佐证）/ 内容非字面量（比对未生效）

环境变量:
  HERMES_DUPCHECK_DISABLE=1    完全禁用（不推荐：关闭全部查重）
  HERMES_DUPCHECK_WARN_ONLY=1  仅提示不阻断（观察模式）

检索工具契约:
  rg 显式传搜索根 + stdin=DEVNULL——rg 在「无路径参数 + stdin 非 TTY」时会转去读
  stdin 并阻塞至超时（实测 0.01s → 超 6s）；工具故障不谎报零命中，
  判定降级为提示档并明示「符号级比对未生效」
"""
import ast
import fnmatch
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

# 以下通配模式的文件免检（测试文件/构建产物等；fnmatch 语义，零正则）
_SKIP_GLOBS = [
    "test_*.py",             # test_foo.py
    "*_test.py",             # foo_test.py
    "conftest.py",           # pytest fixtures
    "__init__.py",           # package marker
    "setup.py",              # package setup
    "migrations/*",          # Django alembic 迁移
    "*alembic/versions/*",   # alembic 迁移文件——任意深度,revision链唯一命名,查重无意义(003 曾被误拦)
]

# 以下文件名前缀不参与匹配（太通用）
_NOISE_PREFIXES = {"test", "tmp", "temp", "util", "helper", "common", "base",
                   # 纯结构目录名——非功能关键词,拿去搜函数名必误中
                   "app", "src", "lib", "backend", "frontend", "server",
                   "client", "api", "core", "main", "versions", "alembic"}

# 按文件豁免标记（行内豁免，家族惯例：`# <gate>-ok: <理由>`，理由 ≥8 字符）
_INLINE_OK_PREFIX = "# dupcheck-ok:"
_INLINE_OK_MIN_REASON = 8

# 框架约定入口名——同类模块人人同名，纳入同名判定必然误报，直接剔除
_ENTRYPOINT_NAMES = {"register", "setup"}

# git ls-files 缓存
_cache: dict[str, dict[str, Any]] = {}


def _disabled() -> bool:
    return os.environ.get("HERMES_DUPCHECK_DISABLE", "").strip() == "1"


def _warn_only() -> bool:
    return os.environ.get("HERMES_DUPCHECK_WARN_ONLY", "").strip() == "1"


def _should_skip(path: str) -> bool:
    """检查是否属于免检模式（fnmatch 通配，零正则）。"""
    if os.path.basename(path).startswith("."):
        return True
    for pat in _SKIP_GLOBS:
        if fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(os.path.basename(path), pat):
            return True
    return False


# ═══════════════════════════════════════════
# 核心检测逻辑
# ═══════════════════════════════════════════


def _normalize_new_py(path: str, cwd: str) -> str:
    """把候选路径规范化为「项目内 + 尚不存在 + 非免检」的 .py 相对路径；否则 ""。

    Contract:
      Postconditions: 返回项目内相对 .py 路径；非 .py / 已存在 / 免检 / 项目外 → ""
      Invariants: 已存在文件属「修改」不属「新建」，不进查重射程；
                  项目外临时脚本（/tmp、~/.hermes、系统目录）一律不参与——
                  git ls-files 看不到它们，任何「匹配」都只能是误报
    """
    if not path or not path.endswith(".py"):
        return ""
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        real_cwd = os.path.realpath(cwd)
        real_full = os.path.realpath(full)
    except OSError as e:
        logger.warning("realpath 解析失败(%s): %s", path, e)
        return ""
    if os.path.exists(real_full) or not real_full.startswith(real_cwd + os.sep):
        return ""
    rel = os.path.relpath(real_full, real_cwd)
    if _should_skip(rel):
        return ""
    return rel


# ── 写通道目标解析（跨通道统一射程）────────────────────────────────


def _literal_str(node: Any) -> Optional[str]:
    """AST 节点取字面量字符串；非字符串字面量返回 None。

    Contract:
      Postconditions: ast.Constant 且值为 str → 返回该串；否则 None
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_from_open(node: ast.Call) -> Optional[tuple[str, str]]:
    """识别 open("x.py", "w"/"a"/"x") 形态（路径必须字面量）。"""
    fn = node.func
    if not (isinstance(fn, ast.Name) and fn.id == "open" and node.args):
        return None
    path = _literal_str(node.args[0])
    mode = _literal_str(node.args[1]) if len(node.args) > 1 else ""
    if path is None or not mode or not any(f in mode for f in ("w", "a", "x")):
        return None
    return path, ""


def _owner_literal_path(node: ast.Call) -> Optional[str]:
    """取方法调用宿主的第一实参字面量（Path("x.py") / open("x.py", "w")）。"""
    owner = node.func.value if isinstance(node.func, ast.Attribute) else None
    if isinstance(owner, ast.Call) and owner.args:
        return _literal_str(owner.args[0])
    return None


def _target_from_writetext(node: ast.Call) -> Optional[tuple[str, str]]:
    """识别 Path("x.py").write_text("...") / write_bytes 形态。"""
    fn = node.func
    if not (isinstance(fn, ast.Attribute) and fn.attr in ("write_text", "write_bytes")):
        return None
    path = _owner_literal_path(node)
    if path is None:
        return None
    content = _literal_str(node.args[0]) if node.args else ""
    return path, content or ""


def _target_from_writefile(node: ast.Call) -> Optional[tuple[str, str]]:
    """识别 write_file("x.py", "...")（hermes_tools 写通道）。"""
    fn = node.func
    if not (isinstance(fn, ast.Name) and fn.id == "write_file" and node.args):
        return None
    path = _literal_str(node.args[0])
    if path is None:
        return None
    content = _literal_str(node.args[1]) if len(node.args) > 1 else ""
    return path, content or ""


def _with_open_bindings(node: ast.AST) -> dict[str, str]:
    """从 `with open("x.py","w") as f` 收集 变量→路径 绑定。"""
    bound: dict[str, str] = {}
    if not isinstance(node, ast.With):
        return bound
    for item in node.items:
        if not isinstance(item.context_expr, ast.Call):
            continue
        opened = _target_from_open(item.context_expr)
        if opened is None or not isinstance(item.optional_vars, ast.Name):
            continue
        bound[item.optional_vars.id] = opened[0]
    return bound


def _assign_open_bindings(node: ast.AST) -> dict[str, str]:
    """从 `f = open("x.py","w")` 赋值收集 变量→路径 绑定。"""
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
        return {}
    opened = _target_from_open(node.value)
    if opened is None:
        return {}
    return {t.id: opened[0] for t in node.targets if isinstance(t, ast.Name)}


def _open_path_bindings(tree: ast.AST) -> dict[str, str]:
    """收集 open 写模式的变量→字面量路径绑定（with / 赋值两形态）。

    Contract:
      Postconditions: 返回 {变量名: 字面量路径}；无绑定 → {}
      Invariants: 只收集写模式 open 且路径为字面量的绑定——动态路径不进表
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        bound.update(_with_open_bindings(node))
        bound.update(_assign_open_bindings(node))
    return bound


def _target_from_write_method(node: ast.Call, bound: dict[str, str]) -> Optional[tuple[str, str]]:
    """识别 f.write("...") 形态（f 来自 open 链式或变量绑定）。"""
    fn = node.func
    if not (isinstance(fn, ast.Attribute) and fn.attr == "write" and node.args):
        return None
    owner = fn.value
    path: Optional[str] = None
    if isinstance(owner, ast.Name):
        path = bound.get(owner.id)
    elif isinstance(owner, ast.Call):
        opened = _target_from_open(owner)
        path = opened[0] if opened is not None else None
    if path is None:
        return None
    content = _literal_str(node.args[0]) if node.args else ""
    return path, content or ""


def _py_write_call_target(node: ast.Call, bound: dict[str, str]) -> Optional[tuple[str, str]]:
    """从单个 ast.Call 识别写目标；非写形态或路径非字面量返回 None。

    Contract:
      Postconditions: 返回 (路径字面量, 内容字面量或 "")；无法识别 → None
      Invariants: 只认四种高置信写法（open / write_text / write_file / write 方法），
                  动态路径一律不猜——猜错会误拦
    """
    for handler in (_target_from_open, _target_from_writetext, _target_from_writefile):
        found = handler(node)
        if found is not None:
            return found
    return _target_from_write_method(node, bound)


def _execute_code_py_targets(code: str) -> list[tuple[str, str]]:
    """从 execute_code 源码静态提取字面量 .py 写目标。

    Contract:
      Preconditions: code 为待执行 Python 源码
      Postconditions: 返回 [(路径, 内容或 "")]；语法错误 / 无字面量目标 → []
      Invariants: 内容不可解析（路径字面量但内容非字面量）时内容段为 ""——调用方降级提示
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning("execute_code 源码解析失败，跳过查重: %s", e)
        return []
    bound = _open_path_bindings(tree)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = _py_write_call_target(node, bound)
            if target is not None and target[0].endswith(".py"):
                found.append(target)
    return found


def _heredoc_body(cmd: str) -> str:
    """取 shell heredoc 正文（<<'EOF' … EOF / <<EOF … EOF）；无则空串。

    Contract:
      Postconditions: 返回定界符之间的正文（不含定界行）；无 heredoc / 未闭合 → ""
    """
    idx = cmd.find("<<")
    if idx < 0:
        return ""
    rest = cmd[idx + 2:].lstrip()
    if not rest:
        return ""
    if rest[0] in ("'", '"'):
        quote = rest[0]
        close = rest.find(quote, 1)
        if close < 0:
            return ""
        marker, after = rest[1:close], close + 1
    else:
        end = 0
        while end < len(rest) and (rest[end].isalnum() or rest[end] in "_-"):
            end += 1
        marker, after = rest[:end], end
    if not marker:
        return ""
    body: list[str] = []
    for line in rest[after:].split("\n")[1:]:
        if line.strip() == marker:
            break
        body.append(line)
    return "\n".join(body)


_TERMINAL_WRITE_OPS = (">", ">>", "tee", "cp", "mv")


def _terminal_py_targets(cmd: str) -> list[tuple[str, str]]:
    """从 terminal 写命令提取字面量 .py 目标（内容取 heredoc 正文）。

    Contract:
      Postconditions: 返回 [(目标, heredoc 正文或 "")]；仅写算子后的 .py token
      Invariants: 只认 > >> tee cp mv 的直接目标；其余形态不猜
    """
    body = _heredoc_body(cmd)
    found: list[tuple[str, str]] = []
    for tokens in (seg.split() for seg in cmd.replace(";", "\n").split("\n")):
        for i, tok in enumerate(tokens):
            target = ""
            if tok in _TERMINAL_WRITE_OPS and len(tokens) > i + 1:
                is_copy = tok in ("cp", "mv") and len(tokens) > i + 2
                target = tokens[i + 2] if is_copy else tokens[i + 1]
            elif tok.startswith(">>") and len(tok) > 2:
                target = tok[2:]
            elif tok.startswith(">") and len(tok) > 1:
                target = tok[1:]
            if target.strip("'\"").endswith(".py"):
                found.append((target.strip("'\""), body))
    return found


def _resolve_new_py_targets(tool_name: str, args: Any, cwd: str) -> list[tuple[str, str]]:
    """跨通道解析本次调用可能新建的项目内 .py 目标。

    Contract:
      Preconditions: cwd 为项目根；args 为工具入参
      Postconditions: 返回 [(项目内相对路径, 内容或 "")]；无可判定目标 → []
      Invariants: 与 write_file 同判据（只判新建，不判修改）；
                  内容不可解析时内容段为 ""，由调用方降级为提示档——不静默
    """
    if not isinstance(args, dict):
        return []
    candidates: list[tuple[str, str]] = []
    if tool_name in ("write_file", "patch"):
        candidates.append((str(args.get("path") or ""),
                           _content_of(args) or str(args.get("new_string") or "")))
    elif tool_name == "execute_code":
        candidates.extend(_execute_code_py_targets(str(args.get("code") or "")))
    elif tool_name == "terminal":
        candidates.extend(_terminal_py_targets(str(args.get("command") or "")))
    resolved: list[tuple[str, str]] = []
    for path, content in candidates:
        rel = _normalize_new_py(path, cwd)
        if rel:
            resolved.append((rel, content))
    return resolved


def _extract_functional_keywords(path: str) -> list[str]:
    """从**文件名**（非全路径）中提取有意义的功能关键词。

    只用 basename：目录名（loom/capabilities/protocols 等）是结构信息而非
    功能语义，混入关键词后策略2的 rg `(def|class)\\s+\\w*(kw)` 会在全库
    目录引用上系统性误中（2026-08-23 实测：qcc.py/risk_classify_step.py
    /graph_penetration_step.py 三连误报，提示文件 penetration 关键词零命中）。
    排除通用前缀（test/base/common等），只保留领域相关的词。
    """
    # 只取文件名（去扩展名）——目录段不参与关键词提取
    clean = os.path.splitext(os.path.basename(path))[0]
    clean = clean.replace("-", "_").replace("/", "_")

    parts = clean.split("_")
    keywords = []
    for part in parts:
        # 跳过太短的词和噪声词
        if len(part) < 3:
            continue
        if part.lower() in _NOISE_PREFIXES:
            continue
        keywords.append(part)

    # 去重，保持顺序
    seen = set()
    result = []
    for kw in keywords:
        if kw.lower() not in seen:
            result.append(kw)
            seen.add(kw.lower())
    return result


def _get_cached_ls_files(cwd: str) -> list[str]:
    """获取项目所有 Python 文件列表（带缓存）。"""
    cache_key = "ls_files"
    if cwd in _cache and cache_key in _cache[cwd]:
        return _cache[cwd][cache_key]

    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
            stdin=subprocess.DEVNULL,
        )
        files = [
            f.strip() for f in result.stdout.splitlines()
            if f.strip() and not _should_skip(f.strip())
        ]
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("git ls-files 不可用(%s),本次跳过库内查重", e)
        files = []

    if cwd not in _cache:
        _cache[cwd] = {}
    _cache[cwd][cache_key] = files
    return files


_RG_TIMEOUT = 12


def _rg_lines(pattern: str, cwd: str) -> Optional[list[str]]:
    """rg 检索，返回「文件:行号:内容」行；工具故障（缺失/超时）返回 None。

    Contract:
      Postconditions: 零命中返回 []；工具故障返回 None（与零命中严格区分，不谎报）
      Invariants: 必须显式传搜索根且 stdin 接 DEVNULL——rg 在「无路径参数 + stdin 非
                  TTY」时会转去读 stdin 并阻塞到超时（实测 0.01s → 超 6s 不发一词）
    """
    try:
        r = subprocess.run(
            ["rg", "-n", "--no-heading", "--type", "py", pattern, "."],
            capture_output=True, text=True, timeout=_RG_TIMEOUT,
            cwd=cwd, stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("rg 检索不可用(pattern=%s): %s", pattern, e)
        return None
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _rg_files(pattern: str, cwd: str) -> Optional[list[str]]:
    """rg 只取命中文件名（去重、剔免检文件）；工具故障返回 None。

    Contract:
      Postconditions: 零命中返回 []；工具故障返回 None
      Invariants: 按出现顺序去重并剥离 "./" 前缀，与 git ls-files 的路径口径一致
    """
    lines = _rg_lines(pattern, cwd)
    if lines is None:
        return None
    files: list[str] = []
    for ln in lines:
        rel = ln.split(":", 1)[0]
        rel = rel[2:] if rel.startswith("./") else rel
        if rel and rel not in files and not _should_skip(rel):
            files.append(rel)
    return files


def _stem_set(name: str) -> set[str]:
    """文件名分词后的 ≥4 字符词干集合（str 扫描，零正则）。

    纯 str 扫描，语义与旧正则实现严格等价：
    lower → 按 _/- 分词 → 每词提取纯字母段 → ≥4 字符保留。
    """
    stems: set[str] = set()
    for w in name.lower().replace("-", "_").split("_"):
        piece = ""
        for ch in w:
            if ch.isalpha():
                piece += ch
            else:
                if len(piece) >= 4:
                    stems.add(piece)
                piece = ""
        if len(piece) >= 4:
            stems.add(piece)
    return stems


def _generic_stems(path: str, all_files: list[str]) -> set[str]:
    """同目录 ≥3 个文件名共享的词干 = 项目级通用词（不具区分性）。

    0824 误报根因修复：hermes_state_cold vs hermes_bootstrap 共享 "hermes"、
    vs hermes_state* 共享 "state"，均为结构性噪声而非等价实现信号。
    """
    stem_file_count: dict[str, int] = {}
    for f in all_files:
        if f == path or os.path.dirname(f) != os.path.dirname(path):
            continue
        for s in _stem_set(os.path.splitext(os.path.basename(f))[0]):
            stem_file_count[s] = stem_file_count.get(s, 0) + 1
    return {s for s, n in stem_file_count.items() if n >= 3}


def _stem_collision_files(path: str, all_files: list[str]) -> list[tuple[str, list[str]]]:
    """策略1 探测：同目录 + 共享 ≥4 字符文件名词干 → [(既有文件, 共享词干)]。

    Contract:
      Postconditions: 零命中返回空列表；命中返回 (相对路径, 共享词干列表)
      Invariants: 只比对同目录文件；通用词干（同目录 ≥3 文件共享）先剔除
    """
    new_stems = _stem_set(os.path.splitext(os.path.basename(path))[0]) - _generic_stems(path, all_files)
    hits: list[tuple[str, list[str]]] = []
    for f in all_files:
        if f == path:
            continue
        if os.path.dirname(f) and os.path.dirname(f) != os.path.dirname(path):
            continue
        common = new_stems & _stem_set(os.path.splitext(os.path.basename(f))[0])
        if common:
            hits.append((f, sorted(common)))
    return hits


def _symbols_in_file(rel: str, cwd: str) -> set[str]:
    """读既有文件声明的顶层 def/class 名（读失败告警并返回空集，不静默）。

    Contract:
      Postconditions: 返回该文件顶层符号名集合；不可读时返回空集并 logger.warning
      Invariants: 复用 _extract_declared_symbols 的顶格/入口名规则，口径与策略2一致
    """
    try:
        with open(os.path.join(cwd, rel), encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        logger.warning("查重：读取既有文件失败，已跳过符号比对: %s", rel, exc_info=True)
        return set()
    return _extract_declared_symbols(text)


def _stem_evidence(
    path: str, symbols: set[str], cwd: str, all_files: list[str],
) -> tuple[list[str], list[str]]:
    """策略1 判定：词干命中 + 同名符号佐证 → 阻断；仅词干命中 → 提示档。

    Contract:
      Preconditions: symbols 为待建文件声明符号；all_files 为项目 .py 相对路径列表
      Postconditions: 返回 (block_evidence, advisory_evidence)，二者不同时非空
      Invariants: 名字档（文件名词干）单独不足以阻断——必须同时具备同名符号内容证据
    """
    hits = _stem_collision_files(path, all_files)
    if not hits:
        return [], []
    for rel, stems in hits:
        shared = _symbols_in_file(rel, cwd) & symbols if symbols else set()
        if shared:
            return [f"文件名共享词干({','.join(stems)}) 且同名符号({','.join(sorted(shared))}): 已有 {rel}"], []
    files = ", ".join(rel for rel, _ in hits[:3])
    stems_txt = ",".join(sorted({s for _, ss in hits for s in ss}))
    return [], [f"文件名共享词干({stems_txt}): {files} —— 无同名符号，按提示档放行"]


def _symbol_of_line(raw: str) -> Optional[str]:
    """单行取顶格 def/class 名（非声明行/框架入口名返回 None）。

    Contract:
      Postconditions: 命中顶格声明且非框架入口名时返回符号名，否则 None
      Invariants: 缩进行（类内方法）与注释行一律不取
    """
    if not raw or raw[0].isspace() or raw.startswith("#"):
        return None
    line = raw.strip()
    for kw in ("async def ", "def ", "class "):
        if line.startswith(kw):
            name = line[len(kw):].split("(")[0].split(":")[0].strip()
            if name and name.isidentifier() and name not in _ENTRYPOINT_NAMES:
                return name
            return None
    return None


def _extract_declared_symbols(content: str) -> set[str]:
    """从新文件内容提取将要定义的**顶层** def/class 名（str 扫描，零正则）。

    Contract:
      Postconditions: 返回模块级符号名集合；无内容或无定义返回空集
      Invariants: 只认顶格（无缩进）声明——类内方法（validate_config/_execute 等
                  框架约定名）人人同名，纳入判定必然误报；
                  框架约定入口名（_ENTRYPOINT_NAMES）一并剔除
    """
    symbols: set[str] = set()
    for raw in (content or "").splitlines():
        name = _symbol_of_line(raw)
        if name:
            symbols.add(name)
    return symbols


def _existing_symbol_files(symbols: set[str], cwd: str) -> tuple[dict[str, list[str]], bool]:
    """策略2：一次 rg 检索全部待建符号，按行解析命中符号名（顶格、词边界精确）。

    Contract:
      Preconditions: symbols 为待建文件将定义的符号名集合
      Postconditions: 返回 ({符号名: [命中文件]}, degraded)；degraded=True 表示检索工具故障
      Invariants: 只认顶格 def/class（缩进方法行由 _symbol_of_line 剔除）；
                  工具故障返回空命中并标记 degraded——不把「查不到」谎报成「零命中」；
                  单次 rg 调用（旧版每符号一次，20 个符号最坏 20 次子进程）
    """
    if not symbols:
        return {}, False
    alt = "|".join(sorted(symbols)[:20])
    lines = _rg_lines(rf"^(async def|def|class)\s+({alt})\b", cwd)
    if lines is None:
        return {}, True
    hits: dict[str, list[str]] = {}
    for ln in lines:
        rel, _, rest = ln.partition(":")
        _, _, body = rest.partition(":")
        rel = rel[2:] if rel.startswith("./") else rel
        sym = _symbol_of_line(body)
        if sym in symbols and not _should_skip(rel):
            bucket = hits.setdefault(sym, [])
            if rel not in bucket:
                bucket.append(rel)
    return {s: f[:3] for s, f in hits.items()}, False


def _distinctive_keywords(keywords: list[str], cwd: str) -> list[str]:
    """筛出「有命中但非通用」的主题词（命中 ≥3 文件 = 项目级通用词，剔除）。

    Contract:
      Postconditions: 返回 ≤2 个 ≥6 字符且命中文件数在 (0,3) 的主题词
      Invariants: 0829 误报根因——通用词（progress/ledger/engine）不参与判定
    """
    out: list[str] = []
    for kw in [k for k in keywords if len(k) >= 6][:2]:
        files = _rg_files(rf"(def|class)\s+\w*{kw}\w*", cwd)
        n = 0 if files is None else len(files)
        if 0 < n < 3:
            out.append(kw)
    return out


def _topic_files(keywords: list[str], cwd: str) -> dict[str, list[str]]:
    """策略2（提示档）：文件名主题词命中既有 def/class 子串。

    仅作提示——主题词命中不等于等价实现（成熟命名空间下必然命中）。
    """
    distinctive = _distinctive_keywords(keywords, cwd)
    if not distinctive:
        return {}
    pattern = "|".join(distinctive)
    files = _rg_files(rf"(def|class)\s+\w*({pattern})\w*", cwd)
    if not files:
        return {}
    return {pattern: files[:3]}


def _inline_ok_reason(content: str) -> Optional[str]:
    """读取按文件豁免标记 `# dupcheck-ok: <理由>`（理由 ≥8 字符）。

    Contract:
      Postconditions: 标记存在且理由足够长时返回理由，否则返回 None
      Invariants: 理由过短视为无效豁免（防无理由放行）
    """
    for raw in (content or "").splitlines()[:40]:
        line = raw.strip()
        if line.startswith(_INLINE_OK_PREFIX):
            reason = line[len(_INLINE_OK_PREFIX):].strip()
            if len(reason) >= _INLINE_OK_MIN_REASON:
                return reason
            return None
    return None


def _topic_evidence(path: str, symbols: set[str], cwd: str) -> list[str]:
    """主题词命中（提示档证据）：附新文件声明符号供人工复核。

    Contract:
      Postconditions: 零命中返回空列表；命中返回 ≤2 条证据文本
      Invariants: 只作提示语义，不含阻断语义
    """
    topic_hits = _topic_files(_extract_functional_keywords(path), cwd)
    if not topic_hits:
        return []
    declared = ", ".join(sorted(symbols)) if symbols else "(未能从内容提取)"
    return [f"主题词 '{kw}' 命中既有符号名（非同名）: {', '.join(files)}"
            for kw, files in topic_hits.items()] + [f"新文件声明符号: {declared}"]


def _judge(path: str, content: str, cwd: str) -> tuple[str, list[str], list[str]]:
    """三层判定：豁免 → 内容级硬证据阻断 → 提示档放行。

    Contract:
      Preconditions: path 为项目内待新建 .py 路径
      Postconditions: 返回 (verdict, evidence, next_steps)，verdict ∈ {pass,advisory,block}
      Invariants: 阻断只在内容级证据（同名符号 / 词干+同名符号佐证）成立时给出；
                  名字档（仅词干、仅主题词）一律 advisory 放行并留痕；
                  检索工具故障必产出一条降级提示——不返回 pass 谎报「零重复」
    """
    reason = _inline_ok_reason(content)
    if reason:
        return "advisory", [f"命中按文件豁免标记: {reason}"], []

    all_files = _get_cached_ls_files(cwd)
    symbols = _extract_declared_symbols(content)

    stem_block, stem_advisory = _stem_evidence(path, symbols, cwd, all_files)
    if stem_block:
        return "block", stem_block, _next_steps(stem_block)

    symbol_hits, degraded = _existing_symbol_files(symbols, cwd)
    if symbol_hits:
        evidence = [f"新文件将定义 {sym} → 同名符号已存在于 {files[0]}"
                    for sym, files in symbol_hits.items()]
        return "block", evidence, _next_steps(evidence)

    advisory = list(stem_advisory) + _topic_evidence(path, symbols, cwd)
    if degraded:
        advisory.append("检索降级：rg 不可用或超时 → 本轮符号级比对未生效，"
                        "结论不完整，请人工复核（勿当「零重复」）")
    if not advisory:
        return "pass", [], []
    return "advisory", advisory, []


def _next_steps(evidence: list[str]) -> list[str]:
    """按证据类型生成分档下一步（阻断档）。

    Contract:
      Postconditions: 返回 4 条有序建议，首条为复用已有实现
      Invariants: 全局禁用开关只作末位提示（不推荐），不当作首选出路
    """
    if any("同名符号" in e for e in evidence):
        first = "复用已有实现：上列文件已有同名符号，用 patch 扩展它，或给新符号换名"
    else:
        first = "复用已有实现：用 patch 扩展上列命中文件，不新建"
    return [
        first,
        "确认是不同能力：改文件名/符号名，避开共享词干与同名",
        "确认非重复但必须保留现状：文件首行加 `# dupcheck-ok: <≥8字理由>`",
        "全局开关 HERMES_DUPCHECK_DISABLE=1（不推荐：会关闭全部查重）",
    ]


def _render(path: str, verdict: str, evidence: list[str], next_steps: list[str]) -> str:
    """渲染用户可见消息（阻断/提示两档）。

    Contract:
      Postconditions: 返回含「判定结论 + 证据 + 下一步」的消息文本
      Invariants: 阻断档必带下一步；提示档必带「放行」措辞
    """
    if verdict == "block":
        lines = ["", "=" * 60,
                 " 查重拦截 — 疑似重复实现 (guards.duplicate_check v4)",
                 f"   新建文件: {path}",
                 "   判定: 存在符号级/词干级重复证据",
                 "   证据:"]
        lines.extend(f"     - {e}" for e in evidence[:5])
        lines.extend(["   ", "   下一步（按推荐顺序，任选其一）:"])
        lines.extend(f"     {i + 1}. {s}" for i, s in enumerate(next_steps))
        lines.append("=" * 60)
        return "\n".join(lines)

    lines = [f"[dupcheck] 放行 {path}（提示档，不阻断）"]
    lines.extend(f"  {e}" for e in evidence[:5])
    lines.append("  如确为重复实现请人工复核；本条仅提示，不阻断写入。")
    return "\n".join(lines)


# ═══════════════════════════════════════════
# Hook
# ═══════════════════════════════════════════


def _content_of(args: Any) -> str:
    """取新文件内容（无则空串）。

    Contract:
      Postconditions: args 为 dict 且 content 为字符串时返回它，否则返回 ""
      Invariants: 类型不符不抛异常（hook 路径须稳）
    """
    if not isinstance(args, dict):
        return ""
    val = args.get("content", "")
    return val if isinstance(val, str) else ""


def _emit(
    path: str, verdict: str, evidence: list[str], next_steps: list[str],
) -> Optional[dict[str, str]]:
    """按判定档位输出：advisory 放行留痕；block 返回阻断（warn-only 时降级放行）。

    Contract:
      Postconditions: 返回 None（放行）或 {"action":"block","message":...}
      Invariants: advisory 档永不阻断；block 档仅 warn-only 模式可降级放行
    """
    msg = _render(path, verdict, evidence, next_steps)
    if verdict == "advisory":
        logger.info("查重提示档放行:\n%s", msg)
        return None
    if _warn_only():
        logger.warning("查重命中但 warn-only 模式放行:\n%s", msg)
        return None
    return {"action": "block", "message": msg}


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    cwd: str = "",
    **_: Any,
) -> Optional[dict[str, str]]:
    """pre_tool_call hook — 新建 Python 文件前查重（v4：四通道统一射程）。"""
    if _disabled():
        return None
    if not cwd:
        cwd = os.getcwd()

    for rel, content in _resolve_new_py_targets(tool_name, args, cwd):
        if not _extract_functional_keywords(rel):
            continue
        if not content:
            _emit(rel, "advisory",
                  [f"{tool_name} 将新建 {rel}，但写入内容非字面量 → 符号级比对未生效，"
                   "请人工确认无等价实现"], [])
            continue
        start = time.time()
        verdict, evidence, next_steps = _judge(rel, content, cwd)
        elapsed_ms = int((time.time() - start) * 1000)
        if elapsed_ms > 500:
            logger.warning("查重耗时 %sms (verdict=%s)", elapsed_ms, verdict)
        if verdict == "pass":
            continue
        blocked = _emit(rel, verdict, evidence, next_steps)
        if blocked is not None:
            return blocked
    return None


def register(ctx: Any) -> None:
    """Plugin entry point — 注册 pre_tool_call hook。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    # 静默注册——不打印，避免每轮输出干扰
