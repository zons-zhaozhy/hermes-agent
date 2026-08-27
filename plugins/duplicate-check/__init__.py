"""
duplicate-check — write_file 前置查重拦截插件 v2.0。

在每次 write_file 调用前，如果是新建 Python 文件，自动搜索项目中
是否已有等价实现，有则阻断并提示已有文件路径。

v2.0 改进:
- 准确性：排除 test_*.py / *_test.py / conftest.py 等测试文件免检
- 准确性：功能关键词提取（从路径+文件名+上下文推断，不只做前缀匹配）
- 效率：git ls-files 结果按 session 缓存
- 子代理：子代理自动继承（同一 HERMES_HOME 下的插件全局生效）

环境变量:
  HERMES_DUPCHECK_DISABLE=1  完全禁用
  HERMES_DUPCHECK_WARN_ONLY=1  仅警告不阻断
"""
import fnmatch
import logging
import subprocess
import os
import time
from typing import Any, Dict, List, Optional, Set

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

# git ls-files 缓存
_cache: Dict[str, Dict[str, Any]] = {}


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


def _is_new_python_file(tool_name: str, args: Any, cwd: str) -> Optional[str]:
    """如果此次 write_file 会新建 Python 文件，返回路径。否则 None。"""
    if tool_name != "write_file":
        return None
    path = ""
    if isinstance(args, dict):
        path = args.get("path", "")
    if not path or not path.endswith(".py"):
        return None
    full = os.path.join(cwd, path) if not os.path.isabs(path) else path
    if os.path.exists(full):
        return None
    if _should_skip(path):
        return None
    # 项目外的临时脚本（/tmp、~/.hermes、系统目录）不参与项目查重——
    # git ls-files 搜不到它们，任何"匹配"都只能是误报。
    try:
        real_cwd = os.path.realpath(cwd)
        real_full = os.path.realpath(full)
        if not real_full.startswith(real_cwd + os.sep) and real_full != real_cwd:
            return None
    except OSError as e:
        logger.warning("realpath 解析失败(%s): %s", path, e)
        return None
    return path


def _extract_functional_keywords(path: str) -> List[str]:
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


def _get_cached_ls_files(cwd: str) -> List[str]:
    """获取项目所有 Python 文件列表（带缓存）。"""
    cache_key = "ls_files"
    if cwd in _cache and cache_key in _cache[cwd]:
        return _cache[cwd][cache_key]

    try:
        result = subprocess.run(
            ["git", "ls-files", "*.py"],
            capture_output=True, text=True, timeout=10, cwd=cwd,
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


def _search_duplicates(path: str, cwd: str) -> List[str]:
    """搜索项目中是否有类似功能的已有文件。返回警告列表。"""
    start = time.time()
    warnings = []

    keywords = _extract_functional_keywords(path)
    if not keywords:
        return warnings  # 无有效关键词，跳过

    all_files = _get_cached_ls_files(cwd)

    # 策略1：文件名词干匹配（真正的相似性，不是同位字符数）
    # 只有当两个文件名共享同一个 ≥4 字符的词干（camelCase/snake_case 分词后）
    # 才算相似——"verify_compression_fix" vs "verification_evidence" 分词后
    # {verify, compression, fix} vs {verification, evidence} 无公共词干，不拦。
    basename = os.path.basename(path)
    name_no_ext = os.path.splitext(basename)[0]

    def _stem_set(name: str) -> Set[str]:
        # 纯 str 扫描，语义与旧正则实现严格等价：
        # lower → 按 _/- 分词 → 每词提取纯字母段 → ≥4 字符保留
        # （旧实现先 lower 再 findall，camelCase 分支从不触发=死代码，此处固化真实行为）
        stems: Set[str] = set()
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

    new_stems = _stem_set(name_no_ext)

    # 区分性过滤（0824 误报根因修复）：一个词干若已出现在同目录 ≥3 个
    # 其他文件名中，它是项目级通用词（hermes/state/common…），不具区分性，
    # 命中只能是误报。hermes_state_cold vs hermes_bootstrap 共享 "hermes"、
    # vs hermes_state* 共享 "state"，均为结构性噪声而非等价实现信号。
    stem_file_count: Dict[str, int] = {}
    for f in all_files:
        if f == path or os.path.dirname(f) != os.path.dirname(path):
            continue
        for s in _stem_set(os.path.splitext(os.path.basename(f))[0]):
            stem_file_count[s] = stem_file_count.get(s, 0) + 1
    generic_stems = {s for s, n in stem_file_count.items() if n >= 3}
    new_stems = new_stems - generic_stems

    for kw in keywords[:2]:
        for f in all_files:
            if f == path:
                continue
            f_basename = os.path.basename(f)
            f_name = os.path.splitext(f_basename)[0]
            # 同目录 + 共享词干才算等价实现候选
            if os.path.dirname(f) and os.path.dirname(f) != os.path.dirname(path):
                continue
            common = new_stems & _stem_set(f_name)
            if common:
                warnings.append(
                    "文件名共享词干({0}): 已有 {1}".format(
                        ",".join(sorted(common))[:40], f))
                break  # 一个关键词匹配一次就够了

    # 策略2：函数/类名关键词匹配（用 rg 而非 git grep，更快）
    if not warnings:  # 如果策略1已命中，策略2可跳过
        # 只用 ≥6 字符的关键词——短词（fix/check/verify/utils）在全库
        # 几乎必然命中，导致大量误报。
        strong_kws = [k for k in keywords if len(k) >= 6][:2]
        if strong_kws:
            pattern = "|".join(strong_kws)
            try:
                result = subprocess.run(
                    ["rg", "-l", "--no-heading", "--type", "py",
                     r"(def|class)\s+\w*({0})\w*".format(pattern)],
                    capture_output=True, text=True, timeout=8, cwd=cwd,
                )
                existing = [
                    f.strip() for f in result.stdout.splitlines()
                    if f.strip() and f.strip() != path
                    and not _should_skip(f.strip())
                ]
                for f in existing[:3]:
                    warnings.append(
                        "函数/类名含关键词 '{0}': 已有 {1}".format(pattern, f))
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.warning("rg 搜索不可用(%s),跳过函数名查重", e)

    elapsed_ms = int((time.time() - start) * 1000)
    if elapsed_ms > 500:
        # 性能日志：超过500ms告警
        logger.warning("查重耗时 %sms (files=%s)", elapsed_ms, len(all_files))

    return warnings


# ═══════════════════════════════════════════
# Hook
# ═══════════════════════════════════════════


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    cwd: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    """pre_tool_call hook — 新建 Python 文件前强制查重。"""
    if _disabled():
        return None
    if not cwd:
        cwd = os.getcwd()

    path = _is_new_python_file(tool_name, args, cwd)
    if path is None:
        return None

    keywords = _extract_functional_keywords(path)
    if not keywords:
        # 无有效关键词（如全噪声词）→ 放行
        return None

    warnings = _search_duplicates(path, cwd)
    if not warnings:
        return None

    lines = [
        "",
        "=" * 60,
        " 编码前强制查重拦截 (duplicate-check v2)",
        "   新建文件: {0}".format(path),
        "   提取关键词: {0}".format(", ".join(keywords)),
        "   ",
        "   检测到项目中可能已有等价实现：",
    ]
    for w in warnings[:5]:
        lines.append("    {0}".format(w))
    lines.extend([
        "   ",
        "   铁律 (SOUL.md  8.5)：创建新文件前必须先搜已有实现。",
        "   ",
        "   正确做法：",
        "     1. 评估已有实现是否可复用/扩展",
        "     2. 如果可以复用，用 patch 修改已有文件而非新建",
        "     3. 如果确定必须新建: HERMES_DUPCHECK_DISABLE=1 跳过（不建议）",
        "     4. commit message 标注「已查重，无现有等价实现」",
        "=" * 60,
    ])

    msg = "\n".join(lines)

    if _warn_only():
        # warn-only 模式：写日志而非阻断（走 logging，不再用 print）
        logger.warning("查重命中但 warn-only 模式放行:\n%s", msg)
        return None

    return {"action": "block", "message": msg}


def register(ctx: Any) -> None:
    """Plugin entry point — 注册 pre_tool_call hook。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    # 静默注册——不打印，避免每轮输出干扰
