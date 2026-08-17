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
import subprocess
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════

# 以下模式的文件免检（测试文件/构建产物等）
_SKIP_PATTERNS = [
    r"^test_.*\.py$",        # test_foo.py
    r".*_test\.py$",         # foo_test.py
    r"^conftest\.py$",       # pytest fixtures
    r"^__init__\.py$",       # package marker
    r"^setup\.py$",          # package setup
    r"^migrations/",         # Django alembic 迁移
    r"^\.",                  # 隐藏文件
]

# 以下文件名前缀不参与匹配（太通用）
_NOISE_PREFIXES = {"test", "tmp", "temp", "util", "helper", "common", "base"}

# git ls-files 缓存
_cache: Dict[str, Dict[str, Any]] = {}


def _disabled() -> bool:
    return os.environ.get("HERMES_DUPCHECK_DISABLE", "").strip() == "1"


def _warn_only() -> bool:
    return os.environ.get("HERMES_DUPCHECK_WARN_ONLY", "").strip() == "1"


def _should_skip(path: str) -> bool:
    """检查是否属于免检模式。"""
    basename = os.path.basename(path)
    for pat in _SKIP_PATTERNS:
        if re.match(pat, basename) or re.match(pat, path):
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
    except OSError:
        return None
    return path


def _extract_functional_keywords(path: str) -> List[str]:
    """从文件路径中提取有意义的功能关键词。

    排除通用前缀（test/base/common等），只保留领域相关的词。
    """
    # 标准化路径：去掉扩展名，替换分隔符
    clean = os.path.splitext(path)[0]
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
    except (subprocess.TimeoutExpired, FileNotFoundError):
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
        # snake/kebab 分词 + camelCase 拆分
        words = re.split(r"[_\-]", name.lower())
        stems: Set[str] = set()
        for w in words:
            if len(w) < 4:
                continue
            # 拆 camelCase（对混合命名）
            for piece in re.findall(r"[a-z]+|[A-Z][a-z]*", w):
                if len(piece) >= 4:
                    stems.add(piece.lower())
        return stems

    new_stems = _stem_set(name_no_ext)

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
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    elapsed_ms = int((time.time() - start) * 1000)
    if elapsed_ms > 500:
        # 性能日志：超过500ms告警
        try:
            print("[dupcheck] 查重耗时 {0}ms (files={1})".format(
                elapsed_ms, len(all_files)), flush=True)
        except Exception:
            pass

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
        print(msg)
        return None

    return {"action": "block", "message": msg}


def register(ctx: Any) -> None:
    """Plugin entry point — 注册 pre_tool_call hook。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    # 静默注册——不打印，避免每轮输出干扰
