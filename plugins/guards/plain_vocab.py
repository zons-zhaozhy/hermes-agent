"""guards.plain_vocab — project deprecated-term vocabulary gate for user-facing writes.

Motivation: plain-language remediation (OntoX lab_standalone 2026-10-25,
commits d15447f68/ab021c4e9: 176 occurrences across frontend/docs/backend
user-visible strings) lives only in history — nothing stops the retired
terms from creeping back in the next write. This guard enforces a
project-owned vocabulary at write time.

Blocked shape (ALL must hold):
  1. tool is ``write_file`` or ``patch`` (first version deliberately does
     NOT cover terminal/execute_code — path resolution is unreliable there)
  2. target file extension is user-facing: .vue/.ts/.tsx/.js/.jsx/.py/.md
  3. a vocabulary file ``.hermes/plain_terms.yaml`` exists on the target
     path's ancestor chain (no file ⇒ fail-open: the gate is opt-in per
     project/repository)
  4. the content being written (write_file ``content`` / patch
     ``new_string``) contains a ``deprecated`` term, and the hit line does
     not carry an inline waiver ``vocab-ok:``

Block message names the term, its replacement and the vocabulary file, so
the fix is mechanical.

Contract:
  Preconditions: plugin system fires pre_tool_call with tool_name/args;
    args carry path plus content/new_string.
  Postconditions: returns None (allow) or {"action":"block","message":str};
    never raises; vocabulary load failures fail open with a log line.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VOCAB_FILENAME = ".hermes/plain_terms.yaml"
_WAIVER_MARK = "vocab-ok"
_USER_FACING_EXT = frozenset({".vue", ".ts", ".tsx", ".js", ".jsx", ".py", ".md"})
_TOOLS = frozenset({"write_file", "patch"})

# 词表缓存: (mtime_ns, size) -> parsed terms。进程内生效,词表变更下次写入可见
# (同 mtime+size 的原地改写理论上可漏,词表是低频人工编辑文件,可接受)。
_cache: dict[str, tuple[tuple[int, int], tuple[tuple[str, str], ...]]] = {}


def _find_vocab(path: str) -> Optional[str]:
    """沿目标路径祖先链找词表文件;找不到返回 None。

    Contract:
      Postconditions: 返回存在的词表绝对路径或 None;不抛异常。
    """
    probe = os.path.abspath(path)
    while True:
        parent = os.path.dirname(probe)
        if parent == probe:  # 到达文件系统根
            return None
        candidate = os.path.join(parent, _VOCAB_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        probe = parent


def _parse_terms(data: Any) -> tuple[tuple[str, str], ...]:
    """从词表 YAML 对象提取合法 (deprecated, replacement) 元组。

    Contract:
      Preconditions: data 为 safe_load 结果(可能为 None/dict)。
      Postconditions: 返回可空元组;非法条目静默跳过,不抛异常。
    """
    terms: list[tuple[str, str]] = []
    if not isinstance(data, dict):
        return ()
    for item in data.get("terms", []) or []:
        if not isinstance(item, dict):
            continue
        dep = str(item.get("deprecated") or "").strip()
        rep = str(item.get("replacement") or "").strip()
        if dep and rep and dep != rep:
            terms.append((dep, rep))
    return tuple(terms)


def _load_terms(vocab_path: str) -> tuple[tuple[str, str], ...]:
    """按 mtime+size 缓存解析词表;任何失败返回空表(fail-open)。

    词表 YAML 形态(terms 列表,每项 deprecated/replacement 必填,note 可选):
        terms:
          - deprecated: 裁决
            replacement: 审批

    Contract:
      Preconditions: vocab_path 存在(调用方已 isfile 判定)。
      Postconditions: 返回可空元组;解析/导入异常吞掉记日志,永不抛出。
    """
    try:
        st = os.stat(vocab_path)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        return ()
    cached = _cache.get(vocab_path)
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        from hermes_yaml import safe_load

        with open(vocab_path, encoding="utf-8") as fh:
            parsed = _parse_terms(safe_load(fh.read()))
    except Exception:
        logger.warning(
            "guards.plain_vocab: vocabulary %s unreadable; failing open", vocab_path, exc_info=True
        )
        return ()
    _cache[vocab_path] = (stamp, parsed)
    return parsed


def _hit_lines(text: str, terms: tuple[tuple[str, str], ...]) -> list[tuple[str, str, int]]:
    """扫描写入内容,返回 (deprecated, replacement, line_no) 命中表(不含豁免行)。

    Contract:
      Preconditions: text 为写入内容字符串(可为空)。
      Postconditions: 返回列表(可能为空);纯扫描无副作用。
    """
    hits: list[tuple[str, str, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _WAIVER_MARK in line:
            continue
        for dep, rep in terms:
            if dep in line:
                hits.append((dep, rep, line_no))
    return hits


def _target_text(tool_name: str, args: dict[str, Any]) -> Optional[tuple[str, str]]:
    """提取 (path, 写入内容);不适用形态返回 None。

    Contract:
      Preconditions: tool_name 已在 _TOOLS 内;args 为 dict。
      Postconditions: 返回非空 path+text 二元组或 None;纯提取无副作用。
    """
    path = str(args.get("path") or "")
    if not path or os.path.splitext(path)[1].lower() not in _USER_FACING_EXT:
        return None
    text = str(args.get("content") or args.get("new_string") or "")
    if not text:
        return None
    return path, text


_BLOCK_TEMPLATE = (
    "[PlainVocab] 写入内容命中项目废弃术语词表,已拦截——白话化术语回潮。\n"
    "词表: {vocab}\n"
    "命中: {hits}\n"
    "处置: 按词表替换词改写后重试;确属技术域/物理名/时点快照等保留场景,"
    "在该行加行内豁免标记 vocab-ok: <理由>。\n"
    "词表本身是项目配置:增删条目直接改该文件,护栏实时生效(fail-open,无词表不执法)。"
)


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook: block deprecated-term writes when a project vocabulary exists.

    Contract:
      Preconditions: kwargs contains tool_name (str) and args (dict).
      Postconditions: returns None (allow) or {"action":"block","message":str}.
    """
    if kwargs.get("tool_name", "") not in _TOOLS:
        return None
    target = _target_text(str(kwargs.get("tool_name")), kwargs.get("args") or {})
    if target is None:
        return None
    path, text = target

    vocab = _find_vocab(path)
    if vocab is None:
        return None
    terms = _load_terms(vocab)
    if not terms:
        return None

    hits = _hit_lines(text, terms)
    if not hits:
        return None

    rendered = "; ".join(f"「{dep}」→「{rep}」(第{no}行)" for dep, rep, no in hits[:8])
    logger.warning(
        "guards.plain_vocab: blocked deprecated terms (%s hits, vocab=%s)", len(hits), vocab
    )
    return {"action": "block", "message": _BLOCK_TEMPLATE.format(vocab=vocab, hits=rendered)}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.plain_vocab registered")
