# dupcheck-ok: 全库 search_files load_section|prompt_sections 命中仅本模块与既有段渲染链路，
# 无现有等价实现（既有段文本全部硬编码在各插件模块的模块级常量里）
"""段文本单一事实源：单文件段库 + 版本指纹。

背景: 段文本原以模块级常量散在各插件内, 进程启动即冻结——改段对在跑的会话完全
无效。实测: 本机 CLI 进程 11:15 启动、diff_debt 段 16:15 提交, 同刻渲染日志里本会话
始终 4 段, 而同一分钟的 cron 新进程 5 段。故段文本外置为段库文件, 由本模块统一读盘。

约定: 段库为 <repo_root>/prompt_sections/sections.json, 形如 {"<id>": "<段文本>"}；
键即 register_system_prompt_section 的 id（1:1, 禁另立映射表）。段文本用 JSON 承载的
原因: 段数量多且单个很短, 单文件便于一次修改多处判据并对账指纹。

用途:
  - load_section(id): 各插件注册段的渲染器读盘取最新——新会话首注即最新;
  - fingerprint(): 段库内容指纹, 供重注入通道判「段是否变过」;
  - section_ids(): 段 id 清单（对账与测试用）。

Contract:
  Preconditions: 段库为 UTF-8 JSON 对象, 值为字符串; id 仅含小写字母/数字/下划线。
  Postconditions: load_section 返回段文本(strip 后); 键缺失或段库不可读返回 ""
    并响亮告警一次; fingerprint 返回 8 位十六进制(库缺失时返回 "");
    section_ids 返回按名排序的 id 列表。
  Invariants: 三函数均不抛异常出界（读盘/解析失败=告警+空结果，不炸主循环）。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LIBRARY = Path(__file__).resolve().parent.parent / "prompt_sections" / "sections.json"
_WARNED: set[str] = set()


def library_path() -> Path:
    """段库文件路径（仓库根下 prompt_sections/sections.json）。

    Returns:
        Path: 段库路径（不保证存在）。
    """
    return _LIBRARY


def _load_library() -> dict[str, str]:
    """读段库并解析为 {id: 段文本}。

    Postconditions: 库缺失/不可读/非对象时返回空字典并告警一次;
      值为非字符串的键被跳过并告警（禁静默丢弃）。
    """
    try:
        raw = _LIBRARY.read_text(encoding="utf-8")
    except FileNotFoundError:
        if "library" not in _WARNED:
            _WARNED.add("library")
            logger.warning(
                "prompt_sections: 段库缺失 %s——文件型段本轮不注入; 若应有段库, "
                "请确认它随插件分发到位。",
                _LIBRARY,
            )
        return {}
    except OSError as exc:
        logger.warning("prompt_sections: 段库读取失败 %s: %s", _LIBRARY, exc, exc_info=True)
        return {}
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("prompt_sections: 段库解析失败 %s: %s", _LIBRARY, exc, exc_info=True)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("prompt_sections: 段库顶层非对象(实为 %s)", type(parsed).__name__)
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(value, str):
            result[key] = value
        else:
            logger.warning("prompt_sections: 段 %s 的值非字符串, 已跳过", key)
    return result


def section_ids() -> list[str]:
    """列出全部段 id（按名排序）。

    Postconditions: 段库缺失或为空时返回空列表。
    """
    return sorted(_load_library())


def load_section(section_id: str) -> str:
    """读段文本（渲染器用；每次调用读盘，段库更新即生效）。

    Preconditions: section_id 为段标识。
    Postconditions: 键存在且非空 → 返回 strip 后的文本; 键缺失 → 告警一次并返回 ""
      （空段被 core 跳过, 不清空既有段）。
    """
    text = _load_library().get(section_id, "")
    if not text and section_id not in _WARNED:
        _WARNED.add(section_id)
        logger.warning(
            "prompt_sections: 段 %s 不在段库中——该段本轮不注入; 若应有该段, "
            "请确认键名与注册 id 一致。",
            section_id,
        )
    return text.strip()


def fingerprint() -> str:
    """段库内容指纹（判「段是否变过」，供重注入通道使用）。

    Postconditions: 返回 8 位十六进制; 库文件缺失或不可读时返回 ""。
    """
    try:
        raw = _LIBRARY.read_bytes()
    except OSError as exc:
        logger.warning("prompt_sections: 指纹计算失败 %s: %s", _LIBRARY, exc, exc_info=True)
        return ""
    return hashlib.sha1(raw).hexdigest()[:8]


def render_all(max_chars: int) -> str:
    """拼接全部段文本（重注入通道用）。

    Preconditions: max_chars 为正整数（注入体量上限）。
    Postconditions: 返回各段以空行分隔的拼接文本; 超 max_chars 即截断并告警
      （截断=尾部段本轮不进上下文, 属两态状态, 故响亮告警）。
    """
    blocks: list[str] = []
    size = 0
    for section_id in section_ids():
        block = load_section(section_id)
        if not block:
            continue
        if size + len(block) + 2 > max_chars:
            logger.warning(
                "prompt_sections: 段拼接超上限 %d 字符, 自 %s 起的段本轮未注入"
                "（压缩段文本或提高上限）",
                max_chars,
                section_id,
            )
            break
        blocks.append(block)
        size += len(block) + 2
    return "\n\n".join(blocks)
