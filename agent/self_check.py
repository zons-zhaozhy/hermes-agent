"""
Agent 自我检查管理器 (SelfCheckManager)

内核组件——在每次工具调用前检查是否匹配已知失败模式。
解析所有已加载 skill 的 AVOID 章节 + 冗余检测规则，
匹配时在工具结果中注入警告（不阻断调用，让模型自主修正）。

约束：不触碰系统提示词（prompt caching 安全），只在 tool result 中注入。
"""

from __future__ import annotations

import json
import logging
import re  # noqa: E402 — pattern parser fundamentally needs regex for AVOID section extraction
from pathlib import Path

logger = logging.getLogger("agent.self_check")

# ── 冗余检测规则函数 ────────────────────────────────────────────────


def _r01_read_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R01: 同一文件被 read_file >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    key = "read_file:" + file_path
    if history.get(key, 0) >= 3:
        return (
            "文件 %s 已被 read_file 了 %d 次。"
            "文件内容不会变，不需要反复读取。每次 read_file 都在消耗 token 预算。"
            % (file_path, history[key])
        )
    return None


def _r02_patch_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R02: 同一文件被 patch >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "patch":
        return None
    key = "patch:" + file_path
    if history.get(key, 0) >= 3:
        return (
            "文件 %s 已被 patch 了 %d 次。"
            "考虑用 write_file 一次性重写整个文件——"
            "反复 patch 意味着第一轮方案没选对，继续 patch 只会越改越乱。"
            % (file_path, history[key])
        )
    return None


def _r03_read_after_edit(tool_name: str, args: dict, history: dict) -> str | None:
    """R03: read_file 刚被编辑过的文件。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    recently_edited = history.get("_recently_edited", set())
    if file_path in recently_edited:
        return (
            "文件 %s 刚被编辑过，不需要重新读取。"
            "上一轮编辑的结果就是当前状态。"
            % file_path
        )
    return None


def _r04_terminal_fragments(tool_name: str, args: dict, history: dict) -> str | None:
    """R04: 5+ terminal 调用。"""
    if tool_name != "terminal":
        return None
    cnt = history.get("_terminal_count", 0) + 1
    if cnt >= 5:
        return (
            "已调用 terminal %d 次。"
            "考虑用 execute_code 编写 Python 脚本替代多个碎片化的 shell 命令——"
            "execute_code 里的 terminal() 在同一进程中运行，结果可以继续加工。"
            % cnt
        )
    return None


def _r05_write_without_read(tool_name: str, args: dict, history: dict) -> str | None:
    """R05: 编码前未读文件（首次写入/修改陌生文件）。"""
    file_path = args.get("path", "")
    if tool_name not in ("write_file", "patch"):
        return None
    all_read = history.get("_all_read", set())
    if file_path and file_path not in all_read and not file_path.startswith("/dev/"):
        return (
            "准备写入 %s 但本 session 尚未读取过它。"
            "先 read_file 确认当前内容，防止覆盖未提交的改动或基于过时的记忆编写代码。"
            % file_path
        )
    return None


# 规则注册表
_RULES = [
    ("R01", "medium", _r01_read_repeat),
    ("R02", "high", _r02_patch_repeat),
    ("R03", "medium", _r03_read_after_edit),
    ("R04", "low", _r04_terminal_fragments),
    ("R05", "medium", _r05_write_without_read),
]

_SEVERITY_EMOJI = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}
_SEVERITY_EMOJI["high"] = "\U0001f534"  # 🔴
_SEVERITY_EMOJI["medium"] = "\U0001f7e1"  # 🟡
_SEVERITY_EMOJI["low"] = "\U0001f7e2"  # 🟢


class SelfCheckManager:
    """内核自检组件——在 tool_executor 中作为 pre-tool 检查运行。

    指标：
    - 零缓存影响：警告注入到 tool result，不碰 system prompt
    - 零阻断：只警告不阻止，模型自主决定是否采纳
    - 零网络调用：所有检查是纯内存操作
    """

    def __init__(self):
        self._avoid_entries: list[tuple[str, str, str]] = []  # [(keyword, text, source)]
        self._call_history: dict[str, int | set] = {}
        self._all_read: set[str] = set()
        self._recently_edited: set[str] = set()
        self._loaded = False

    # ── 加载 ────────────────────────────────────────────────────────

    def load(self, agent=None) -> None:
        """从所有已加载 skill 中解析 AVOID 章节。agent 参数可选。"""
        count = 0
        skill_dirs_seen: set[str] = set()

        try:
            from agent.skill_commands import scan_skill_commands
            commands = scan_skill_commands()
            for _slug, info in commands.items():
                skill_dir_str = info.get("skill_dir", "")
                if not skill_dir_str or skill_dir_str in skill_dirs_seen:
                    continue
                skill_dirs_seen.add(skill_dir_str)
                count += self._load_skill_avoid(Path(skill_dir_str), info.get("name", _slug))
        except Exception as e:
            logger.warning("SelfCheck: repo skills scan skipped: %s", e)

        # 用户安装的 skill
        try:
            from hermes_constants import get_hermes_home
            user_skills = get_hermes_home() / "skills"
            if user_skills.exists():
                for skill_md in user_skills.rglob("SKILL.md"):
                    skill_dir_str = str(skill_md.parent)
                    if skill_dir_str in skill_dirs_seen:
                        continue
                    skill_dirs_seen.add(skill_dir_str)
                    count += self._load_skill_avoid(skill_md.parent, skill_md.parent.name)
        except Exception as e:
            logger.warning("SelfCheck: user skills scan skipped: %s", e)

        self._loaded = True
        if count:
            logger.info("SelfCheckManager: loaded %d AVOID entries from skills", count)

    def _load_skill_avoid(self, skill_dir: Path, name: str) -> int:
        """加载单个 skill 的 AVOID 章节。返回条目数。"""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return 0
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("SelfCheck: cannot read %s: %s", skill_md, e)
            return 0
        return self._parse_avoid_section(content, name)

    def _parse_avoid_section(self, content: str, source: str) -> int:
        """从 markdown 中解析 ## AVOID 章节。返回条目数。"""
        avoid_match = re.search(r'## AVOID\s*\n((?:(?:[-*]\s+).*\n?)+)', content)
        if not avoid_match:
            return 0
        avoid_text = avoid_match.group(1)
        count = 0
        for line in avoid_text.strip().split("\n"):
            line = line.strip()
            if not line or not re.match(r'^[-*]\s', line):
                continue
            item = line[2:].strip()
            if not item:
                continue
            # 清理格式
            clean = re.sub(r'^(AVOID|avoid):\s*', "", item)
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r'[`"\']', "", clean)
            keywords = clean.lower()
            self._avoid_entries.append((keywords, item, source))
            count += 1
        return count

    # ── 检查 ────────────────────────────────────────────────────────

    def check(self, tool_name: str, args: dict) -> str | None:
        """检查当前工具调用是否匹配已知失败模式。

        Returns:
            None if clean, or a warning string to prepend to the tool result.
        """
        if not self._loaded:
            self.load()

        self._update_history(tool_name, args)
        warnings: list[str] = []

        # 1. 冗余检测规则（确定性匹配）
        for rule_id, severity, check_fn in _RULES:
            try:
                warning = check_fn(tool_name, args, self._call_history)
                if warning:
                    emoji = _SEVERITY_EMOJI.get(severity, "\u26aa")
                    warnings.append("%s [%s] %s" % (emoji, rule_id, warning))
            except Exception as e:
                logger.warning("SelfCheck: rule %s check raised: %s", rule_id, e)

        # 2. AVOID 关键词匹配（模糊匹配，需重合度）
        tool_desc = (tool_name + " " + json.dumps(args, ensure_ascii=False)).lower()
        for keywords, item, source in self._avoid_entries:
            kw_parts = keywords.split()
            if len(kw_parts) >= 3:
                matches = sum(1 for kw in kw_parts if kw in tool_desc)
                if matches >= 2:
                    warnings.append("\U0001f4cb [%s] \U0001f4a1 %s" % (source, item))

        if warnings:
            return "[SelfCheck]\n" + "\n".join(warnings)
        return None

    def _update_history(self, tool_name: str, args: dict) -> None:
        """更新调用计数器。"""
        file_path = args.get("path", "")
        if file_path:
            if tool_name in ("read_file", "patch", "write_file"):
                key = tool_name + ":" + file_path
                self._call_history[key] = self._call_history.get(key, 0) + 1  # type: ignore[operator]
            if tool_name == "read_file":
                self._all_read.add(file_path)
            if tool_name in ("write_file", "patch"):
                self._recently_edited.add(file_path)
                self._all_read.add(file_path)

        if tool_name == "terminal":
            cnt = self._call_history.get("_terminal_count", 0)
            self._call_history["_terminal_count"] = int(cnt) + 1  # type: ignore[arg-type]

        # 传递引用给检查函数
        self._call_history["_recently_edited"] = self._recently_edited  # type: ignore[index]
        self._call_history["_all_read"] = self._all_read  # type: ignore[index]

    # ── 查询 ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回自检统计。"""
        return {
            "avoid_entries": len(self._avoid_entries),
            "call_history_keys": len(self._call_history),
            "files_read": len(self._all_read),
            "files_edited": len(self._recently_edited),
        }


# 单例 —— 由 AIAgent 持有
_global_manager: SelfCheckManager | None = None


def get_self_check() -> SelfCheckManager | None:
    return _global_manager


def set_self_check(manager: SelfCheckManager | None) -> None:
    global _global_manager
    _global_manager = manager
