"""
Agent 自我检查管理器 (SelfCheckManager)

内核组件——在每次工具调用前检查是否匹配已知失败模式。
解析所有已加载 skill 的 AVOID 章节 + 冗余检测规则，
匹配时在工具结果中注入警告（不阻断调用，让模型自主修正）。

设计原则：上下文的每一 token 都必须是高信号。
- 警告一行一条，无 emoji/header 废话
- AVOID 匹配要求全关键词命中（防止不相关 skill 误报）
- 每次调用最多 1 条 AVOID 警告（取最相关的）
- 不触碰系统提示词（prompt caching 安全）
"""

from __future__ import annotations

import json
import logging
import re  # noqa: E402 — pattern parser fundamentally needs regex for AVOID section extraction
from pathlib import Path

logger = logging.getLogger("agent.self_check")


# ── 领域推断 ────────────────────────────────────────────────────────
# skill 目录名 → 领域标签映射。路径中的目录名段匹配即推断。
_SKILL_DIR_TO_DOMAIN: dict[str, str] = {
    "aml": "aml", "dingtalk-bot": "dingtalk", "dingtalk": "dingtalk",
    "devops": "devops", "docker": "devops", "deployment": "devops",
    "github": "github", "git": "github",
    "frontend": "frontend", "vue": "frontend", "react": "frontend",
    "java-development": "java", "java": "java",
    "sql": "database", "database": "database", "db": "database",
    "security": "security", "license": "security",
    "mlops": "mlops", "ml": "mlops",
    "productivity": "productivity",
    "research": "research",
}

# 文件扩展名 → 领域
_EXT_TO_DOMAIN: dict[str, str] = {
    ".java": "java", ".jsx": "frontend", ".tsx": "frontend", ".vue": "frontend",
    ".sql": "database", ".tf": "devops", ".dockerfile": "devops",
    ".proto": "java",
}

# 命令关键词 → 领域
_CMD_KEYWORD_TO_DOMAIN: list[tuple[str, str]] = [
    ("docker", "devops"), ("kubectl", "devops"), ("helm", "devops"),
    ("git ", "github"), ("gh ", "github"),
    ("npm ", "frontend"), ("yarn ", "frontend"), ("pnpm ", "frontend"),
    ("mvn ", "java"), ("gradle ", "java"),
    ("psql ", "database"), ("mysql ", "database"),
    ("curl ", "network"), ("wget ", "network"),
]


def _infer_domain_from_path(path: str) -> str:
    """从文件路径推断操作领域。返回领域标签或空字符串。"""
    if not path:
        return ""
    path_lower = path.lower()
    # 1. 扩展名匹配
    for ext, domain in _EXT_TO_DOMAIN.items():
        if path_lower.endswith(ext):
            return domain
    # 2. 路径段匹配
    parts = path_lower.replace("\\", "/").split("/")
    for part in parts:
        if part in _SKILL_DIR_TO_DOMAIN:
            return _SKILL_DIR_TO_DOMAIN[part]
    # 3. 路径段包含关键词
    for part in parts:
        for key, domain in _SKILL_DIR_TO_DOMAIN.items():
            if key in part:
                return domain
    return ""


def _infer_domain_from_operation(tool_name: str, args: dict) -> str:
    """从当前操作推断领域。优先级：文件路径 > 命令内容 > 工具名。"""
    # 1. 文件路径
    file_path = args.get("path", "") or args.get("url", "")
    domain = _infer_domain_from_path(file_path)
    if domain:
        return domain
    # 2. 命令内容（terminal/execute_code）
    cmd = args.get("command", "") or args.get("code", "")
    if cmd:
        cmd_lower = cmd.lower()
        for keyword, dom in _CMD_KEYWORD_TO_DOMAIN:
            if keyword in cmd_lower:
                return dom
    # 3. 无推断
    return ""


def _infer_domain_from_skill_dir(skill_dir: str, skill_name: str) -> str:
    """从 skill 目录路径推断领域。"""
    combined = (skill_dir + "/" + skill_name).lower().replace("\\", "/")
    parts = combined.split("/")
    for part in parts:
        if part in _SKILL_DIR_TO_DOMAIN:
            return _SKILL_DIR_TO_DOMAIN[part]
    for part in parts:
        for key, domain in _SKILL_DIR_TO_DOMAIN.items():
            if key in part:
                return domain
    return "general"


# ── 冗余检测规则函数 ────────────────────────────────────────────────


def _r01_read_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R01: 同一文件被 read_file >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    key = "read_file:" + file_path
    if history.get(key, 0) >= 3:
        return "文件 %s 已读 %d 次，内容不会变。" % (file_path, history[key])
    return None


def _r02_patch_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R02: 同一文件被 patch >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "patch":
        return None
    key = "patch:" + file_path
    if history.get(key, 0) >= 3:
        return "文件 %s 已 patch %d 次，考虑 write_file 整体重写。" % (file_path, history[key])
    return None


def _r03_read_after_edit(tool_name: str, args: dict, history: dict) -> str | None:
    """R03: read_file 刚被编辑过的文件。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    recently_edited = history.get("_recently_edited", set())
    if file_path in recently_edited:
        return "文件 %s 刚被编辑过，不需要重读。" % file_path
    return None


def _r04_terminal_fragments(tool_name: str, args: dict, history: dict) -> str | None:
    """R04: 5+ terminal 调用。"""
    if tool_name != "terminal":
        return None
    cnt = history.get("_terminal_count", 0) + 1
    if cnt >= 5:
        return "已调用 terminal %d 次，考虑用 execute_code 合并。" % cnt
    return None


def _r05_write_without_read(tool_name: str, args: dict, history: dict) -> str | None:
    """R05: 编码前未读文件（首次写入/修改陌生文件）。"""
    file_path = args.get("path", "")
    if tool_name not in ("write_file", "patch"):
        return None
    all_read = history.get("_all_touched_before_update", set())
    if file_path and file_path not in all_read and not file_path.startswith("/dev/"):
        return "准备写入 %s 但本 session 尚未读取过它，先 read_file 确认当前内容。" % file_path
    return None


# ── R06: 推责/归属语言检测 ─────────────────────────────────────────
_BLAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"不是我(这次|的)?(改的|改出来的|写的|弄的|引入的|代码)"), "不推责——视野内破损即责任"),
    (re.compile(r"这是(上游|历史|之前|其他人|别的).{0,20}的(问题|代码|改动|bug)"), "不推责——追究来源不如直接修复"),
    (re.compile(r"不属于本[次轮个](修改|改动|修复|任务)"), "不推责——看到就修"),
    (re.compile(r"这个是.{0,200}留[下存]的|(从以前|旧版本|版本)就有的"), "不推责——历史债务当场清"),
]


# ── R07: 第一性原理——跳结论/凭推断 ──────────────────
_R07_SPECULATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"应该(是|可以|没问题)|大概(是|没问题)|一般来说|正常情况下|大概率"), "推断不是验证，加 [实测] 或 [文档] 标注"),
    (re.compile(r"(直接|简单|快速).{0,10}(修复|改|处理|解决)(一下|即可|就好)"), "'直接修复'前先拆解根因"),
]


# ── R08: 推责给用户 ────────────────────────
_R08_USER_BLAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"需要你做的|需用户.{0,5}(操作|执行|运行|配置)"), "禁止推责——自己排查修复"),
    (re.compile(r"^(要不要|是否需要|要不要我).{0,20}[？?]$", re.MULTILINE), "禁止推责——分析清楚直接做"),
    (re.compile(r"请确认一下|请确认是否|请帮我确认"), "禁止推责——自己验证"),
    (re.compile(r"我无法直接执行|不支持.{0,10}(这个|该)操作"), "禁止推责——换一条路执行"),
    (re.compile(r"(你要|你想)我修(哪个|哪|什么|哪些)"), "禁止推责——主动修全部"),
    (re.compile(r"(还需|仍需|要)手动.{0,10}(操作|执行|处理|改|配|运行)"), "禁止推责——找程序化替代方案"),
]


# ── R09: 验证驱动——声称结论缺少证据标注 ──────────────────────────
def _has_evidence(content: str) -> bool:
    """启发式判断段落是否包含可验证的证据。纯 Python，零依赖。"""
    if re.search(r"\[实测\]|\[文档\]|\[推断\]", content):
        return True
    if re.search(r"\d+/\d+\s*(通过|pass|ok|green|passed)", content, re.I):
        return True
    if re.search(r"(exit.?code|返回码|status.?code)\s*[=:]\s*\d+", content, re.I):
        return True
    if re.search(r"(passed|failed|skipped|error)\s*[=:]\s*\d+", content, re.I):
        return True
    if re.search(r"\d+\s*(通过|失败|跳过|pass|fail|skip)", content, re.I):
        return True
    if re.search(r"[/\w]+\.(py|java|go|ts|tsx|js|rs|yaml|yml|sql):\d+", content):
        return True
    if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", content):
        return True
    if re.search(r"Traceback\s*\(most recent call last\)", content, re.I):
        return True
    if re.search(r"(stdout|stderr|output)\s*:", content, re.I):
        return True
    if re.search(r"\d+\s*(个|条|次|行|s|ms|MB|KB|并发|qps|tps)", content):
        return True
    if re.search(r"\d+\s*(files|tests|errors|matches|results|records)", content, re.I):
        return True
    if re.search(r"[/\w]+\s*(存在|不存在|已创建|已删除|is\s+at|located\s+at)", content):
        return True
    return False


_R09_EVIDENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(根因|原因)(是|在于|出在).{0,60}[。，\n]"), "声称根因但没有验证证据"),
    (re.compile(r"(已经|已)(修复|解决|完成|搞定).{0,40}[。，\n]"), "声称完成但没有验证证据"),
    (re.compile(r"(确认|确定|保证|肯定|绝对).{0,40}[。，\n]"), "断言确认但没有 [实测] 证据"),
]


# ── R11: 判断阶段——给方案但没列替代方案对比 ──────────────────────
_R11_JUDGMENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(方案|做法|方法)\s*(很简单|就是简单|没什么|非常容易)"), "'方案很简单'=跳过复杂性分析"),
    (re.compile(r"(直接|只要|仅仅|简单)(改|修复|用|换|替换|写).{0,20}(就行|就好|即可|可以)"), "直接跳到实施没有分析路径"),
]


# ── R10: 链式工具路由——3+ 步应合并为 execute_code ──────────────────
_CHAIN_WINDOW: int = 4

_CHAIN_TRIGGER_SEQUENCES: list[tuple[tuple[str, ...], str]] = [
    (
        ("search_files", "read_file", "patch", "terminal"),
        "search→read→patch→验证 可走 execute_code 一次性流水线",
    ),
    (
        ("search_files", "read_file", "write_file", "terminal"),
        "search→read→write→验证 可走 execute_code 一次性流水线",
    ),
    (
        ("search_files", "read_file", "patch"),
        "search→read→patch 三次往返，合并为 execute_code",
    ),
    (
        ("search_files", "read_file", "write_file"),
        "search→read→write 三次往返，合并为 execute_code",
    ),
    (
        ("read_file", "patch", "terminal"),
        "读→改→验证可合并为 execute_code",
    ),
    (
        ("read_file", "write_file", "terminal"),
        "读→写→验证可合并为 execute_code",
    ),
]


def _r10_chain_route(tool_name: str, _args: dict, history: dict) -> str | None:
    """R10: 链式工具路由。"""
    recent: list[str] = history.get("_recent_tools", [])  # type: ignore[assignment]
    if len(recent) < 3:
        return None

    for sequence, hint in _CHAIN_TRIGGER_SEQUENCES:
        seq_len = len(sequence)
        if len(recent) < seq_len:
            continue
        window = recent[-seq_len:]
        if tuple(window) == sequence[:seq_len]:
            return hint

    terminal_streak = 0
    for t in reversed(recent):
        if t == "terminal":
            terminal_streak += 1
        else:
            break
    if terminal_streak >= 3:
        return "连续 %d 次 terminal，合并为 execute_code 更快" % terminal_streak

    return None


# ── R13: 任务漂移——写入与已操作文件零目录重叠 ────────────────────
def _r13_task_drift(tool_name: str, args: dict, history: dict) -> str | None:
    """R13: 任务漂移检测。"""
    if tool_name not in ("write_file", "patch"):
        return None

    file_path = args.get("path", "")
    if not file_path or file_path.startswith("/dev/"):
        return None

    all_touched = history.get("_all_touched_before_update", set())
    if len(all_touched) < 3:
        return None

    if file_path in all_touched:
        return None

    from pathlib import PurePosixPath
    new_dir = str(PurePosixPath(file_path).parent)

    overlap = any(
        str(PurePosixPath(f).parent) == new_dir
        for f in all_touched
        if f and f != file_path
    )
    if overlap:
        return None  # 同目录 → 相关

    return "写入 %s 与已操作文件零目录重叠，检查是否 drive-by edit" % file_path


# ── R14: 操作结果忽略——工具输出含错误但回复无引用 ─────────────────
_R14_ERROR_SIGNALS: list[re.Pattern] = [
    re.compile(r"Traceback\s*\(most recent call last\)", re.I),
    re.compile(r"\bERROR\b.*[:：]", re.I),
    re.compile(r"\bFAILED\b", re.I),
    re.compile(r"\bError:\s", re.I),
    re.compile(r"exit_code\s*[=:]\s*[1-9]"),
]

_R14_ACK_PATTERNS: re.Pattern = re.compile(
    r"错误|失败|异常|报错|exit.?code|错误码|error|failed|traceback|stack.?trace|出错|没通过|不通过|崩溃",
    re.I,
)


# 规则注册表
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "search_files",
    "read_file",
    "web_search",
    "web_extract",
    "browser_snapshot",
    "browser_console",
    "browser_back",
    "browser_scroll",
    "browser_get_images",
    "session_search",
    "skill_view",
    "skills_list",
    "memory",
    "vision_analyze",
})

_RULES = [
    ("R01", _r01_read_repeat),
    ("R02", _r02_patch_repeat),
    ("R03", _r03_read_after_edit),
    ("R04", _r04_terminal_fragments),
    ("R05", _r05_write_without_read),
    ("R10", _r10_chain_route),
    ("R13", _r13_task_drift),
]
# 回复级规则在 check_response() 中处理
_RESPONSE_RULE_IDS: frozenset[str] = frozenset({"R06", "R07", "R08", "R11", "R14"})


class SelfCheckManager:
    """内核自检组件——在 tool_executor 中作为 pre-tool 检查运行。

    指标：
    - 零缓存影响：警告注入到 tool result，不碰 system prompt
    - 零阻断：只警告不阻止，模型自主决定是否采纳
    - 零网络调用：所有检查是纯内存操作
    - 零噪音：警告一行一条，无 emoji/header 废话
    """

    def __init__(self):
        self._avoid_entries: list[tuple[str, str, str, str]] = []  # [(domain, keywords, text, source)]
        self._call_history: dict[str, int | set] = {}
        self._all_read: set[str] = set()
        self._recently_edited: set[str] = set()
        self._recent_tools: list[str] = []
        self._loaded = False
        self._last_tool_output: str = ""
        self._all_touched_before_update: set[str] = set()
        self._active_domain: str = ""  # 当前操作的领域标签
        self._loaded_skill_dirs: set[str] = set()  # 去重：已加载 AVOID 的 skill 目录

    # ── 加载 ────────────────────────────────────────────────────────

    def load(self, agent=None) -> None:
        """初始化——不再全量扫描所有 skill。

        AVOID 条目改为按需加载：只有 skill 被 skill_view 加载到当前会话时，
        它的 AVOID 条目才会被激活。这避免了不相关 skill 的噪音注入。

        唯一预加载的例外：coding-conventions skill（通用编码铁律）。
        """
        # 预加载 coding-conventions（通用规则，适用于所有场景）
        _BASE_SKILLS = ["coding-conventions"]
        for skill_name in _BASE_SKILLS:
            try:
                from agent.skill_commands import scan_skill_commands
                commands = scan_skill_commands()
                for _slug, info in commands.items():
                    if info.get("name", "") == skill_name:
                        skill_dir = Path(info["skill_dir"])
                        self._load_skill_avoid(skill_dir, skill_name)
                        break
            except Exception as e:
                logger.warning("SelfCheck: base skill preload failed for '%s': %s", skill_name, e)

        self._loaded = True
        logger.info("SelfCheckManager: initialized (%d base AVOID entries, on-demand loading active)", len(self._avoid_entries))

    def load_skill_on_demand(self, skill_dir: str, skill_name: str) -> int:
        """按需加载单个 skill 的 AVOID 条目。

        在 skill_view 加载 skill 时调用此方法，使该 skill 的 AVOID 条目
        参与后续的 SelfCheck 匹配。这样只有实际被加载到会话的 skill 的
        AVOID 规则才会激活，不相关 skill 的规则不会注入噪音。
        """
        if skill_dir in self._loaded_skill_dirs:
            return 0
        self._loaded_skill_dirs.add(skill_dir)
        count = self._load_skill_avoid(Path(skill_dir), skill_name)
        if count:
            logger.info("SelfCheckManager: on-demand loaded %d AVOID entries from skill '%s'", count, skill_name)
        return count

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
        domain = _infer_domain_from_skill_dir(str(skill_dir), name)
        return self._parse_avoid_section(content, name, domain)

    def _parse_avoid_section(self, content: str, source: str, domain: str = "general") -> int:
        """从 markdown 中解析 ## AVOID 章节。返回条目数。

        支持 bullet 列表（- / *）和编号列表（1. / 2.），以及带括号后缀的标题
        如"## AVOID（铁律）"。段落间空行不影响解析。
        """
        # 策略：捕获 ## AVOID 到下一个 ## header 之间的全部内容
        section_match = re.search(r'## AVOID[^\n]*\n(.*?)(?=^## |\Z)', content, re.DOTALL | re.MULTILINE)
        if not section_match:
            return 0
        section_text = section_match.group(1)
        count = 0
        for line in section_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 匹配 bullet (-, *) 或编号 (1., 2., ...)
            if re.match(r'^[-*]\s+', line) or re.match(r'^\d+\.\s+', line):
                item = re.sub(r'^\d+\.\s+', "", line)
                item = re.sub(r'^[-*]\s+', "", item)
                if not item:
                    continue
                clean = re.sub(r'^(AVOID|avoid):\s*', "", item)
                clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
                clean = re.sub(r'[`"\']', "", clean)
                keywords = clean.lower()
                self._avoid_entries.append((domain, keywords, item, source))
                count += 1
        return count
        avoid_text = avoid_match.group(1)
        count = 0
        for line in avoid_text.strip().split("\n"):
            line = line.strip()
            if not line or not re.match(r'^[-*]\s', line):
                continue
            item = line[2:].strip()
            if not item:
                continue
            clean = re.sub(r'^(AVOID|avoid):\s*', "", item)
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r'[`"\']', "", clean)
            keywords = clean.lower()
            self._avoid_entries.append((domain, keywords, item, source))
            count += 1
        return count

    # ── 检查 ────────────────────────────────────────────────────────

    def check(self, tool_name: str, args: dict) -> str | None:
        """检查当前工具调用是否匹配已知失败模式。

        Returns:
            None if clean, or a compact warning string (one line per rule).
        """
        if not self._loaded:
            self.load()

        # R13 需要在 update_history 前做快照
        self._all_touched_before_update = (self._all_read | self._recently_edited).copy()
        self._call_history["_all_touched_before_update"] = self._all_touched_before_update  # type: ignore[index]

        self._update_history(tool_name, args)
        warnings: list[str] = []

        # 1. 冗余检测规则
        for rule_id, check_fn in _RULES:
            try:
                warning = check_fn(tool_name, args, self._call_history)
                if warning:
                    warnings.append("[%s] %s" % (rule_id, warning))
            except Exception as e:
                logger.warning("SelfCheck: rule %s check raised: %s", rule_id, e)

        # 2. AVOID 领域过滤 + 关键词匹配
        # 核心相关性：先按领域过滤（不相关领域的 AVOID 条目直接跳过），再做关键词匹配
        if tool_name not in _READ_ONLY_TOOLS:
            op_domain = _infer_domain_from_operation(tool_name, args)
            # 提取实际写入内容（不匹配 JSON 元数据）
            content_str = ""
            if tool_name in ("write_file", "patch"):
                content_str = args.get("content", "") or args.get("new_string", "")
            elif tool_name == "terminal":
                content_str = args.get("command", "")
            elif tool_name == "execute_code":
                content_str = args.get("code", "")
            else:
                try:
                    content_str = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    content_str = ""
            tool_desc = (content_str or "").lower()

            best_avoid = None
            best_score = 0
            for entry_domain, keywords, item, source in self._avoid_entries:
                # 领域过滤核心逻辑：
                # - op_domain 已识别 → 只匹配同 domain + general 的条目
                # - op_domain 为空（无法推断）→ 只匹配 general 条目
                #   这防止不相关 skill 的 AVOID 规则在通用场景注入
                if op_domain:
                    if entry_domain != op_domain and entry_domain != "general":
                        continue
                else:
                    if entry_domain != "general":
                        continue
                kw_parts = keywords.split()
                if not kw_parts:
                    continue
                # 全关键词命中
                matched = [kw for kw in kw_parts if kw in tool_desc]
                if len(matched) == len(kw_parts):
                    score = len(kw_parts)
                    if score > best_score:
                        best_score = score
                        best_avoid = "[AVOID/%s] %s" % (source, item)
            if best_avoid:
                warnings.append(best_avoid)

        if warnings:
            return "\n".join(warnings)
        return None

    def check_response(self, assistant_content: str | None) -> str | None:
        """检查 assistant 回复文本是否包含推责/推断/缺证据等行为模式。

        Returns:
            None if clean, or a compact warning string (one line per rule).
        """
        if not assistant_content:
            return None

        try:
            return self._do_check_response(assistant_content)
        except Exception:
            logger.exception("SelfCheck: check_response raised — returning None")
            return None

    def _do_check_response(self, assistant_content: str) -> str | None:
        """check_response 的核心逻辑。"""
        warnings: list[str] = []
        for pattern, hint in _BLAME_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("[R06] %s" % hint)
        for pattern, hint in _R07_SPECULATION_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("[R07] %s" % hint)
        for pattern, hint in _R08_USER_BLAME_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("[R08] %s" % hint)
        # R09 removed: evidence-tagging enforcement produces more noise than value.
        # Pattern matching can't reliably distinguish "has evidence" from "doesn't".
        # The system prompt already instructs evidence-tagging; SelfCheck can't enforce it.
        for pattern, hint in _R11_JUDGMENT_PATTERNS:
            if pattern.search(assistant_content):
                has_comparison = bool(re.search(r"方案\s*[A-Za-z①②③]|①|②|③|\bvs\b|比较|权衡|对比|另一个|替代方案|哪个更好", assistant_content, re.I))
                if not has_comparison:
                    warnings.append("[R11] %s" % hint)
        # R14: 操作结果忽略
        if self._last_tool_output:
            has_error = any(p.search(self._last_tool_output) for p in _R14_ERROR_SIGNALS)
            if has_error:
                acknowledges = bool(_R14_ACK_PATTERNS.search(assistant_content))
                if not acknowledges:
                    warnings.append("[R14] 上次工具输出含错误但回复未引用")

        if warnings:
            return "\n".join(warnings)
        return None

    def record_tool_result(self, tool_name: str, tool_output: str) -> None:
        """R14 支撑：记录最近一次工具输出文本。"""
        if not tool_output:
            return
        self._last_tool_output = str(tool_output)[:4000]

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

        self._call_history["_recently_edited"] = self._recently_edited  # type: ignore[index]
        self._call_history["_all_read"] = self._all_read  # type: ignore[index]

        self._recent_tools.append(tool_name)
        if len(self._recent_tools) > _CHAIN_WINDOW:
            self._recent_tools = self._recent_tools[-_CHAIN_WINDOW:]
        self._call_history["_recent_tools"] = self._recent_tools  # type: ignore[index]

    # ── 查询 ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回自检统计。"""
        return {
            "avoid_entries": len(self._avoid_entries),
            "call_history_keys": len(self._call_history),
            "files_read": len(self._all_read),
            "files_edited": len(self._recently_edited),
        }

    # ── 自审计：验证每条规则确实在工作 ──────────────────────────────
    _CANARY_TOOL_CASES: list[dict] = [
        {"rule": "R01", "type": "positive", "desc": "read_file×3 同文件",
         "ops": [("read_file", {"path": "/canary/r01.py"})] * 3,
         "final": ("read_file", {"path": "/canary/r01.py"})},
        {"rule": "R01", "type": "negative", "desc": "read_file×1",
         "ops": [("read_file", {"path": "/canary/r01_neg.py"})],
         "final": ("read_file", {"path": "/canary/r01_neg.py"})},
        {"rule": "R02", "type": "positive", "desc": "patch×3 同文件",
         "ops": [("patch", {"path": "/canary/r02.py", "old_string": "a", "new_string": "b"})] * 3,
         "final": ("patch", {"path": "/canary/r02.py", "old_string": "a", "new_string": "b"})},
        {"rule": "R03", "type": "positive", "desc": "编辑后重读",
         "ops": [("write_file", {"path": "/canary/r03.py", "content": "x"}), ("read_file", {"path": "/canary/r03.py"})],
         "final": ("read_file", {"path": "/canary/r03.py"})},
        {"rule": "R04", "type": "positive", "desc": "terminal×5",
         "ops": [("terminal", {"command": "ls"})] * 4,
         "final": ("terminal", {"command": "ls"})},
        {"rule": "R05", "type": "positive", "desc": "write 未读文件",
         "ops": [],
         "final": ("write_file", {"path": "/canary/r05_new.py", "content": "x"})},
        {"rule": "R05", "type": "negative", "desc": "write 已读文件",
         "ops": [("read_file", {"path": "/canary/r05_read.py"})],
         "final": ("write_file", {"path": "/canary/r05_read.py", "content": "x"})},
        {"rule": "R10", "type": "positive", "desc": "search→read→patch",
         "ops": [("search_files", {"pattern": "x"}), ("read_file", {"path": "/canary/r10.py"})],
         "final": ("patch", {"path": "/canary/r10.py", "old_string": "a", "new_string": "b"})},
        {"rule": "R13", "type": "positive", "desc": "3文件/src 写/docs",
         "ops": [("read_file", {"path": "/src/a.py"}), ("read_file", {"path": "/src/b.py"}), ("patch", {"path": "/src/c.py", "old_string": "x", "new_string": "y"})],
         "final": ("write_file", {"path": "/docs/drift.md", "content": "x"})},
        {"rule": "R13", "type": "negative", "desc": "同目录不漂移",
         "ops": [("read_file", {"path": "/src/a.py"}), ("read_file", {"path": "/src/b.py"}), ("patch", {"path": "/src/c.py", "old_string": "x", "new_string": "y"})],
         "final": ("write_file", {"path": "/src/d.py", "content": "x"})},
    ]

    _CANARY_RESPONSE_CASES: list[dict] = [
        {"rule": "R06", "type": "positive", "text": "不是我改的bug。"},
        {"rule": "R06", "type": "negative", "text": "根因在匹配逻辑。"},
        {"rule": "R07", "type": "positive", "text": "应该没问题。"},
        {"rule": "R07", "type": "negative", "text": "已修复。[实测]"},
        {"rule": "R08", "type": "positive", "text": "需要你做的：改配置。"},
        {"rule": "R08", "type": "negative", "text": "我来修复。"},
        # R09 removed — evidence-tagging enforcement removed
        # {"rule": "R09", "type": "positive", "text": "根因是连接池太小。"},
        # {"rule": "R09", "type": "negative", "text": "根因是连接池 [实测]：500并发通过。"},
        {"rule": "R11", "type": "positive", "text": "方案很简单，加个白名单就行。"},
        {"rule": "R11", "type": "negative", "text": "方案A加白名单；方案B加分类器。选A因为更简洁。"},
    ]

    def audit(self) -> dict:
        """自审计：验证每条规则的金丝雀测试是否通过。"""
        results: list[dict] = []
        rule_status: dict[str, str] = {}

        for case in self._CANARY_TOOL_CASES:
            rule_id = case["rule"]
            ctype = case["type"]
            mgr = SelfCheckManager()
            mgr._loaded = True
            for tn, args in case["ops"]:
                mgr.check(tn, args)
            final_tn, final_args = case["final"]
            result = mgr.check(final_tn, final_args)
            hit = bool(result) and ("[%s]" % rule_id in result)
            results.append({"rule": rule_id, "type": ctype, "hit": hit, "desc": case["desc"]})

        for case in self._CANARY_RESPONSE_CASES:
            rule_id = case["rule"]
            ctype = case["type"]
            mgr = SelfCheckManager()
            mgr._loaded = True
            result = mgr.check_response(case["text"])
            hit = bool(result) and ("[%s]" % rule_id in result)
            results.append({"rule": rule_id, "type": ctype, "hit": hit, "desc": case["text"][:40]})

        # R14 特殊审计
        mgr14p = SelfCheckManager()
        mgr14p._loaded = True
        mgr14p.record_tool_result("terminal", "Traceback (most recent call last)\nError: x")
        r14p = mgr14p.check_response("已修复。")
        results.append({"rule": "R14", "type": "positive", "hit": bool(r14p) and "[R14]" in r14p, "desc": "ERROR无引用"})
        mgr14n = SelfCheckManager()
        mgr14n._loaded = True
        mgr14n.record_tool_result("terminal", "exit_code=1\nFAILED")
        r14n = mgr14n.check_response("上次命令失败了，原因分析中。")
        results.append({"rule": "R14", "type": "negative", "hit": bool(r14n) and "[R14]" in r14n, "desc": "ERROR有引用"})

        all_rules = set(r["rule"] for r in results)
        for rule_id in all_rules:
            positives = [r for r in results if r["rule"] == rule_id and r["type"] == "positive"]
            negatives = [r for r in results if r["rule"] == rule_id and r["type"] == "negative"]
            pos_ok = all(r["hit"] for r in positives) if positives else False
            neg_ok = all(not r["hit"] for r in negatives) if negatives else True
            if pos_ok and neg_ok:
                rule_status[rule_id] = "alive"
            elif not pos_ok:
                rule_status[rule_id] = "broken"
            else:
                rule_status[rule_id] = "broken"

        alive = sum(1 for v in rule_status.values() if v == "alive")
        broken = sum(1 for v in rule_status.values() if v == "broken")

        return {
            "alive": alive,
            "broken": broken,
            "total": len(rule_status),
            "status": rule_status,
            "details": results,
        }


# 单例 —— 由 AIAgent 持有
_global_manager: SelfCheckManager | None = None


def get_self_check() -> SelfCheckManager | None:
    return _global_manager


def set_self_check(manager: SelfCheckManager | None) -> None:
    global _global_manager
    _global_manager = manager
