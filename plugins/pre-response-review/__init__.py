"""
回复前自审外部校验插件 v2.0
==============================
transform_llm_output 钩子：
每次 LLM 生成回复后，扫描自审标记（改前验证/不交付半成品/正则），
用 session_id 查 state.db 验证对应工具调用是否真实存在。

假✓检测逻辑：
  - "改前验证✓" → 查当前会话是否有 read_file/search_files 调用
  - "不交付半成品✓" → 查当前会话是否有 terminal/execute_code 验证调用
  - "正则✓" 或 "正则：无✓" → 扫描回复中的 Python 代码是否用了 re 模块

无 state.db 访问权限时静默降级为纯文本扫描（仅检测正则违规）。
"""

import json
import os
import sqlite3


def _get_session_tool_calls(session_id: str, limit: int = 50) -> list[str]:
    """查询当前会话最近的工具调用名称列表。"""
    try:
        hermes_home = os.path.expanduser("~/.hermes")
        db_path = os.path.join(hermes_home, "state.db")
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                """
                SELECT tool_calls, tool_name
                FROM messages
                WHERE session_id = ?
                  AND (tool_calls IS NOT NULL OR tool_name IS NOT NULL)
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            conn.close()

        tool_names: list[str] = []
        for tool_calls_json, tool_name in rows:
            if tool_name:
                tool_names.append(tool_name)
            if tool_calls_json:
                try:
                    calls = json.loads(tool_calls_json)
                    for tc in calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        name = fn.get("name", "")
                        if name:
                            tool_names.append(name)
                except (json.JSONDecodeError, TypeError):
                    pass
        return tool_names
    except Exception:
        return []


def _check_false_claims(response_text: str, session_id: str | None) -> list[str]:
    """检测假✓——自审标记与实际工具调用不匹配。返回警告列表。"""
    warnings: list[str] = []

    # 纯文本检查：正则违规（不需要 DB）
    # 如果回复说"正则✓"或"正则：无✓"或"正则：禁止✓"
    # 但代码中包含 import re / re.match / re.search / re.sub 等
    _REGEX_MARKERS = ["正则", "regex"]
    has_regex_claim = any(
        marker in response_text and "✓" in response_text
        for marker in _REGEX_MARKERS
    )
    if has_regex_claim:
        _REGEX_USAGE = ["import re", "re.", "re.match", "re.search", "re.sub", "re.compile", "re.findall"]
        regex_violations = [pat for pat in _REGEX_USAGE if pat in response_text]
        if regex_violations:
            warnings.append(
                f"⚠️ 假✓: 标记'正则✓'但代码中使用了 {regex_violations}"
            )

    # DB 驱动检查：需要 session_id
    if not session_id:
        return warnings

    tool_calls = _get_session_tool_calls(session_id)
    tool_set = set(tool_calls)

    # 检查"改前验证✓"——应有 read_file 或 search_files
    if "改前验证" in response_text and "✓" in response_text:
        verification_tools = {"read_file", "search_files", "terminal"}
        if not (tool_set & verification_tools):
            warnings.append(
                "⚠️ 假✓: 标记'改前验证✓'但当前会话无 read_file/search_files/terminal 调用记录"
            )

    # 检查"不交付半成品✓"——应有 terminal 或 execute_code 验证
    if "不交付半成品" in response_text and "✓" in response_text:
        verification_tools = {"terminal", "execute_code"}
        if not (tool_set & verification_tools):
            warnings.append(
                "⚠️ 假✓: 标记'不交付半成品✓'但当前会话无 terminal/execute_code 验证调用记录"
            )

    return warnings


def register(ctx):
    """注册 transform_llm_output 钩子。"""

    def review_response(response_text, session_id=None, model=None, platform=None, **kwargs):
        if not response_text:
            return None

        warnings = _check_false_claims(response_text, session_id)
        if not warnings:
            return None  # 透传，不做任何修改

        # 在回复末尾追加假✓警告
        warning_block = (
            "\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔍 自审外部校验——检测到假✓：\n"
            + "\n".join(f"  {w}" for w in warnings)
            + "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        return response_text + warning_block

    ctx.register_hook("transform_llm_output", review_response)
