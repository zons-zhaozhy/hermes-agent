"""coding-standards-guard plugin — 事前拦截编码规范违规（集中管理）。

用 Python ast 模块精准检测代码中的违规模式（非正则）。
所有规则来源于系统提示词、coding-conventions skill、用户多次强调的铁律。
新规则只需：写检测函数 + 加到 _RULES 注册表，自动生效。

拦截的规则（用 AST 精准检测，非正则）：

  ── 吞异常系列（5 条）──
  R001  except Exception: pass          — 吞异常无日志（铁律）
  R002  except: pass                    — 裸 except 吞一切（铁律）
  R003  except Exception as e: pass     — 有变量名仍 pass
  R005  except handler 只有 pass         — 所有 except body 只含 Pass
  R008  静默降级（Exception+return 常量）— 只抓宽异常+静默返回
  R013  静默吞异常（Exception+return 无日志）— 只抓宽异常
  R015  except 块只有 debug/info 无 warning/error — 只抓纯低级别

  ── 默认值兜底系列（3 条）──
  R009  os.environ.get 带硬编码默认值   — 缺失必须报错引导
  R016  getattr(config, key, default)   — 只抓变量名含 config/settings/env
  R018  config.get("old") or config.get("new") — 别名兼容禁止

  ── 硬编码系列（4 条）──
  R007  硬编码密码/密钥                 — password="xxx" 赋值字面量
  R010  硬编码 IP/端口                   — "192.168.x.x" / "localhost:5432"
  R011  硬编码数据库连接串               — "postgresql://..." 赋值字面量
  R012  硬编码部署路径                   — "/opt/..." 赋值字面量

  ── 安全系列（1 条）──
  R014  裸 eval()                        — 安全风险，用 ast.literal_eval 替代

  ── 代码质量系列（2 条）──
  R017  str(val or "") 仅数值语义        — 只抓变量名含 row/col/field 等数值场景
  R019  函数内 sys.exit() 非入口函数     — 放过 main/cli/__main__

  ── 其他 ──
  R006  import *                         — 禁止 from X import *

ACTIVATION: ON by default. Set CODING_STANDARDS_GUARD_DISABLE=1 to turn off.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})


def _plugin_disabled() -> bool:
    return os.environ.get("CODING_STANDARDS_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


# ═══════════════════════════════════════════════════════════════════════
# Violation 数据结构 + 辅助函数
# ═══════════════════════════════════════════════════════════════════════

class Violation:
    __slots__ = ("rule_id", "line", "col", "severity", "message", "snippet")

    def __init__(self, rule_id: str, line: int, col: int,
                 severity: str, message: str, snippet: str = ""):
        self.rule_id = rule_id
        self.line = line
        self.col = col
        self.severity = severity
        self.message = message
        self.snippet = snippet

    def __repr__(self):
        return f"Violation({self.rule_id} L{self.line}: {self.message})"


def _snippet(lines: List[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].rstrip()
    return ""


def _assign_target_names(node: ast.Assign) -> str:
    names = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return ", ".join(names) if names else "?"


def _has_logger_call(body: List[ast.stmt], min_level: str = "debug") -> bool:
    """检查 except body 中是否有 logger 调用。min_level 限定最低级别。"""
    _LEVELS = ("debug", "info", "warning", "error", "critical", "exception")
    min_idx = _LEVELS.index(min_level) if min_level in _LEVELS else 0
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr in _LEVELS:
                if _LEVELS.index(func.attr) >= min_idx:
                    return True
    return False


def _is_broad_exception(node: ast.ExceptHandler) -> bool:
    """判断是否是宽异常（Exception/BaseException/裸 except）。
    ValueError/TypeError/OSError/KeyError 等特定异常不算。
    """
    if node.type is None:
        return True  # bare except
    if isinstance(node.type, ast.Name):
        return node.type.id in ("Exception", "BaseException")
    return False


def _exc_desc(node: ast.ExceptHandler) -> str:
    if node.type is None:
        return "bare except"
    return ast.unparse(node.type)


# ═══════════════════════════════════════════════════════════════════════
# R001-R005: 吞异常 — except body 只有 pass
# ═══════════════════════════════════════════════════════════════════════

def _check_except_pass(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R001-R005: 所有 except handler 的 body 只含 Pass → 违规。"""
    violations = []
    severity = "error"  # 默认 error，R005 覆盖为 warning
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            if node.type is None:
                rule_id = "R002"
                exc_desc = "bare except"
            elif isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException"):
                rule_id = "R001" if node.name is None else "R003"
                exc_desc = f"except {node.type.id}"
            else:
                # 收集异常类型名（处理 Name 和 Tuple）
                _SAFE_PASS_EXCEPTIONS = frozenset({
                    "KeyboardInterrupt", "SystemExit", "GeneratorExit",
                    "CancelledError", "FileNotFoundError",
                    "ProcessLookupError", "NoSuchProcess",
                    "PermissionError", "ImportError", "ModuleNotFoundError",
                    "OSError",  # 文件系统操作的标准安全 pass
                    "json.JSONDecodeError",  # JSON 解析失败的标准安全 pass
                    "UnicodeDecodeError",  # 编码问题的标准安全 pass
                })
                exc_names = set()
                if isinstance(node.type, ast.Name):
                    exc_names.add(node.type.id)
                elif isinstance(node.type, ast.Tuple):
                    for elt in node.type.elts:
                        if isinstance(elt, ast.Name):
                            exc_names.add(elt.id)
                # 如果所有异常都在安全列表中，放过
                if exc_names and exc_names.issubset(_SAFE_PASS_EXCEPTIONS):
                    continue
                # 如果 tuple 中包含 Exception（宽异常），即使混合也报
                if "Exception" in exc_names or "BaseException" in exc_names:
                    rule_id = "R001" if node.name is None else "R003"
                    exc_desc = "except (" + ", ".join(sorted(exc_names)) + ")"
                    violations.append(Violation(
                        rule_id=rule_id, line=node.lineno, col=node.col_offset,
                        severity="error",
                        message=f"{exc_desc} 的 body 只有 pass — 含宽异常，吞异常必须至少 logger.warning 或 raise",
                        snippet=_snippet(lines, node.lineno),
                    ))
                    continue
                rule_id = "R005"
                exc_desc = "except (" + ", ".join(sorted(exc_names)) + ")"
            violations.append(Violation(
                rule_id=rule_id, line=node.lineno, col=node.col_offset,
                severity=severity,
                message=f"{exc_desc} 的 body 只有 pass — 吞异常必须至少 logger.warning 或 raise",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R006: import *
# ═══════════════════════════════════════════════════════════════════════

def _check_import_star(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R006: from X import * — 污染命名空间。"""
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    violations.append(Violation(
                        rule_id="R006", line=node.lineno, col=node.col_offset,
                        severity="warning",
                        message=f"from {node.module or ''} import * — 禁止 import *，污染命名空间",
                        snippet=_snippet(lines, node.lineno),
                    ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R007: 硬编码密码/密钥
# ═══════════════════════════════════════════════════════════════════════

_SECRET_NAME_KEYWORDS = frozenset({
    "password", "passwd", "secret", "api_key", "apikey",
    "token", "access_key", "private_key", "secret_key",
    "jwt_secret", "db_password", "admin_secret",
})


def _check_hardcoded_secret(tree: ast.AST, lines: List[str], skip_tests: bool = True) -> List[Violation]:
    """R007: password="xxx" 直接赋值字面量。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = ""
            if isinstance(target, ast.Name):
                name = target.id.lower()
            elif isinstance(target, ast.Attribute):
                name = target.attr.lower()
            if not any(kw in name for kw in _SECRET_NAME_KEYWORDS):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                val = node.value.value
                if val and not val.startswith("${"):
                    violations.append(Violation(
                        rule_id="R007", line=node.lineno, col=node.col_offset,
                        severity="error",
                        message=f"硬编码密码/密钥: {name} = '{val[:20]}...' — 必须从 .env 读取",
                        snippet=_snippet(lines, node.lineno),
                    ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R008: 静默降级 — 宽异常 + return 常量
# ═══════════════════════════════════════════════════════════════════════

_SILENT_CONSTANTS = (None, "", 0, False)


def _is_silent_return_value(val: ast.expr) -> bool:
    """检查 return 值是否是静默常量/空容器。"""
    if isinstance(val, ast.Constant) and val.value in _SILENT_CONSTANTS:
        return True
    if isinstance(val, (ast.List, ast.Tuple, ast.Set)) and not val.elts:
        return True
    if isinstance(val, ast.Dict) and not val.keys:
        return True
    return False


def _check_silent_downgrade(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R008: 宽异常 (Exception/BaseException/bare) + return 常量值 + 无日志。

    收窄：只对宽异常报警，放过 except ValueError: return False 等正常输入校验。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_broad_exception(node):
            continue
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            continue
        stmt_val = node.body[0].value
        if _is_silent_return_value(stmt_val):
            violations.append(Violation(
                rule_id="R008", line=node.lineno, col=node.col_offset,
                severity="error",
                message=f"{_exc_desc(node)} 的 body 只有 return 常量 — 静默降级！必须至少 logger.warning()",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R009: 默认值兜底 — os.environ.get("X", "default")
# ═══════════════════════════════════════════════════════════════════════

def _check_env_default_fallback(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R009: os.environ.get 带硬编码默认值。缺失必须报错引导。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get"):
            continue
        if not (isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"):
            continue
        if len(node.args) >= 2:
            default_val = node.args[1]
            if isinstance(default_val, ast.Constant) and isinstance(default_val.value, str):
                val = default_val.value
                if val:
                    violations.append(Violation(
                        rule_id="R009", line=node.lineno, col=node.col_offset,
                        severity="warning",
                        message=f"os.environ.get 硬编码默认值 '{val}' — 缺失必须报错引导，不能用默认值兜底",
                        snippet=_snippet(lines, node.lineno),
                    ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R010: 硬编码 IP/端口
# ═══════════════════════════════════════════════════════════════════════

_DOTDECIMAL_PREFIXES = (
    "192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
    "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
    "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
    "172.30.", "172.31.",
)


def _looks_like_ip_literal(s: str) -> bool:
    if not s:
        return False
    if s.startswith("localhost:"):
        rest = s[len("localhost:"):]
        if rest and rest.isdigit():
            return True
    for prefix in _DOTDECIMAL_PREFIXES:
        if s.startswith(prefix):
            rest = s[len(prefix):]
            if rest and rest[0].isdigit():
                return True
    return False


def _check_hardcoded_ip(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R010: HOST = "192.168.1.1" / URL = "localhost:5432"。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            val = node.value.value
            if _looks_like_ip_literal(val):
                violations.append(Violation(
                    rule_id="R010", line=node.lineno, col=node.col_offset,
                    severity="warning",
                    message=f"硬编码 IP/端口: {_assign_target_names(node)} = '{val}' — 必须从 .env 读取",
                    snippet=_snippet(lines, node.lineno),
                ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R011: 硬编码数据库连接串
# ═══════════════════════════════════════════════════════════════════════

_DB_URL_PREFIXES = ("postgresql://", "mysql://", "oracle://", "mongodb://",
                    "postgres://", "sqlite://", "redis://", "amqp://")


def _check_hardcoded_db_url(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R011: DATABASE_URL = "postgresql://user:pass@host/db"。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            val = node.value.value.lower()
            for prefix in _DB_URL_PREFIXES:
                if val.startswith(prefix):
                    violations.append(Violation(
                        rule_id="R011", line=node.lineno, col=node.col_offset,
                        severity="error",
                        message=f"硬编码数据库连接串: {_assign_target_names(node)} — 必须从 .env 读取",
                        snippet=_snippet(lines, node.lineno),
                    ))
                    break
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R012: 硬编码部署路径
# ═══════════════════════════════════════════════════════════════════════

_DEPLOY_PATH_PREFIXES = ("/opt/", "/var/", "/etc/", "/usr/local/", "/srv/",)


def _check_hardcoded_deploy_path(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R012: DEPLOY_DIR = "/opt/ontox/deploy" 等硬编码部署路径。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            val = node.value.value
            for prefix in _DEPLOY_PATH_PREFIXES:
                if val.startswith(prefix) and len(val) > len(prefix):
                    name = _assign_target_names(node).lower()
                    path_keywords = ("dir", "path", "home", "root", "deploy", "data",
                                     "config", "log", "cache", "cert")
                    if any(kw in name for kw in path_keywords):
                        violations.append(Violation(
                            rule_id="R012", line=node.lineno, col=node.col_offset,
                            severity="warning",
                            message=f"硬编码部署路径: {_assign_target_names(node)} = '{val}' — 必须从 .env 或相对路径读取",
                            snippet=_snippet(lines, node.lineno),
                        ))
                    break
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R013: 静默吞异常 — 宽异常 + return + 无 logger.warning
# ═══════════════════════════════════════════════════════════════════════

def _check_silent_return(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R013: 宽异常 (Exception/BaseException/bare) + return + 无 logger.warning。

    收窄：只对宽异常报警。
    except ValueError: return False 是正常输入校验，不报。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_broad_exception(node):
            continue
        if len(node.body) < 1:
            continue
        has_return = any(isinstance(s, ast.Return) for s in node.body)
        if has_return and not _has_logger_call(node.body, min_level="warning"):
            has_raise = any(isinstance(s, ast.Raise) for s in node.body)
            if has_raise:
                continue
            violations.append(Violation(
                rule_id="R013", line=node.lineno, col=node.col_offset,
                severity="error",
                message=f"{_exc_desc(node)} 有 return 但无 logger.warning — 静默吞异常！必须 logger.warning()",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R014: 裸 eval()
# ═══════════════════════════════════════════════════════════════════════

def _check_bare_eval(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R014: 裸 eval() — 安全风险，用 ast.literal_eval 替代。"""
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "eval":
            violations.append(Violation(
                rule_id="R014", line=node.lineno, col=node.col_offset,
                severity="error",
                message="裸 eval() — 安全风险！用 ast.literal_eval() 或 json.loads() 替代",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R015: except 块只有 logger.debug/info 无 warning/error
# ═══════════════════════════════════════════════════════════════════════

def _check_except_low_log_level(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R015: except 块中有 logger.debug/info 但**没有** logger.warning/error。

    收窄：只在 body 中没有任何 warning/error 级别日志时才报 debug。
    如果同时有 debug 和 warning，不报（warning 已覆盖）。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        has_debug_or_info = False
        has_warning_or_above = False
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if isinstance(func, ast.Attribute):
                    if func.attr in ("debug", "info"):
                        has_debug_or_info = True
                    elif func.attr in ("warning", "error", "critical", "exception"):
                        has_warning_or_above = True
        if has_debug_or_info and not has_warning_or_above:
            violations.append(Violation(
                rule_id="R015", line=node.lineno, col=node.col_offset,
                severity="warning",
                message=f"except 块中只有 logger.debug/info — 生产环境不输出 = 变相静默！改为 logger.warning/error",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R016: getattr(config, key, default) 静默降级
# ═══════════════════════════════════════════════════════════════════════

_CONFIG_LIKE_NAMES = frozenset({
    "config", "settings", "cfg", "conf", "options",
    "params", "configuration", "preferences",
})


def _check_getattr_default(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R016: getattr(config_like_obj, key, default) — 配置缺失静默降级。

    收窄：只对第一个参数是含 config/settings/env 等的变量名才报。
    getattr(obj, "attr", None) 是 Python 标准惯用法，不报。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "getattr"):
            continue
        if len(node.args) < 3:
            continue
        if not isinstance(node.args[2], ast.Constant):
            continue
        # 检查第一个参数（对象）是否是配置类变量
        first_arg = node.args[0]
        obj_name = ""
        if isinstance(first_arg, ast.Name):
            obj_name = first_arg.id.lower()
        elif isinstance(first_arg, ast.Attribute):
            obj_name = first_arg.attr.lower()
        if not any(kw in obj_name for kw in _CONFIG_LIKE_NAMES):
            continue
        # 放过 self._config / self.config 等内部配置缓存 — 安全属性访问
        if obj_name in ("config", "_config"):
            if isinstance(first_arg, ast.Attribute):
                continue
        violations.append(Violation(
            rule_id="R016", line=node.lineno, col=node.col_offset,
            severity="warning",
            message=f"getattr({ast.unparse(first_arg)}, key, default) — 配置缺失静默降级！必须显式检查并报错",
            snippet=_snippet(lines, node.lineno),
        ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R017: str(val or "") — 仅数值语义场景
# ═══════════════════════════════════════════════════════════════════════

_NUMERIC_SEMANTIC_KEYWORDS = frozenset({
    "row", "rows", "col", "cols", "column", "columns", "field", "fields",
    "amount", "count", "num", "number", "total",
    "price", "rate", "score", "quantity", "balance", "age", "age_months",
    "weight", "height", "width", "length", "size", "ratio",
})

# 明确是字符串语义的变量 — str(or "") 是安全的，不报
_STRING_SAFE_KEYWORDS = frozenset({
    "message_id", "msg_id", "thread_id", "chat_id", "session_id", "task_id",
    "user_id", "channel_id", "role_id", "bot_id", "file_id", "photo_id",
    "ts", "message_type", "bot_token", "baseurl", "base_url", "api_key",
    "phone_number", "email", "url", "uri", "path", "name", "text", "content",
    "description", "desc", "title", "subject", "label", "tag", "type", "status",
    "model", "provider", "field_type", "field_desc", "user_message",
    "dialect", "response", "reply", "comment", "note", "reason", "error",
    "emoji", "sticker", "query", "command", "action", "event_type",
    "operator_type", "emoji_type", "original_message_ts",
})


def _get_call_context_var_name(node: ast.Call) -> str:
    """提取函数调用所在上下文的赋值目标变量名（如果有）。"""
    # 往上走太复杂，这里简单检查调用是否是 str(...) 的 arg
    return ""


def _check_falsy_or_empty_string(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R017: str(val or "") — 0/False 变成空字符串，数据静默丢失。

    收窄：只对赋值目标变量名含数值语义关键词时才报。
    str(name or "") 是安全的（name 不可能是 0/False）。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Name) and func.id == "str"):
            continue
        if len(node.value.args) != 1:
            continue
        arg = node.value.args[0]
        if not (isinstance(arg, ast.BoolOp) and isinstance(arg.op, ast.Or)):
            continue
        has_empty_str = any(
            isinstance(v, ast.Constant) and v.value == ""
            for v in arg.values
        )
        if not has_empty_str:
            continue
        # 检查赋值目标是否含数值语义（放过字符串语义变量）
        target_name = _assign_target_names(node).lower()
        if any(kw in target_name for kw in _STRING_SAFE_KEYWORDS):
            continue
        if any(kw in target_name for kw in _NUMERIC_SEMANTIC_KEYWORDS):
            violations.append(Violation(
                rule_id="R017", line=node.lineno, col=node.col_offset,
                severity="warning",
                message=f"str(val or \"\") — 0/False 会被吞成空字符串！用 str(val) 或 str(val if val is not None else \"\")",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R018: config.get("old_key") or config.get("new_key") — 别名兼容禁止
# ═══════════════════════════════════════════════════════════════════════

def _check_config_alias_compat(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R018: config.get("old") or config.get("new") — 别名兼容禁止。

    收窄：只对 .get() 调用对象是 dict/数据类（非 msg/row/record 等业务对象）
    且变量名含 config/settings/cfg 时才报。
    msg.get("content") or msg.get("value") 是合理字段降级，不报。
    """
    violations = []
    _CONFIG_GET_OBJECTS = frozenset({"config", "cfg", "conf", "settings", "params", "options", "env", "environ"})
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        get_calls = []
        for val in node.values:
            if isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute):
                if val.func.attr == "get" and len(val.args) >= 1:
                    if isinstance(val.args[0], ast.Constant) and isinstance(val.args[0].value, str):
                        obj_name = ""
                        if isinstance(val.func.value, ast.Name):
                            obj_name = val.func.value.id.lower()
                        elif isinstance(val.func.value, ast.Attribute):
                            obj_name = val.func.value.attr.lower()
                        get_calls.append((val.args[0].value, obj_name))
        if len(get_calls) >= 2:
            # 至少一个 get 的对象是 config 类变量
            config_like = any(name for _, name in get_calls if any(kw in name for kw in _CONFIG_GET_OBJECTS))
            if config_like:
                keys = [k for k, _ in get_calls[:2]]
                violations.append(Violation(
                    rule_id="R018", line=node.lineno, col=node.col_offset,
                    severity="error",
                    message=f"config.get({keys[0]!r}) or config.get({keys[1]!r}) — 别名兼容禁止！只认一个键名，旧键必须报错提示新键",
                    snippet=_snippet(lines, node.lineno),
                ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R019: 函数内 sys.exit() — 非入口函数
# ═══════════════════════════════════════════════════════════════════════

_ENTRY_FUNCTION_NAMES = frozenset({
    "main", "cli", "run", "start", "serve", "__main__",
    "app", "entrypoint", "bootstrap",
})  # 入口函数名关键词（部分匹配）
_ENTRY_FUNCTION_KEYWORDS = ("command", "inner", "handler", "callback", "listener")


def _check_sys_exit_in_function(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R019: 函数内 sys.exit() — 放过入口函数。

    main()/cli()/run() 等入口函数中 sys.exit 是正常退出。
    只在普通工具函数/库函数中报。
    """
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.lower() in _ENTRY_FUNCTION_NAMES:
            continue
        # 函数名含入口关键词（如 _gateway_command_inner）也放过
        if any(kw in node.name.lower() for kw in _ENTRY_FUNCTION_KEYWORDS):
            continue
        # 检查装饰器中是否有 @app.command / @click.command 等入口标记
        is_entry = False
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr in ("command", "group", "cli"):
                    is_entry = True
            elif isinstance(dec, ast.Attribute):
                if dec.attr == "command":
                    is_entry = True
        if is_entry:
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "sys"
                        and func.attr == "exit"):
                    violations.append(Violation(
                        rule_id="R019", line=node.lineno, col=node.col_offset,
                        severity="warning",
                        message=f"函数 {node.name}() 内 sys.exit() — 中断调用链！改为 return",
                        snippet=_snippet(lines, child.lineno),
                    ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# 规则注册表 — 集中管理所有编码规范
# ═══════════════════════════════════════════════════════════════════════

_RULES = [
    # ── 吞异常系列（5 条）──
    ("R001-R005", _check_except_pass,             "except handler 只有 pass — 吞异常铁律"),
    ("R008",      _check_silent_downgrade,         "静默降级 — 宽异常 + return 常量"),
    ("R013",      _check_silent_return,            "静默吞异常 — 宽异常 + return 无 logger"),
    ("R015",      _check_except_low_log_level,     "except 块只有 debug/info 无 warning"),
    # ── 默认值兜底系列（3 条）──
    ("R009",      _check_env_default_fallback,    "默认值兜底 — os.environ.get 带硬编码默认值"),
    ("R016",      _check_getattr_default,           "getattr 静默降级 — 只抓 config/settings/env"),
    ("R018",      _check_config_alias_compat,      "别名兼容 — config.get(old) or config.get(new)"),
    # ── 硬编码系列（4 条）──
    ("R007",      _check_hardcoded_secret,          "硬编码密码/密钥"),
    ("R010",      _check_hardcoded_ip,              "硬编码 IP/端口"),
    ("R011",      _check_hardcoded_db_url,           "硬编码数据库连接串"),
    ("R012",      _check_hardcoded_deploy_path,      "硬编码部署路径"),
    # ── 安全系列（1 条）──
    ("R014",      _check_bare_eval,                 "裸 eval() — 安全风险"),
    # ── 代码质量系列（2 条）──
    ("R017",      _check_falsy_or_empty_string,     "str(val or \"\") — 仅数值语义"),
    ("R019",      _check_sys_exit_in_function,      "函数内 sys.exit() — 非入口函数"),
    # ── 其他（1 条）──
    ("R006",      _check_import_star,               "from X import * — 禁止星号导入"),
]


def _run_all_checks(source: str, *, skip_tests: bool = False) -> List[Violation]:
    """用 AST 解析源码，运行所有规则检查。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # noqa: D5 — unparseable code has no AST to check
        return []

    lines = source.split("\n")
    all_violations = []
    for _rule_ids, check_fn, _desc in _RULES:
        # R007/R010/R011/R014 跳过测试文件（测试中的硬编码/fake key/eval 是测试 fixture）
        if skip_tests and _rule_ids in ("R007", "R010", "R011", "R014"):
            continue
        all_violations.extend(check_fn(tree, lines))
    return all_violations


# ═══════════════════════════════════════════════════════════════════════
# Hook 实现
# ═══════════════════════════════════════════════════════════════════════

def _check_content(content: str, target: str) -> Optional[Dict[str, Any]]:
    """检查写入内容是否违反编码规范。"""
    if not content:
        return None
    if target and not target.endswith((".py", ".pyi")):
        return None

    skip_tests = bool(target and ("tests/" in target.replace(os.sep, "/") or "/tests/" in target.replace(os.sep, "/")))
    violations = _run_all_checks(content, skip_tests=skip_tests)
    if not violations:
        return None

    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    if not errors:
        return None

    details = "\n".join(
        f"  L{v.line} [{v.rule_id}] {v.message}"
        + (f"\n       {v.snippet}" if v.snippet else "")
        for v in errors[:8]
    )
    count_text = f"\n  （还有 {len(errors) - 8} 处 error 未显示）" if len(errors) > 8 else ""
    warning_note = f"\n  另有 {len(warnings)} 处 warning" if warnings else ""

    return {
        "action": "block",
        "message": (
            f"[CodingStandardsGuard] 检测到 {len(errors)} 处编码规范违规（error 级）！\n"
            f"  文件: {target or '(execute_code)'}\n"
            f"  违规:\n{details}{count_text}{warning_note}\n\n"
            f"  用户铁律：以下行为绝对不能容忍——\n"
            f"    • except: pass / return 无 logger.warning → 必须至少 logger.warning(exc_info=True) 或 raise\n"
            f"    • except 块中只有 logger.debug/info → 必须改为 logger.warning/error\n"
            f"    • 硬编码密码/IP/连接串/路径 → 必须从 .env / 环境变量读取\n"
            f"    • 默认值兜底/别名兼容/getattr 默认值 → 缺失必须报错引导\n"
            f"    • eval() → 用 ast.literal_eval() 替代\n"
            f"    • import * → 必须显式导入\n"
            f"  修复后重试。"
        ),
    }


def on_pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """拦截写入工具中的编码规范违规。"""
    if _plugin_disabled():
        return None

    tool_name = kwargs.get("tool_name", "")
    if tool_name not in _WRITE_TOOLS:
        return None

    args = kwargs.get("args") or {}

    if tool_name == "write_file":
        content = args.get("content", "")
        target = args.get("path", "")
        return _check_content(content, target)

    elif tool_name == "patch":
        new_string = args.get("new_string", "")
        target = args.get("path", "")
        return _check_content(new_string, target)

    elif tool_name == "execute_code":
        code = args.get("code", "")
        return _check_content(code, "")

    return None


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
