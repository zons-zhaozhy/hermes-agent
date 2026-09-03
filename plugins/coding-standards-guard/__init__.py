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
  R020  变换异常信息 — except 内 raise 新异常不带 from e（error）

  ── 默认值兜底系列（3 条）──
  R009  os.environ.get 带硬编码默认值   — 缺失必须报错引导（error）
  R016  getattr(config, key, default)   — 只抓变量名含 config/settings/env（error）
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

  ── 正则使用系列（1 条）──
  R021  正则使用                          — 优先 str 方法/结构化解析，豁免=re-ok

  ── 诊断输出系列（1 条）──
  R022  诊断输出截断                      — print/logger/raise 消息 [:N] 切片，豁免=trunc-ok

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


def _iter_name_value_pairs(tree: ast.AST):
    """产出 (name_lower, value_node) 对 — 覆盖所有硬编码值出现形态。

    形态覆盖：Assign 顶层、AnnAssign、调用 kwarg、dict 字面量 value。
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield target.id.lower(), node.value, node
                elif isinstance(target, ast.Attribute):
                    yield target.attr.lower(), node.value, node
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            yield elt.id.lower(), node.value, node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                yield node.target.id.lower(), node.value, node
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg:
                    yield kw.arg.lower(), kw.value, node
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    yield key.value.lower(), value, node


def _check_hardcoded_secret(tree: ast.AST, lines: List[str], skip_tests: bool = True) -> List[Violation]:
    """R007: password="xxx" 直接赋值/kwarg/dict 值字面量。

    Contract:
      Preconditions: tree 为合法 AST
      Postconditions: 返回的每条 Violation 的 value 均为非空 str 字面量且非 ${...} 占位
    """
    violations = []
    for name, value, node in _iter_name_value_pairs(tree):
        if not any(kw in name for kw in _SECRET_NAME_KEYWORDS):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            val = value.value
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
                        severity="error",
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
    """R010: HOST = "192.168.1.1" / connect(host="10.0.0.1") / {"host": "localhost:5432"}。"""
    violations = []
    for _name, value, node in _iter_name_value_pairs(tree):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            val = value.value
            if _looks_like_ip_literal(val):
                violations.append(Violation(
                    rule_id="R010", line=node.lineno, col=node.col_offset,
                    severity="error",
                    message=f"硬编码 IP/端口: '{val}' — 必须从 .env 读取",
                    snippet=_snippet(lines, node.lineno),
                ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R011: 硬编码数据库连接串
# ═══════════════════════════════════════════════════════════════════════

_DB_URL_PREFIXES = ("postgresql://", "mysql://", "oracle://", "mongodb://",
                    "postgres://", "sqlite://", "redis://", "amqp://")


def _check_hardcoded_db_url(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R011: DATABASE_URL = "postgresql://user:pass@host/db" — 含 kwarg/dict 形态。"""
    violations = []
    for _name, value, node in _iter_name_value_pairs(tree):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            val = value.value.lower()
            for prefix in _DB_URL_PREFIXES:
                if val.startswith(prefix):
                    violations.append(Violation(
                        rule_id="R011", line=node.lineno, col=node.col_offset,
                        severity="error",
                        message=f"硬编码数据库连接串: '{value.value[:40]}' — 必须从 .env 读取",
                        snippet=_snippet(lines, node.lineno),
                    ))
                    break
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R012: 硬编码部署路径
# ═══════════════════════════════════════════════════════════════════════

_DEPLOY_PATH_PREFIXES = ("/opt/", "/var/", "/etc/", "/usr/local/", "/srv/",)


def _check_hardcoded_deploy_path(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R012: DEPLOY_DIR = "/opt/ontox/deploy" 等硬编码部署路径 — 含 kwarg/dict 形态。"""
    violations = []
    path_keywords = ("dir", "path", "home", "root", "deploy", "data",
                     "config", "log", "cache", "cert")
    for name, value, node in _iter_name_value_pairs(tree):
        if not any(kw in name for kw in path_keywords):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            val = value.value
            for prefix in _DEPLOY_PATH_PREFIXES:
                if val.startswith(prefix) and len(val) > len(prefix):
                    violations.append(Violation(
                        rule_id="R012", line=node.lineno, col=node.col_offset,
                        severity="error",
                        message=f"硬编码部署路径: {name} = '{val}' — 必须从 .env 或相对路径读取",
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
            severity="error",
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
# R020: 变换异常信息 — except 内 raise 新异常不带 from e（丢 __cause__ 链）
# ═══════════════════════════════════════════════════════════════════════

def _check_raise_without_cause(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R020: except 块内 raise 新异常但不带 from e — 异常堆栈链断裂。

    合法形态（放行）：
      - raise                      # 裸 re-raise，完整堆栈透传
      - raise e                    # re-raise 捕获的异常本体
      - raise NewError(...) from e # 显式链，__cause__ 保留
    违规形态（error 阻断）：
      - raise NewError(...)        # Call 形式新异常，无 cause → 堆栈被改写
      - raise NewError             # Name 形式新异常（非 handler 名），无 cause
    警告形态（warning）：
      - raise NewError(...) from None  # 刻意压制 __context__，堆栈信息丢失
    """
    violations = []
    parent_map: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[id(child)] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        # 沿父链找最近的 ExceptHandler，确认 raise 在 except 块内
        cur = parent_map.get(id(node))
        in_handler = None
        while cur is not None:
            if isinstance(cur, ast.ExceptHandler):
                in_handler = cur
                break
            cur = parent_map.get(id(cur))
        if in_handler is None:
            continue

        # raise（裸 re-raise）→ 放行
        if node.exc is None:
            continue
        # raise <handler 名>（re-raise 本体）→ 放行
        if (isinstance(node.exc, ast.Name) and in_handler.name is not None
                and node.exc.id == in_handler.name):
            continue

        # from None — 刻意压制上下文=程序员显式声明,不再告警
        # (0901 实测 mcp_server 3 处 from None 防敏感信息外泄属合法防御形态,
        #  warning 刷屏无增量信息——显式语法本身即审计可见)
        if isinstance(node.cause, ast.Constant) and node.cause.value is None:
            continue

        # 已带 from <expr> → 放行
        if node.cause is not None:
            continue

        # 豁免通道: 行尾 `# raise-ok` + 理由(对齐 trunc-ok/re-ok/ts-ok 惯例)
        # 适用形态: except 内恢复路径/业务校验的分支 raise——与捕获异常无因果,
        # from e 反而错误关联(0901 实测 dbchat example/metric 查重即此形态)
        if "# raise-ok" in lines[node.lineno - 1]:
            continue

        # raise Call(...) 或 raise 其它 Name，无 cause → error
        if isinstance(node.exc, (ast.Call, ast.Name)):
            violations.append(Violation(
                rule_id="R020", line=node.lineno, col=node.col_offset,
                severity="error",
                message="except 内 raise 新异常不带 from e — 异常链断裂（丢 __cause__），堆栈必须原样透传",
                snippet=_snippet(lines, node.lineno),
            ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R021: 正则使用 — 减少正则，优先 str 方法 / 结构化解析
# ═══════════════════════════════════════════════════════════════════════

_REGEX_FUNC_ATTRS = frozenset({
    "match", "fullmatch", "search", "split", "sub", "subn", "findall", "finditer",
})

# 合法消费方：被调用对象本身是 re 模块（re.match）或已编译 pattern（p.match）。
# 已编译 pattern 无法从 AST 区分，按命名约定识别。
_PATTERN_NAME_HINTS = ("re_", "regex", "pattern", "pat_", "rx")


def _is_regex_call(func: ast.expr) -> bool:
    """Contract:
      Preconditions: func 为任意 AST 表达式
      Postconditions: True 当且仅当调用形如 re.xxx(...) / <re_/pattern 变量>.xxx(...)
    """
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in _REGEX_FUNC_ATTRS:
        return False
    if isinstance(func.value, ast.Name) and func.value.id == "re":
        return True
    if isinstance(func.value, ast.Name):
        low = func.value.id.lower()
        return any(low.startswith(h) or h in low for h in _PATTERN_NAME_HINTS)
    if isinstance(func.value, ast.Call):
        # re.compile(...).match(...) 链式
        callee = func.value.func
        if isinstance(callee, ast.Attribute) and callee.attr == "compile":
            if isinstance(callee.value, ast.Name) and callee.value.id == "re":
                return True
    return False


def _line_has_regex_ok(lines: List[str], lineno: int) -> bool:
    if 1 <= lineno <= len(lines):
        if "# re-ok" in lines[lineno - 1]:
            return True
    return False


def _check_regex_usage(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R021: re.sub/re.match/re.compile 等正则使用 — 正则是最后手段。

    优先顺序：str.startswith/endswith/split/replace/in、str 方法组合、
    结构化解析（json/csv/xml/ast）、pathlib。确需正则时行尾加 `# re-ok` 显式豁免。
    """
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_regex_call(node.func):
            if _line_has_regex_ok(lines, node.lineno):
                continue
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else "regex"
            violations.append(Violation(
                rule_id="R021", line=node.lineno, col=node.col_offset,
                severity="error",
                message=f"正则使用 {attr}() — 正则是最后手段，优先 str 方法/结构化解析；"
                        f"确需正则请行尾加 `# re-ok` 显式豁免并写明理由",
                snippet=_snippet(lines, node.lineno),
            ))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = ""
            if isinstance(node, ast.Import):
                imported_re = any(alias.name == "re" for alias in node.names)
            else:
                imported_re = node.module == "re" or (node.module or "").startswith("re.")
            if imported_re and not _line_has_regex_ok(lines, node.lineno):
                violations.append(Violation(
                    rule_id="R021", line=node.lineno, col=node.col_offset,
                    severity="warning",
                    message="import re — 先确认 str 方法/结构化解析无法解决；"
                            "确需正则请在使用处加 `# re-ok` 并将 import 行也加 `# re-ok`",
                    snippet=_snippet(lines, node.lineno),
                ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R022: 诊断输出截断 — print/logger/raise 消息里的切片
# ═══════════════════════════════════════════════════════════════════════

_DIAG_FUNC_HINTS = ("print", "log", "logger", "warning", "error", "info",
                    "debug", "critical", "exception")


def _is_diag_call(func: ast.expr) -> bool:
    """Contract:
      Preconditions: func 为任意 AST 表达式
      Postconditions: True 当且仅当调用目标名含 print/log/logger/warning/
      error/info/debug/critical/exception（日志与打印类调用）
    """
    if isinstance(func, ast.Name):
        return func.id in _DIAG_FUNC_HINTS
    if isinstance(func, ast.Attribute):
        return func.attr in _DIAG_FUNC_HINTS
    return False


def _check_diag_truncation(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R022: print/logger/raise 消息里出现 [:N] / [-N:] 切片 — 诊断证据被截断。

    教训(2026-08-28):dispatch 派发行 issue[:120] 切掉证据尾部,排障 20 分钟
    找不到根因。截断就是埋坑——诊断输出必须全文;回灌 LLM 的功能性上限
    须单独注明且日志侧同步落全文。豁免=行尾 `# trunc-ok` 加理由。
    """
    violations = []
    for node in ast.walk(tree):
        # print(...)/logger.xxx(...) 参数里含 Subscript 切片
        if isinstance(node, ast.Call) and _is_diag_call(node.func):
            for arg in node.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
                        if "# trunc-ok" in lines[sub.lineno - 1]:
                            continue
                        violations.append(Violation(
                            rule_id="R022", line=sub.lineno, col=sub.col_offset,
                            severity="error",
                            message="诊断输出截断 [:N] — 截断就是埋坑(实测排障被误导20分钟);"
                                    "诊断消息必须全文;确属功能性上限请行尾 `# trunc-ok` 加理由,"
                                    "且日志侧同步落全文",
                            snippet=_snippet(lines, sub.lineno),
                        ))
        # raise Xxx(...) 消息里含切片
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
                        if "# trunc-ok" in lines[sub.lineno - 1]:
                            continue
                        violations.append(Violation(
                            rule_id="R022", line=sub.lineno, col=sub.col_offset,
                            severity="error",
                            message="异常消息截断 [:N] — 报错信息是定位根因的第一证据,"
                                    "禁止切片;确需上限行尾 `# trunc-ok` 加理由",
                            snippet=_snippet(lines, sub.lineno),
                        ))
    return violations


# ═══════════════════════════════════════════════════════════════════════
# R023: 信息维度丢弃 — 日志采集调用缺时间戳参数
# ═══════════════════════════════════════════════════════════════════════

_LOG_COLLECT_CMDS = ("logs", "journalctl")


def _check_log_collect_timestamp(tree: ast.AST, lines: List[str]) -> List[Violation]:
    """R023: docker/kubectl logs、journalctl 采集缺时间戳 — 有而不取。

    契约(2026-09-01 用户拍板,普适编程规约): 信息持有的定位维度(时间/位置/主体),
    不取/不用/不传/丢弃,每一环都必须有显式声明的理由;无声丢弃=违规。
    本规则拦第一环「有而不取」: 日志采集工具原生支持 --timestamps(docker/kubectl)
    或 --output=short-precise(journalctl),不取=错误何时发生永久丢失。
    教训: ontoX doctor 全部 6 处 docker logs 均未带 --timestamps,诊断报告里的
    错误行是无时间光杆行(commit e94255d6a 修)。
    豁免=行尾 `# ts-ok` 加理由(如采集目的是纯计数)。

    Contract:
      Preconditions: tree 为可解析 AST
      Postconditions: 命中当且仅当 Call 参数含字面量 'logs'/'journalctl' 且
      全部字面量参数无 --timestamps/-t/--output=short-precise 前缀;豁免行不报
    """
    violations = []
    _TS_ARGS = ("--timestamps", "-t", "--output=short-precise",
                "-o", "--output")
    # 命令首位须是采集工具本身——排除 os.path.join(dir,"logs")/argparse("logs")/
    # getattr(cfg,"log_dir") 等同名巧合(负向验证 4 处误报的根因,0901 实测)
    _TOOL_PREFIX = ("docker", "kubectl", "podman", "nerdctl", "journalctl",
                    "sudo", "timeout")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # 字面量参数采集: 直接字符串常量 + 列表字面量内的字符串常量
        # (命令形态典型为 ["docker","logs",...] 包在 List 里——R023 首版漏此形态)
        lits = []
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                lits.append(a.value)
            elif isinstance(a, ast.List):
                lits.extend(e.value for e in a.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str))
        if not lits:
            continue
        first = lits[0]
        if first not in _TOOL_PREFIX:
            continue
        # 找到工具后的首个非包装词(sudo/timeout 后面才是真命令)
        cmd_chain = [v for v in lits if v in _TOOL_PREFIX or v in _LOG_COLLECT_CMDS]
        if not any(v in _LOG_COLLECT_CMDS for v in cmd_chain):
            continue
        if any(v.startswith(_TS_ARGS) for v in lits):
            continue  # 已带时间戳参数
        if "# ts-ok" in lines[node.lineno - 1]:
            continue
        cmd = next(v for v in cmd_chain if v in _LOG_COLLECT_CMDS)
        violations.append(Violation(
            rule_id="R023", line=node.lineno, col=node.col_offset,
            severity="error",
            message=f"日志采集({cmd})缺时间戳参数 — 无声丢弃是唯一不可逆动作;"
                    "气门(按可逆性三选一):①补 --timestamps(docker/kubectl)/"
                    "--output=short-precise(journalctl) 全取;②确知下游不需要"
                    "时间→行尾 `# ts-ok` 加理由(保留豁免痕迹,后续可翻案);"
                    "③暂不确定→同样补参数取全,下游自滤(多收可逆,少收不可逆)",
        ))
    return violations


_RULES = [
    # ── 吞异常系列（5 条）──
    ("R001-R005", _check_except_pass,             "except handler 只有 pass — 吞异常铁律"),
    ("R008",      _check_silent_downgrade,         "静默降级 — 宽异常 + return 常量"),
    ("R013",      _check_silent_return,            "静默吞异常 — 宽异常 + return 无 logger"),
    ("R015",      _check_except_low_log_level,     "except 块只有 debug/info 无 warning"),
    ("R020",      _check_raise_without_cause,      "变换异常信息 — raise 新异常不带 from e"),
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
    # ── 正则使用系列（1 条）──
    ("R021",      _check_regex_usage,               "正则使用 — 优先 str 方法/结构化解析，豁免=re-ok"),
    # ── 诊断输出系列（1 条）──
    ("R022",      _check_diag_truncation,           "诊断输出截断 — print/logger/raise 消息切片，豁免=trunc-ok"),
    # ── 信息维度系列（1 条）──
    ("R023",      _check_log_collect_timestamp,     "日志采集缺时间戳 — 有而不取，豁免=ts-ok"),
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
