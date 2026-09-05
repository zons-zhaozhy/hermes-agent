"""discipline —— L3 执行纪律护栏套件（收拢统一入口）。

前身 8 个独立插件目录(no-guessing / db-safety / error-discipline /
curl-safety / log-first-diagnosis / tool-safety / no-bypass / no-pushback)
收拢为单一套件:一个 plugin.yaml、config 一个开关 ``discipline``。
同层同性质——都管「怎么跑命令/怎么面对错误」:

  no_guessing      R1失败原样重试拦/R2同错2次/R3服务名--list核验/R4日志裸奔/R5 sleep干等/R6 2>/dev/null
  db_safety        SQL 查询前必须先确认 schema(禁猜表名列名)
  error_discipline 连续错误未诊断即拦(强制读错误→读代码→追根因)
  curl_safety      HTTP 请求前强制确认 API 签名(禁瞎猜路径参数)
  log_first        修复前先读日志(log-first diagnosis)
  tool_safety      灾难删除/批量替换无验证等工具滥用拦截
  no_bypass        被护栏拦截后禁换通道绕过
  no_pushback      输出推责话术拦截

分层依据见 docs/guard-system-architecture.md。

Contract:
  Preconditions: plugin system 提供 pre_tool_call / post_tool_call / pre_llm_call 钩子。
  Postconditions: 每个子模块 register 被调用一次;单个失败仅告警不连坐。
  Invariants: 子模块代码纯移动,检测逻辑与迁移前逐字一致。
"""

from __future__ import annotations

import logging

from plugins.discipline import (
    curl_safety,
    db_safety,
    error_discipline,
    log_first,
    no_bypass,
    no_guessing,
    no_pushback,
    tool_safety,
)

logger = logging.getLogger(__name__)

_SUB_GUARDS = (
    ("no_guessing", no_guessing),
    ("db_safety", db_safety),
    ("error_discipline", error_discipline),
    ("curl_safety", curl_safety),
    ("log_first", log_first),
    ("tool_safety", tool_safety),
    ("no_bypass", no_bypass),
    ("no_pushback", no_pushback),
)


def register(ctx):
    """套件入口——逐子护栏注册,单个失败不连坐。"""
    for name, mod in _SUB_GUARDS:
        try:
            mod.register(ctx)
        except Exception:
            logger.warning("discipline: sub-guard %s failed to register", name, exc_info=True)
