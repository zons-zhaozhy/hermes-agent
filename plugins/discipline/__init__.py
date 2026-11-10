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
  blame_shield     生成前自检注入(推责/免责/偷懒/理想化,默认注入,封闭闲聊豁免)
  failure_preflight 动手前失败预演注入(失败说明书/断电开关/翻转护栏)
  declare_act_guard 宣告-动手配对守卫(LLM 语义判定;post_llm_call 预热缓存)
  negative_conclusion_guard 否定性结论前强制权威入口(禁单目录探测断言不存在)
  rule_reinjection 规则常驻重注入(.hermes-rules.md 每轮权威版注入,防纪律衰减)
  test_discipline  禁裸 pytest——一律 scripts/run_tests.sh(CI 环境同步)
  section_reinjection 守卫段重注入(段库版本指纹变更即注入 + 每 10 轮补注, 让段
                   修改对运行中的会话也生效——system prompt 段在会话起点冻结)
  workflow_router  普通会话首轮注入一次工程工作流路由(非阻断, 指向 bundled skills)

分层依据见 docs/guard-system-architecture.md。

Contract:
  Preconditions: plugin system 提供 pre_tool_call / post_tool_call / pre_llm_call 钩子。
  Postconditions: 每个子模块 register 被调用一次;单个失败仅告警不连坐。
  Invariants: 子模块代码纯移动,检测逻辑与迁移前逐字一致。
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

_SUB_GUARD_NAMES = (
    "no_guessing",
    "db_safety",
    "error_discipline",
    "curl_safety",
    "log_first",
    "tool_safety",
    "no_bypass",
    "no_pushback",
    "blame_shield",
    "failure_preflight",
    "declare_act_guard",
    "negative_conclusion_guard",
    "rule_reinjection",
    "test_discipline",
    "section_reinjection",
    "workflow_router",
)

# 有拦截、无前置提示且属「高频×可预判」的子护栏——段文本见段库
# prompt_sections/sections.json; 其余子护栏不配段(不为覆盖率加段)。
_SECTION_IDS = ("db_safety", "test_discipline")

# 子代理与批处理会话无写代码主循环,段体诱导空转——按 platform 跳过。
_EXCLUDED_PLATFORMS = frozenset({"subagent", "batch"})


def _render_section(section_id: str) -> Callable[[Mapping[str, Any]], str]:
    """生成文件型段渲染器（读段库取最新；排除平台返回空段）。

    Contract:
      Preconditions: section_id 为段库内键名。
      Postconditions: 返回渲染器——常规会话返回段文本, 排除平台返回 ""。
    """
    def _render(session_info: Mapping[str, Any]) -> str:
        if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
            return ""
        from plugins.prompt_sections import load_section  # 延迟导入: 仅段渲染路径需要

        return load_section(section_id)

    return _render


def register(ctx: Any) -> None:
    """套件入口——逐子护栏注册 + 注册文件型段(单个失败不连坐)。"""
    for name in _SUB_GUARD_NAMES:
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
            mod.register(ctx)
        except Exception:
            logger.warning(
                "discipline: sub-guard %s failed to import/register; siblings continue",
                name,
                exc_info=True,
            )
    for section_id in _SECTION_IDS:
        try:
            ctx.register_system_prompt_section(section_id, _render_section(section_id))
        except Exception:
            logger.warning(
                "discipline: prompt section %s failed to register", section_id, exc_info=True
            )
