"""enterprise-quality-gate — 企业级质量纲领强制贯彻插件。

双层结构（对齐规范预注入法：规范上移 system prompt 稳定段，数据走纲领文件）：

  L1 稳定段（register_system_prompt_section）
    四层 23 维对表纪律框架注入每个新会话——不依赖 skill 是否被加载。
    只注入纪律框架（≤400 字），纲领正文不复制（唯一事实源在项目仓库
    docs/standards/ENTERPRISE_GRADE_STANDARD.md）。
  L2 finish 检查（pre_verify）
    回复声称「企业级/可交付/验收完成」而项目无 QUALITY_SCORECARD.md
    计分卡时，注入补办提示（同会话一次；连续空口声明下轮升级）。

  设计约束：fail-open（插件异常不阻断会话）；subagent/batch 平台排除；
  纲领路径按项目解析（多仓适用），不存在时只注入纪律不注入路径。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

from plugins import _shared_state

logger = logging.getLogger(__name__)

_NAMESPACE = "enterprise_quality_gate"

_EXCLUDED_PLATFORMS = {"subagent", "batch"}

# 纲领判据：声称完成类关键词（命中且无计分卡时提示）
_CLAIM_KEYWORDS = (
    "企业级",
    "可交付",
    "验收完成",
    "达到生产",
    "质量达标",
)

# 计分卡文件名（纲领五-贯彻挂点约定）
_SCORECARD_NAME = "QUALITY_SCORECARD.md"

# 稳定段纪律框架（≤400 字，只讲规矩不讲数据——数据在纲领文件里）
_QUALITY_RULE = """\
[企业级质量纪律]
开发/交付/验收任何企业级应用前，先对四层 23 维质量纲领对表：
L1 业务完备（功能/流程/领域覆盖/角色/数据契约）
L2 体验（反馈/效率/信息架构/视觉/性能感知）
L3 工程构造（契约确定性/真相零漂移/测试证据真实/零静默/可审计/可维护/兼容）
L4 运行存活（容量/预警先知/安全/恢复/可演进/交付赋能）。
铁律：①「已做」须当轮工具原始输出背书，纸面自评无效 ②每维验证挂点缺失须显式
标「待门禁化」 ③事实变更当轮重算计分卡对应层得分 ④引用外部标准必核版本链
（DCMM 现行=GB/T 36073-2025）⑤Security（信息安全性）与 Safety（使用安全）是
两个特性禁混译。纲领唯一事实源=项目仓 docs/standards/ENTERPRISE_GRADE_STANDARD.md；
各应用计分卡=QUALITY_SCORECARD.md。"""


def _plugin_disabled() -> bool:
    """按环境变量 ENTQG_DISABLE 快速关闭（调试用）。

    Contract:
      Postconditions: 返回 bool；环境变量缺失视为未关闭；永不抛异常
    """
    raw = os.environ.get("ENTQG_DISABLE")
    if raw is None:
        return False
    try:
        return bool(int(raw))
    except ValueError:
        logger.warning("ENTQG_DISABLE 非法值 %r，视为未关闭", raw)
        return False


def _find_scorecard(cwd: Optional[str]) -> Optional[Path]:
    """从 cwd 向上最多 4 层找 QUALITY_SCORECARD.md。

    Contract:
      Precondition: cwd 为合法目录路径或 None
      Postconditions: 返回存在的计分卡路径或 None；异常时 warning 并返回 None
    """
    if not cwd:
        return None
    try:
        p = Path(cwd).resolve()
        for _ in range(4):
            cand = p / _SCORECARD_NAME
            if cand.is_file():
                return cand
            cand2 = p / "docs" / "standards" / _SCORECARD_NAME
            if cand2.is_file():
                return cand2
            # 应用子目录形态（monorepo）：apps/*/docs/QUALITY_SCORECARD.md
            apps = p / "apps"
            if apps.is_dir():
                for app_doc in apps.glob("*/docs/" + _SCORECARD_NAME):
                    if app_doc.is_file():
                        return app_doc
            if p.parent == p:
                break
            p = p.parent
        return None
    except Exception as exc:
        logger.warning("enterprise-quality-gate 计分卡查找失败: %s", exc, exc_info=True)
        return None


def _quality_section(session_info: Mapping[str, Any]) -> str:
    """渲染稳定段纪律框架（subagent/batch 无对表需求，返回空被 core 跳过）。

    Contract:
      Postconditions: 主会话返回 _QUALITY_RULE 原文；排除平台返回 ""
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    return _QUALITY_RULE


def register(ctx: Any) -> None:
    """注册 system prompt 稳定段与 pre_verify 检查。

    Contract:
      Postconditions: 两挂点已注册；钩子内部异常不外泄（fail-open）
    """

    def check_finish(**kwargs: Any) -> Optional[dict[str, Any]]:
        """finish 前检查：声称企业级完成而无计分卡 → 注入补办提示。

        Contract:
          Postconditions: 返回 None 或 {"context": str}；异常不外泄
        """
        if _plugin_disabled():
            return None
        try:
            reply = str(kwargs.get("response_text") or "")
            if not any(k in reply for k in _CLAIM_KEYWORDS):
                return None
            cwd = kwargs.get("cwd") or os.getcwd()
            scorecard = _find_scorecard(cwd)
            if scorecard is not None:
                return None
            # 同会话只提示一次（状态键跨轮持久）
            sid = kwargs.get("session_id") or ""
            st = _shared_state.get_session_state(_NAMESPACE, sid)
            if st.get("reminded"):
                return None
            st["reminded"] = True
            return {
                "context": (
                    "[enterprise-quality-gate] 本轮声明含企业级质量口径，但项目内未找到 "
                    "QUALITY_SCORECARD.md 计分卡。按纲领贯彻挂点：先建计分卡"
                    "（四层 23 维对表+差距清单，模板见纲领文件第五节），再声明质量结论。"
                )
            }
        except Exception as exc:  # fail-open
            logger.warning("enterprise-quality-gate check failed: %s", exc, exc_info=True)
            return None

    ctx.register_system_prompt_section("enterprise_quality_gate", _quality_section)
    ctx.register_hook("pre_verify", check_finish)
    logger.info("enterprise-quality-gate: 稳定段 + finish 计分卡检查已注册")
