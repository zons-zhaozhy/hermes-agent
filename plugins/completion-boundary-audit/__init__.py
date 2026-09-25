"""completion-boundary-audit plugin — 反面检查检查点（反者道之动）

judge 判定最终回复是否"包含交付完成声明且未披露任何未验证边界"（人话语义
不可穷举，禁关键词/正则匹配），命中不修改用户可见回复，只记入会话状态；
下一轮 pre_llm_call 把红牌注入给 agent（与 reply-certainty-checker 同范式），
强制 agent 在下一轮补披露未验证边界。

消项闭环（2026-09-25 根治改造）：用户实录——AI 每轮都披露未验证边界但
披露完就撂挑子，下一轮把同一批边界原样再抄。根因：旧判据「完成声明且
未披露」才标记，披露了就放行——披露成了免罪金牌。改造：披露了边界+
完成声明 → 记 pending；下一轮注入消项红牌（能验证的当轮贴原始输出，
外部依赖项写依赖方+等待条件即算消项）；同批 pending 连续 ≥2 轮未消 →
红牌升级点名「连续未消」；回复不再含边界且含完成声明 → 已消项清状态。

设计修正史：
- v1 transform_llm_output 直接把红牌追加到用户可见回复尾部 → 污染用户
  输出，"该由 agent 补的边界"被甩给用户看。改为记状态→下轮注入给 agent。
- v1.5 只管补披露 → 半截规则，披露即履约，AI 拿边界清单当免责声明
  （2026-09-25 用户怒斥"压根不管不顾就撂挑子"）。升级为跨轮消项闭环。

设计依据：落实笔记-七心法七功法.md 第五节——把用户"完成了吗"式人工
纠偏变成 agent 自检。与 UX 审计铁律（"完成了吗"=追加审查信号）同构。

成本控制：回复 hash 去重 + judge 调用每会话硬上限 30 + 红牌每会话
上限 3 + 短回复（<80字符，闲聊）直接跳过 + fail-open。

缓存安全：不改 system prompt、不改历史消息、不换 toolset——
per-conversation prompt caching 不受影响（注入走 request-scoped context）。

ACTIVATION: 需在 config.yaml plugins.enabled 中添加 "completion-boundary-audit"。
Set COMPLETION_BOUNDARY_AUDIT_DISABLE=1 to turn off.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state

# 回复侧合并判定：uncertain/needs_audit 一次调用（唯一出入口收拢）。
# 延迟导入：避免插件间循环依赖；certainty 插件未启用时
# judge_reply_side 不可用，此时本插件回退独立判定（llm_judge_bool 原路径）。
from importlib import import_module as _import_module
import sys

logger = logging.getLogger(__name__)

_NAMESPACE = "completion_boundary_audit"

# 短回复（闲聊/确认）不触发
_MIN_LENGTH = 80
_MAX_REMINDERS = 3
_MAX_JUDGE_CALLS = 30

_JUDGE_SYSTEM = (
    "你是交付审查哨兵。判断下面这条 AI 最终回复是否同时满足："
    "1) 包含交付完成声明（声称任务/修复/测试/部署已完成、全部通过、已交付）；"
    "2) 未披露任何未验证边界（未列出未测路径/环境/已知风险/局限/未覆盖）。"
    "两条都满足才答 true。只是进度汇报、已含边界声明、闲聊、提问都不算。"
    "只回答 JSON：{\"needs_audit\": true} 或 {\"needs_audit\": false}"
)

_INJECT = (
    "【反面检查】你的上一条回复包含交付完成声明，但未披露任何未验证边界。"
    "本轮回复必须补充：1) 哪些路径/环境/方向未实测；2) 验证覆盖到哪里、"
    "之外是推断还是实测；3) 最可能的失败场景与触发条件。"
    "依据：交付验证铁律——报成功必报边界，未列边界=未审计。"
)

# 消项红牌：披露≠履约——披露的边界是工作清单不是免责声明
_INJECT_RESOLVE = (
    "【消项闭环】你的上一条回复披露了未验证边界并声称交付完成——"
    "披露的边界是工作清单，不是免责声明。本轮必须逐项消项："
    "1) 能自己验证的当轮就验，回复正文贴原始输出；"
    "2) 依赖外部（对端服务/用户操作/等待窗口）的，写明依赖方+等待条件+"
    "预计何时可验，才算处置完毕；"
    "3) 不许把同一批边界原样再抄一遍充当披露。"
    "依据：列边界=起点，逐项消=履约，连续两轮只列不消=撂挑子。"
)

# 升级红牌：同批边界连续 ≥2 轮未消——撂挑子实锤
_INJECT_STALE = (
    "【消项闭环·升级】你披露的同一批未验证边界已连续多轮原样未消——"
    "这是把披露当免责声明、列完就撂挑子。本轮禁止再抄边界清单收尾："
    "能验证的立即验证并贴原始输出；外部依赖项写明依赖方+等待条件；"
    "两者都不做的项必须说明为什么本轮做不了。再原样罗列不消项，"
    "等于承认该交付未完成。"
)


def _plugin_disabled() -> bool:
    return os.environ.get("COMPLETION_BOUNDARY_AUDIT_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _state(sid: str) -> Dict[str, Any]:
    return get_session_state(sid or "_global", _NAMESPACE)


def _judge_reply_side(text: str):
    # 运行时插件模块名 = hermes_plugins.<slug>（连字符转下划线，
    # 见 hermes_cli/plugins.py _directory_module_name）；依次尝试两个命名空间。
    for modname in ("hermes_plugins.reply_certainty_checker",
                    "plugins.reply_certainty_checker"):
        try:
            mod = sys.modules.get(modname) or _import_module(modname)
            return mod.judge_reply_side(text)
        except Exception as e:
            logger.warning("reply-side merge via %s unavailable: %s", modname, e)
    return None


def needs_boundary_audit(text: str) -> Optional[bool]:
    """judge 判"完成声明且无边界披露"；None=fail-open。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 空/短文本 → False；否则 True/False/None，绝不 raise
    """
    if not text or len(text) < _MIN_LENGTH:
        return False
    merged = _judge_reply_side(text)
    if merged is not None and merged.get("needs_audit") is not None:
        return merged.get("needs_audit")
    return llm_judge_bool(
        task="completion_boundary_audit",
        system=_JUDGE_SYSTEM,
        text=text,
        true_key="needs_audit",
    )


def _resolve_verdict(text: str) -> Dict[str, Optional[bool]]:
    """合并 judge 判 done_claim/has_boundary 两键；异常/缺键 → 全 None。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 空/短文本 → {False, False}；judge 通道异常 →
                     {None, None}（fail-open，调用方须零状态改动）
    """
    if not text or len(text) < _MIN_LENGTH:
        return {"done_claim": False, "has_boundary": False}
    merged = _judge_reply_side(text)
    if merged is not None and merged.get("done_claim") is not None:
        return {"done_claim": merged.get("done_claim"),
                "has_boundary": merged.get("has_boundary")}
    logger.warning("completion-boundary-audit: 合并判定缺 done_claim 键，fail-open")
    return {"done_claim": None, "has_boundary": None}


def register(ctx) -> None:
    """注册 transform_llm_output（检测+记状态）与 pre_llm_call（注入）。

    Contract:
      Postconditions: 两个钩子均已注册；钩子内部任何异常不外泄（fail-open）
    """

    def audit_boundary(response_text, session_id=None, model=None,
                       platform=None, **kwargs) -> Optional[str]:
        """judge 判定 → 状态机转移；用户可见回复零改动。

        状态机（消项闭环）：
        - 完成声明 + 无披露 → 记 pending（下轮补披露红牌，旧路径）
        - 完成声明 + 有披露 → 记 pending + streak=1（下轮消项红牌）
        - 有 pending 时再次完成声明+有披露 → streak+1（连续未消）
        - 有 pending 时完成声明+无边界 → 已消项，清状态
        - 无完成声明（进行中汇报）→ 不打扰（但 pending 保留，不重置）

        Contract:
          Postconditions: 恒返回 None（绝不修改用户回复）；异常不外泄；
                          judge fail-open（None）→ 状态零改动
        """
        if _plugin_disabled():
            return None
        try:
            text = response_text or ""
            if len(text) < _MIN_LENGTH:
                return None
            sid = session_id or ""
            st = _state(sid)
            if int(st.get("count", 0)) >= _MAX_REMINDERS:
                return None
            if int(st.get("judge_calls", 0)) >= _MAX_JUDGE_CALLS:
                return None
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            seen = st.get("seen")
            if not isinstance(seen, set):
                seen = set()
                st["seen"] = seen
            if h in seen:
                return None
            seen.add(h)
            st["judge_calls"] = int(st.get("judge_calls", 0)) + 1
            verdict = _resolve_verdict(text)
            done = verdict.get("done_claim")
            boundary = verdict.get("has_boundary")
            if done is None or boundary is None:
                return None  # fail-open：判定失败零状态改动
            if not done:
                return None  # 进行中汇报不打扰（pending 保留跨轮）
            if boundary:
                # 披露≠履约：记 pending，连续披露未消 → streak 累加
                st["count"] = int(st.get("count", 0)) + 1
                st["pending_reminder"] = True
                st["streak"] = int(st.get("streak", 0)) + 1
                logger.info(
                    "completion-boundary-audit: 完成声明+边界披露已标记"
                    "（streak=%d，下轮注入消项红牌）", st["streak"]
                )
            else:
                # 无披露且含完成声明：有 pending=已消项；无 pending=旧路径
                if st.get("pending_reminder") or int(st.get("streak", 0)) > 0:
                    st["pending_reminder"] = False
                    st["streak"] = 0
                    logger.info(
                        "completion-boundary-audit: 边界已消项，状态清零"
                    )
                elif needs_boundary_audit(text) is True:
                    st["count"] = int(st.get("count", 0)) + 1
                    st["pending_reminder"] = True
                    logger.info(
                        "completion-boundary-audit: 完成声明未披露边界已标记"
                        "（下轮注入红牌）"
                    )
            return None
        except Exception as e:  # 绝不因插件自身错误破坏回复
            logger.warning("completion-boundary-audit skipped: %s", e,
                           exc_info=True)
        return None  # 透传

    def inject_reminder(**kwargs) -> Optional[Dict[str, Any]]:
        """上一轮被标记 → 注入红牌给 agent。

        红牌分三档：旧路径（未披露→补披露）；消项（披露未消→逐项消）；
        升级（streak≥2 连续未消→点名撂挑子）。注入一次即消费 pending 标记
        （streak 保留，由消项成功时清零）。

        Contract:
          Postconditions: 返回 None 或 {"context": str}；注入一次即消费标记
        """
        if _plugin_disabled():
            return None
        try:
            sid = kwargs.get("session_id") or ""
            st = _state(sid)
            if not st.get("pending_reminder"):
                return None
            del st["pending_reminder"]
            if int(st.get("streak", 0)) >= 2:
                return {"context": _INJECT_STALE}
            if int(st.get("streak", 0)) >= 1:
                return {"context": _INJECT_RESOLVE}
            return {"context": _INJECT}
        except Exception as exc:  # fail-open
            logger.warning("completion-boundary-audit inject failed: %s", exc)
            return None

    ctx.register_hook("transform_llm_output", audit_boundary)
    ctx.register_hook("pre_llm_call", inject_reminder)
    logger.info("completion-boundary-audit registered")
