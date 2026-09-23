"""
失败预演护栏插件（failure_preflight）
====================================

思想源：查理·芒格逆向思维法——先写「怎样一定会失败」，再避开最危险的几步。
出处：docs/research/2026-08-30-thinking-systems-hermes-insights.md A-1 条。

机制：pre_llm_call hook。检测到「动手类任务」指令时，向本轮上下文注入
「失败说明书」生成要求：开工前先写失败 top3、断电开关、翻转护栏。
与 goal-judge（事后审）互补，本插件是事前防。

设计原则：
1. 只在任务首轮注入——检测到这是新任务的开端，追问/确认/闲聊不注入
2. 注入的是「元要求」不是死步骤，agent 按任务性质自行裁剪
3. 轻量：单条规则文本，不读文件不查状态，零副作用
"""

import logging
import re

logger = logging.getLogger(__name__)

_PREFLIGHT_RULE = """\
[动手前预检——失败说明书，跳过即视为未评估风险]
本轮是动手类任务。在产出计划/执行之前，先用 3-5 行完成失败预演：
1. 失败说明书：这件事最可能怎么烂尾？列 top2-3（写具体条件，不写「没做好」这类空话）
2. 断电开关：哪一处失败会让后续全部白做？（识别关键依赖单点）
3. 翻转护栏：把 top1 风险翻转成护栏——触发条件 + 一句补救动作
预演结论并入计划前置输出；若任务足够简单（单命令/纯只读），可声明「风险可忽略」并一句话说明理由后跳过。
"""

# 动手类任务信号：动词性指令词（动词打头的祈使句，排除疑问/闲聊）
_ACTION_RE = re.compile(
    r"^(动手|开始|做|写|建|创建|改|修|重构|部署|实现|开发|迁移|删除|清理|配置|安装|跑|执行|测试|上线|发布|提交|推送|打包|生成|搭|搭一个|出一个|出一份|出一张|画)"
)

# 疑问/讨论类信号——出现则不注入（即使含动作词）
_DISCUSS_RE = re.compile(r"^(什么是|是什么|为啥|为什么|怎么理解|怎么看|如何理解|能不能|能不能实现|是否|啥意思|听说过|介绍一下|评价|对比)")


def _is_action_task(user_message: str) -> bool:
    """判断是否为动手类任务的开端（而非追问/讨论/闲聊）。

    Contract:
        Preconditions: user_message 为字符串（可为空）
        Postconditions: 返回 True 当且仅当消息以动作动词开头且非讨论句式
    """
    if not user_message:
        return False
    msg = user_message.strip()
    if _DISCUSS_RE.match(msg):
        return False
    return bool(_ACTION_RE.match(msg))


def _on_pre_llm_call(**kwargs) -> dict:
    """pre_llm_call 回调：动手类任务注入失败说明书要求。"""
    user_message = kwargs.get("user_message", "")
    if not _is_action_task(user_message):
        return {}
    logger.debug(
        "failure_preflight: 注入失败预演要求 (user_message: %s)",
        user_message,
    )
    return {"context": _PREFLIGHT_RULE}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("failure_preflight 插件已注册——失败预演护栏就绪")
