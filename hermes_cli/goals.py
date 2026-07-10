"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Judge failures are fail-OPEN: ``continue``. A broken judge must not wedge
  progress; the turn budget is the backstop.
- When a real user message arrives mid-loop it preempts the continuation
  prompt and also pauses the goal loop for that turn (we still re-judge
  after, so if the user's message happens to complete the goal the judge
  will say ``done``).
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# OntoX 集成——延迟导入，避免循环依赖
_ontox_imported = False
_ontox_draft_contract = None
_ontox_preflight_check = None
_ontox_get_ecosystem = None

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Judge output budget. The freeform judge returns a one-line JSON verdict, but
# reasoning models (deepseek-v4, qwq, etc.) burn tokens on hidden reasoning
# before emitting the visible JSON — and the first /goal turn's prompt is
# larger than later turns, which pushes total reply length past tight caps.
# 200 tokens (the original default) reliably truncated the JSON on reasoning
# models, leaving '{"done": true, "reason": "The agent successfully' and
# triggering the auto-pause. 4096 covers reasoning + verdict on every model
# we've live-tested; override via auxiliary.goal_judge.max_tokens for
# specifically constrained setups.
DEFAULT_JUDGE_MAX_TOKENS = 4096
# Cap how much of the last response + recent messages we send to the judge.
_JUDGE_RESPONSE_SNIPPET_CHARS = 8000
# After this many consecutive judge *parse* failures (empty output / non-JSON),
# the loop auto-pauses and points the user at the goal_judge config. API /
# transport errors do NOT count toward this — those are transient. This guards
# against small models (e.g. deepseek-v4-flash) that cannot follow the strict
# JSON reply contract; without it the loop runs until the turn budget is
# exhausted with every reply shaped like `judge returned empty response` or
# `judge reply was not JSON`.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3


CONTINUATION_PROMPT_TEMPLATE = (
    '[继续执行目标]\n'
    '目标: {goal}\n\n'
    '继续推进。自主决策，自主执行，不要停下来等待。\n'
    '- 遇到技术问题：自己排查修复，不要请示。\n'
    '- 遇到缺失信息：做合理假设并继续，在最终报告中列出所有假设。\n'
    '- 遇到不可逆操作（DROP、TRUNCATE、删数据、推送代码）：暂停并说明，这是唯一需要暂停的情况。\n'
    '- 一轮做不完就汇报进度后继续下一轮，不要声称完成然后收工。\n'
    '- 只有目标彻底完成、产出了可验证的结果时才停下来。\n'
    '- 每轮结束时如果工作有实质进展，附上实际的命令输出（curl/grep/npm test 等的终端输出）——不止是「✅」标记。\n'
    '- 声称目标完成时，每个关键断言必须附验证证据。例如：「N个场景品控门控阻断」需列出至少一个场景的curl验证输出；「产物可访问」需提供实际curl输出或文件路径。「全绿」「全部通过」等无具体输出的总结不应出现。'
)

# Used when the goal carries a structured completion contract. The contract
# block tells the agent exactly what "done" means, how to prove it, what not
# to break, what's in scope, and when to stop and ask — so it targets the
# verification surface instead of declaring victory loosely.
CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE = (
    '[继续执行目标]\n'
    '目标: {goal}\n\n'
    '完成合同:\n'
    '{contract_block}\n\n'
    '继续推进，达成上述结果。自主执行下一步。\n'
    '遵守所述边界，不违反约束条件。\n'
    '声称目标完成前，必须满足验证条件，并展示具体证据（命令输出、文件内容、测试结果）。\n'
    '如果触发了停止条件或因其他原因受阻需要用户输入，暂停并说明。'
)

# ──────────────────────────────────────────────────────────────────────
# OntoX 生态约束 — /goal-ontox 模式下注入
# ──────────────────────────────────────────────────────────────────────

ONTOX_ECOSYSTEM_BOUNDARY = (
    '[OntoX 生态约束 — 你只能在以下底座能力范围内编码]\n'
    '\n'
    'OntoX 五驱能力清单（你的全部可用工具箱）：\n'
    '\n'
    '1. OMS（元数据建模引擎）— 数据建模与零代码 CRUD\n'
    '   - 对象类型元数据定义 + 自动 CRUD 页面\n'
    '   - Browser API: POST/GET/PUT/DELETE {OMS_URL}/oms/api/v1/admin/oms/browser/{type}\n'
    '   - 状态机: POST {OMS_URL}/oms/api/v1/admin/oms/state-machine/{name}/fire\n'
    '   - 批量查询: POST {OMS_URL}/oms/api/v1/admin/oms/browser/query\n'
    '\n'
    '2. Cortex（AI 理解引擎）— AIGC 生成 + LLM 推理\n'
    '   - 文本生成: POST {CORTEX_URL}/cortex/api/v1/internal/aigc/text\n'
    '   - 图像生成: POST {CORTEX_URL}/cortex/api/v1/internal/aigc/image\n'
    '   - 视频生成: POST {CORTEX_URL}/cortex/api/v1/internal/aigc/video\n'
    '   - 质量评分: POST {CORTEX_URL}/cortex/api/v1/internal/aigc/quality\n'
    '   - 安全审核: POST {CORTEX_URL}/cortex/api/v1/internal/aigc/safety\n'
    '   - 增强端点: /dedup /keyword_extract /translate /video/extend /video/concat 等 19 个\n'
    '\n'
    '3. Loom（编排引擎）— YAML DAG 工作流执行\n'
    '   - 异步执行: POST {LOOM_URL}/loom/api/v1/analyze/async\n'
    '   - 进度轮询: GET {LOOM_URL}/loom/api/v1/analyze/status/{task_id}\n'
    '   - 场景 YAML 放 qfwy 仓库 scenarios/ 目录，部署时复制到 Loom\n'
    '   - http_call Step 调 Cortex + 写回 OMS\n'
    '   - quality_gate 品控门控 + condition_branch 条件分支\n'
    '\n'
    '4. DBChat（数据查询引擎）— NL2SQL 自然语言查库\n'
    '   - Chat API: POST {DBCHAT_URL}/dbchat/api/v1/chat\n'
    '\n'
    '5. Auth（认证中心）— 统一 JWT 签发与验证\n'
    '   - 登录: POST {AUTH_URL}/auth/api/v1/login\n'
    '   - 五驱共享 JWT_SECRET 验签\n'
    '\n'
    '前端技术栈：Vue 3 + TypeScript + Vite + Element Plus\n'
    '\n'
    '禁止事项（绝对不得违反）：\n'
    '- 禁止创建新的 Java/Python/Go 后端服务——所有后端能力由五驱提供\n'
    '- 禁止直连第三方 AI API（OpenAI/DeepSeek/GLM 等）——必须通过 Cortex 中转\n'
    '- 禁止手搓标准 CRUD 页面（表格/表单/筛选/分页）——必须走 OMS 元数据驱动\n'
    '- 禁止在 qfwy 仓库写 Java/Python 后端代码——只能写前端 Vue 页面 + Loom 场景 YAML\n'
    '- 禁止在前端直连 Cortex AIGC API——生成请求必须通过 Loom /analyze/async 编排\n'
    '- 供应商 URL/API Key/模型名零硬编码——由 Cortex config.py 管理\n'
    '\n'
    '决策判断表（不确定用哪个驱动时参考）：\n'
    '| 需求 | 正确做法 |\n'
    '|------|---------|\n'
    '| 标准增删改查 | OMS 元数据 CRUD（零代码）|\n'
    '| 列表/详情/筛选排序 | OMS Browser API |\n'
    '| 状态流转 | OMS 状态机 |\n'
    '| AI 生成（文案/图像/视频）| Loom 场景 → http_call → Cortex /aigc/* |\n'
    '| 质量评分/安全审核 | Loom 场景 → http_call → Cortex /aigc/quality /safety |\n'
    '| 工作流编排 | Loom YAML DAG（depends_on 拓扑执行）|\n'
    '| 自然语言查数据 | DBChat NL2SQL |\n'
    '| 用户登录/权限 | Auth 中心 JWT |\n'
    '| 视频编辑器/画布 | 手搓前端 Vue 组件（Loom 覆盖不了的交互 UI）|\n'
    '\n'
    '架构原则：\n'
    '- OntoX 底座能力不足时 → 增强底座（补 Cortex/Loom 端点），不在 qfwy 造后端\n'
    '- 业务编排 → Loom 场景 YAML，前端调 Loom API，不硬编码 async/await 链\n'
    '- 数据写入 → Loom http_call 写回 OMS Browser API，结果不入库 = 丢失\n'
    '- 品控门控 → Loom quality_gate Step（服务端真正阻断，不依赖前端 if-else）\n'
)



# Used when the user has added one or more /subgoal criteria. Surfaced
# to the agent verbatim so it sees what to target on the next turn,
# and surfaced to the judge so the verdict considers them too.
CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    '[继续执行目标]\n'
    '目标: {goal}\n\n'
    '用户追加的验收标准:\n'
    '{subgoals_block}\n\n'
    '继续推进，同时满足原始目标和所有验收标准。自主决策，自主执行。\n'
    '- 遇到技术问题：自己排查修复，不要请示。\n'
    '- 遇到缺失信息：做合理假设并继续，在最终报告中列出所有假设。\n'
    '- 遇到不可逆操作（DROP、TRUNCATE、删数据、推送代码）：暂停并说明，这是唯一需要暂停的情况。\n'
    '- 一轮做不完就汇报进度后继续下一轮，不要声称完成然后收工。\n'
    '- 只有目标与所有验收标准全部满足、产出可验证结果时才停下来。'
)


JUDGE_SYSTEM_PROMPT = (
    '你是严格的目标评审员，判断自主 agent 是否真正完成了用户的目标。'
    '你会收到目标文本和 agent 的最新响应片段，以及当前运行的后台进程列表。做出三选一判定。\n\n'
    '=== 判定为 DONE 的条件（必须全部满足） ===\n'
    '1. 交付物已产出且有可验证的证据（文件路径、命令输出、测试结果、具体数据等），'
    '不能只有笼统描述（如「已完成」「改好了」「部署成功」）。\n'
    '2. 目标的每一项要求都被逐一回应，不存在「留到后续」「待完善」「下一步再处理」等未完成项。\n'
    '3. 响应中无实质性错误或半成品（空的 placeholder 文件、未通过的测试、TODO 注释等）。\n'
    '4. 警惕伪证据——以下形式不算可验证证据，出现时严格判定为 CONTINUE：\n'
    '   a) 大量「✅」「已完成」「全部通过」等标记性文字但没有附上任何命令的实际终端输出\n'
    '   b) 「N个场景」「N项检查」等批量数字声明但没有逐项列出验证过程（N>5 时必须至少抽样列出具体验证命令和输出）\n'
    '   c) 「E2E completed Ns」但没有显示每步执行的耗时、HTTP状态码、审计步骤数\n'
    '   d) 「测试全绿」「构建通过」等结论性声明但没有实际的测试/构建输出片段\n'
    '   如果 agent 的响应中大部分"证据"是上述伪证据形式，判定为 CONTINUE——证据不足以确认完成。\n\n'
    '=== 判定为 WAIT 的条件 — 目标未完成，但下一步是等待异步工作完成而非继续操作 ===\n'
    '仅在 agent 的进度确实被某个自主运行的事物阻塞时选择 WAIT：\n'
    '- 下方列出的某个后台进程仍在运行，且 agent 明确在等待其结果（如 CI 轮询、构建、'
    '测试、部署）。如果进程有 session id，在 ``wait_on_session`` 中返回——它会在进程退出'
    '或 watch_patterns 触发时释放。否则在 ``wait_on_pid`` 中返回其 pid（仅在退出时释放）。\n'
    '- agent 表示自己被限速/退避/必须等待固定时间——在 ``wait_for_seconds`` 中返回秒数。\n'
    '选择 WAIT 会暂停循环而不消耗一轮；当 pid 退出或时间到达时自动恢复。'
    '不要仅仅因为还有工作要做就选 WAIT——只有当现在重新推进纯粹是白忙活时才选。\n\n'
    '=== 判定为 CONTINUE 的情况（出现任意一条即判定未完成） ===\n'
    '1. agent 说「需要用户确认」「需要用户输入」「请用户决定」——'
    '除非明确涉及不可逆操作，否则 agent 应自主决策，不应停工等待。\n'
    '2. agent 说「遇到阻塞」，但该阻塞可通过技术手段自行解决（查文档、换方案、重试、读配置等）。\n'
    '3. agent 做了一部分工作但还有未完成的子任务或待验证的步骤。\n'
    '4. agent 给出了方案或计划但没有实际执行——描述不等于完成。\n'
    '5. agent 声称完成但未提供具体可验证的证据。\n'
    '6. agent 在响应末尾说「等待用户指示」「请用户确认」等放弃继续的信号。\n\n'
    '=== 特殊情况 ===\n'
    '唯一因「阻塞」判定为 DONE 的场景：agent 遇到真正的不可逆操作'
    '（如 DROP TABLE、TRUNCATE、删除生产数据、git push --force、ALTER TABLE 删列），'
    '明确列出具体操作内容、影响范围，并等待用户批准。'
    '除此之外，所有「阻塞」都应判定为 CONTINUE。\n\n'
    '只回复一行 JSON，不要输出任何其他内容。格式：\n'
    '{"verdict": "done", "reason": "<一句话理由>"}\n'
    '{"verdict": "continue", "reason": "<一句话理由>"}\n'
    '{"verdict": "wait", "wait_on_session": "<id>", "reason": "<一句话理由>"}\n'
    '{"verdict": "wait", "wait_on_pid": <int>, "reason": "<一句话理由>"}\n'
    '{"verdict": "wait", "wait_for_seconds": <int>, "reason": "<一句话理由>"}\n'
    '旧格式 {"done": <true|false>, "reason": "..."} 仍然兼容（true=done, false=continue）。'
)


# Rendered into the judge prompt when the agent has background processes
# running. Gives the judge the context it needs to decide WAIT vs CONTINUE
# (and which pid to wait on) without it having to probe anything itself.
JUDGE_BACKGROUND_BLOCK_TEMPLATE = (
    "Agent 当前运行的后台进程（可能在等待其中某一个）:\n{background_lines}\n\n"
)


JUDGE_USER_PROMPT_TEMPLATE = (
    "目标:\n{goal}\n\n"
    "Agent 最新响应:\n{response}\n\n"
    "{background_block}"
    "当前时间: {current_time}\n\n"
    "目标是否已完成 — done、continue 还是 wait？"
)

# Used when the user has added /subgoal criteria. The judge must
# evaluate ALL of them being met, not just the original goal.
JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "目标:\n{goal}\n\n"
    "用户追加的验收标准（全部满足才算 DONE）:\n{subgoals_block}\n\n"
    "Agent 最新响应:\n{response}\n\n"
    "{background_block}"
    "当前时间: {current_time}\n\n"
    "逐条检查：在响应中找到每条验收标准的具体满足证据（文件内容、命令输出、"
    "测试结果等），不接受「所有要求已满足」「看起来完成了」等笼统说法。"
    "如果任意一条标准缺少具体证据，判定为 CONTINUE（如果被后台进程阻塞则判 WAIT）。\n\n"
    "目标与所有验收标准是否全部完成 — done、continue 还是 wait？"
)


# Used when the goal carries a structured completion contract. The judge
# decides DONE strictly against the Verification criterion and refuses to
# accept completion when a constraint was violated.
JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "目标:\n{goal}\n\n"
    "完成合同（done 的权威定义）:\n"
    "{contract_block}\n\n"
    "Agent 最新响应:\n{response}\n\n"
    "{background_block}"
    "当前时间: {current_time}\n\n"
    "判定规则:\n"
    "- 仅当验证条件满足且响应中有具体证据（命令结果、文件内容、测试/基准输出）时，"
    "目标才算 DONE——不接受「已完成」「所有测试通过」等空洞声明。\n"
    "- 如果违反了任何约束条件，目标未完成 — CONTINUE。\n"
    "- 如果 agent 正在等待后台进程完成验证（如 CI 仍在运行），返回 WAIT——"
    "现在重新推进只会白忙活。\n"
    "- 如果响应表明工作受阻/无法完成/需要用户输入（如触发了停止条件），"
    "判定为 DONE 并说明阻塞原因。\n"
    "- 否则目标未完成 — CONTINUE。\n\n"
    "根据完成合同，目标是否已完成 — done、continue 还是 wait？"
)


# System prompt for /goal draft — turns a plain-language objective into a
# structured completion contract the user can review before activating.
# Adapted from Codex's "let Codex draft the goal" guidance.
DRAFT_CONTRACT_SYSTEM_PROMPT = (
    "You turn a user's plain-language objective into a structured completion "
    "contract for an autonomous coding agent. The contract has five fields:\n"
    "- outcome: the single end state that must be true when done\n"
    "- verification: the specific test / command / artifact that PROVES the "
    "outcome (must be concrete and checkable)\n"
    "- constraints: what must NOT change or regress\n"
    "- boundaries: which files, dirs, tools, or systems are in scope\n"
    "- stop_when: the condition under which the agent should stop and ask "
    "for human input instead of pushing on\n\n"
    "Infer sensible, specific values from the objective and any project "
    "context implied by it. Prefer concrete verification (a named test "
    "command, a build, a benchmark) over vague phrases. Keep each field to "
    "one or two sentences. If a field genuinely cannot be inferred, use an "
    "empty string for it.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"outcome": "...", "verification": "...", "constraints": "...", '
    '"boundaries": "...", "stop_when": "..."}'
)


# ──────────────────────────────────────────────────────────────────────
# Completion contract
# ──────────────────────────────────────────────────────────────────────

# The five contract fields, in display order. Adapted from OpenAI Codex's
# "strong goal" guidance: a durable objective works best when it names what
# "done" means, how to prove it, what must not regress, what tools/paths are
# in bounds, and when to stop and ask. A bare free-form goal (no contract)
# stays fully supported — every field defaults empty and is simply omitted
# from the prompts when unset.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

# Human labels for rendering and for the inline `field: value` parser.
_CONTRACT_LABELS = {
    "outcome": "Outcome",
    "verification": "Verification",
    "constraints": "Constraints",
    "boundaries": "Boundaries",
    "stop_when": "Stop when blocked",
}

# Inline-input aliases the user may type before a value, mapped to the
# canonical field name. e.g. `verify: tests pass` or `done when: ...`.
_CONTRACT_ALIASES = {
    "outcome": "outcome",
    "goal": "outcome",
    "done": "outcome",
    "done when": "outcome",
    "verification": "verification",
    "verify": "verification",
    "verified by": "verification",
    "evidence": "verification",
    "proof": "verification",
    "constraints": "constraints",
    "constraint": "constraints",
    "preserve": "constraints",
    "must not": "constraints",
    "do not change": "constraints",
    "boundaries": "boundaries",
    "boundary": "boundaries",
    "scope": "boundaries",
    "allowed": "boundaries",
    "files": "boundaries",
    "stop when": "stop_when",
    "stop_when": "stop_when",
    "blocked": "stop_when",
    "stop if blocked": "stop_when",
    "give up when": "stop_when",
}


@dataclass
class GoalContract:
    """Optional structured completion contract for a goal.

    Each field is free-form prose the user (or :func:`draft_contract`)
    supplies. Empty fields are omitted everywhere — a goal with no contract
    behaves exactly like the original free-form goal. The contract is woven
    into both the continuation prompt (so the agent targets the verification
    surface and respects constraints) and the judge prompt (so "done" is
    decided against evidence, not vibes).
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> Dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})

    def render_block(self) -> str:
        """Render non-empty contract fields as a labelled block. Empty
        contract → empty string (callers skip the section entirely)."""
        lines = []
        for f in _CONTRACT_FIELDS:
            val = getattr(self, f).strip()
            if val:
                lines.append(f"- {_CONTRACT_LABELS[f]}: {val}")
        return "\n".join(lines)


def parse_contract(text: str) -> Tuple[str, GoalContract]:
    """Split user-typed goal text into a headline + structured contract.

    Supports inline ``field: value`` lines so power users can type a full
    contract in one shot, e.g.::

        Migrate auth to JWT
        verify: the auth test suite passes
        constraints: keep the public /login response shape unchanged
        boundaries: only touch services/auth and its tests
        stop when: a schema change needs product sign-off

    The first non-field line(s) become the goal headline; recognized
    ``field:`` lines populate the contract. Lines for the same field are
    joined. Unrecognized prefixes stay part of the headline, so a plain
    free-form goal with an incidental colon (``Fix bug: the parser``)
    is NOT mangled — only lines whose prefix matches a known alias are
    pulled out. Returns ``(headline, contract)``.
    """
    if not text:
        return "", GoalContract()

    headline_parts: List[str] = []
    fields: Dict[str, List[str]] = {f: [] for f in _CONTRACT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        if ":" in line:
            prefix, _, value = line.partition(":")
            key = _CONTRACT_ALIASES.get(prefix.strip().lower())
            if key is not None and value.strip():
                fields[key].append(value.strip())
                matched = True
        if not matched:
            headline_parts.append(line)

    headline = " ".join(headline_parts).strip()
    contract = GoalContract(
        **{f: " ".join(v).strip() for f, v in fields.items()}
    )
    # If a headline was given but no explicit `outcome:` field, the headline
    # IS the outcome — don't duplicate it into the contract block (the goal
    # text already carries it), so leave outcome empty in that case.
    return headline, contract


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # User-added criteria appended mid-loop via the /subgoal command.
    # When non-empty the judge prompt and continuation prompt both
    # include them so the agent works toward them and the judge factors
    # them into the verdict. Backwards-compatible: defaults to empty so
    # old state_meta rows load unchanged.
    subgoals: List[str] = field(default_factory=list)
    # Wait barrier: when the agent is blocked on long-running async work
    # (CI poller, build, test run, deploy, rate-limit cooldown) the goal loop
    # PARKS instead of being re-poked every turn into busy-work. Two barrier
    # kinds, set automatically by the judge (which now sees the live
    # background-process list and can return a ``wait`` verdict) or manually
    # via ``/goal wait``:
    #   • ``waiting_on_pid`` — park until that process exits.
    #   • ``waiting_on_session`` — park until that process_registry session's
    #     OWN trigger fires: it exits, OR (if it has watch_patterns) its
    #     pattern matches. Covers long-lived watchers/servers that signal
    #     mid-run via a trigger and may never exit. Preferred over raw pid
    #     when the agent set up a watch_patterns/notify_on_complete process.
    #   • ``waiting_until``  — park until this wall-clock epoch (time backoff).
    # While ANY is active, ``evaluate_after_turn`` short-circuits to
    # should_continue=False without burning a turn or calling the judge. The
    # barrier auto-clears when the pid exits / the trigger fires / the deadline
    # passes, then the next turn resumes normal judging. Cleared by that,
    # ``/goal unwait``, pause, resume, or clear. Backwards-compatible: old
    # state_meta rows load with no barrier.
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    # Optional structured completion contract (outcome / verification /
    # constraints / boundaries / stop_when). Empty by default; a goal with
    # no contract behaves exactly like the original free-form goal.
    contract: GoalContract = field(default_factory=GoalContract)
    ontox_mode: bool = False  # /goal-ontox 模式——注入 OntoX 生态约束

    def to_json(self) -> str:
        data = asdict(self)
        # asdict already recursed GoalContract into a plain dict.
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            consecutive_parse_failures=int(data.get("consecutive_parse_failures", 0) or 0),
            subgoals=subgoals,
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_until=float(data.get("waiting_until", 0.0) or 0.0),
            waiting_reason=data.get("waiting_reason"),
            waiting_since=float(data.get("waiting_since", 0.0) or 0.0),
            contract=GoalContract.from_dict(data.get("contract")),
            ontox_mode=bool(data.get("ontox_mode", False)),
        )

    # --- contract helpers -------------------------------------------------

    def has_contract(self) -> bool:
        return self.contract is not None and not self.contract.is_empty()

    # --- subgoals helpers -------------------------------------------------

    def render_subgoals_block(self) -> str:
        """Render the subgoals as a numbered ``- N. text`` block. Empty
        when no subgoals exist."""
        if not self.subgoals:
            return ""
        return "\n".join(f"- {i}. {text}" for i, text in enumerate(self.subgoals, start=1))


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    SessionDB has no built-in singleton, but opening a new connection per
    /goal call would thrash the file. We cache one instance per
    ``hermes_home`` path so profile switches still pick up the right DB.
    Defensive against import/instantiation failures so tests and
    non-standard launchers can still use the GoalManager.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


def migrate_goal_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /goal from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session,
    but ``load_goal`` does a flat ``goal:<session_id>`` lookup with no
    parent-lineage walk — so an active goal silently dies at the
    compaction boundary (#33618). Copy the goal onto the new session and
    archive the old row as ``cleared`` so exactly one active goal row
    exists per logical conversation (avoids the "two active goals"
    hazard of a pure copy).

    Returns True when a goal was migrated, False when there was nothing
    to migrate or the DB was unavailable. Best-effort and never raises —
    a failure here must not block compression.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_goal(old_session_id)
        if state is None or getattr(state, "status", None) == "cleared":
            return False
        # Don't clobber a goal already set on the child (e.g. a resumed
        # lineage that re-established its own goal).
        if load_goal(new_session_id) is not None:
            return False
        save_goal(new_session_id, state)
        # Archive the parent's row so it isn't double-counted as active.
        clear_goal(old_session_id)
        logger.debug(
            "GoalManager: migrated goal %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("GoalManager: goal migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive.

    Delegates to ``gateway.status._pid_exists`` — the canonical,
    cross-platform, footgun-safe liveness check (psutil with a ctypes /
    POSIX fallback). Critically this avoids ``os.kill(pid, 0)``, which on
    Windows is NOT a no-op: it routes to ``CTRL_C_EVENT`` and hard-kills the
    target's console process group (bpo-14484). Any error resolves to False
    (treat unknown as dead) so a stale barrier never wedges the loop — the
    worst case is the goal resumes one turn early, which is safe.
    """
    if not pid or pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(int(pid)))
    except Exception:
        pass
    # Last-resort fallback if gateway.status is unavailable: psutil directly.
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _session_waiting(session_id: str) -> bool:
    """Whether a goal parked on a process_registry session should stay parked.

    Delegates to ``process_registry.is_session_waiting`` — True while the
    session is running and (if it has watch_patterns) its trigger hasn't fired.
    Fail-safe: any import/registry error yields False (don't wait) so a stale
    barrier can never wedge the loop.
    """
    if not session_id:
        return False
    try:
        from tools.process_registry import process_registry

        return bool(process_registry.is_session_waiting(session_id))
    except Exception:
        return False


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _goal_judge_max_tokens() -> int:
    """Resolve auxiliary.goal_judge.max_tokens, falling back to the default.

    ``load_config()`` is cached on the config file's (mtime, size), so calling
    this once per judge turn is cheap. A non-positive or non-int value falls
    back to the default rather than crashing the goal loop.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = (
            (cfg.get("auxiliary") or {})
            .get("goal_judge", {})
            .get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)
        )
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_JUDGE_MAX_TOKENS


def _parse_judge_response(raw: str) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Parse the judge's reply. Fail-open on unusable output.

    Returns ``(verdict, reason, parse_failed, wait_directive)`` where:
      - ``verdict`` is ``"done"``, ``"continue"``, or ``"wait"``.
      - ``parse_failed`` is True when the judge returned output that couldn't
        be interpreted as the expected JSON verdict (empty body, prose,
        malformed JSON). Callers use it to auto-pause after N consecutive
        parse failures so a weak judge model doesn't silently burn the budget.
      - ``wait_directive`` is set only for ``verdict == "wait"``: a dict with
        ``{"pid": int}`` or ``{"seconds": int}`` (whichever the judge supplied).
        ``None`` otherwise. If a wait verdict carries neither a usable pid nor
        seconds, it is downgraded to ``continue`` (can't park on nothing).

    Accepts both the new ``{"verdict": ...}`` shape and the legacy
    ``{"done": <bool>}`` shape.
    """
    if not raw:
        return "continue", "judge returned empty response", True, None

    text = raw.strip()

    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        # Peel off leading json/JSON/etc tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # First try: parse the whole blob.
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        # Second try: pull the first JSON object out.
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return "continue", f"judge reply was not JSON: {_truncate(raw, 200)!r}", True, None

    reason = str(data.get("reason") or "").strip() or "no reason provided"

    # Determine verdict — prefer the explicit "verdict" field, fall back to
    # the legacy "done" boolean.
    verdict_raw = data.get("verdict")
    if isinstance(verdict_raw, str):
        verdict = verdict_raw.strip().lower()
    else:
        done_val = data.get("done")
        if isinstance(done_val, str):
            done = done_val.strip().lower() in {"true", "yes", "1", "done"}
        else:
            done = bool(done_val)
        verdict = "done" if done else "continue"

    if verdict not in {"done", "continue", "wait"}:
        verdict = "continue"

    if verdict != "wait":
        return verdict, reason, False, None

    # Wait verdict: extract a concrete directive (pid or seconds). Accept a
    # few key spellings the model might emit.
    def _first_int(*keys: str) -> Optional[int]:
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            try:
                iv = int(v)
                if iv > 0:
                    return iv
            except (TypeError, ValueError):
                continue
        return None

    # Prefer a session-id directive (releases on the process's own trigger —
    # exit OR watch-pattern match), then pid (exit only), then seconds.
    sess = data.get("wait_on_session") or data.get("session_id") or data.get("wait_session")
    if isinstance(sess, str) and sess.strip():
        return "wait", reason, False, {"session_id": sess.strip()}
    pid = _first_int("wait_on_pid", "pid", "wait_pid")
    if pid is not None:
        return "wait", reason, False, {"pid": pid}
    seconds = _first_int("wait_for_seconds", "seconds", "wait_seconds")
    if seconds is not None:
        return "wait", reason, False, {"seconds": seconds}
    # Wait with no usable target — can't park on nothing; treat as continue.
    return "continue", f"{reason} (wait verdict had no target — continuing)", False, None


def _render_background_block(background_processes: Optional[List[Dict[str, Any]]]) -> str:
    """Render the live background-process list for the judge prompt.

    Each entry is a ``process_registry.list_sessions()`` dict. Only RUNNING
    processes are worth showing (an exited one is nothing to wait on). Returns
    an empty string when there's nothing running, so the judge prompt is
    byte-identical to the no-background case (no behavior change for the
    common path).
    """
    if not background_processes:
        return ""
    lines: List[str] = []
    for p in background_processes:
        if not isinstance(p, dict):
            continue
        if p.get("status") == "exited":
            continue
        pid = p.get("pid")
        if not pid:
            continue
        cmd = _truncate(str(p.get("command") or "").replace("\n", " ").strip(), 120)
        uptime = p.get("uptime_seconds")
        tail = _truncate(str(p.get("output_preview") or "").replace("\n", " ").strip(), 120)
        sid = p.get("session_id")
        line = f"- pid {pid}"
        if sid:
            line += f" / session {sid}"
        line += f": {cmd}"
        if uptime is not None:
            line += f" (running {uptime}s)"
        # Surface the process's own trigger so the judge can wait on a
        # mid-run signal (watch-pattern) or completion, not just exit.
        wps = p.get("watch_patterns")
        if wps:
            hit = " [already matched]" if p.get("watch_hit") else ""
            line += f" | watch_patterns={wps}{hit}"
        elif p.get("notify_on_complete"):
            line += " | notify_on_complete"
        if tail:
            line += f" | recent output: {tail}"
        lines.append(line)
    if not lines:
        return ""
    return JUDGE_BACKGROUND_BLOCK_TEMPLATE.format(background_lines="\n".join(lines))


def _pre_judge_confidence(last_response: str) -> Tuple[bool, str]:
    """扫描 Agent 响应中是否包含真实证据，返回 ``(有证据, 原因)``。

    在调 Judge LLM 之前用纯文本规则扫描，不依赖模型判断力。
    检测三类伪证据：
    - 全 ✅ 标记但无命令输出
    - 批量数字声明（N个/N项）但无抽样验证
    - "全绿""全部通过"等结论性文字但无测试输出片段
    """
    check_count = last_response.count('\u2705')  # ✅
    has_cmd_output = bool(re.search(  # noqa: 多模式正则匹配验证证据，工具本质需要
        r'(curl\s+\S+|grep\s+|npm\s+test|pytest|docker\s+exec|HTTP/\d|exit.*code|'
        r'rows?\s+\d+|FAILED|PASSED|Traceback|Error:.*\n)',
        last_response
    ))
    has_actual_data = bool(re.search(  # noqa: 多模式正则匹配验证证据，工具本质需要
        r'(http_status|status_code|elapsed|response\.json|'
        r'\"score\"\s*:\s*\d+|\"passed\"\s*:\s*(?:true|false)|'
        r'output_text|output_url|task_id.*qfwy)',
        last_response
    ))

    if check_count > 3 and not has_cmd_output and not has_actual_data:
        return False, (
            f"响应含 {check_count} 个 \u2705 标记但无命令输出或实际数据——疑似未验证声明"
        )

    # 检测 "全绿""全部通过""所有测试通过" 但无测试输出片段
    blanket_patterns = [r'全绿', r'全部通过', r'所有测试.*通过', r'all.*pass']
    for pat in blanket_patterns:
        if re.search(pat, last_response, re.IGNORECASE):  # noqa: 动态模式列表匹配，工具本质需要
            if not has_cmd_output and check_count >= 1:
                return False, '响应声称「全绿/全部通过」但无测试输出片段——证据不足'

    return True, ""


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    subgoals: Optional[List[str]] = None,
    background_processes: Optional[List[Dict[str, Any]]] = None,
    contract: Optional[GoalContract] = None,
    ontox_mode: bool = False,
) -> Tuple[str, str, bool, Optional[Dict[str, Any]]]:
    """Ask the auxiliary model whether the goal is satisfied.

    Returns ``(verdict, reason, parse_failed, wait_directive)`` where verdict
    is ``"done"``, ``"continue"``, ``"wait"``, or ``"skipped"`` (when the
    judge couldn't be reached). ``wait_directive`` is set only for ``"wait"``
    (``{"pid": int}`` or ``{"seconds": int}``); ``None`` otherwise.

    ``parse_failed`` is True only when the judge call succeeded but its output
    was unusable (empty or non-JSON). API/transport errors return False — they
    are transient and should fail-open silently. Callers use this flag to
    auto-pause after N consecutive parse failures (see
    ``DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES``).

    ``subgoals`` is an optional list of user-added criteria (from
    ``/subgoal``) factored into the verdict. ``background_processes`` is the
    live ``process_registry.list_sessions()`` snapshot; when the agent is
    waiting on one (a CI poller, build, etc.) the judge can return a ``wait``
    verdict naming its pid, parking the loop instead of re-poking.
    ``contract`` is an optional structured completion contract; when present
    the judge decides DONE strictly against its Verification criterion and
    refuses completion when a Constraint was violated. All three are additive
    — a contract, subgoals, and a background-process list can coexist in one
    judge prompt; when none are set, behavior is identical to the original
    free-form judge.

    This is deliberately fail-open: any error returns ``("continue", ..., False, None)``
    so a broken judge doesn't wedge progress — the turn budget and the
    consecutive-parse-failures auto-pause are the backstops.
    """
    if not goal.strip():
        return "skipped", "empty goal", False, None
    if not last_response.strip():
        # No substantive reply this turn — almost certainly not done yet.
        return "continue", "empty response (nothing to evaluate)", False, None

    try:
        from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return "continue", "auxiliary client unavailable", False, None

    if client is None or not model:
        return "continue", "no auxiliary client configured", False, None

    # Maker≠Grader 自动保护：主模型与 Judge 同厂商时，尝试切到 cross-review 厂商
    try:
        from agent.config_loader import get_active_model_config
        main_cfg = get_active_model_config()
        main_provider = (main_cfg.get("provider") or "").lower() if main_cfg else ""
        judge_provider = ""
        try:
            from hermes_config.runtime import get_config_snapshot
            snap = get_config_snapshot()
            aux = snap.get("auxiliary", {})
            judge_provider = (aux.get("goal_judge", {}).get("provider", "") or "").lower()
        except Exception as _e:
            logger.warning("goal: judge provider check failed: %s", _e, exc_info=True)
        if main_provider and judge_provider and main_provider == judge_provider:
            logger.warning(
                "Maker=Grader violation: main(%s) == judge(%s). "
                "Consider /model to a different provider or set goal_judge.provider differently.",
                main_provider, judge_provider
            )
    except Exception as _e:
        logger.warning("goal: Maker!=Grader check failed: %s", _e, exc_info=True)

    # Build the prompt. Priority: contract > subgoals > plain. When both a
    # contract and subgoals exist, the subgoals are appended into the
    # contract block as extra criteria so the judge sees a single source of
    # truth.
    clean_subgoals = [s.strip() for s in (subgoals or []) if s and s.strip()]
    background_block = _render_background_block(background_processes)
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    # ═══ OntoX 验证管道（在 Judge 之前运行） ═══
    pipeline_evidence = ""
    if ontox_mode and contract is not None and not contract.is_empty():
        try:
            _ensure_ontox_imports()
            if _ontox_imported:
                from ontox_verification_pipeline import run_verification_pipeline
                pipeline = run_verification_pipeline(
                    agent_response=last_response,
                    goal=goal,
                    contract_verification=contract.verification,
                    enable_cross_review=False,  # 默认关闭交叉审查，避免额外 API 调用
                    enable_build_test=True,
                    enable_e2e=True,
                )
                if not pipeline.all_passed:
                    logger.warning(
                        "OntoX pipeline FAILED (%d/%d gates): %s",
                        pipeline.passed_count, len(pipeline.gates),
                        [g.name for g in pipeline.failed_gates]
                    )
                    return (
                        "continue",
                        f"验证管道失败 ({pipeline.passed_count}/{len(pipeline.gates)}): "
                        + "; ".join(g.name for g in pipeline.failed_gates),
                        False, None
                    )
                pipeline_evidence = pipeline.evidence
                logger.info("OntoX pipeline PASSED: %d/%d gates", 
                           pipeline.passed_count, len(pipeline.gates))
        except Exception as e:
            logger.warning("OntoX pipeline error (fail-open): %s", e)

    if contract is not None and not contract.is_empty():
        contract_block = contract.render_block()
        if clean_subgoals:
            extra = "\n".join(
                f"- Extra criterion {i}: {text}"
                for i, text in enumerate(clean_subgoals, start=1)
            )
            contract_block = f"{contract_block}\n{extra}"
        # 如果管道通过，附加证据到 judge prompt
        if pipeline_evidence:
            contract_block = f"{contract_block}\n\n验证管道结果:\n{pipeline_evidence}"
        prompt = JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            contract_block=_truncate(contract_block, 2500),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    elif clean_subgoals:
        subgoals_block = "\n".join(
            f"- {i}. {text}" for i, text in enumerate(clean_subgoals, start=1)
        )
        prompt = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )
    else:
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            background_block=background_block,
            current_time=current_time,
        )

    # 预判 Agent 响应证据质量，低置信度时在 prompt 中注入警告
    # 不依赖 Judge 模型判断力——代码层直接扫描文本模式
    has_evidence, evidence_note = _pre_judge_confidence(last_response)
    if not has_evidence:
        prompt += (
            f"\n\n⚠️ 预检警告：Agent 响应的证据质量可疑——{evidence_note}。"
            "请严格审查，优先判定为 CONTINUE。"
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:
        logger.info("goal judge: API call failed (%s) — falling through to continue", exc)
        return "continue", f"judge error: {type(exc).__name__}", False, None

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    verdict, reason, parse_failed, wait_directive = _parse_judge_response(raw)
    logger.info(
        "goal judge: verdict=%s reason=%s%s",
        verdict, _truncate(reason, 120),
        f" wait={wait_directive}" if wait_directive else "",
    )
    return verdict, reason, parse_failed, wait_directive


def gather_background_processes(task_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the live background-process snapshot for the goal judge.

    Thin, fail-safe wrapper over ``process_registry.list_sessions(task_id)``.
    Returns only RUNNING processes (an exited one is nothing to wait on) and
    never raises — any import/registry failure yields ``[]`` so the goal loop
    degrades to its pre-wait-barrier behavior (judge just won't see processes).
    The drivers (CLI + gateway) call this and pass the result into
    ``GoalManager.evaluate_after_turn(background_processes=...)``.
    """
    try:
        from tools.process_registry import process_registry

        sessions = process_registry.list_sessions(task_id=task_id) or []
    except Exception as exc:
        logger.debug("gather_background_processes failed: %s", exc)
        return []
    return [s for s in sessions if isinstance(s, dict) and s.get("status") != "exited"]


def draft_contract(objective: str, *, timeout: float = DEFAULT_JUDGE_TIMEOUT) -> Optional[GoalContract]:
    """Expand a plain-language objective into a structured completion contract.

    Uses the ``goal_judge`` auxiliary task (main-model-first, cache-safe — it
    is a side LLM call, not a conversation turn). Returns a populated
    :class:`GoalContract` on success, or ``None`` when the auxiliary client is
    unavailable or the model's reply can't be parsed. Callers fall back to a
    bare free-form goal in that case, so a missing/weak aux model never blocks
    setting a goal.
    """
    objective = (objective or "").strip()
    if not objective:
        return None

    try:
        from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal draft: auxiliary client import failed: %s", exc)
        return None

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal draft: get_text_auxiliary_client failed: %s", exc)
        return None

    if client is None or not model:
        return None

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DRAFT_CONTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Objective:\n{_truncate(objective, 4000)}"},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
        )
    except Exception as exc:
        logger.info("goal draft: API call failed (%s)", exc)
        return None

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        logger.debug("goal draft: reply was not JSON: %r", _truncate(raw, 200))
        return None
    contract = GoalContract.from_dict(data)
    return None if contract.is_empty() else contract


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort: pull the first JSON object out of a model reply.

    Shares the fence-stripping + first-object fallback logic used by the
    judge parser, but returns the dict (or None) rather than a verdict.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None
    return data if isinstance(data, dict) else None


# ──────────────────────────────────────────────────────────────────────
# GoalManager — the orchestration surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


def _ensure_ontox_imports():
    """延迟导入 OntoX 模块——只在 /goal-ontox 使用时加载"""
    global _ontox_imported, _ontox_draft_contract, _ontox_preflight_check, _ontox_get_ecosystem
    if _ontox_imported:
        return True
    try:
        ontox_refs = os.path.expanduser(
            "~/.hermes/skills/autonomous-ai-agents/hermes-agent/references"
        )
        # 加入 references 目录自身（直接 import 文件名）
        if ontox_refs not in sys.path:
            sys.path.insert(0, ontox_refs)
        # 也加入父目录（支持 from references.xxx import 格式）
        ontox_parent = os.path.dirname(ontox_refs)
        if ontox_parent not in sys.path:
            sys.path.insert(0, ontox_parent)
        from ontox_capability_registry import get_ecosystem as _ge
        from ontox_contract_drafter import draft_contract as _dc, preflight_check as _pc
        _ontox_get_ecosystem = _ge
        _ontox_draft_contract = _dc
        _ontox_preflight_check = _pc
        _ontox_imported = True
        return True
    except ImportError as e:
        logger.warning("OntoX imports failed: %s — ontox_mode degraded to static boundary", e)
        return False


class GoalManager:
    """Per-session goal state + continuation decisions.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:

    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[GoalState] = load_goal(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def has_contract(self) -> bool:
        return self._state is not None and self._state.has_contract()

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared",}:
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        con = ", contract" if self.has_contract() else ""
        meta = f"{turns}{sub}{con}"
        if s.status == "active":
            if s.waiting_on_session and _session_waiting(s.waiting_on_session):
                wr = s.waiting_reason or f"session {s.waiting_on_session}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_on_pid and _pid_alive(s.waiting_on_pid):
                wr = s.waiting_reason or f"pid {s.waiting_on_pid}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_until and time.time() < s.waiting_until:
                remaining = int(s.waiting_until - time.time())
                wr = s.waiting_reason or f"{remaining}s"
                return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
            return f"⊙ Goal (active, {meta}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {meta}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({meta}): {s.goal}"
        return f"Goal ({s.status}, {meta}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None, contract: Optional[GoalContract] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        # 自动检测 OntoX 标记 → 激活生态约束模式
        # 支持: [ontox] / ontox: / @ontox / 【ontox】
        ontox_mode = False
        for prefix in ("[ontox]", "[OntoX]", "ontox:", "@ontox", "【ontox】"):
            if goal.lower().startswith(prefix.lower()):
                ontox_mode = True
                goal = goal[len(prefix):].strip()
                break
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            contract=contract if contract is not None else GoalContract(),
            ontox_mode=ontox_mode,
        )
        # [ontox] 前缀激活时，如果未显式传入 contract，自动起草合同
        if ontox_mode and (contract is None or contract.is_empty()):
            if _ensure_ontox_imports() and _ontox_draft_contract:
                try:
                    eco = _ontox_get_ecosystem()
                    draft = _ontox_draft_contract(goal, eco=eco)
                    state.contract = GoalContract(
                        outcome=draft.get("outcome", ""),
                        verification=draft.get("verification", ""),
                        constraints=draft.get("constraints", ""),
                        boundaries=draft.get("boundaries", ""),
                        stop_when=draft.get("stop_when", ""),
                    )
                except Exception as e:
                    logger.warning("OntoX auto-contract draft failed: %s", e)
        self._state = state
        save_goal(self.session_id, state)
        return state

    def set_ontox(self, goal: str, *, max_turns: Optional[int] = None) -> GoalState:
        """Set a goal with OntoX ecosystem constraints enabled.

        Automatically drafts a GoalContract by analyzing the goal against
        the OntoX capability registry, generating concrete verification
        commands. If contract drafting fails, falls back to static boundary.
        """
        # 尝试智能起草合同
        contract = GoalContract()
        preflight_result = None
        if _ensure_ontox_imports() and _ontox_draft_contract:
            try:
                eco = _ontox_get_ecosystem()
                preflight_result = _ontox_preflight_check(goal, eco=eco)
                draft = _ontox_draft_contract(goal, eco=eco)
                contract = GoalContract(
                    outcome=draft.get("outcome", ""),
                    verification=draft.get("verification", ""),
                    constraints=draft.get("constraints", ""),
                    boundaries=draft.get("boundaries", ""),
                    stop_when=draft.get("stop_when", ""),
                )
                if draft.get("warnings"):
                    logger.info("OntoX contract warnings: %s", draft["warnings"])
            except Exception as e:
                logger.warning("OntoX contract draft failed: %s", e)

        state = self.set(goal, max_turns=max_turns, contract=contract)
        state.ontox_mode = True
        save_goal(self.session_id, state)
        return state

    def ontox_preflight(self, goal: str) -> Optional[dict]:
        """运行预检——分析目标能否通过 OntoX 实现"""
        if _ensure_ontox_imports() and _ontox_preflight_check:
            return _ontox_preflight_check(goal)
        return None

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        """Attach or replace the completion contract on the active goal.

        Returns the updated state, or None when there is no goal to attach to.
        """
        if self._state is None:
            return None
        self._state.contract = contract or GoalContract()
        save_goal(self.session_id, self._state)
        return self._state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        # A wait barrier is meaningless once paused — drop it.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        # Resuming starts fresh — clear any stale barrier.
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> str:
        """Append a user-added criterion to the active goal. Requires
        ``has_goal()``; raises ``RuntimeError`` otherwise.

        Returns the cleaned text so the caller can show it back to the user.
        """
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        self._state.subgoals.append(text)
        save_goal(self.session_id, self._state)
        return text

    def remove_subgoal(self, index_1based: int) -> str:
        """Remove a subgoal by 1-based index. Returns the removed text."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.subgoals):
            raise IndexError(
                f"index out of range (1..{len(self._state.subgoals)})"
            )
        removed = self._state.subgoals.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_subgoals(self) -> int:
        """Wipe all subgoals. Returns the previous count."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.subgoals)
        self._state.subgoals = []
        save_goal(self.session_id, self._state)
        return prev

    def render_subgoals(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- /goal wait barrier -------------------------------------------

    def wait_on(self, pid: int, reason: str = "") -> GoalState:
        """Park the goal loop on a background process PID.

        While the PID is alive, ``evaluate_after_turn`` returns
        ``should_continue=False`` without burning a turn or calling the
        judge — the loop quiesces instead of re-poking the agent into busy
        work. The barrier auto-clears when the process exits. Requires an
        active goal. For a process with a watch_patterns/notify_on_complete
        trigger, prefer ``wait_on_session`` so a mid-run trigger (not just
        exit) releases the barrier.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")
        self._state.waiting_on_pid = pid
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_on_session(self, session_id: str, reason: str = "") -> GoalState:
        """Park the goal loop on a process_registry session's OWN trigger.

        Unlike ``wait_on`` (which releases only on PID exit), this releases
        when the session's trigger fires: it exits, OR — if it was started
        with ``watch_patterns`` — its pattern matches. This is the right
        barrier for a long-lived watcher/server/poller that signals mid-run
        and may never exit. Requires an active goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string")
        self._state.waiting_on_session = session_id
        self._state.waiting_on_pid = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def wait_for_seconds(self, seconds: int, reason: str = "") -> GoalState:
        """Park the goal loop until ``seconds`` from now have elapsed.

        Time-based counterpart to ``wait_on`` — for backoff / cooldown waits
        where there's no process to track (e.g. the agent is rate-limited).
        The barrier auto-clears once the deadline passes. Requires an active
        goal.
        """
        if self._state is None or self._state.status != "active":
            raise RuntimeError("no active goal to park")
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = time.time() + seconds
        self._state.waiting_reason = (reason or "").strip() or None
        self._state.waiting_since = time.time()
        save_goal(self.session_id, self._state)
        return self._state

    def stop_waiting(self) -> bool:
        """Clear any active wait barrier (pid / session / time). Returns True
        if one was cleared."""
        if self._state is None:
            return False
        if (
            self._state.waiting_on_pid is None
            and self._state.waiting_on_session is None
            and not self._state.waiting_until
        ):
            return False
        self._state.waiting_on_pid = None
        self._state.waiting_on_session = None
        self._state.waiting_until = 0.0
        self._state.waiting_reason = None
        self._state.waiting_since = 0.0
        save_goal(self.session_id, self._state)
        return True

    def is_waiting(self) -> bool:
        """True iff a barrier is set AND not yet satisfied.

        Session barrier: active until the process exits or its watch-pattern
        trigger fires. Pid barrier: active while the process is alive. Time
        barrier: active until the deadline passes. Side effect: a satisfied
        barrier is cleared here (lazy auto-clear) so the next evaluation
        resumes normal judging.
        """
        s = self._state
        if s is None:
            return False
        if s.waiting_on_session is not None:
            if _session_waiting(s.waiting_on_session):
                return True
            self.stop_waiting()  # session exited or trigger fired
            return False
        if s.waiting_on_pid is not None:
            if _pid_alive(s.waiting_on_pid):
                return True
            self.stop_waiting()  # process gone
            return False
        if s.waiting_until:
            if time.time() < s.waiting_until:
                return True
            self.stop_waiting()  # deadline passed
            return False
        return False

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        ``background_processes`` is the live ``process_registry.list_sessions()``
        snapshot for this session. It's handed to the judge so it can decide
        to WAIT on an in-flight process (CI poller, build, ...) instead of
        re-poking the agent — the automatic counterpart to ``/goal wait``.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "wait" | "skipped" | "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        # Wait barrier: if the loop is parked (on a live process OR a time
        # deadline that hasn't passed), quiesce — do NOT burn a turn or call
        # the judge. Resumes automatically once the barrier clears.
        if self.is_waiting():
            if state.waiting_on_session is not None:
                tgt = f"session {state.waiting_on_session}"
            elif state.waiting_on_pid is not None:
                tgt = f"pid {state.waiting_on_pid}"
            else:
                remaining = max(0, int(state.waiting_until - time.time()))
                tgt = f"{remaining}s remaining"
            reason = state.waiting_reason or tgt
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "waiting",
                "reason": reason,
                "message": f"⏳ Goal parked — waiting on {tgt}: {reason}",
            }

        # Count the turn that just finished.
        state.turns_used += 1
        state.last_turn_at = time.time()

        verdict, reason, parse_failed, wait_directive = judge_goal(
            state.goal,
            last_response,
            subgoals=state.subgoals or None,
            background_processes=background_processes,
            contract=state.contract if state.has_contract() else None,
            ontox_mode=state.ontox_mode,
        )
        state.last_verdict = verdict
        state.last_reason = reason

        # Track consecutive judge parse failures. Reset on any usable reply,
        # including API / transport errors (parse_failed=False) so a flaky
        # network doesn't trip the auto-pause meant for bad judge models.
        if parse_failed:
            state.consecutive_parse_failures += 1
        else:
            state.consecutive_parse_failures = 0

        # WAIT verdict: the judge decided the agent is blocked on async work
        # and re-poking now would be busy-work. Set the barrier and park —
        # the turn we just counted stands (the judge call happened), but no
        # continuation fires. The loop resumes automatically when the pid
        # exits or the deadline passes (next evaluate_after_turn falls through
        # the is_waiting() short-circuit once the barrier clears).
        if verdict == "wait" and wait_directive:
            if wait_directive.get("session_id"):
                self.wait_on_session(str(wait_directive["session_id"]), reason=reason)
                tgt = f"session {wait_directive['session_id']}"
            elif wait_directive.get("pid"):
                self.wait_on(int(wait_directive["pid"]), reason=reason)
                tgt = f"pid {wait_directive['pid']}"
            else:
                self.wait_for_seconds(int(wait_directive["seconds"]), reason=reason)
                tgt = f"{wait_directive['seconds']}s"
            return {
                "status": "active",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "wait",
                "reason": reason,
                "message": f"⏳ Goal parked (judge) — waiting on {tgt}: {reason}",
            }

        if verdict == "done":
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        # Auto-pause when the judge model can't produce the expected JSON
        # verdict N turns in a row. Points the user at the goal_judge config
        # so they can route this side task to a model that follows the
        # contract (e.g. google/gemini-3-flash-preview). Without this guard,
        # weak judge models burn the entire turn budget returning prose or
        # empty strings.
        if state.consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge model returned unparseable output {state.consecutive_parse_failures} turns in a row"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — the judge model ({state.consecutive_parse_failures} turns) "
                    "isn't returning the required JSON verdict. Route the judge to a stricter "
                    "model in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: openrouter\n"
                    "      model: google/gemini-3-flash-preview\n"
                    "Then /goal resume to continue."
                ),
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        save_goal(self.session_id, state)
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        # Contract takes priority: it carries the verification surface and
        # constraints the agent must target. Subgoals fold in as extra
        # criteria appended to the contract block.
        if self._state.ontox_mode:
            # OntoX 模式：智能合同优先，静态边界降级
            if self._state.has_contract():
                contract_block = self._state.contract.render_block()
                # 附加能力注册表摘要
                if _ensure_ontox_imports() and _ontox_get_ecosystem:
                    try:
                        eco = _ontox_get_ecosystem()
                        running = [s.name for s in eco.services if s.running]
                        contract_block += (
                            f"\\n当前运行服务: {', '.join(running) if running else '仅PG+Redis'}"
                        )
                    except Exception as _e:
                        logger.warning("goal: running services detection failed: %s", _e, exc_info=True)
                if self._state.subgoals:
                    extra = "\\n".join(
                        f"- Extra criterion {i}: {text}"
                        for i, text in enumerate(self._state.subgoals, start=1)
                    )
                    contract_block = f"{contract_block}\\n{extra}"
                return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                    goal=self._state.goal,
                    contract_block=contract_block,
                )
            # 降级：用静态边界
            ontox_goal = f"{self._state.goal}\\n\\n{ONTOX_ECOSYSTEM_BOUNDARY}"
            if self._state.subgoals:
                return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                    goal=ontox_goal,
                    subgoals_block=self._state.render_subgoals_block(),
                )
            return CONTINUATION_PROMPT_TEMPLATE.format(goal=ontox_goal)

        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        if self._state.subgoals:
            return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)

    def render_contract(self) -> str:
        """Public helper for the /goal show + /goal draft slash commands."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.has_contract():
            return "(no completion contract — set one with /goal draft <objective> or inline field: value lines)"
        return self._state.contract.render_block()


# ──────────────────────────────────────────────────────────────────────
# Kanban worker goal loop
# ──────────────────────────────────────────────────────────────────────

# Continuation prompt fed back to a kanban goal-mode worker that has not
# yet completed/blocked its task. The card's own acceptance criteria are
# the goal — the worker already has the full task body in its first turn,
# so we keep this short and point it back at the lifecycle contract.
KANBAN_GOAL_CONTINUATION_TEMPLATE = (
    "[Continuing toward this kanban task — judge says it is not done yet]\n"
    "Reason: {reason}\n\n"
    "Take the next concrete step toward completing the task. When the work "
    "is genuinely finished, call kanban_complete with a summary. If you are "
    "blocked and need human input, call kanban_block with a reason. Do not "
    "stop without calling one of them."
)

# Fed when the judge believes the work is done but the worker never called
# kanban_complete / kanban_block. One explicit nudge to terminate the task
# the right way before the loop gives up.
KANBAN_GOAL_FINALIZE_TEMPLATE = (
    "[The work looks complete, but the task is still open]\n"
    "Reason: {reason}\n\n"
    "If the task is genuinely done, call kanban_complete now with a short "
    "summary of what you did. If something still blocks completion, call "
    "kanban_block with the reason instead."
)


def run_kanban_goal_loop(
    *,
    task_id: str,
    goal_text: str,
    run_turn,
    task_status_fn,
    block_fn,
    max_turns: int = DEFAULT_MAX_TURNS,
    first_response: str = "",
    log=None,
) -> Dict[str, Any]:
    """Drive a kanban worker through a Ralph-style goal loop.

    The dispatcher spawns a goal-mode worker exactly like a normal worker
    (``hermes -p <profile> chat -q "work kanban task <id>"``). The worker's
    first turn has already run by the time this is called; ``first_response``
    is that turn's reply. From here we:

    1. Check whether the worker already terminated the task (called
       ``kanban_complete`` / ``kanban_block``). If so, stop — nothing to do.
    2. Otherwise judge the latest response against ``goal_text`` (the card's
       title + body). ``continue`` → feed a continuation prompt and run
       another turn IN THE SAME SESSION via ``run_turn``. ``done`` but the
       task is still open → one explicit "call kanban_complete" nudge.
    3. When the turn budget is exhausted and the worker still hasn't
       terminated the task, ``block_fn`` is invoked so the card lands in a
       sticky ``blocked`` state for human review (NOT a silent exit).

    This function performs NO SessionDB persistence — a worker process is
    ephemeral, so the turn budget lives in a local counter. It is fully
    decoupled from the CLI for testability: callers inject ``run_turn``
    (str -> str), ``task_status_fn`` (() -> str|None), and ``block_fn``
    (reason: str -> None).

    Returns a decision dict: ``{"outcome", "turns_used", "reason"}`` where
    outcome is one of ``"completed_by_worker"``, ``"blocked_budget"``,
    ``"blocked_by_worker"``, or ``"stopped"``.
    """

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    max_turns = int(max_turns or DEFAULT_MAX_TURNS)
    if max_turns < 1:
        max_turns = DEFAULT_MAX_TURNS

    last_response = first_response or ""
    # The first turn already consumed one unit of budget.
    turns_used = 1
    nudged_to_finalize = False

    while True:
        # Did the worker terminate the task itself this turn?
        try:
            status = task_status_fn()
        except Exception as exc:
            _log(f"kanban goal loop: status check failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": "status check failed"}

        if status == "done":
            _log(f"kanban goal loop: task {task_id} completed by worker after {turns_used} turn(s)")
            return {"outcome": "completed_by_worker", "turns_used": turns_used, "reason": "worker completed the task"}
        if status == "blocked":
            _log(f"kanban goal loop: task {task_id} blocked by worker after {turns_used} turn(s)")
            return {"outcome": "blocked_by_worker", "turns_used": turns_used, "reason": "worker blocked the task"}
        if status not in ("running", "ready"):
            # Reclaimed / archived / unexpected — let the dispatcher own it.
            _log(f"kanban goal loop: task {task_id} status={status!r}; stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"status={status}"}

        # Still open — judge whether the latest response satisfies the card.
        # The kanban worker loop has no wait-barrier concept (workers finish
        # via kanban_complete / kanban_block, not by parking), so a WAIT
        # verdict is treated as CONTINUE here.
        verdict, reason, _parse_failed, _wait = judge_goal(goal_text, last_response)
        if verdict == "wait":
            verdict = "continue"
        _log(f"kanban goal loop: turn {turns_used}/{max_turns} verdict={verdict} reason={_truncate(reason, 120)}")

        if verdict == "done":
            if nudged_to_finalize:
                # Already asked once to call kanban_complete and it still
                # didn't — block for review rather than spin.
                _log(f"kanban goal loop: task {task_id} judged done but worker won't finalize; blocking")
                try:
                    block_fn(
                        f"Goal-mode worker's output looked complete but it never "
                        f"called kanban_complete after a finalize nudge ({reason})."
                    )
                except Exception as exc:
                    _log(f"kanban goal loop: block_fn failed ({exc})")
                return {"outcome": "blocked_budget", "turns_used": turns_used, "reason": "judged done, never finalized"}
            prompt = KANBAN_GOAL_FINALIZE_TEMPLATE.format(reason=_truncate(reason, 400))
            nudged_to_finalize = True
        else:
            prompt = KANBAN_GOAL_CONTINUATION_TEMPLATE.format(reason=_truncate(reason, 400))

        # Budget check BEFORE spending another turn.
        if turns_used >= max_turns:
            _log(f"kanban goal loop: task {task_id} exhausted {turns_used}/{max_turns} turns; blocking")
            try:
                block_fn(
                    f"Goal-mode worker exhausted its turn budget "
                    f"({turns_used}/{max_turns}) without completing the task. "
                    f"Last judge verdict: {_truncate(reason, 300)}"
                )
            except Exception as exc:
                _log(f"kanban goal loop: block_fn failed ({exc})")
            return {"outcome": "blocked_budget", "turns_used": turns_used, "reason": "turn budget exhausted"}

        # Run another turn in the same session.
        try:
            last_response = run_turn(prompt) or ""
        except Exception as exc:
            _log(f"kanban goal loop: run_turn failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"run_turn error: {type(exc).__name__}"}
        turns_used += 1


__all__ = [
    "GoalState",
    "GoalContract",
    "GoalManager",
    "parse_contract",
    "draft_contract",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE",
    "JUDGE_USER_PROMPT_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_CONTRACT_TEMPLATE",
    "DRAFT_CONTRACT_SYSTEM_PROMPT",
    "KANBAN_GOAL_CONTINUATION_TEMPLATE",
    "KANBAN_GOAL_FINALIZE_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "load_goal",
    "save_goal",
    "clear_goal",
    "migrate_goal_to_session",
    "judge_goal",
    "run_kanban_goal_loop",
]
