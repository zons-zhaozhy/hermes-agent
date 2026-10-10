"""terminology-guard — 术语一致性守护（治 LLM 用词漂移）。

三层机制，全部零核心改动：
  L0 单一事实源：~/.hermes/terminology.yaml（canonical 规范名 + 禁用别名 +
     语体注记）。治理一个术语 = 加一行数据，不改代码。
  L1 每轮锚定：pre_llm_call 每轮注入紧凑术语表（字符预算封顶）——注入在
     上下文压缩后依然存在，补上"早期用词被压缩打掉"的根因。
  L3 双通道收口：transform_llm_output 对回复做确定性别名字面量检测
     （机器消费的确定性文本表，str 方法匹配合规）+ judge 判开放集称谓混用；
     命中不改用户可见文本，下一轮注入针对性纠正提醒。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from plugins._llm_judge import llm_judge_json, llm_judge_multi
from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "terminology_guard"
_MAX_JUDGE_CALLS = 40
# 每轮注入预算：中文 ~0.65 token/字符，2000 字符 ≈ 1.3k token/轮；pre_llm_call
# 注入只进当轮请求不进历史，60+ 条术语可全覆盖（YAML 顺序=注入优先级）
_INJECT_BUDGET = 2000

# ASCII 别名走整词比对；_ 与 . 保留在词内，代码标识符（ontox_protocols 等）不被拆词误伤
_SPLIT_CHARS = ",;:()[]{}<>\"'`|=+*&^%$#@!?~\n\t/\\-"

# 进程级缓存：mtime 未变不重读盘
_GLOSSARY_CACHE: dict[str, Any] = {"mtime": None, "entries": []}

_INJECT_HEAD = "【术语一致性】本会话术语唯一规范写法（禁用所列变体，含同义词/旧写法/混称）："

_TENSE_RULE = (
    "时态纪律：状态描述与事实一致——已完成用「已/完成」，进行中用「正在」，"
    "计划中用「将/计划」；同一任务的状态表述前后一致，禁在未新证据时翻转。"
)

_JUDGE_SYSTEM = (
    "你是术语一致性审查员。判定下面这段 AI 回复是否存在'同一概念混用不同"
    "称谓/不同词形'——同义词互换（如'名单筛查'与'名单扫描'指同一操作）、"
    "同一对象中英混称、或同一名词形态不一致。以下不算混用：引用代码标识符、"
    "文件路径、引号内的原文引用、对不同事物的合理区分表述。"
    '只答 JSON：{"naming_drift": true} 或 {"naming_drift": false}'
)


def _plugin_disabled() -> bool:
    return os.environ.get("TERMINOLOGY_GUARD_DISABLE") == "1"


def _state(sid: str) -> dict[str, Any]:
    return get_session_state(sid, _NAMESPACE)


def _load_glossary() -> list[dict[str, Any]]:
    """读术语 SSOT；文件 mtime 未变走缓存；任何失败 fail-open 按空表。

    Contract:
      Postconditions: 返回 list[dict]（可能为空）；绝不 raise；
                      条目均含非空 canonical 与非空 aliases 列表
    """
    try:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "terminology.yaml"
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime == _GLOSSARY_CACHE["mtime"]:
            return _GLOSSARY_CACHE["entries"]
        entries: list[dict[str, Any]] = []
        if path.exists():
            import yaml

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            # aliases 为空列表合法（纯规范名锚定，无禁用变体）；只拦结构残缺
            entries = [
                e
                for e in (raw.get("terms") or [])
                if isinstance(e, dict)
                and e.get("canonical")
                and isinstance(e.get("aliases"), list)
            ]
        _GLOSSARY_CACHE.update(mtime=mtime, entries=entries)
        return entries
    except Exception as exc:
        logger.warning(
            "terminology-guard: 术语表加载失败，按上次结果/空表处理: %s", exc, exc_info=True
        )
        return _GLOSSARY_CACHE["entries"]


def _tokens(text: str) -> set[str]:
    """小写 + 把标点/空白换成空格后切词；_ 与 . 留在词内护住标识符。

    Contract:
      Postconditions: 返回小写 token 集合，不含空串
    """
    flat = text.lower()
    for ch in _SPLIT_CHARS:
        flat = flat.replace(ch, " ")
    return {tok for tok in flat.split() if tok}


def detect_alias_drift(
    text: str, entries: list[dict[str, Any]]
) -> list[tuple[str, str]]:
    """确定性别名检测：CJK 别名=子串包含；ASCII 别名=整词比对。

    Contract:
      Preconditions: text 为 str（可为空）；entries 为 _load_glossary 产物
      Postconditions: 返回 [(alias, canonical), ...]（保持条目顺序）；绝不 raise
    """
    lowered = text.lower()
    token_set = _tokens(text)
    hits: list[tuple[str, str]] = []
    for entry in entries:
        canonical = str(entry["canonical"])
        for raw_alias in entry.get("aliases") or []:
            alias = str(raw_alias).strip()
            if not alias:
                continue
            low = alias.lower()
            matched = (low in lowered) if not low.isascii() else (low in token_set)
            if matched:
                hits.append((alias, canonical))
    return hits


def _judge_naming_drift(text: str) -> Optional[bool]:
    """judge 判开放集称谓混用（表内别名之外的语义漂移）；None=fail-open。

    Contract:
      Preconditions: text 为非空 str
      Postconditions: 返回 True/False/None；绝不 raise
    """
    if len(text) < 20:
        return False
    result = llm_judge_multi(
        task="terminology_drift",
        system=_JUDGE_SYSTEM,
        text=text,
        keys=["naming_drift"],
    )
    return result.get("naming_drift")


def _format_entry(entry: dict[str, Any]) -> str:
    """单条术语 → 注入行（note 是给人看的编辑注记，不进注入省预算）。

    Contract: Postconditions: 返回非空 str。
    """
    aliases = "/".join(str(a) for a in entry.get("aliases") or [])
    return f"- {entry['canonical']}（禁用：{aliases}）"


def _budgeted_terms(entries: list[dict[str, Any]]) -> list[str]:
    """预算内尽量多放术语行。Contract: Postconditions: 各行总长 ≤ _INJECT_BUDGET。"""
    pieces: list[str] = []
    used = len(_INJECT_HEAD)
    for entry in entries:
        piece = _format_entry(entry)
        if used + len(piece) + 1 > _INJECT_BUDGET:
            break
        pieces.append(piece)
        used += len(piece) + 1
    return pieces


def build_injection(
    entries: list[dict[str, Any]],
    drifts: list[tuple[str, str]],
    open_drift: bool,
    ledger: Optional[list[str]] = None,
) -> Optional[str]:
    """组装注入文本：术语表（预算内）+ 漂移纠正行 + 会话账本行 + 时态纪律行。

    Contract:
      Preconditions: entries/drifts 为上序产物；open_drift 为 bool；ledger 为账本
      Postconditions: 四者皆空 → None；否则返回非空 str
    """
    if not entries and not drifts and not open_drift and not ledger:
        return None
    lines = [_INJECT_HEAD] + _budgeted_terms(entries)
    if drifts:
        detail = "；".join(f"「{a}」→ 用「{c}」" for a, c in drifts[:5])
        lines.append(f"你上一条回复称谓漂移：{detail}。本轮起统一用规范写法。")
    elif open_drift:
        lines.append("你上一条回复存在同一概念混用不同称谓/词形。本轮统一用词，同一概念前后同一写法。")
    ledger_text = _ledger_line(ledger or [])
    if ledger_text:
        lines.append(ledger_text)
    lines.append(_TENSE_RULE)
    return "\n".join(lines)


def _collect_reply(response_text: str, sid: str, st: dict[str, Any]) -> None:
    """别名检测 + judge 判定 + 会话用词账本记账 → 写会话状态。

    Contract: Postconditions: 绝不 raise。
    """
    drifts = detect_alias_drift(response_text, _load_glossary())
    if drifts:
        st["pending_drifts"] = drifts
        logger.info("terminology-guard: 别名漂移已标记（下轮注入纠正）: %s", drifts)
    if int(st.get("judge_calls", 0)) < _MAX_JUDGE_CALLS:
        st["judge_calls"] = int(st.get("judge_calls", 0)) + 1
        if _judge_naming_drift(response_text):
            st["open_drift"] = True
            logger.info("terminology-guard: 开放集称谓漂移已标记（下轮注入纠正）")
        _update_ledger(response_text, st)


_LEDGER_SYSTEM = (
    "你是会话用词审查员。从下面这段 AI 回复中抽取'本会话确立的关键用词'——"
    "领域术语、组件名、操作名、指标名等会贯穿后续对话的名词。忽略：代词、"
    "通用词（问题/系统/数据/文件）、代码标识符、只出现一次的临时引用。"
    "只回答一个 JSON 对象，含全部键，值为字符串数组："
    '{"key_terms": ["术语1", "术语2"]}'
)
_LEDGER_CAP = 24
_LEDGER_BUDGET = 300


def _update_ledger(response_text: str, st: dict[str, Any]) -> None:
    """抽取本轮关键用词 → 并入会话账本（首见即锁，上限截断）。

    L2 会话用词账本：开放集漂移的根治件——会话内首次用词即成为契约，
    每轮注入「已确立用词」，把开放集问题在会话内转化为封闭集。

    Contract:
      Preconditions: response_text 为 str；st 为会话状态 dict
      Postconditions: st["ledger"] 为 list[str] 且长度 ≤ _LEDGER_CAP；绝不 raise
    """
    if len(response_text) < 40:
        return
    result = llm_judge_json(
        task="terminology_ledger",
        system=_LEDGER_SYSTEM,
        text=response_text,
        keys=["key_terms"],
    )
    fresh = result.get("key_terms") or []
    if not fresh:
        return
    ledger: list[str] = list(st.get("ledger") or [])
    for term in fresh:
        if term not in ledger:
            ledger.append(term)
    st["ledger"] = ledger[:_LEDGER_CAP]


def _ledger_line(ledger: list[str]) -> Optional[str]:
    """账本 → 注入行（300 字符预算；空账本 → None）。

    Contract:
      Postconditions: 返回 None 或非空 str；行长 ≤ _LEDGER_BUDGET + 表头
    """
    if not ledger:
        return None
    head = "本会话已确立用词（首次使用即锁，后文同一概念一律沿用该写法，禁同义改写）："
    used = len(head)
    kept: list[str] = []
    for term in ledger:
        if used + len(term) + 3 > _LEDGER_BUDGET + len(head):
            break
        kept.append(term)
        used += len(term) + 3
    if not kept:
        return None
    return head + "、".join(kept)


def register(ctx: Any) -> None:
    """注册 transform_llm_output（检测+记账）与 pre_llm_call（每轮注入）。

    Contract:
      Postconditions: 两钩子已注册；钩子内部异常不外泄（fail-open）
    """

    def check_reply(
        response_text: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> Optional[str]:
        """检测 → 记状态供下轮注入；用户可见回复零改动。

        Contract:
          Postconditions: 恒返回 None；异常不外泄
        """
        if _plugin_disabled():
            return None
        try:
            sid = session_id or ""
            _collect_reply(response_text or "", sid, _state(sid))
            return None
        except Exception as exc:  # fail-open
            logger.warning("terminology-guard check failed: %s", exc, exc_info=True)
            return None

    def inject_terminology(**kwargs: Any) -> Optional[dict[str, Any]]:
        """每轮注入术语表锚定 + 消费上轮漂移标记。

        Contract:
          Postconditions: 返回 None 或 {"context": str}；漂移标记注入一次即消费
        """
        if _plugin_disabled():
            return None
        try:
            sid = kwargs.get("session_id") or ""
            st = _state(sid)
            drifts = st.pop("pending_drifts", []) or []
            open_drift = bool(st.pop("open_drift", False))
            ledger = st.get("ledger") or []
            text = build_injection(_load_glossary(), drifts, open_drift, ledger)
            return {"context": text} if text else None
        except Exception as exc:  # fail-open
            logger.warning("terminology-guard inject failed: %s", exc, exc_info=True)
            return None

    ctx.register_hook("transform_llm_output", check_reply)
    ctx.register_hook("pre_llm_call", inject_terminology)
