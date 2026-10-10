"""terminology-guard — 术语一致性守护（治 LLM 用词漂移）。

理论基础与三层结构（对齐记忆科学 standard model of memory consolidation,
McGaugh 2000; 词汇锚定见 Hoey, Lexical Priming, 2005; 晋升判定见 W-TinyLFU
admission control, Einziger et al. 2017）:

  策展层 curated lexicon（~/.hermes/terminology.yaml）
    人工维护的规范名+禁用别名 SSOT。每次注入全部锚定（预算内）。
  情景层 session working lexicon（会话内）
    每轮回复抽取关键用词，首见即锁——把开放集问题在会话内转化为封闭集。
  语义层 consolidated lexicon（~/.hermes/terminology_consolidated.yaml）
    跨会话 admission：词出现于 ≥3 个不同会话（持久化计数，进程重启不丢）
    即固化为全局锚定词；last_seen 超期标 stale 不再注入（老化衰减）。

  每轮注入（pre_llm_call）三层合并锚定 + 时态纪律；transform_llm_output
  做确定性别名检测 + judge 开放集判定，命中不改用户可见文本，下轮注入纠正。
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
        _update_ledger(response_text, sid, st)


_LEDGER_SYSTEM = (
    "你是会话用词审查员。从下面这段 AI 回复中抽取'本会话确立的关键用词'——"
    "领域术语、组件名、操作名、指标名等会贯穿后续对话的名词。忽略：代词、"
    "通用词（问题/系统/数据/文件）、代码标识符、只出现一次的临时引用。"
    "只回答一个 JSON 对象，含全部键，值为字符串数组："
    '{"key_terms": ["术语1", "术语2"]}'
)
_LEDGER_CAP = 24
_LEDGER_BUDGET = 300


def _update_ledger(response_text: str, sid: str, st: dict[str, Any]) -> None:
    """抽取本轮关键用词 → 并入会话账本（首见即锁，上限截断）→ 毕业检查。

    L2 会话用词账本：开放集漂移的根治件——会话内首次用词即成为契约，
    每轮注入「已确立用词」，把开放集问题在会话内转化为封闭集。
    跨会话固化（consolidation）：词出现于 ≥3 个不同会话 → 固化入
    terminology_consolidated.yaml，新会话开局即有锚，不再从零。

    Contract:
      Preconditions: response_text 为 str；sid 为会话 id；st 为会话状态 dict
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
    _consolidate(fresh, sid)


# ── 语义层：跨会话固化（admission control, W-TinyLFU 同构）────────────────
# 固化词典 terminology_consolidated.yaml 与策展层同格式；admission 计数持久化
# 在同一文件的 candidates 段（词 → {sessions: [sid], last_seen: 日期}），
# 进程重启不丢；候选 last_seen 超 30 天视为陈旧，不再参与晋升（老化衰减）。
_CONSOLIDATED_FILE = "terminology_consolidated.yaml"
_CONSOLIDATE_THRESHOLD_SESSIONS = 3
_CONSOLIDATED_MAX = 100
_CONSOLIDATED_BUDGET = 400
_CANDIDATE_STALE_DAYS = 30


def _consolidated_path() -> Any:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _CONSOLIDATED_FILE


def _load_consolidated_raw() -> dict[str, Any]:
    """读固化词典 raw（terms=已固化词, candidates=admission 计数）；文件不
    存在则建骨架。IO/解析异常向上抛，由调用方 fail-open。

    Contract:
      Postconditions: 返回 dict 含 terms/candidates 键
    """
    import yaml

    path = _consolidated_path()
    if not path.exists():
        path.write_text(
            "# 固化词典（terminology-guard 语义层，自动维护）\n"
            "# 词出现于 ≥3 个不同会话（admission）即固化，全局锚定；\n"
            "# candidates 段为 admission 计数（sessions/last_seen）；\n"
            "# 可直接编辑 terms 删词；删文件=重置全部固化状态。\n"
            "terms: []\n"
            "candidates: {}\n",
            encoding="utf-8",
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_consolidated_raw(raw: dict[str, Any]) -> None:
    """固化词典整体落盘。Contract: Postconditions: 文件与 raw 一致；异常上抛。"""
    import yaml

    _consolidated_path().write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _is_stale(last_seen: str, today: str) -> bool:
    """候选老化判定：last_seen 距 today 超 _CANDIDATE_STALE_DAYS 天。

    Contract:
      Preconditions: 两参均为 YYYY-MM-DD
      Postconditions: 超期返回 True；解析失败按未老化（保守，不丢计数）
    """
    from datetime import date

    try:
        delta = date.fromisoformat(today) - date.fromisoformat(last_seen)
    except ValueError:
        return False
    return abs(delta.days) > _CANDIDATE_STALE_DAYS


def _consolidate(fresh: list[str], sid: str) -> None:
    """admission 记账：词 → 会话集合累积（持久化）；达标词晋升固化。

    对应记忆巩固（consolidation）：情景痕迹（会话内用词）经跨会话重复激活
    固化为语义知识（全局锚定词）。计数持久化在 candidates 段，进程重启不丢。

    Contract:
      Preconditions: fresh 为本轮抽取词列表；sid 为会话 id
      Postconditions: 计数/固化词典落盘（terms ≤_CONSOLIDATED_MAX）；stale
                      候选跳过晋升；任何异常仅记日志，绝不 raise
    """
    if not sid:
        return
    try:
        import time as _time

        raw = _load_consolidated_raw()
        today = _time.strftime("%Y-%m-%d")
        terms = [e for e in (raw.get("terms") or []) if isinstance(e, dict)]
        candidates: dict[str, Any] = raw.get("candidates") or {}
        known = {e.get("canonical") for e in terms}
        promoted: list[str] = []
        for term in fresh:
            if term in known:
                continue
            rec = candidates.get(term) or {}
            last_seen = str(rec.get("last_seen") or "")
            if last_seen and _is_stale(last_seen, today):
                continue
            sessions = [s for s in (rec.get("sessions") or []) if s != sid]
            sessions.append(sid)
            rec["sessions"] = sessions
            rec["last_seen"] = today
            candidates[term] = rec
            if len(set(sessions)) >= _CONSOLIDATE_THRESHOLD_SESSIONS:
                promoted.append(term)
        for term in promoted:
            if len(terms) < _CONSOLIDATED_MAX:
                terms.append({
                    "canonical": term,
                    "aliases": [],
                    "note": f"固化（{today}）——出现于 {len(candidates[term]['sessions'])} 个会话",
                })
                logger.info("terminology-guard: 用词固化入语义层: %s", term)
            candidates.pop(term, None)
        raw["terms"] = terms
        raw["candidates"] = candidates
        _save_consolidated_raw(raw)
    except Exception as exc:
        logger.warning(
            "terminology-guard: 固化词典写入失败（fail-open）: %s", exc, exc_info=True
        )


def _load_consolidated() -> list[str]:
    """读固化词典词表（canonical）；失败 fail-open 按空表。

    Contract:
      Postconditions: 返回 list[str]（≤_CONSOLIDATED_MAX）；绝不 raise
    """
    try:
        raw = _load_consolidated_raw()
        return [
            str(e["canonical"])
            for e in (raw.get("terms") or [])
            if isinstance(e, dict) and e.get("canonical")
        ][:_CONSOLIDATED_MAX]
    except Exception as exc:
        logger.warning(
            "terminology-guard: 固化词典读取失败（fail-open）: %s", exc, exc_info=True
        )
        return []


def _consolidated_line(terms: list[str]) -> Optional[str]:
    """固化词典 → 注入行（400 字符预算；空表 → None）。

    Contract:
      Postconditions: 返回 None 或非空 str；总长受 _CONSOLIDATED_BUDGET 约束
    """
    if not terms:
        return None
    head = "跨会话固化词（多会话验证的全局规范写法，禁同义改写）："
    used = len(head)
    kept: list[str] = []
    for term in terms:
        if used + len(term) + 3 > _CONSOLIDATED_BUDGET:
            break
        kept.append(term)
        used += len(term) + 3
    if not kept:
        return None
    return head + "、".join(kept)


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
            consolidated_text = _consolidated_line(_load_consolidated())
            if consolidated_text and text:
                text = text + "\n" + consolidated_text
            elif consolidated_text:
                text = consolidated_text
            return {"context": text} if text else None
        except Exception as exc:  # fail-open
            logger.warning("terminology-guard inject failed: %s", exc, exc_info=True)
            return None

    ctx.register_hook("transform_llm_output", check_reply)
    ctx.register_hook("pre_llm_call", inject_terminology)
