"""
轻量级质量审计 —— 每轮回复后用辅助 LLM 做十维质量评估。

不 fork AIAgent，不走 conversation replay。直接构造 prompt，HTTP 调 aux 模型。
配置走 auxiliary.quality_auditor（建议配 glm-4.7 等便宜模型）。

结果追加写入 ~/.hermes/state/quality_audit.jsonl，供日聚合分析。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_AUDIT_ENABLED = True

# ============================================================
# 十维评估 prompt
# ============================================================

_QUALITY_AUDIT_PROMPT = """你是一个严谨的 Hermes Agent 回复质量审计员。评估以下 AI 助手对用户消息的回复质量，从 10 个维度打分（每维 0-10 分）。

## 用户消息
{user_message}

## AI 回复
{assistant_response}
{extra_context}

## 评估维度

### 通用维度
1. 事实准确性：结论是否有可验证证据？有没有"看起来对但未实测"的断言？
2. 完整性：是否覆盖所有子问题？有无遗漏？
3. 效率：是否精简？有没有"理解你的需求"之类的废话？
4. 风格：语气、详略度、代码格式是否适当？

### Agent 特有维度
5. 验证卫生：声称的修复/发现是否有实测输出/日志/traceback 证据？
6. 根因深度：修 bug 是否追到根因？同类问题是否全局搜索一次修完？
7. 工具效率：是否避免了大文件全量读取、多轮重复调用、不必要的大搜索？
8. 交付诚实："已修复"有没有跑验证？"已测试"有没有贴输出？
9. 静默降级防范：有没有静默兼容、静默 pass、fallback 没日志、异常吞掉？
10. 推进力：用户给定方向后是直接执行还是反复确认/停下请示？

## 输出格式（严格 JSON，只输出 JSON 对象）

```json
{{
  "scores": {{
    "accuracy": <int>, "completeness": <int>, "efficiency": <int>, "tone": <int>,
    "verification": <int>, "root_cause": <int>, "tool_efficiency": <int>,
    "delivery_honesty": <int>, "silent_degradation": <int>, "decisiveness": <int>
  }},
  "issues": ["<具体问题--每条说明扣分原因>"],
  "strengths": ["<做得好的地方>"],
  "suggestions": ["<可落地的改进建议>"],
  "overall_assessment": "<一句话总体评价>",
  "fatal_issues": ["<原则性错误--编造证据/静默降级/绕路等，无则空数组>"]
}}
```"""


def _build_audit_prompt(user_message, assistant_response, tool_call_count=0, tool_names=None):
    MAX_USER, MAX_ASSISTANT = 1500, 3000
    user = user_message[:MAX_USER]
    assistant = assistant_response[:MAX_ASSISTANT]
    if len(user_message) > MAX_USER:
        user += "...(截断)"
    if len(assistant_response) > MAX_ASSISTANT:
        assistant += "...(截断)"
    extra = ""
    if tool_call_count > 0:
        extra = f"\n\n## 本轮工具调用\n共 {tool_call_count} 次"
        if tool_names:
            extra += f": {', '.join(tool_names[:15])}"
        extra += "\n"
    return _QUALITY_AUDIT_PROMPT.format(user_message=user, assistant_response=assistant, extra_context=extra)


def _resolve_auditor_runtime():
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task = aux.get("quality_auditor", {}) if isinstance(aux.get("quality_auditor"), dict) else {}
    provider = str(task.get("provider", "")).strip() or None
    model = str(task.get("model", "")).strip() or None
    if not (provider and provider != "auto" and model):
        br = aux.get("background_review", {})
        if isinstance(br, dict):
            provider = str(br.get("provider", "")).strip() or "auto"
            model = str(br.get("model", "")).strip() or ""
    if not (provider and provider != "auto" and model):
        return {}
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        rp = resolve_runtime_provider(
            requested=provider, target_model=model,
            explicit_api_key=task.get("api_key") or None,
            explicit_base_url=task.get("base_url") or None,
        )
        return {"provider": provider, "model": model, "base_url": rp.get("base_url", ""), "api_key": rp.get("api_key", "")}
    except Exception:
        return {}


def _call_auxiliary_llm(prompt, timeout=60):
    import urllib.request
    runtime = _resolve_auditor_runtime()
    base_url = runtime.get("base_url", "") if runtime else ""
    api_key = runtime.get("api_key", "") if runtime else ""
    model = runtime.get("model", "") if runtime else ""

    if not base_url or not model:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            provider = cfg.get("provider", "")
            model = cfg.get("model", {}).get("default", "") or model
            providers = cfg.get("providers", {})
            if provider in providers:
                base_url = providers[provider].get("base_url", "")
                if not api_key:
                    api_key = providers[provider].get("api_key", "")
        except Exception as _e:
            logger.warning("quality_auditor: config fallback failed: %s", _e, exc_info=True)
    if not model or not base_url:
        logger.debug("quality_auditor: 无可用模型/base_url")
        return None

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', '')}"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1200, "temperature": 0.2}
    try:
        req = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers)
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(resp.read().decode())
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return None
        json_str = content.strip()
        i = json_str.find("{")
        if i > 0:
            json_str = json_str[i:]
        j = json_str.rfind("}")
        if j >= 0:
            json_str = json_str[:j + 1]
        return json.loads(json_str)
    except Exception as e:
        logger.warning("quality_auditor: LLM 调用/解析失败: %s", e, exc_info=True)
        return None


def _write_audit_entry(entry):
    audit_file = get_hermes_home() / "state" / "quality_audit.jsonl"
    audit_file.parent.mkdir(parents=True, exist_ok=True)
    entry["_ts"] = time.time()
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as _e:
        logger.warning("quality_auditor: 审计数据写入失败: %s", _e, exc_info=True)


def _build_audit_entry(user_message, assistant_response, session_id, model, tool_call_count, tool_names):
    """执行一次审计（同步），返回 entry dict 或 None。"""
    try:
        prompt = _build_audit_prompt(user_message, assistant_response, tool_call_count, tool_names)
        result = _call_auxiliary_llm(prompt)
        if result is None:
            return None
        scores = result.get("scores", {})
        dims = ["accuracy","completeness","efficiency","tone","verification","root_cause","tool_efficiency","delivery_honesty","silent_degradation","decisiveness"]
        valid = [scores.get(d, 0) for d in dims if isinstance(scores.get(d), (int, float))]
        total = sum(valid) / max(len(valid), 1) if valid else 0
        entry = {
            "session_id": session_id, "model": model, "total_score": round(total, 1),
            "scores": {d: scores.get(d) for d in dims},
            "issues": result.get("issues", []), "strengths": result.get("strengths", []),
            "suggestions": result.get("suggestions", []),
            "overall_assessment": result.get("overall_assessment", ""),
            "fatal_issues": result.get("fatal_issues", []),
        }
        if entry["fatal_issues"]:
            logger.warning("quality_auditor: FATAL! %.1f/10, fatal: %s", total, "; ".join(entry["fatal_issues"]))
        elif total < 5.0:
            logger.warning("quality_auditor: 低分 %.1f/10, issues: %s", total, ", ".join(entry["issues"][:3]) if entry["issues"] else "无")
        return entry
    except Exception:
        return None


def _audit_worker(user_message, assistant_response, session_id, model, tool_call_count, tool_names):
    """后台线程：执行审计并写入 JSONL。"""
    entry = _build_audit_entry(user_message, assistant_response, session_id, model, tool_call_count, tool_names)
    if entry:
        _write_audit_entry(entry)


def fire_quality_audit(user_message, assistant_response, session_id="", model="", tool_call_count=0, tool_names=None, fast_timeout=0):
    """启动质量审计。
    
    fast_timeout > 0: 同步等待最多 fast_timeout 秒，返回 entry dict 或 None。
                      用于在回复末尾标注分数（尽力而为，超时即跳过）。
    fast_timeout = 0: 纯后台 daemon 线程，不阻塞（默认）。
    """
    if not _AUDIT_ENABLED:
        return None
    if not assistant_response or len(assistant_response.strip()) < 50:
        return None
    if tool_names is None:
        tool_names = []
    
    if fast_timeout > 0:
        # 快速同步通道：用线程 + join(timeout) 实现有超时的等待
        result_holder = {"entry": None}
        def _runner():
            result_holder["entry"] = _build_audit_entry(
                user_message, assistant_response, session_id, model, tool_call_count, tool_names
            )
        t = threading.Thread(target=_runner, daemon=True, name="quality-audit-fast")
        t.start()
        t.join(timeout=fast_timeout)
        entry = result_holder["entry"]
        if entry:
            _write_audit_entry(entry)
        return entry
    else:
        t = threading.Thread(
            target=_audit_worker,
            args=(user_message, assistant_response, session_id, model, tool_call_count, tool_names),
            daemon=True, name="quality-audit"
        )
        t.start()
        return None


def aggregate_daily(output_file=None):
    audit_file = get_hermes_home() / "state" / "quality_audit.jsonl"
    if not audit_file.exists():
        return {"total": 0, "msg": "无审计数据文件"}
    cutoff = time.time() - 86400
    entries = []
    try:
        with open(audit_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line[0] != "{":
                    continue
                try:
                    e = json.loads(line)
                    if e.get("_ts", 0) >= cutoff:
                        entries.append(e)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {"total": 0, "msg": "读取失败"}
    if not entries:
        return {"total": 0, "msg": "过去24h无数据"}
    total = len(entries)
    avg_score = round(sum(e.get("total_score", 0) for e in entries) / total, 1)
    min_score = round(min(e.get("total_score", 0) for e in entries), 1)
    max_score = round(max(e.get("total_score", 0) for e in entries), 1)
    fatal_count = sum(1 for e in entries if e.get("fatal_issues"))
    dim_sums = {}
    for e in entries:
        for d, v in (e.get("scores") or {}).items():
            if isinstance(v, (int, float)):
                dim_sums[d] = dim_sums.get(d, 0) + v
    dim_avgs = {d: round(s / total, 1) for d, s in dim_sums.items()}
    iss_counts, sug_counts = {}, {}
    for e in entries:
        for iss in e.get("issues", []):
            k = iss[:100].strip(); iss_counts[k] = iss_counts.get(k, 0) + 1
        for sug in e.get("suggestions", []):
            k = sug[:100].strip(); sug_counts[k] = sug_counts.get(k, 0) + 1
    report = {
        "total": total, "avg_score": avg_score, "min_score": min_score, "max_score": max_score,
        "fatal_count": fatal_count, "dimension_averages": dim_avgs,
        "top_issues": sorted(iss_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_suggestions": sorted(sug_counts.items(), key=lambda x: x[1], reverse=True)[:10],
    }
    if output_file:
        from pathlib import Path
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report
