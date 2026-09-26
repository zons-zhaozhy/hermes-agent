#!/usr/bin/env python3
"""paradigm_triage — 范式级信号分诊件（判断题用小模型，SystemOne 纪律）。

问题：每日采集数百篇论文，漏斗比 1630:1（周采集 9780 vs 深度消化 6）。
日学习 job 按星期轮排选题，范式级物种（如 SystemOne/Jev 类）要碰巧撞上
「对的方向日」才被深学——被动响应跟不上节奏。

本件=判断题（这篇是否范式级信号），按纪律禁用大模型生成再解析，用本地
小模型（Ollama）对每日新增论文打分排队，产出 top 队列供日学习优先消化。

评分标尺（few-shot 锚定，输出仅 0-9 单字符，无需 JSON 解析）：
  0-2 增量改进（新基准/新数据集/某任务+2点）
  3-5 显著但沿用既有范式（更强的 LoRA/更好的 RAG 混合）
  6-9 范式级信号（改变问题定义/推理形态/系统架构的物种）

用法:
    python paradigm_triage.py [--papers-dir PATH] [--days N] [--top K] [--json]
    Ollama 不可用时降级为信号词预分诊并打 DEGRADED 标记（exit 0 仍出结果，
    但 stderr 明示降级——禁静默）。

Contract:
  Preconditions: papers 目录存在（否则 exit 2）
  Postconditions: 永不 raise；输出 top-K 列表（分数/标题/文件），
                  Ollama 不可用时 DEGRADED 且 stderr 告警

已查重：scripts/ 下无论文分诊工具；日学习 job prompt 无分诊逻辑；
本件与 flywheel_freshness（产出时效）职责互斥。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PAPERS_DIR = Path("/Users/stan/Knowledge-Base/ai-research/papers")
OLLAMA_URL = "http://localhost:11434/api/generate"
TRIAGE_MODEL = "qwen2.5:0.5b"  # 非思考模型: qwen3.5 是思考模型, num_predict 预算烧在 thinking 字段致 response 空(2026-09-26 实测)

# 范式级信号词（预分诊层：零成本粗筛，命中直接进候选池）
_PARADIGM_SIGNALS = (
    "paradigm", "new framework species", "redefines", "first-of-its-kind",
    "system one", "system two", "metacognitive", "self-evolv", "self-improv",
    "emergent", "foundation agent", "world model", "scratchpad",
    "chain-of-thought", "test-time compute", "inference-time",
    "redefine the problem", "changes how we", "no longer needs",
    "replaces fine-tuning", "tool use as", "agent kernel",
)

_PROMPT_TEMPLATE = """你是技术雷达分诊员。给下面论文打范式级信号分（只回一个数字）。

标尺：
0-2=增量改进（新基准/新数据集/某任务+2点）
3-5=显著但沿用既有范式（更强LoRA/更好RAG混合）
6-9=范式级信号（改变问题定义/推理形态/系统架构）

标题: {title}
摘要: {abstract}

只输出一个 0-9 的数字，不要任何其他文字。"""


def _load_recent(papers_dir: Path, days: int) -> List[Dict[str, Any]]:
    """加载近 N 天新增论文（按 mtime），提取标题/摘要/路径。

    Contract:
      Preconditions: papers_dir 为 Path
      Postconditions: 返回 [{file,title,abstract,age_hours}]，永不 raise
    """
    cutoff = time.time() - days * 86400
    out: List[Dict[str, Any]] = []
    for p in papers_dir.glob("*.md"):
        mtime = p.stat().st_mtime
        if mtime < cutoff:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(text)
        abstract = _extract_section(text, "摘要", max_chars=1200)
        out.append({
            "file": str(p),
            "title": title or p.stem,
            "abstract": abstract or "",
            "age_hours": round((time.time() - mtime) / 3600, 1),
        })
    out.sort(key=lambda x: x["age_hours"])
    return out


def _extract_title(text: str) -> str:
    """取文件首个「# 标题」行（str 方法，零正则）。"""
    for line in text.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip()
    return ""


def _extract_section(text: str, header: str, max_chars: int) -> str:
    """取「## 摘要」段落到下一个标题（str 方法）。"""
    lines = text.splitlines()
    capture = False
    buf: List[str] = []
    for line in lines:
        if line.strip().startswith("## ") and header in line:
            capture = True
            continue
        if capture:
            if line.strip().startswith("## ") or line.strip().startswith("# "):
                break
            buf.append(line)
    joined = "\n".join(buf).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "…"  # trunc-ok: LLM 输入预算上限1200字符,全文在源文件
    return joined


def _pretriage(paper: Dict[str, Any]) -> int:
    """信号词粗筛计数（零成本预分诊层）。"""
    text = (paper["title"] + " " + paper["abstract"]).lower()
    return sum(1 for s in _PARADIGM_SIGNALS if s in text)


def _ollama_score(title: str, abstract: str, timeout: int = 60) -> Optional[int]:
    """本地小模型打分（0-9），失败返回 None。

    Contract:
      Preconditions: Ollama 在线（离线由调用方降级）
      Postconditions: 返回 0-9 或 None；输出剥离只取首个数字字符
    """
    prompt = _PROMPT_TEMPLATE.format(title=title, abstract=abstract)
    payload = json.dumps({
        "model": TRIAGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 4},
    })
    try:
        proc = subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), "-H", "Content-Type: application/json",
             "-d", payload, OLLAMA_URL],
            capture_output=True, timeout=timeout + 10)
    except FileNotFoundError as e:
        raise RuntimeError("curl 不可用——分诊件依赖 curl 调 Ollama") from e
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        resp = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        # 全文落盘供事后排查（诊断输出禁截断，写文件而非切片打印）
        dump_path = Path("/tmp/paradigm_triage_ollama_raw.txt")
        dump_path.write_bytes(proc.stdout)
        print(f"Ollama 响应非 JSON，全文已落盘 {dump_path}", file=sys.stderr)
        raise RuntimeError("Ollama 响应解析失败") from e
    raw = unicodedata.normalize("NFKC", str(resp.get("response", "")).strip())
    for ch in raw:  # str 扫描取首个数字字符（零正则）
        if ch.isdigit():
            return int(ch)
    return None


def triage(papers_dir: Path, days: int, top: int, quiet: bool = False) -> Dict[str, Any]:
    """主分诊：预筛 → 小模型打分 → 排序输出。

    Contract:
      Preconditions: papers_dir 存在
      Postconditions: 返回 {generated_at, papers_seen, candidates, degraded, top_k}，永不 raise
    """
    papers = _load_recent(papers_dir, days)
    candidates = [p for p in papers if _pretriage(p) >= 1] or papers[:20]
    degraded = False
    scored: List[Dict[str, Any]] = []
    for p in candidates[:60]:  # 预算上限:60 篇/轮，超出留给下一轮
        score = _ollama_score(p["title"], p["abstract"])
        if score is None:
            degraded = True
            score = min(9, _pretriage(p) * 2)  # 降级=信号词计数×2 封顶 9
        scored.append({**p, "score": score, "pretriage_hits": _pretriage(p)})
    scored.sort(key=lambda x: (-x["score"], x["age_hours"]))
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "papers_seen": len(papers),
        "candidates": len(candidates),
        "degraded": degraded,
        "top_k": scored[:top],
    }
    if degraded and not quiet:
        print("DEGRADED: Ollama 不可用，退化为信号词预分诊", file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="范式级信号分诊（判断题→本地小模型）")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--days", type=int, default=1, help="回看天数")
    parser.add_argument("--top", type=int, default=5, help="输出 top-K")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    if not papers_dir.is_dir():
        print(f"papers 目录不存在: {papers_dir}", file=sys.stderr)
        return 2

    result = triage(papers_dir, args.days, args.top)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"分诊 {result['papers_seen']} 篇 → 候选 {result['candidates']} → top{args.top}:")
        if result["degraded"]:
            print("  [DEGRADED 模式]")
        for i, p in enumerate(result["top_k"], 1):
            print(f"  {i}. [分{p['score']}] {p['title']}")
            print(f"     {p['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
