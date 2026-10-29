# 200k 上下文窗口实测：对第一轮优胜模型发 ~200k token 输入的压缩请求
# 用法: python3 /Users/stan/.hermes/cache/scratch/compression_probe_200k.py
# 结果: /Users/stan/.hermes/cache/scratch/compression_probe_200k_results.json
import json, time, os, subprocess, urllib.request, logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("probe_200k")

_cfg = subprocess.run(["grep", "-A3", "yinfeng:", os.path.expanduser("~/.hermes/config.yaml")],
                      capture_output=True, text=True, check=False).stdout
BASE = _cfg.split("base_url:")[-1].strip().splitlines()[0].strip()
KEY = subprocess.run(["sh", "-c", "grep -E 'yinfeng|YINFENG' ~/.hermes/.env | head -1 | cut -d= -f2"],
                     capture_output=True, text=True, check=False).stdout.strip()
assert BASE.startswith("http"), f"base_url 提取失败: {BASE!r}"

# 第一轮优胜（首轮实测 finish_reason=stop 且延迟最低的候选）
CANDIDATES = [
    "deepseek-v4.1-flash", "deepseek-v4-flash", "qwen3.7-flash",
    "glm-5.2", "glm-5.3-flash", "kimi-k3",
]

para = (
    "会话记录段：工具 terminal 返回 34 行输出，包含三张表的行数对账结果。"
    "下游消费方为报表模块与跑批引擎。本轮修改涉及适配器层与契约层，"
    "契约文件位于 tools/ci/contracts/，生成物目录为 schemas/generated/。"
    "验证步骤：先跑窄集测试，再跑全量回归，最后真机走查三角色旅程。"
)
# para 实测 ~94 token/段（首轮 8499/90）；段数为命令行参数（必填，缺失报错引导）
import sys
if len(sys.argv) < 2 or not sys.argv[1].isdigit():
    raise SystemExit("用法: python3 compression_probe_200k.py <段数>  # 2000 段 ≈ 19 万 token")
SEGS = int(sys.argv[1])
long_input = "\n".join(f"第{i}段 {para}" for i in range(SEGS))

def probe_model(model: str) -> dict:
    """Contract: Preconditions — BASE/KEY 就绪, long_input 已超 200k token; Postconditions — 返回 ok/latency_s/finish_reason/prompt_tokens 或 error"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是上下文压缩摘要器。"},
            {"role": "user", "content": (
                "将以下会话记录压缩为结构化摘要，分五个小节，保留全部关键数字与文件路径：\n"
                + long_input)},
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "stream": False,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        log.warning("probe_200k %s failed: %s: %s", model, type(e).__name__, str(e), exc_info=True)
        return {"model": model, "ok": False, "latency_s": round(time.time() - t0, 1),
                "error": f"{type(e).__name__}: {e!s}"}
    dt = time.time() - t0
    ch = body.get("choices", [{}])[0]
    usage = body.get("usage", {})
    return {
        "model": model, "ok": True, "latency_s": round(dt, 1),
        "finish_reason": ch.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }

if __name__ == "__main__":
    results = []
    for m in CANDIDATES:
        r = probe_model(m)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
    with open("/Users/stan/.hermes/cache/scratch/compression_probe_200k_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("DONE", len(results))
