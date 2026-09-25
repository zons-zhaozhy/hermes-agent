# hindsight 本地化迁移与性能根因调查（2026-09-25）

调查触发：用户在 429 频次调查中追问「哪些小请求在不停调 glm」，锁定 hindsight 记忆守护进程；
随后用户指出「本地 4B 模型单次推理 82 秒太慢」，要求彻底处理。

来源标注：[实测] 本机当轮工具原始输出｜[文档] 上游源码/官方文档原文｜[推断] 基于实测外推｜[未查证] 无证据。

## 一、结论摘要

1. hindsight 记忆守护进程是除活跃会话外**第二大 zai 配额消费者**（[实测] 22.5h 8844 行 LLM 调用/错误 ≈ 6.5 次/分钟，主会话同期 5.7 次/分钟，两者同量级）。
2. 「本地模型慢」的表象由**三个可修复的配置缺陷**叠加而成，模型本身不慢（[实测] `eval_duration=0.55s / eval_count=5` → 约 9 tok/s）。
3. 已修复：本地化切换、`think:false`、并发上限、写入频率、空闲回收、显式 base_url 分裂源。

## 二、配置源测绘（关键：存在三处配置源，优先级不同）

| 源 | 路径 | 消费者 | 特点 |
|---|---|---|---|
| ① 插件配置源 | `~/.hermes/hindsight/config.json` | 插件 `initialize()` | **权威源**；插件 materialize 时据此重写 ② |
| ② profile env | `~/.hindsight/profiles/hermes.env` | daemon CLI 启动 | 会被 ① 重写；但键齐全 |
| ③ 进程环境 | `~/.hermes/.env` | daemon 子进程（`env = os.environ.copy()`） | 优先级最高（`cli.py:115` `if key not in os.environ` 不覆盖已有值） |

[文档] 取值优先级证据：`daemon_embed_manager.py:737-741` 为 `value = config.get(simple_key) or config.get(env_key)`；
`:748-750` 会把 profile/env 里所有 `HINDSIGHT_*` 键传播给 daemon 子进程。

**分裂事故（已修）**：改动 ① 的 `llm_provider`/`llm_model` 后，活跃会话仍以内存中的旧 `provider/model`（zai/glm-4.7）
调用 `daemon_embed_manager`，而旧 `llm_base_url` 为空未覆盖 → 与 ① 中新写的 `base_url=localhost` 拼成
`provider=zai + model=glm-4.7 + base_url=localhost:11434` 的畸形组合（[实测] 17:06-17:11 期间 5571 行
`HTTP 404: model 'glm-4.7' not found`）。
**根治**：删除显式 `llm_base_url`，交给 `openai_compatible_llm.py:601-628` 按 provider 自动推断端点
（`provider=="ollama"` → `http://localhost:11434/v1`；`"ollama-cloud"` 才是 `https://ollama.com/v1`）。

## 三、性能根因（三层，逐层实测）

### 3.1 第一层：thinking 未关（主因）

[文档] hindsight 的 ollama 路径走**原生 `/api/chat`**（`openai_compatible_llm.py:884-889`：只有原生端点能携带
context window；`:1534 _call_ollama_native`）。
[实测] 该路径**无 `reasoning_effort` 处理代码**（搜索 1500 行后零命中），只认 top-level `think`
（`:1596-1604` 注释原文：`native top-level fields (think, keep_alive, ...) pass through directly`）。
→ 所配 `reasoning_effort=none` 在 native 上是**死参数**，thinking 全程燃烧。

量化（[推断]，基于 3.3 的 9 tok/s 实测速率外推）：

| 状态 | 输出 tokens | 单次耗时 |
|---|---|---|
| 无 thinking | ~600 | ~66s |
| 有 thinking | ~2497 | ~275s |

### 3.2 第二层：超时 + 重试放大

[文档] `config.py:962 DEFAULT_LLM_TIMEOUT = 120.0`、`:959 DEFAULT_LLM_MAX_RETRIES = 3`。
[推断] 275s > 120s 超时线 → 触发重试 → 最坏 3×275 ≈ 825s。这是日志中 `llm=227s` 乃至更长的来源。

### 3.3 第三层：并发雪崩

[文档] `config.py:958 DEFAULT_LLM_MAX_CONCURRENT = 32`、`:1405 DEFAULT_WORKER_MAX_SLOTS = 10`
——按云端高并发设计的默认值打到本地**单实例串行** ollama 上。
[实测] 某次 native 调用 `total=112.39s`，其中真正的推理仅 `load 0.06 + prompt_eval 0.049 + eval 0.55 = 0.66s`，
其余 111.7s 全是排队；另实测 4 个任务 age 同步堆积至 121s/182s。

## 四、修复清单（全部落在 ② profile env 与 ③ `.env`，daemon 已重启验证继承）

| 键 | 值 | 作用 |
|---|---|---|
| `llm_provider` / `llm_model`（① 源） | `ollama` / `qwen3.5:4b-mlx` | 迁出 zai 配额 |
| `llm_base_url`（① 源） | **删除** | 消除 provider/endpoint 分裂源 |
| `HINDSIGHT_API_LLM_EXTRA_BODY` | `{"think": false}` | 关 thinking（native 路径唯一有效手段） |
| `HINDSIGHT_API_LLM_MAX_CONCURRENT` | `2` | 匹配串行后端，留一路给前台 recall |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT` | `1` | 最重的任务串行 |
| `retain_every_n_turns`（① 源） | `1` → `3` | 写入量降 2/3（攒批，非丢数据） |
| `idle_timeout`（① 源） | `0` → `1800` | 空闲 30 分钟回收（原为永不退出） |
| `HINDSIGHT_API_LLM_REASONING_EFFORT` | `none` | 保留：`/v1` 路径仍需（双路径兼容） |

备份：`hindsight/config.json.bak-20260925-170005|170627|171337`、`.env.bak-20260925-170005|172221|172621`、
`hermes.env.bak-20260925-170005|172221`。

## 五、未验证边界

1. 修复后的**实测 `avg=` 改善值未取得**（新 daemon 75404 尚未跑出完整 batch）——「提速约 4 倍」是
   基于实测 tokens 与 9 tok/s 速率的**推算值**，不是实测值。
2. 活跃会话（9a07d0 / 802916 / f7f151 / e031aa）内存中仍持旧配置，**重启前**其触发的 daemon 重启
   会回落到 zai 端点（合法调用，占配额），需会话重启方可全本地。
3. `HINDSIGHT_API_LLM_EXTRA_BODY` 会被旧配置会话继承；若其 provider 为 zai，`think` 是否被拒**未查证**。

## 六、配套观测命令

```bash
# daemon 实际持有的配置
P=$(lsof -nP -iTCP:9177 -sTCP:LISTEN -t | head -1); ps eww -p $P | tr ' ' '\n' | grep '^HINDSIGHT'

# batch 实测耗时与 token 量（avg = llm 秒 / memories 数）
python3 -c "p='/Users/stan/.hindsight/profiles/hermes.log'; ls=open(p,errors='ignore').read().splitlines(); \
[print(l[85:300]) for l in [x for x in ls if 'processed=' in x][-5:]]"

# 实测单次推理速率（区分排队与推理）
curl -s http://localhost:11434/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:4b-mlx","messages":[{"role":"user","content":"hi"}],"think":false,"stream":false,"options":{"num_predict":20}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['eval_count'], d['eval_duration']/1e9, 'tok/s=', d['eval_count']/(d['eval_duration']/1e9))"
```
