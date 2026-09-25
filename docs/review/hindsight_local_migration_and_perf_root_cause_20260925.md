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

## 三、性能根因（四层，逐层实测）

### 3.1 第一层：thinking 参数在 native 路径失效（机制成立，但实测证明**不是主因**）

[文档] hindsight 的 ollama 路径走**原生 `/api/chat`**（`openai_compatible_llm.py:884-889`：只有原生端点能携带
context window；`:1534 _call_ollama_native`）。
[实测] 该路径**无 `reasoning_effort` 处理代码**（搜索 1500 行后零命中），只认 top-level `think`
（`:1596-1604` 注释原文：`native top-level fields (think, keep_alive, ...) pass through directly`）。
→ 所配 `reasoning_effort=none` 在 native 上是**死参数**。该机制成立，故仍补配 `think:false`。

**[实测修正] 本报告初版的推断已被推翻。** 初版据 `llm=227s ÷ 9 tok/s` 反推「输出约 2497 tokens、
thinking 全程燃烧」，但 `llm_requests` 真实台账显示 `output_tokens` 仅 **346-1200**（见 3.5），
属事实抽取的正常量级，**无 thinking 膨胀迹象**。故 `think:false` 修的是一个真实缺陷，
但**不是本次「慢」的主因**。真正的瓶颈见 3.3 与 3.5。

### 3.2 第二层：超时 + 重试放大

[文档] `config.py:962 DEFAULT_LLM_TIMEOUT = 120.0`、`:959 DEFAULT_LLM_MAX_RETRIES = 3`。
[实测] 台账中存在单次 `169780ms`（169.8s）> 120s 的记录，证明该超时线确会被触及；
`llm=227s` 这类总量亦符合「一次超时 + 重试」的叠加形态。
[修正] 触发条件不是初版推断的「275s」，而是**并发争抢把单次耗时推过 120s**（见 3.5 慢侧样本）。

### 3.3 第三层：并发雪崩

[文档] `config.py:958 DEFAULT_LLM_MAX_CONCURRENT = 32`、`:1405 DEFAULT_WORKER_MAX_SLOTS = 10`
——按云端高并发设计的默认值打到本地**单实例串行** ollama 上。
[实测] 某次 native 调用 `total=112.39s`，其中真正的推理仅 `load 0.06 + prompt_eval 0.049 + eval 0.55 = 0.66s`，
其余 111.7s 全是排队；另实测 4 个任务 age 同步堆积至 121s/182s。

### 3.4 第四层（最隐蔽，也是真正的自我维持机制）：daemon 自杀螺旋

[文档] `daemon_embed_manager.py:92-97` 上游自述：daemon 的 `/health` 与 LLM 调用**共用同一个 asyncio
事件循环**，慢 provider 调用会把响应 stall 数十秒；且**探测超时过短的代价是「杀掉健康的 listener」**。
[文档] `:98-101 HEALTH_PROBE_TIMEOUT = 10.0`；`:587-590` 文档字符串写明该 slack 是故意留的
(`That slack is deliberate: it is what lets a daemon stalled on a slow LLM call answer before we call it stale`)。
[文档] `:645-647` 一旦判定不健康即 `Clearing unhealthy process on port ... (PID ...)` 后 `_kill_process`。

因果闭环：LLM 慢 → `/health` 被同一事件循环占住 → 10s 探测超时 → 判「不健康」→ **杀 daemon**
→ in-flight 任务被打断、recovery attempts +1 → 满 3 次即 `moved to 'failed'` → 新任务重新排队 → 回到起点。

[实测] 证据：daemon PID 在**无人干预下自行变化**（75404 → 77723）；日志两次出现
`Worker zons-2.local moved 2 tasks to 'failed' (exceeded 3 recovery attempts in schema None)`。

**推论**：这解释了为什么「4 个任务占满串行队列」只是表象——真正的自维持机制是 daemon 被误判死亡后
反复自杀重启，而每次重启都把任务推向 failed 并让新任务从零开始。因此慢调用治理与
`HEALTH_PROBE_TIMEOUT`（治误判）**必须成对修复**，单独修任一侧都无法终止螺旋。

### 3.5 真正的瓶颈（`llm_requests` 台账实测，修正 3.1）

[实测] 全部 ollama 记录（11 条，均 success），DB `llm_requests` 表原始数据：

| 时间(UTC) | scope | duration_ms | input | output | 实测 tok/s |
|---|---|---|---|---|---|
| 09:01:23 | consolidation | 227116 | 8682 | 1200 | 5.3 |
| 09:11:25 | retain_extract_facts | 104669 | 3126 | 659 | 6.3 |
| 09:14:23 | retain_extract_facts | 48926 | 2512 | 394 | 8.1 |
| 09:14:23 | retain_extract_facts | 115550 | 2663 | 346 | 3.0 |
| 09:16:46 | retain_extract_facts | 61205 | 2388 | 516 | 8.4 |
| 09:16:46 | retain_extract_facts | 116203 | 2617 | 453 | 3.9 |
| 09:23:13 | retain_extract_facts | 169780 | 2388 | 406 | 2.4 |
| 09:23:13 | retain_extract_facts | 77934 | 2617 | 615 | 7.9 |
| 09:27:05 | retain_extract_facts | 74688 | 2388 | 692 | 9.3 |
| 09:31:41 | retain_extract_facts | 99948 | 2754 | 785 | 7.9 |
| 09:34:11 | retain_extract_facts | 88553 | 2853 | 692 | 7.8 |

**三条实测结论**：

1. **并发争抢是主要拖累**：09:14 / 09:16 / 09:23 各有一对同一时刻的记录，慢侧 115.6s / 116.2s / 169.8s
   对快侧 48.9s / 61.2s / 77.9s，**慢 2.2-2.4 倍**，tok/s 从 8-9 掉到 2.4-3.9。
2. **单条稳态 ≈ 88s**，构成 = prefill(2853 tokens ≈ 1.2s，按本机 curl 实测 0.43ms/token)
   + decode(692 tokens ÷ 8.7 tok/s ≈ 79s)。瓶颈是 **4B MLX 的解码物理速率**（skill 记录理想值 11 tok/s）。
3. **对照云端**（同表 zai/glm-4.7）：retain 均 29.7s（151 条）、consolidation 均 46.2s（111 条）。
   **本地比云端慢 2-3 倍**，换来的是不占配额、不再触发 429。

### 3.6 failed 任务的来源（自我归因，非系统缺陷）

[实测] `async_operations` 状态分布：completed 580 / **failed 397** / pending 7 / processing 5；
failed 时间戳集中在 09:08-09:15 UTC（= 17:08-17:15 CST），错误信息统一为
`exceeded max recovery attempts (retry_count >= 3)`。

该时间窗正是本调查中**反复 kill/重启 daemon** 的时段，每次 SIGTERM 都会打断 in-flight 任务并累加
recovery attempts，满 3 次即转 failed。**这 397 条失败由调查动作自身造成**，非系统固有缺陷。

[文档] `config.py:1410 DEFAULT_OPERATION_RETENTION_DAYS = 0`，注释明确
`operation history is a user-visible audit trail`——**上游设计即不自动清理**，故保留不删。

### 3.7 配置源的真实优先级（实测纠正）

[实测] `.env` 才是**权威源**：daemon CLI 启动时自行加载它（`hindsight_embed/cli.py:115` 的
`if key not in os.environ` 语义），故写在 `.env` 的键会进入 daemon 环境。
[实测] profile env（`~/.hindsight/profiles/hermes.env`）会被插件 materialize **按 `config.json` 重写**，
写在其中的、`config.json` 不存在的键（如 `HINDSIGHT_API_LLM_EXTRA_BODY`）会被抹掉——
但这**不影响 daemon**，因为 `.env` 优先且 daemon 自己会加载。
[实测] 验证方式与结果：`ps eww -p <daemon_pid> | tr " " "\n" | grep ^HINDSIGHT` 四个键齐备。

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
| `HINDSIGHT_EMBED_HEALTH_PROBE_TIMEOUT` | `10` → `120` | 防 daemon 被误判「不健康」而自杀重启（见 3.4） |
| `daemon_embed_manager.py:99,100` 默认值 | `10.0` → `120.0` | **改 venv 源码**：活跃会话 import 时已缓存旧值，改 env 救不了它们（见 3.4） |

备份：`hindsight/config.json.bak-20260925-170005|170627|171337`、`.env.bak-20260925-170005|172221|172621`、
`hermes.env.bak-20260925-170005|172221`。

## 五、验证状态与残余边界

### 已闭环（均有当轮实测输出）

1. **实测耗时台账已取得**（见 3.5）：单条稳态 88s、解码 7.8-9.3 tok/s。初版「提速约 4 倍」的推算
   **已被推翻**（见 3.1 修正），真实瓶颈是 4B MLX 的解码物理速率与并发争抢。
2. **并发限制生效**：task stage 出现 `.queued` 后缀（信号量饱和才产生），见 3.3 / 3.4。
3. **daemon 配置继承**：4 个关键键经 `ps eww -p <pid>` 逐项确认（daemon 78296）。
4. **默认值改动生效**：`env -u HINDSIGHT_EMBED_HEALTH_PROBE_TIMEOUT python -c "import
   hindsight_embed.daemon_embed_manager as d; print(d.HEALTH_PROBE_TIMEOUT)"` → `120.0`。
5. **failed 任务归因**：397 条源于本调查反复重启 daemon，非系统固有缺陷（见 3.6）。

### 残余边界（仍存在，需知悉）

1. **单条 88s 无法通过配置压缩**：它是 4B MLX 解码速率（8-9 tok/s vs skill 记录理想值 11）的结果。
   若要更快只有两条路——换更小/更快的本地模型（质量权衡）或回到云端（占配额、会 429）。
2. **活跃会话（9a07d0 / 802916 / f7f151 / e031aa）内存持旧配置**：daemon 侧被误杀的风险已通过
   改默认值消除（`import` 时即读到 120），但会话自身在重启前仍按旧行为工作，建议空闲时重启。
3. **venv 源码改动会在 `hindsight_embed` 升级时被覆盖**：升级后需按本节记录重放
   （`:99,100` 由 `10.0` 改 `120.0`）；`.env` 中的 `HINDSIGHT_EMBED_HEALTH_PROBE_TIMEOUT=120` 构成第二道保险。

## 六、配套观测命令

```bash
# daemon 实际持有的配置
P=$(lsof -nP -iTCP:9177 -sTCP:LISTEN -t | head -1); ps eww -p $P | tr ' ' '\n' | grep '^HINDSIGHT'

# 默认值是否生效（不受 env 影响）
env -u HINDSIGHT_EMBED_HEALTH_PROBE_TIMEOUT python -c "import hindsight_embed.daemon_embed_manager as d; print(d.HEALTH_PROBE_TIMEOUT)"

# 实测耗时台账（DB 直查，权威于日志）
psql -h 127.0.0.1 -p 5432 -U hindsight -d hindsight -c \
  "SELECT started_at, scope, duration_ms, input_tokens, output_tokens FROM llm_requests WHERE provider='ollama' ORDER BY started_at DESC LIMIT 10"

# 任务状态分布
psql -h 127.0.0.1 -p 5432 -U hindsight -d hindsight -c \
  "SELECT status, count(*) FROM async_operations GROUP BY status"

# 实测单次推理速率（区分排队与推理）
curl -s http://localhost:11434/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5:4b-mlx","messages":[{"role":"user","content":"hi"}],"think":false,"stream":false,"options":{"num_predict":20}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['eval_count'], d['eval_duration']/1e9, 'tok/s=', d['eval_count']/(d['eval_duration']/1e9))"
```
