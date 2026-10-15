# Hermes 上下文暴涨与压缩失效诊断报告

日期：2026-10-15
调查对象：澳门极易付反洗钱会话（session_id `20261014_013602_d17250`，cwd `/Users/stan/code/ai/cnb/ontox`）
调查方式：运行态日志、state.db 用量表、源码路径追踪、API 对照实测
相关既有文档：`docs/review/glm_coding_plan_rate_limit_investigation_20260925.md`（本文不重复其已覆盖的 zai 额度机制，只引用结论）

## 一、结论

极易付会话的上下文在 31 分钟内从 125,974 tokens 单调升至 422,114 tokens（18:24:08–18:55:32），期间压缩尝试无一生效；该会话于 18:56:49 以 `new_session` 结束。

根因是两条上下文回收路径同时失效：

1. **LLM 摘要路径**：压缩摘要的辅助模型 zai/glm-5.3-flash 因账户 5 小时额度耗尽返回 429，代码降级回退到主模型 deepseek-flash；deepseek-flash 的非思考模式输出上限为 8K（官方默认值），而摘要输出需求紧贴该上限，导致摘要被截断（`finish_reason=length`），压缩整体中止并进入 60/300/900 秒阶梯冷却。
2. **本地确定性裁剪路径**：`proactive_prune_min_result_chars` 门槛为 8,000 字符，而该会话活跃工具结果平均仅 1,179 字符，无一条达到门槛，裁剪判定为 `prune:nothing_eligible`。

两条路径同时不可用，上下文只能单向增长。

## 二、实测数据

### 2.1 极易付会话上下文轨迹（API 实测输入 token，来源 agent.log）

| 时刻 | API call | 输入 tokens | 事件 |
|---|---:|---:|---|
| 18:07:05 | #168 | 156,736 | 会话起始段 |
| 18:09:31 | #180 | 191,562 | |
| 18:15:13 | #200 | 149,723 | 压缩生效，下降约 42K |
| 18:24:08 | #220 | 125,974 | 压缩后低点 |
| 18:27:01 | #240 | 170,336 | |
| 18:32:05 | #260 | 214,949 | |
| 18:37:50 | #280 | 259,463 | |
| 18:43:50 | #300 | 327,279 | 本地裁剪判定 `prune:nothing_eligible` |
| 18:46:41 | #320 | 375,494 | |
| 18:54:15 | #340 | 408,027 | |
| 18:55:32 | #351 | 422,114 | 会话终止（18:56:49, end_reason=new_session）|

自 18:24:08 起 31 分钟内由 125,974 升至 422,114，增量 296,140 tokens，期间无一次成功回收。

### 2.2 压缩健康度（今日 agent.log 全量）

- 压缩提交结果：`committed` 9 次 / `aborted` 28 次 → 成功率 24%
- 失败类型单一：`summary_truncated_failure`，共 43 次（含 errors.log 轮转前记录）
- 冷却阶梯触发：60 秒 ×20、300 秒 ×9、900 秒 ×4

### 2.3 压缩辅助调用的路由统计（今日）

| 事件 | 次数 |
|---|---:|
| `Auxiliary compression: using zai (glm-5.3-flash)` | 64 |
| `Auxiliary compression: rate limit on zai`（错误码 1308） | 102 |
| `Auxiliary compression: using deepseek (deepseek-flash)` | 6 |
| 全部降级路径耗尽 | 5 |
| `skipping deepseek (request timed out)` | 3 |

zai 错误原文：`已达到 5 小时的使用上限。您的限额将在 2026-10-15 20:16:06 重置。`

### 2.4 压缩任务的实际输出规模（state.db `session_model_usage` 表）

| task | model | 调用次数 | 输入 tokens | 输出 tokens | 推理 tokens | 平均输出 |
|---|---|---:|---:|---:|---:|---:|
| compression | glm-5.3-flash | 325 | 17,624,119 | 1,874,514 | 12,437 | 5,768 |
| compression | deepseek-flash | 204 | 5,630,539 | 1,520,909 | **0** | **7,455** |

deepseek-flash 的压缩调用推理 tokens 为 0，说明思考已被关闭；但平均输出 7,455 tokens 紧贴 8K 上限，12 个活跃会话的分布全部落在 6,844–7,855 区间，属撞上限特征。

### 2.5 官方上限依据

DeepSeek Chat Completions API 文档（api-docs.deepseek.com）：max_tokens「取值范围为 1 到 384K；未设置时，非思考模式默认 8K，思考模式默认 64K」。

压缩摘要调用不传 max_tokens，见 `agent/context_compressor.py:3891` 注释原文：

```
# NO max_tokens: Anthropic/NIM wires forward it and a hard cap truncates summaries
# (thinking models burn it on reasoning). Timeout comes from call_llm config.
```

叠加 `auxiliary.compression.extra_body.thinking.type: disabled` 已配置，请求落入非思考模式分支，上限即 8,192。

### 2.6 极易付会话上下文构成（state.db messages 表，活跃消息）

| 角色 | 条数 | content 字符 | reasoning 字符 | tool_calls 字符 | 合计字符 |
|---|---:|---:|---:|---:|---:|
| assistant | 158 | 67,831 | 161,906 | 540,423 | 770,160 |
| tool | 174 | 205,154 | 0 | 0 | 205,154 |
| user | 14 | 18,549 | 0 | 0 | 18,549 |

会话累计消息 20,977 条，其中已压缩标记 8,531 条 assistant、9,456 条 tool；活跃仅 346 条，与 `message_count=346` 一致——压缩机制本身在工作，是 18:24 后失效。

### 2.7 本地裁剪门槛不可达

`agent/agent_init.py:1565-1570` 读取配置，未配置时用默认值：

```
proactive_prune_min_result_chars = 8000   （默认）
proactive_prune_min_reclaim_tokens = 4096 （默认）
```

`agent/context_compressor.py:3438` 在无可裁剪项时记 `prune:nothing_eligible`；`:3444` 在回收量不足时记 `prune:reclaim_below_minimum`。该会话活跃工具结果 205,154 字符 / 174 条 = 平均 1,179 字符，全部低于 8,000 字符门槛。

## 三、并发因素（用户问及的其他活动会话）

| session_id | 标题 | 最近一次输入 tokens |
|---|---|---:|
| 20261014_013602_d17250 | 澳门极易付反洗钱应用 | 422,114 |
| 20261015_185649_2f0024 | OntoX 表号段规范 | 342,426 |
| 20261014_111226_89cc53 | 成都农商反洗钱项目 | 276,711 |
| 20261015_190156_3494af | 实验台状态检查 | 245,072 |
| 20261011_144537_ab0721 | 评估 rentahuman.ai | 240,323 |

同时在线 10 个 hermes CLI 进程（`ps` 实测）。5 个活跃会话上下文全部超过 200,000 阈值。

今日主循环 token 消耗（state.db `session_model_usage`）：glm-5.3（zai）64,030 次调用 / 267,770,164 输入 tokens / 9,092,559 推理 tokens。zai 的 5 小时额度由这些并发会话共同消耗，因而在 17:30 前后耗尽。

zai 429 时间分布（每 10 分钟桶）：17:30 起密集（47/31/20/44/46/38/46/25/45/55/23/28 次），与极易付会话上下文失守的起始时刻吻合；早于该时段的 errors.log.2 中截断失败为 0 次。

额度双窗口与 credit 消耗速率的完整调查见 `docs/review/glm_coding_plan_rate_limit_investigation_20260925.md` 第二、三节（结论：按该文实测速率外推，Max 档也会被 3.6 倍速度烧穿）。

## 四、次要缺陷（同批发现）

`auxiliary` 任务未配置 `fallback_chain`，当主模型不可用时无路可走。日志原文：

```
Auxiliary terminology_ledger: main provider deepseek is unavailable and no fallback_chain / fallback_providers is configured — refusing
Auxiliary declare_act_guard: main provider deepseek is unavailable and no fallback_chain / fallback_providers is configured — refusing
```

`auxiliary.<task>.fallback_chain` 为受支持配置项（`agent/auxiliary_client.py:4383 _try_configured_fallback_chain`，每个条目需 provider，model/base_url/api_key 可选）。写入时注意：`hermes config set` 不支持 `[0]` 数组下标语法，会写入字面键（见 9-25 报告第六节实测记录）。

## 五、已执行修复

配置项：`auxiliary.compression.extra_body.max_tokens = 16384`

```
hermes config set auxiliary.compression.extra_body.max_tokens 16384
✓ Set auxiliary.compression.extra_body.max_tokens = 16384 in /Users/stan/.hermes/config.yaml
```

构造层验证（走 hermes 自身请求构造路径，非手工拼装）：

```
1) _get_task_extra_body('compression') = {'thinking': {'type': 'disabled'}, 'max_tokens': 16384}

2) deepseek/deepseek-flash
   extra_body  : {'thinking': {'type': 'disabled'}, 'max_tokens': 16384}
2) zai/glm-5.3-flash
   extra_body  : {'thinking': {'type': 'disabled'}, 'max_tokens': 16384}
```

要点：

- `thinking: disabled` 覆盖了 provider profile 的默认 enabled 投影（未配置时 deepseek profile 投影为 `{'thinking': {'type': 'enabled'}}`，已实测）
- `max_tokens: 16384` 进入请求体，输出上限由 8,192 抬至 16,384
- 顶层无 `max_tokens` 键（值为 None 时未写入），不与 extra_body 冲突
- 配置仅在会话启动时读取，对运行中的会话不生效；zai 额度于 20:16:06 重置后，压缩主路径恢复走 zai

## 六、待决策项

以下两项需要用户拍板，不属于可自行决断的可控可回滚范围：

1. **zai Coding Plan 额度池共用的量化事实**。三条实测证据：① 端点唯一——主对话 provider zai（`~/.hermes/config.yaml:53`）与 auxiliary.compression（`:237`）等全部 zai 辅助任务同为 `https://open.bigmodel.cn/api/coding/paas/v4`，全配置 20 余处 base_url 无一处指向按量端点；② 重置时刻唯一——837 条 429 原文中重置时刻只有一个取值 `2026-10-15 20:16:06`（主对话 `conversation_loop` 318 条 + 辅助 `auxiliary_client` 92 条）；③ 兜底链同指向——56 处 `fallback_chain` 条目（350–402 行区间）全部 `provider: zai` + 同一 coding 端点。

   累计输入 tokens 占比（zai 通道）：主对话 302,383,879（84.1%）、辅助合计 57,225,249（15.9%），其中 `background_review` 37,056,376（10.3%）为辅助最大消耗者，`compression` 18,022,921（5.0%）。

   该量化推翻了「为压缩任务拆出独立额度即可解决」的设想——拆走压缩收益上限仅 5.0%，额度主体是主对话的 84.1%（即并发会话数本身）。建议按 9-25 报告第六节处置：会话数收敛（只保留 1-2 个会话走 zai，其余切 deepseek）+ 下调 `threshold_tokens`。默认动作（用户不回复时）：保持现状，仅保留本次 max_tokens 修复。

2. **本地裁剪门槛调整**。将 `compression.proactive_prune_min_result_chars` 由 8,000 下调至 1,500 字符，可让本地确定性裁剪成为摘要失败时的第二道防线；代价是裁剪提交会破坏 prompt 缓存前缀（`agent/context_compressor.py:3399` 注释：「A commit breaks the prompt cache」）。默认动作（用户不回复时）：不下调，因当前 prompt 缓存命中率接近满值（实测 280,064 / 280,247）。

9-25 报告建议的 `compression.threshold_tokens` 下调（当前 200,000）同样涉及对话体验，仍需用户裁定。

## 七、复现与验证命令

```bash
# 压缩健康度
grep -c "committed" ~/.hermes/logs/agent.log; grep -c "aborted" ~/.hermes/logs/agent.log

# 截断失败原文
grep "summary was truncated" ~/.hermes/logs/errors.log*

# 压缩任务输出规模
sqlite3 "file:$HOME/.hermes/state.db?mode=ro" \
  "SELECT task,model,SUM(api_call_count),SUM(output_tokens),SUM(reasoning_tokens) \
   FROM session_model_usage WHERE task='compression' GROUP BY model;"

# 会话消息构成
sqlite3 "file:$HOME/.hermes/state.db?mode=ro" \
  "SELECT role,COUNT(*),SUM(LENGTH(COALESCE(content,''))),SUM(LENGTH(COALESCE(tool_calls,''))) \
   FROM messages WHERE session_id='20261014_013602_d17250' AND active=1 GROUP BY role;"

# 裁剪判定原因
grep "reclamation did not run" ~/.hermes/logs/agent.log
```
