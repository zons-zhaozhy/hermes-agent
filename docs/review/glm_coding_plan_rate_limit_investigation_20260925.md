# GLM Coding Plan 限速根因调查报告（2026-09-25）

调查触发：2026-09-24 深夜用户报「其他几个会话用 glm 非常慢」，要求查清是 Hermes 逻辑问题还是模型慢。
本报告全部结论标注来源：[实测] 本机当轮工具原始输出｜[文档] Z.AI 官方文档原文｜[推断] 基于实测外推｜[未查证] 无证据。

## 一、结论摘要

1. 慢的主因是 z.ai Coding Plan 的 **credit 额度耗尽**（HTTP 429 / code 1302），不是网络慢、不是模型推理慢、不是 Hermes 逻辑慢。
2. 次要因素是 Hermes 侧两个纯本地开销源（ontox MCP 每天数千次启动失败、skill 漂移检查每晚数十万条 WARNING），已修复。
3. 缓存命中率是决定性的成本杠杆（缓存倍率 1.7 vs 输入 6.9，差 4 倍）；实测命中率 92.9%，属良好。

## 二、官方限速规则（[文档] https://docs.z.ai/devpack/overview.md 原文）

### 2.1 支持模型

* All plans support **GLM-5.3**, GLM-5.3-Flash.
* Requests for GLM-5.2/GLM-5.1 will be automatically routed to GLM-5.3, requests for GLM-4.7 will automatically be routed to GLM-5.3-Flash.

### 2.2 额度双窗口

Each plan is subject to both a 5-hour usage limit and a weekly usage limit.

| Plan Type | 5-Hour Credits | Weekly Credits |
| :-------: | :------------: | :------------: |
|    Lite   |      2,000     |     10,000     |
|    Pro    |     12,000     |     60,000     |
|    Max    |     28,000     |     140,000    |

* 5-hour credits: Dynamically refreshed; credit quota resets 5 hours after consumption.（滚动窗口，非整点清零）
* Weekly credits: Activated upon subscription; resets every 7 days.

### 2.3 credit 计算公式

* Model credit usage = (Input tokens × Input multiplier + Cached Input tokens × Cached Input multiplier + Output tokens × Output multiplier) / 10,000
* MCP tool credit usage = Number of calls × Output multiplier

| Product Type | Product | Input | Cached Input | Output |
| --- | --- | --- | --- | --- |
| Model | GLM-5.3 | 6.9 | 1.7 | 24 |
| Model | GLM-5.3-Flash（含视觉理解 MCP） | 2.3 | 0.56 | 8 |
| MCP Server | Web Search / Web Reader / Zread | — | — | 1.2 |

### 2.4 计费时段

* **Peak hours**: Monday to Friday, 14:00–18:00 Singapore Standard Time (UTC+8). 其余时段按 **50%** 计费。
* 特惠：2026-09-25 至 2026-10-07 全天按非高峰费率（50%）。
* GLM-5.3-Flash 活动：2026-09-03 至 2026-10-07 每天 23:00–次日 09:00，通过 ZCode 无限量使用 GLM-5.3-Flash，其他 agent 享双倍额度。

### 2.5 额度耗尽行为

额度用尽后只能等待 5 小时窗口刷新，不会扣账户余额——没有兜底路径，这就是 code 1302 的直接含义。

## 三、本机实测

### 3.1 端点直测（[实测] curl）

    23:47  coding 端点           → HTTP 429，body code=1302「您的账户已达到速率限制」，TTFB 0.19s
    23:48  coding 端点复测        → HTTP 200，2.21s
    23:48  按量端点 /api/paas/v4  → HTTP 200，0.76s
    01:01  coding + glm-5.3-flash → HTTP 200，3.00s

TTFB 0.19s 说明网络不是瓶颈；1 分钟内 429→200 说明限流随窗口滚动恢复。

### 3.2 模型路由实测（[实测] 看响应 model 字段）

    模型名            响应中的 model 字段       结论
    glm-5.3          glm-5.3                   原样
    glm-5.3-flash    glm-5.3-flash             原样（官方支持）
    glm-4.7          glm-5.3-flash             被路由（与文档一致）
    glm-5-turbo      glm-5.3-flash             被路由
    glm-4.5-flash    glm-4.5-flash             **未被路由**，不在官方支持列表

`glm-4.5-flash` 的套餐归属 [未查证]：官方 Supported Models 未列该模型，其资源来源（套餐内 / 按量扣费）无账单侧证据。

### 3.3 credit 消耗精算（[实测] agent.log 埋点，16 分钟窗口）

    zai 调用         33 次（glm-5.3 ×24、glm-5.3-flash ×6、glm-5 ×3）
    输入 tokens      3,609,996（其中缓存命中 3,353,152 = 92.9%）
    输出 tokens      17,616
    credit           790
    折算             2,961 credit/小时（高峰全额）｜1,480 credit/小时（非高峰 50%）

按实测速率外推（[推断]）：71,100 credit/天 → 497,700 credit/周。
对照额度表：Lite 需 ~50 倍、Pro 需 ~8 倍、Max 需 ~3.6 倍。
即按当前用法，即使 Max 档也会被 3.6 倍速度烧穿（五折期压到 ~1.8 倍，仍超）。

### 3.4 单次调用的上下文构成（[实测] agent.log + state.db）

    16 分钟窗口：主模型调用 67 次 + 辅助模型调用 6 次；模型分布 deepseek-v4-flash 25、
    glm-5.3 23、deepseek-v4-pro 9、glm-5.3-flash 6、glm-5 3、aux 6
    主模型延迟：n=67  p50=8.2s  p90=85.8s  max=186.8s  avg=26.5s
    输入 token：p50=131,571  p90=181,796  max=198,167  合计 8,143,894

上下文八类（`agent/context_breakdown.py`，即 `/context` 命令的实现）：
System prompt / Tool definitions / Rules / Skills / MCP / Subagent definitions / Memory / Conversation。

官方口径（[文档]）：一个 prompt 平均触发 15-20 次模型调用——即"模型回话 → 执行工具 → 回喂结果 → 再问模型"的循环。

会话消息实测（[实测] state.db）：
    会话 f7f151（glm-5.3）：总 540 条消息 / 417,274 字符；活跃 243 条（tool 119 / assistant 118 / user 6）
    会话 8835f9（deepseek）：总 415 条消息 / 531,607 字符；活跃 73 条

压缩阈值（[实测] config.yaml:200-203）：threshold 0.66 / threshold_tokens 200000 —— 阈值贴着实测最大调用量（198,167），故上下文长期停在 15-20 万 token 不压缩，每轮全量重发。

辅助调用（[实测] agent.log）：devil_advocate_audit / reply_side_guards / declare_act_guard 全部走本地
`qwen3.5:4b-mlx`（http://localhost:11434/v1，Ollama），不消耗 zai credit。

另一类"慢"（[实测] agent.log）：`Stream stale for 120s (threshold 120s) — no chunks received. model=glm-5.3 context=~87,083 tokens. Killing connection.` —— 上游 120 秒不吐 chunk，Hermes 主动杀连接重试。

## 四、Hermes 侧两个本地开销源（已修）

### 4.1 ontox MCP server 每天启动失败数千次

    [实测] mcp-stderr.log：ontox 启动尝试 124,986 次、失败行 227,122；按日分布 9/22 达 10,740 次，9/24 已 5,679 次
    根因  ~/.hermes/config.yaml 的 mcp_servers.ontox.env 缺 MCP_RATE_LIMIT 与 MCP_RATE_HEAVY；
          ontox-mcp-server/server.py:38/80/81 为 fail-closed（缺键即 RuntimeError 拒绝启动）
    修复  hermes config set mcp_servers.ontox.env.MCP_RATE_LIMIT 60 / MCP_RATE_HEAVY 15
    验证  [实测] 用配置完整 env 启动 server.py → returncode=0、RuntimeError 0 行
    生效条件  需会话重启（运行中进程仍持旧 env）

### 4.2 skill 漂移检查每晚刷数十万条 WARNING

    [实测] gateway.error.log 单晚 181,315 行 skill_sha_drift WARNING；agent.log 33,428 行
    根因  scripts/skill_sha_drift.py:106/118 把「路径不属于该 git 仓库 / 不属于 skills 目录」
          这一正常分支（调用方拿到 False 即跳过该 ref）记成 WARNING
    修复  两处降为 logger.debug；真正失败路径（读缓存/读文件）的告警保留
          提交 74c996fef6
    验证  [实测] 单轮全量扫描输出 998 行、WARNING 0 条；报告内容不变（230 skill(s) with drift）

未处理项：该脚本仍用正则把 skill 正文里的示例路径/占位符当"引用"逐个查 git（白烧 CPU/IO）。
改解析规则有漏掉真漂移的风险，未擅自改动。

## 五、未验证边界

1. 用户当前所处档位（Lite/Pro/Max）未知，无法从 API 读取。
2. `glm-4.5-flash` 的套餐归属未查证（不在官方支持列表，路由未生效；是否扣余额无账单证据）。
3. 92.9% 缓存命中率来自 Hermes 记录的 provider 返回值，非服务端账单。
4. 窗口真实长度未测出——只观测到单次「429 → 1 分钟后 200」。
5. loom 今日 5,962 次 glm-5.3-flash 调用是否全走套餐通道，未验证。
6. 辅助调用走本地的结论基于今晚 6 条日志样本，不代表全部钩子。
7. 报告中 credit 速率外推假设消耗速率恒定，未做全天采样。

## 六、处置建议（按性价比）

1. 把 auxiliary 段 8 个 `glm-4.5-flash` 与 3 个 `glm-5-turbo` 统一改成 `glm-5.3-flash`（套餐内，倍率低 3 倍）。
2. 降低 `compression.threshold_tokens`（当前 200,000），直接压缩每轮重复发送的 token 量。
   代价是压缩更频繁、历史细节更早丢失，涉及对话体验，需用户裁定。
3. 会话数收敛：只保留 1-2 个会话用 glm-5.3，其余切 deepseek（实测 deepseek 会话全程未被 429 阻断）。
4. 利用折扣期：2026-09-25 至 10-07 全天五折；23:00-09:00 用 glm-5.3-flash 享双倍额度。
