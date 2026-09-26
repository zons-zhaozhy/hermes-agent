# 跟上 AI 节奏：采集-分诊-消化三层机制（2026-09-26）

## 问题（第一性原理拆解）

「AI 发展快，要跟得上节奏」的本质不是「多看」，是三段链路的吞吐：
采集密度 × 分诊优先级 × 消化深度。实测基线（2026-09-26）：

- 采集层密：RSS 60min / arXiv 每日 / 雷达 2h——不缺
- 消化层窄：日学习 1 题/天 + 深度研究 6 篇/7 天
- 漏斗比 1630:1（周采集 9780 vs 深度消化 6）
- **无分诊层**：日学习按星期几轮流选题（周一小模型/周二Agent…），范式级物种要碰巧撞上「对的方向日」才被深学

当天还挖出两个结构性断点（都 6 天级静默）：
1. arXiv 采集断流 6 天零产出（papers mtime 停在 09-23）
2. 哨兵只盯自我监督层，知识采集层无监测

## 修复一：采集传输层根修（Knowledge-Base 5f0dbbccd）

**根因链**（双客户端对照矩阵实测闭合）：
Fastly 回源传输 ~33KB/s → 单日 cs 全量 3MB 需 ~150s →
urllib timeout=60 大响应必挂死 → cron 3600s 击杀 →
失败账本并入次日 → 欠账滚雪球（13 天）→ 每轮必超时

**修法**：`oai_harvester.py::fetch_oai` 拆双层——`_curl_fetch`（curl 子进程，
--max-time 240）+ `fetch_oai`（406/429 退避语义不变）。

**实测战果**：单轮 156 分钟回收 13 天欠账，新增 1808 篇
（papers 35486→37293）；残余 09-22/09-24 两天进账本次日自动重试。

**排障教训**（已回填 skill:arxiv-oai-harvest-ops）：
- CDN 缓存命中时 curl 秒回会误导定位——测速必须看回源窗口
- 「客户端差异」表象下先量传输速率再换客户端
- qwen3.5:4b-mlx 是思考模型：num_predict 预算烧在 thinking 字段，
  response 为空——分诊/判定件选模型必须非思考款（qwen2.5:0.5b 实测可用）

## 修复二：哨兵补知识采集层（hermes 9eca9ca236）

`plugins/outcome-collector/flywheel_freshness.py` 新增第 4 检查项
`arxiv-papers`：papers 目录最新 mtime >50h 即告警（每日 07:10 采集节奏 +2h 宽限）。

**上线即咬中真缺陷**：首跑报 `age=75.9h FLYWHEEL STALE`（真实断流）；
根修完成后自动转绿（age=0.9h OK）——发现→根修→恢复→确认全闭环实证。

同日阈值对齐：daily-audit/regression-alerts 36h→26h（cron 已改每日后
旧阈值脱配）；regression cron `0 7 * * 1-5`→`0 7 * * *`（周末 48h 盲区消除）。

## 修复三：范式分诊件（hermes cba41a4730）

`scripts/paradigm_triage.py`——判断题（是否范式级信号）按 SystemOne 纪律
走本地小模型，禁大模型生成再解析：

1. **预筛**：22 个范式信号词（paradigm/self-evolv/world model/agent kernel…）
   零成本粗筛进候选池
2. **打分**：Ollama qwen2.5:0.5b，标尺 0-2 增量/3-5 显著沿用/6-9 范式级，
   输出仅 0-9 单字符（few-shot 锚定，无 JSON 解析面）
3. **排序**：top-K 队列，预算上限 60 篇/轮

**降级**：Ollama 离线 → 信号词计数×2 封顶 + DEGRADED 标记 + stderr 明示（禁静默）。

**实测**：3 天窗口 1519 篇 → 候选 159 → top6 出列（20s 内）。

## 接线（全部已验证活）

| 环节 | 机制 | 节奏 | 状态 |
|---|---|---|---|
| 采集 | oai_harvest_recent7.sh cron | 每日 07:10 | 修复后首轮回补 1808 篇 |
| 断流监测 | flywheel_freshness 第4项 + watchdog 检查4 | 每 30min | 75.9h 告警→根修→0.9h 转绿 |
| 告警出口 | watchdog/mirror-drift cron deliver=feishu 群 | exit 1 即投递 | 0926 修复：原 deliver:local 告警只落库无人知（近 7 天 4 次全 suppressed） |
| 分诊 | 范式分诊·日跑 cron（38171bcd607b） | 每日 12:40 | 首跑 completed（17s） |
| 消化 | 日学习 job prompt 前插优先队列指令 | 每日 12:30 | 队列≥7 分优先于星期轮排 |

设计原则：星期轮排是均衡器，范式队列是优先级覆盖——范式级物种不再等
「对的方向日」。

## 维护要点

- 分诊模型换型前先测：思考模型的 thinking 字段会吃掉 num_predict 预算
- 哨兵 papers 阈值 50h 与采集 cron 节奏耦合：改采集频率须同步改阈值
- 账本重试链是活的：收割失败次日自动并入，禁手动清账本
- 告警出口通道改动（cron deliver）须同步本表：哨兵有效=检测+投递两半都在，
  上午文档初版只记了检测半（deliver:local 断点下午才挖出）——教训：接线表
  必须覆盖到「人看见」为止，落库≠送达
