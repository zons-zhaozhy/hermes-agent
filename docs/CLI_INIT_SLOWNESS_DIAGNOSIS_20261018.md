# CLI「正在初始化代理」数分钟等待诊断（2026-10-18）

会话：local-4e411b51（deepseek-flash，yf-ai-delivery / PROGRAM_PROMPT_v1.0 自主推进任务）
现象：用户重启 CLI 后输入「继续」，屏幕停留在「正在初始化代理...」数分钟。

## 实测时间线（agent.log / gateway.log）

| 时刻 | 事件 | 证据位置 |
|---|---|---|
| 09:50:00 | 收到「继续，不要留尾巴…」，轮次开始 | gateway.log:4501 |
| 09:54:02.288 | 上下文压缩尝试 aborted，failure_class=explicit_interrupt | agent.log:8449 |
| 09:54:02.295 | 轮次被中断结束：reason=interrupted_by_user，242.2s、35 次 API 调用、106 个工具轮、response=0 字符 | agent.log:8450-8451 |
| 09:54:02.416 | 收到 session_replay_gap 重放消息，新轮次启动（history=223） | gateway.log:4638, agent.log:8452-8453 |
| 09:54:02.514 | 预检压缩触发：~201,738 token ≥ 阈值 200,000（模型窗口 1,000,000） | agent.log:8454 |
| 09:54:02-09:55:29 | 压缩执行：摘要模型 glm-5.3-flash，prompt 32,206 token，耗时 84,663ms + 提交 1,529ms；224→80 条消息，201,738→112,208 token | agent.log:8696-8697 |
| 09:54:04.651 | 旧 CLI WebSocket 异常断开（code=1006） | agent.log:8459 |
| 09:54:14.916 | 新 CLI WebSocket 接入（用户重启并输入「继续」） | agent.log:8541 |
| 09:55:47 | 压缩后轮次恢复（API #282，缓存命中率仅 55%，前一轮为 99%） | agent.log:8716 |
| 09:56-10:03 | 轮次持续运行（含 193.42s 的 terminal 工具调用） | agent.log:9154 等 |
| 同期 | 网关单进程并行执行另两个重会话（315.3s/285.0s 轮次）+ 2 个 cron 任务（10:00:50、10:01:18 各自 agent_init） | gateway.log:4508/4681, agent.log:9413/9825 |

压缩次数：该会话状态栏显示🗜️4（本日又发生 1 次预检压缩）。

## 根因分层

1. **用户中断作废整轮**（242s / 35 次 API 调用零输出）——中断后 replay 重放消息立即触发新一轮。
2. **压缩阈值与窗口比例失衡**：`compression.threshold_tokens: 200000`（config.yaml:200）封顶了 `threshold: 0.66` 的百分比语义；对 ctx=1,000,000 的 deepseek-flash 实际在 20% 窗口即压缩。每次压缩代价 = 84.6s 辅助模型调用 + 缓存失效（压缩后首次调用 cache 55% vs 常态 99%）+ 上下文信息损失。
3. **单网关进程承载全部并发会话**：诊断窗口内同时跑 2-3 个长轮次 + cron，CLI 侧初始化（会话接入 + 223 条历史加载）与流式反馈被同进程负载拖慢，「正在初始化代理」标签覆盖了压缩等待期，无进度提示。

## 处置建议（待裁定，默认保持现状）

- 候选：`threshold_tokens` 200,000 → 500,000。收益：1M 窗口模型压缩次数 4→0-1，每次省 ~87s 停顿；压缩摘要走 zai（glm-5.3-flash，32K prompt/次），少压缩=少消耗 zai 额度。代价：轮次重发 token 上升，但 deepseek 实测缓存命中 99%（agent.log:9161 起），增量成本小。
- 既有裁定：docs/HERMES_CONTEXT_GROWTH_DIAGNOSIS_20261015.md:168 与 docs/review/glm_coding_plan_rate_limit_investigation_20260925.md:158 均将 threshold_tokens 列为「需用户裁定」项，默认动作=保持 200,000 不变。

数据来源：~/.hermes/logs/agent.log、~/.hermes/logs/gateway.log（2026-10-18 09:50-10:05 窗口），均为实测日志原文。
