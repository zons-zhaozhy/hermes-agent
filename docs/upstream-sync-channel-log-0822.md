# 上游同步通道实测记录（2026-08-22 下午，晚高峰）

## 环境
- 本地: macOS, /Users/stan/code/ai/github/fork/hermes-agent（本地已有完整上游历史，upstream/main=eb63c254bc 08-18）
- 上游真 tip (gh api 实测): 9098f6777b…（比 eb63c254bc 领先约 1099 commits）
- 真实落后量 ≈ 1118 commits（本地main+19b17baf…，origin/main 已领先本地 2 commits）

## 通道实测（全部带时间戳）
| 通道 | 结果 |
|---|---|
| github.com HTTPS 直连 | 200 但 9-10s 压线，raw.githubusercontent 10.9KB/s —— 慢而不死 |
| github.com:22 SSH | 死（5min 零字节） |
| ssh.github.com:443 | TCP 通，认证 OK，git fetch 传输中（tmp_pack 24.8M 增长） |
| cnb pipeline | 根组织 CPU 配额耗尽（0.67 核时都冻结不了）→ 全部 pipeline Prepare 阶段挂 |
| 云服务器 43.143.225.4 | SSH 通（时好时坏）；到 github.com HTTPS 200/15s；codeload 20s 拉 100KB 被限速 |
| gh api api.github.com | 1.5s 快（本地+云都通）——数据面小、控制面快 |
| ghproxy.net (本地) | 早上 2.6s，下午 20s 超时——公共加速器晚高峰全降级 |
| moeyy/ghfast/gitclone 等 | 全部 000 超时或 DNS 解析失败 |
| origin (git@github.com:zons-zhaozhy) SSH | 未单独测（推测同 :22 死） |

## 根因层
1. 国际出口带宽被运营商掐（QoS 限速），SSH:22 直接不可达；HTTPS/443 慢而不断；ssh.github.com:443 慢而活。
2. cnb 配额: 根组织 CPU 核时耗尽，pipelines 全部在 Prepare 挂（skill 已记录此高频故障）。
3. 云服务器国际出口同样被限速（codeload 100KB/20s）——云中继不可用。
云中继不可用。

## 方案层
- 主线: ssh.github.com:443 fetch（进行中，tmp_pack 增长）
- 备用: gh api 逐 commit 拉补丁（api 面快，~1100 commits × N 次请求，可行但工程量大）
- 长期: 配额恢复后启用 cnb 全量历史管道（.cnb.yml sync-full-history 已就位）

## 历史教训（先前会话）
- `timeout N git fetch | tail` 管道使退出码=tail 的 0，fetch 被 kill 也显示成功——假成功。必须 FETCH_EXIT=$? 紧跟或无管道直接跑。
- fetch 需要长 timeout（≥240s），不要重试短超时。
- cnb .cnb.yml 顶层键=分支名匹配模式，`probe:` 不匹配 cnb-probe 分支报 CONFIG_EVENT_EMPTY。
- cnb 日志端点: /build/logs/stage/{sn}/{pid}/{stage_id}（skill:cnb-pipeline-ops 的脚本里有）
- ghproxy 是 URL 重写镜像不是正向代理（前缀模式），CONNECT 險道 400。
- 云服务器 /tmp/up-bundle.git clone 卡 11M（同样被限速），已 kill。
</content>
