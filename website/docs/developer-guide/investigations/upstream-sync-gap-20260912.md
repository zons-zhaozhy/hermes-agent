# 官方上游未同步更新盘点 — 2026-09-12

数据来源：GitHub compare API `d20a8e4475...5aa17c0590`（gh 认证通道，原始输出已核）
本地同步锚点：5201699045 merge upstream/main(d20a8e4475)，2026-09-05 18:45 +0800
官方最新 main：5aa17c0590（2026-09-12T08:35Z）

## 总量

- 官方 main 领先本地分叉点：1861 个提交（compare ahead_by 原文）
  - 其中 2026-09-05 同步之后产生：1717 个（严格未同步量）
  - 其余 141 个为 09-05 之前的旧提交（官方历史整理/分叉窗口重叠，非新增量）
- 未同步窗口：2026-09-05 → 2026-09-12，共 8 天
- 官方节奏：日均 200+ 提交（09-07 单日 420、09-09 单日 320）
- 本地同期自有提交（5201699045 之后）：53 个

## 未同步提交类型分布（09-05 后 1717 个）

| 类型 | 数量 |
|---|---|
| fix | 966 |
| tests | 264 |
| refactor | 111 |
| feat | 104 |
| other | 115 |
| chore/ci | 81 |
| docs | 76 |

改动范围 top：desktop(280)、gateway(147)、agent(79)、state(47)、js(45)、cli(43)、cron(30)、tools(27)、tui(27)、sessions(26)、models(23)、honcho(23)、auth(20)、delegation(19)、bot-mode(19)

## 核心能力面新增（feat，103 条核心面，摘录）

子代理/委派：
- 子代理绝对上下文上限压缩（delegation.compression_threshold_tokens）
- fan-out 中失败子代理立即上报父代理
- per-task completion groups——未分组子代理完成即返回
- 子代理后台进程移交父代理；notify 残留进程具名上报
- CLI/TUI/Desktop 全线 subagent dock（F7 折叠、live tail、steer/stop）

记忆/上下文：
- 压缩 usage anchor 跨 DB 重载与进程重启存活
- 按图计价 token 成本从 provider usage 学习
- provider 级 model_thresholds 键（"<provider>:<substr>"）
- mem0 同步字符上限可配（mem0.json）

模型/供应商：
- gemini-3.7-flash / gemini-3.8-flash、GPT-6 Astra、DeepSeek V4.1 Flash
- GPT Image 2.5 生成+编辑（OpenAI + FAL）
- openrouter per-model provider_routing 覆盖
- nous anthropic_wire=auto 会话级线协议自动判定

账号/密钥：
- 凭据池：重置单个凭证不清兄弟冷却、CLI 选优先级、刷新单个 OAuth grant
- MCP 服务器 device-code 授权（CLI）
- vault：1Password/Bitwarden 集成、支付卡/地址填充、双因素码自动读取、CLI+浏览器+Desktop 三端
- Nous 免费层：免费推理+连接器，/login 一条命令登录（CLI/Desktop/gateway）

插件生态：
- plugin-catalog 成唯一发现系统：requires_hermes 门禁、live catalog 6h 缓存、CLI search/browse/install/doctor、私有 git 仓库安装、commit 钉住
- on_room_member_activity 钩子（群聊成员事件投影给插件）

其他：
- gateway MEDIA 文件投递（远程终端沙箱内文件）
- webhook 标准签名校验
- cron 可创建 paused 任务无调度竞态
- Matrix LaTeX 渲染、Signal 表格等宽对齐
- Hermes Collective Wisdom Agent V1（#94266）
- GPT-Live 全双工语音模式（Desktop）
- 动态 workflow 编排 skill
- multiplex!: 移除 gateway.multiplex_profile_allowlist——所有 profile 全 serve（breaking）
- auth: hermes auth list 显示 entry id 与优先级

## 同步风险提示

- desktop(280)/gateway(147)/agent(79) 是冲突高发区；本地 53 个自有提交集中在
  discipline 套件、rule-reinjection、guard 体系，与官方 plugin-catalog/requires_hermes
  门禁存在结构性交集，合并时 plugins/ 目录需重点对账。
- 官方 feat(multiplex)! 为 breaking change，合并前需评估本地 profile 路由
  （docs/profile-routing.md）是否受影响。

## 复核命令

```bash
gh api "repos/NousResearch/hermes-agent/compare/d20a8e4475...5aa17c0590" --jq '{ahead_by,total_commits}'
```

原始清单：/tmp/upstream_ahead.json（1858 条去重，含 sha/date/msg/author）
