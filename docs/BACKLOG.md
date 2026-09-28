# Hermes fork 维护 Backlog

来源：2026-09-28 深夜 multi-bot 运维会话（bot_mode_dm 修复 e394e62c02、periodic_scheduler 锁序修复 05c4ff8e7a 的收尾盘点）。

## 待办

### 1. [Lark] SDK logger 桥接 gateway 根 logger
- 优先级：中（观测面缺陷，已造成哨兵一次误报）
- 问题：lark-oapi SDK 内部 logger 未接入 gateway 根 logger 配置，其 INFO 级输出被 stdout 块缓冲吞掉，keepalive 断开/重连期间零日志——19:20 飞书「入站死亡」误报的根源之一（另一根源=判活依赖日志解析而非控制面 status）。
- 方案：按 #73779 已有的 SDK 接线先例，把 lark-oapi logger 挂到 root（propagate=True + 合理 level），使 _try_connect 失败必打 ERROR 的设计真正可见。
- 验收：飞书链路断开时 errors.log/info 日志可见 _try_connect ERROR 行；重连成功可见 on_reconnected 打点。
- 备注：重试看门狗立项已驳回（#73779+#113662 已有既有看门狗，重复建设）。

### 2. telemetry 面板：「.dev_logs/oms.log 未按 profile 隔离」事实核查
- 优先级：低（哨兵 09-27 通报，未复核）
- 内容：哨兵称 .dev_logs/oms.log 无 profile 前缀，多 profile 共用一文件。需读 OMS dev_start 逻辑核实后决定是否修。

## 已结案（防重题）

- 429/1302 整点齐射 → 4 任务错峰已落地（:04/:07/:10/:11 分配表，哨兵回读核实）
- bot_mode_dm 投递挂死（管道 EOF 等待）→ e394e62c02，退出码权威范式
- periodic_scheduler 锁序死锁（65848 teardown）→ 05c4ff8e7a，dispatch 移出 _cond
- 飞书「入站死亡」 → 误报，链路 24 秒自愈，不重启（控制面 status 权威）
