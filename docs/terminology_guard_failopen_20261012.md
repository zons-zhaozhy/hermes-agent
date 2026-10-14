# terminology-guard fail-open 事件复盘（2026-10-12 夜）

## 结论

报错 `No module named 'yaml'` 的根因不是上游打包缺 hermes_yaml shim，而是 **installs 运行副本滞留 10-10 10:02 旧版插件**（裸 `import yaml`，隔离解释器无 PyYAML）。修复版（commit 2c451f6378，依赖切 hermes_yaml/ruamel）10-10 11:39 已进 fork 并推双远端，但 install 建环境的插件复制源停留在旧版，从未随修复刷新。

哨兵 23:12 报告的两处误判，实测推翻：
1. 「hermes_yaml 模块目录不存在」——ls 实证全部 workspace 的 hermes_yaml.py 均在（shim 内容=ruamel 兼容层）。
2. 「22:54-23:09 新增 34 条」——gateway.log 实测最后一条报错停在 22:50:44，22:54 重启后 0 条；当前活跃链 f4e1d3f7 的插件副本 22:54 已被同步为修复版（mtime 22:54、0 处旧 import），terminology-guard 恢复正常打 INFO 日志（00:35 起多条「别名漂移已标记」），llm_judge 链 22:54 后 0 降级。

## 事件时间线（均 gateway.log / mtime 实测）

| 时间 | 事件 |
|---|---|
| 10-10 10:02 | 旧版（8f9f819f2f 前身）写入各 install 复制源 |
| 10-10 11:39 | fork commit 2c451f6378 修复（hermes_yaml SSOT），双远端已推 |
| 10-12 20:20 | 41bc58c 环境 f4e1d3f7 旧副本触发首条报错 |
| 10-12 20:33 | hermes update 新装 41bc58c3072daa3c |
| 10-12 22:30 | 插件同步波（f4e1d3f7 plugins/ 目录 mtime） |
| 22:50:44 | 活跃链最后一条 yaml 报错 |
| 10-12 22:51 | hermes update 新装 b5de26ce9e813444 |
| 22:54:09 | SIGUSR1 优雅重启，gateway 自身同步波将 f4e1d3f7 副本换为修复版 |
| 22:54 后 | 报错 0 条，terminology-guard / llm_judge 全恢复 |

## 修复动作（10-13 01:5x 执行，19 份全过）

- 漂移面清点（`import yaml` 为旧版标志）：terminology-guard 5 份（b5de26ce×3 + bc4d0312×2）+ 同类漂移 sandwich-inject 14 份（41bc58c 内 13 + bc4d0312 内 1，含 fork 正路参照 sandwich-inject:124 的 hermes_yaml SSOT 用法）。
- 全部从 fork HEAD 同步替换，逐份 compile 验证 19/19 OK，全副本 `import yaml` 残留 0，抽查 sha256 与 fork HEAD 一致。
- 每份留 `.driftbak-1013` 备份；三个环境 venv 实测含 ruamel.yaml-0.18.16，依赖齐备。
- 受 PmTree 禁令约束未动 pm-runtime；活跃链运行时零影响（当前 41bc58c/f4e1d3f7 本来就是修复版）。

## 遗留风险与根治方向

install 建环境时的插件复制源刷新机制是深层根因——本次修复的是存量 19 份，未动该机制。若下次 hermes update 再从旧缓存复制，同类漂移会重现。观察点：下次 install 的 environments/*/workspace/plugins/ 是否再出现旧版（import yaml 检测）。

## 根治闭环（2026-10-14，双案驱动）

第二案（goal 双缺口补丁 4df33bdc24 在 29 个 workspace 中 0 个生效）确认漂移为结构性问题。根治=用户拍板切换单一真相源，详见 docs/single_source_gateway_20261014.md：

- ~/Library/LaunchAgents/ai.hermes.gateway.plist 启动命令改指 fork 仓 /.hermes/bin/hermes（备份 ai.hermes.gateway.plist.bak-workspace-1014），gateway 拉起的全部会话直读 fork 源码，workspace 快照不再被生产路径加载。
- PM 依赖环境按树路径 sha256 前 16 位登记（41bc58c3072daa3c 即 fork 树），依赖环境不变、只换源码树，零迁移成本。
- 本节前述「遗留风险」已随切换消解：复制源刷新机制不再被生产路径依赖。tools/workspace_drift_check.py 退役保留（docstring 已标注）。

## 附：夜间出网抖动归属（哨兵 01:40 第二案）

非 hermes 侧问题 [实测]：7890 端口 verge-mihomo（PID 22709）正常监听；hermes 进程 env 无代理变量；10-10 09:00 历史故障时段 error.log 显示当时 7890 连接拒绝（代理客户端死），而今晚 curl 短请求全通。今晚故障模式为「短通长断」（流式长连接持续断），属代理/链路层间歇丢长连接，建议用户层检查 verge-mihomo 配置或 Wi-Fi 路由器，晨报已带六波时间线。
