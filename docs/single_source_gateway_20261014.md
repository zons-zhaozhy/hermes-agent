# 单一真相源：gateway 直读 fork 源码（2026-10-14 切换）

> 适用范围：本机（stan 的 macOS）fork 直跑形态。上游官方安装形态（`~/.hermes/hermes-agent` + installs）不受影响，本文件描述的是本机定制。

## 一、背景：为什么切换

双真相漂移两案实测（详见 docs/terminology_guard_failopen_20261012.md）：

1. 2026-10-12 terminology-guard fail-open：install 建环境复制的插件停留在旧版，fork 修复从未回流，手工同步 19 份。
2. 2026-10-14 goal 双缺口：fork 补丁（commit 4df33bdc24）在全部 29 个 workspace 快照中 0 个生效。

结构性根因：hermes 官方安装形态下，gateway 守护进程的环境变量 PYTHONPATH 指向 `~/.hermes/installs/<key>/environments/<env>/workspace`（源码快照），fork 仓的修复不会自动刷新快照——「源码真相」与「运行真相」长期分离。

## 二、机制：切换怎么生效

### 启动链（切换后）

```
launchd (ai.hermes.gateway)
  └─ osascript 包装（stderr_timestamp 时间戳）
      └─ /Users/stan/code/ai/github/fork/hermes-agent/.hermes/bin/hermes   ← 唯一改动点
          └─ store python -I + bootstrap（sys.path 首位插 fork 根）
              └─ gateway run → 拉起全部 CLI 会话 → 直读 fork 源码
```

### 依赖链（不变）

PM 依赖环境按「树路径 sha256 前 16 位」登记：`41bc58c3072daa3c` = sha256("/Users/stan/code/ai/github/fork/hermes-agent") 前 16 位。该 key 下的 `~/.hermes/installs/41bc58c3072daa3c/environments/*/venv` 依赖环境继续服务 fork 树——切换只换源码树，不换依赖环境，零迁移。

fork 仓 `venv/`（uv sync --frozen 管理）服务仓库内测试与脚本。

## 三、切换事实清单（全部实测 2026-10-14）

| 项 | 值 |
|---|---|
| plist | ~/Library/LaunchAgents/ai.hermes.gateway.plist，ProgramArguments 两处路径改指 fork |
| 备份 | 同目录 ai.hermes.gateway.plist.bak-workspace-1014（完整原文） |
| 生效验证 | 新 gateway 进程 launcher 内嵌 fork 根，命令行 workspace 引用 0 处 |
| 平台连接 | 飞书 websocket / 微信 / API server(127.0.0.1:8642) 全部 Connected |
| 依赖加载 | "Turn machinery warmed (21 tool schema)" 成功 |
| 切换时序 | 旧 gateway SIGTERM 排空约 50s，期间新实例首拉被残留 host record 拒绝一次，launchd KeepAlive 排空后自动重拉成功（plist 原设计自愈路径） |

## 四、使用纪律（切换后的行为变化）

1. **改码即生产**：fork 仓保存的任何代码，gateway 新拉起的会话立即加载。修复即时生效；反面是改到一半的代码也会进生产会话。纪律：先跑测试再落盘；落盘错文件 git checkout 秒级回滚。
2. **upstream merge 期间**：冲突中间态会被新会话加载——低频场景，merge 完成即愈。
3. **workspace 快照退役**：`~/.hermes/installs/*/environments/*/workspace` 不再被加载，留着不碍事，可择机清理。
4. **工具退役**：tools/workspace_drift_check.py 无对账对象，保留作 installs 形态排查兜底。
5. **hermes update 不再触达源码**：更新 fork 代码走 git（upstream merge / 手工修复）；PM 依赖更新走 `hermes pm install` 或仓内 uv sync。

## 五、回滚（一条通道，两条命令）

```bash
cp ~/Library/LaunchAgents/ai.hermes.gateway.plist.bak-workspace-1014 ~/Library/LaunchAgents/ai.hermes.gateway.plist
launchctl bootout gui/$(id -u)/ai.hermes.gateway; sleep 3
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

恢复后回到 installs 快照形态（双真相随之回归，drift 工具恢复职责）。

## 六、验证命令（引用数字前先复演）

```bash
# gateway 从哪棵树启动
ps aux | grep "gateway run" | grep -v grep | head -2   # 应见 fork/hermes-agent
# 平台连接
grep -E "Connected|listening" ~/.hermes/logs/gateway.log | tail -5
# install_key 归属
python3 -c "import hashlib; print(hashlib.sha256('/Users/stan/code/ai/github/fork/hermes-agent'.encode()).hexdigest()[:16])"  # = 41bc58c3072daa3c
```
