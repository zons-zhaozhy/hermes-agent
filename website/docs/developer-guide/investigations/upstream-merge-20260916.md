# 上游同步合并报告 2026-09-16/17

## 同步范围
- 上游 HEAD: 6bea68a1de (2026-09-16, fix(desktop): route project writes to the live profile in All profiles #112943)
- 上次同步点: b7b35a84b7 (2026-09-12)
- 同步提交数: 2040
- 双方都改过的文件: 83; git 冲突文件: 26 (9 代码 + 17 文档)
- 备份分支: backup/pre-sync-20260916

## 网络通道实录
GitHub SSH 443 挂死(零字节超时), HTTPS 多次断流(Empty reply / curl 18 /
收尾 EOF)。最终方案: HTTP/1.1 降级 + depth=2500 浅 fetch——对象下载完整
落地后连接断在引用更新阶段, 手动 git update-ref 指向已下载 tip, 四层完整
性核验通过(对象类型全合法 / rev-list 28095 全链可走 / ls-tree 112 顶层
条目 / cat-file 零损坏)。

## 冲突逐块裁定(9 个代码文件)

| 文件 | 冲突 | 裁定 |
|---|---|---|
| tools/file_tools.py | 1 | 并集: 上游 _mark_full_write_baseline(防连续写误拦) + fork _fp_record_write(指纹重打戳), 二者职责不同共存 |
| tools/memory_tool_store.py | 1 | 并集: 上游 mkdir_under_hermes_home(profile 安全) + fork _ledger_snapshot(账本) |
| tools/skill_manager_tool.py | 1 | 上游为结构重构(锁窗口+审计账本+tool_error), fork 侧 tool_rejection 被上游等价 tool_error 取代 → 取上游 |
| tools/checkpoint_manager.py | 6 | 上游将 _prune/_enforce_size_cap 内联循环提炼为 _ref_commit_count/_rewrite_ref_to/_gc_store/_shrink_store_to_cap 等辅助函数 + gc 从 checkpoint 路径移到周期 prune(_mark_gc_pending) → 全取上游结构, fork 无独立改动; 补 _MB 常量 |
| tools/process_registry.py | 1 | 并集: fork 的 live-child guard(detached-alive 防假退出通知) + 上游"已知退出态直接记录"警告 → fork 逻辑为主干, 吸收上游 warning |
| hermes_cli/goals.py | 1 | __all__ 并集(fork 多 4 个符号: workspace_fingerprint/judge_reason_fingerprint 等, 上游压缩格式) |
| 3 个测试文件 | - | 直接采纳(add)——测试期望冲突以实现侧为准重跑验证 |

## 文档冲突(17 个)
docs/ → website/docs/developer-guide/ 目录改名(AU 状态): fork 侧文档内容
完整保留, 按上游新目录结构归位。

## 上游重点变更(与 fork 相关)
- agent/: 114 文件(会话提示索引不水合全文提速 / 压缩 provider-scoped
  model_thresholds / observer 口径)
- vault: 检测到的密码管理器默认启用(opt-out 契约)
- checkpoint: gc 移出工具调用路径(修复长阻塞)
- process_registry: PTY 回收/脱离会话语义
- apps/desktop: 大量修复+性能(虚拟化/共享 composer/approval 栈)——fork
  不用桌面端, 低风险
- Docker 路由 profile 注入、Cloudflare Access OAuth

## 验证
- 逐文件语法核验(ast.parse)全过
- 全量测试套件: 见提交后补记(报告先行, 测试跑完后 commit 补录结果)

## fork 保留差异清单(未采纳上游)
- plugins/ 全部 guard 体系(fork 专有, 上游无对应)
- .hermes-rules.md / memory 纪律条目
- tool_rejection → tool_error 转换点已在上游等价实现, fork 不再保留双轨
