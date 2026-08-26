# 自建守卫插件群审计与加固（2026-08-26）

> 实测环境：源码仓 `~/code/ai/github/fork/hermes-agent`（push cnb），运行副本 `~/.hermes/hermes-agent`（merge 同步）。
> 所有结论基于当日 agent.log / 进程内直调 / pytest 实测，数据窗口 16:56–17:42。

## 一、启用插件真机验证结论

5 个新启用插件全部实证生效（非摆设）：

| 插件 | 触发方式 | 实测证据 |
|---|---|---|
| curl-safety | terminal 执行 curl/httpx | agent.log `curl-safety: 注入 HTTP 请求安全提醒`；对照组 ls 无注入 |
| security-guidance | write_file 内容含 pickle.loads/eval( | 工具结果注入 `pickle_deserialization`/`eval_injection` 警告（warn 不阻断） |
| engine-invariants | post_llm_call 哨兵 | 注册成功；历史干净时零输出（设计行为） |
| tool-safety | 同工具连续 3+ 次失败 | 本会话 terminal 连错 3 次触发 `same_tool_failure_warning` |
| skill-drift-check | 会话启动 | 注册成功；skill 引用源码 SHA 漂移时才警告 |

启用机制：bundled 插件必须 `hermes plugins enable <name>` 进 `~/.hermes/config.yaml` 的 `plugins.enabled` 白名单，否则 discover 静默跳过。

## 二、发现并修复的漏洞：source-code-write-guard 逃生舱误放行

**漏洞**（进程内实测实锤）：`_is_escape_hatch` 旧逻辑只查命令是否**包含** `plugins/` 等关键词——`ls plugins/` 出现在命令任意位置即整条命令放行，同一命令里重定向写任意无关源码文件也被放走。当日日志 escape hatch 触发 8 次、真拦截 0 次。

**修复**（commit 8e63d165，运行副本 c10ccfcb2）：逃生舱收窄为「所有可检测写入目标都是护栏自身文件才放行」。新增 `_split_compound`（复合命令 &&;| 切分，引号感知）/`_extract_all_write_targets`/`_is_guard_owned_path`。无写入目标可提取时退回关键词匹配（保留护栏自指合法通道）。

**回归测试**：`tests/test_source_code_write_guard.py` 5 用例全绿——漏洞场景（关键词在场写无关源码→拦截）、对照（无关键词→拦截）、护栏自指（写 plugins/ 下→放行）、只读（→放行）、混合目标（写护栏+写无关→拦截）。

## 三、辅助模型守卫链超时处置

**问题**：4 个 LLM 守卫（yinyang_restate_guard / devil_advocate_audit / reply_certainty_checker / completion_boundary_audit）走 scnet DeepSeek-V4-Flash-0731，55 次调用 16 次失败（18%，全为超时/502）。旧 `transient_retries: 1` 下同供应商 1 秒后原样重发再撞 15s 超时墙，单次失败净耗 32s；失败后 fail-open，审计白丢。

**处置**（config.yaml，已实测验证）：
1. `auxiliary.transient_retries: 1 → 0` —— 对端 502/超时时同供应商立即重试纯浪费。
2. 4 个任务各配 `fallback_chain` → zai glm-4.5-flash（per-entry timeout 20s，引擎原生支持，见 `agent/auxiliary_client.py` `_fallback_timeout_for`；官方文档 fallback-providers.md「per-entry timeout」节与本行为一致）。
3. `allow_main_model_fallback: false` 保持——fallback 链不抢 glm-5.3 主模型。

**验证**：配置生效后 judge 失败数 0；直调 4 任务全通；reply_certainty_checker 实测 26.5s（>15s 主超时）拿到结果——超时→fallback 切换成功的实证。

## 四、文档对齐记录

- `plugins/source_code_write_guard/plugin.yaml` description 补齐三层检测 + 逃生舱语义（本轮同步）。
- 官方站 `website/docs/user-guide/features/fallback-providers.md`（及 zh-Hans 镜像）对 `auxiliary.<task>.fallback_chain`/per-entry timeout 的描述与引擎实际行为一致，无需修改（逐节比对实测）。
- `transient_retries` 在官方站无文档页（grep 0 命中）——引擎默认值 2，本机显式改 0，属本机策略不进上游文档。

## 五、遗留观察项

- decision_tree / adaptive_reasoning：注册后日志零行为记录，观察窗不足未定性。
- 4 个集成类插件（chronos/google_meet/langfuse/indexer-sync 等）仍闲置（未配外部服务）。
- no-pushback（正则误拦面宽）/ sandwich-inject（缺 sandwich.yaml）按拍板暂缓。
