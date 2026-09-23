# sandwich-inject

三明治架构插件 —— 确定性代码包抄概率 LLM。

参考架构来自《我给Agent装了「三明治」架构》：

- **上层 Pre-hook（强制注入）**：每轮 LLM 请求前把场景 SOP + 知识库内容注入
  user message，并附"禁止自行猜测"工作守则。知识已注入，Agent 无需自行
  检索 —— 把"要不要查库"的选择题变成确定性事实。
- **中层 推理层**：Agent 只做推理。
- **下层 Post-hook（工具契约校验）**：校验上一轮是否调用了场景必需工具，
  未调用则在下一轮注入【校验报错】要求补做；连续 N 轮违规输出
  `NEED_HUMAN_INTERVENTION` 转人工。

## 实现通道（零核心改动）

只挂一个 `pre_llm_call` hook —— 这是 Hermes 框架中**唯一**返回值被消费的
纯插件通道（`agent/turn_context.py` 将返回值注入 user message，含超大输出
spill 保护）：

| 通道 | 返回值消费 | 结论 |
|---|---|---|
| `pre_llm_call` | 注入 user message | ✅ 本插件使用 |
| `pre_verify` | 触发重试 | ❌ 仅编辑代码轮次触发，纯查询场景失效 |
| `post_tool_call` | 丢弃（observer） | ❌ 不能拦截 |
| `pre_api_request` | 丢弃（observer） | ❌ 不能注入 |

## 配置

默认读取 `~/.hermes/sandwich.yaml`（可用 `HERMES_SANDWICH_CONFIG` 覆盖）。
参考 `sandwich.example.yaml`。

场景字段：

| 字段 | 说明 |
|---|---|
| `name` | 场景名（内部计数用） |
| `match_keywords` | 用户消息命中任一关键词即激活 |
| `sop` | 强制流程，注入 `【强制SOP】` 块 |
| `knowledge` | 静态知识，注入 `【已注入的最新知识】` 块 |
| `knowledge_files` | 知识文件列表（mtime 缓存，改文件自动刷新） |
| `required_tools` | 上轮必须调用过的工具名列表 |
| `max_retries` | 连续违规上限，缺省 3 |

## 行为语义

- 场景**会话级锁定**：首个命中后该会话持续注入，后续轮次关键词消失也不失活
- 违规计数**每会话每场景隔离**，多会话并发安全（全程持锁）
- 转人工后停止自增计数，持续输出 `NEED_HUMAN_INTERVENTION` 信号
- 会话追踪**有界**（上限 512），长跑进程（gateway）不泄漏内存
- 无配置文件 = 零行为（插件注册但每次调用返回 None）

## 测试

```bash
scripts/run_tests.sh tests/plugins/test_sandwich_inject.py
```
