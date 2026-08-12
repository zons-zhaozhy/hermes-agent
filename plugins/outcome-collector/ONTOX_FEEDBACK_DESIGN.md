# OntoX 结构化成败回收机制方案

> 受 BigBang-V1 data-layer RSI 启发，为 OntoX 四驱家族设计的信号回收闭环。
> Hermes 侧已实现（outcome-collector plugin + analyze.py），本方案为 OntoX 侧蓝图。

## 核心差距诊断

### 信号断点全景

```
              信号产生              信号采集         信号聚合         信号分析        行动反馈
              ────────              ────────         ────────         ────────        ────────
Loom     ✅ step执行结果         ❌ 丢弃            ❌               ❌              ❌
DBChat   ✅ SQL执行结果          ❌ 仅日志          ❌               ❌              ❌
OMS      ✅ HTTP状态码           ❌ 仅日志          ❌               ❌              ❌
```

**核心问题：OntoX 四驱都在产生丰富的成败信号，但没有一个子系统将这些信号结构化持久化、聚合分析、反馈改进。**

这是 BigBang 方法论揭示的最大浪费——不是缺少数据，而是缺少闭环。

---

## 方案分述

### 1. Loom：激活死代码 + post-execution collector

#### 现状（源码验证 2026-07-16）

- `_normalize()` 存储了 check/on_missing/on_failure 字段，但执行层从不消费
- `_run_serial` 的 `if not ok: break` 对所有步骤一视同仁，无 per-step 失败策略
- Loom 场景执行后的结果（哪个 step 失败、为什么失败、输入输出是否匹配 StepContract 声明）全部蒸发

#### 改动方案：post-execution collector（不改引擎核心）

新增 `loom_outcome_collector.py` 模块，在场景执行完成后（不侵入引擎执行路径）旁路收集信号。

**PG 表结构**（Docker 容器内 PG，禁止本地 PG）：

```sql
-- 场景级结果
CREATE TABLE loom_scenario_outcomes (
    id BIGSERIAL PRIMARY KEY,
    scenario_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,         -- success | partial | failed
    total_steps INTEGER,
    succeeded_steps INTEGER,
    first_failure_step VARCHAR(255),     -- 第一个失败 step（根因定位关键）
    failure_summary TEXT,
    duration_ms BIGINT,
    tenant_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 步骤级结果
CREATE TABLE loom_step_outcomes (
    id BIGSERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES loom_scenario_outcomes(run_id),
    step_id VARCHAR(255) NOT NULL,
    step_type VARCHAR(100),              -- llm_chat | data_batch | condition | ...
    status VARCHAR(50) NOT NULL,
    contract_match BOOLEAN,              -- StepContract 声明 vs 实际输入输出
    error_message TEXT,
    duration_ms BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_loom_so_scenario ON loom_scenario_outcomes(scenario_id);
CREATE INDEX idx_loom_so_status ON loom_scenario_outcomes(status);
CREATE INDEX idx_loom_step_run ON loom_step_outcomes(run_id);
```

**collector 接口**（伪代码）：

```python
class ScenarioOutcomeCollector:
    def collect(self, scenario_id: str, execution_result) -> dict:
        steps_data = []
        for s in execution_result.steps:
            steps_data.append({
                'step_id': s.step_id,
                'step_type': s.step_type,
                'status': s.status,
                'duration_ms': s.duration_ms,
                'error': s.error_message,
                'contract_match': self._check_contract(s),
            })
        return {
            'scenario_id': scenario_id,
            'status': execution_result.overall_status,
            'steps': steps_data,
            'total_steps': len(execution_result.steps),
            'succeeded_steps': sum(1 for s in steps_data if s['status'] == 'success'),
            'first_failure_step': self._find_first_failure(steps_data),
        }
```

**最小侵入接入点**：在 `_run_serial()` 返回结果后，调用 collector：

```python
# loom engine, after execution completes
result = self._run_serial(...)
# 旁路收集（不阻塞返回路径，收集失败不影响主流程）
try:
    collect_scenario_outcome(scenario_id, result)
except Exception:
    logger.warning("outcome collection failed", exc_info=True)
```

#### 分析查询

```sql
-- 哪些场景失败率最高？
SELECT scenario_id,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    ROUND(AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END) * 100, 1) as failure_rate,
    MODE() WITHIN GROUP (ORDER BY first_failure_step) as common_fail_step
FROM loom_scenario_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY scenario_id
HAVING COUNT(*) >= 3
ORDER BY failure_rate DESC;

-- 哪些 step_type 最常失败？
SELECT step_type,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE status != 'success') as failures,
    ROUND(AVG(CASE WHEN status != 'success' THEN 1.0 ELSE 0.0 END)*100,1) as failure_rate
FROM loom_step_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY step_type
ORDER BY failure_rate DESC;

-- Contract 不匹配的步骤（StepContract 声明 vs 实际输入输出）
SELECT step_id, step_type, error_message
FROM loom_step_outcomes
WHERE contract_match = false
ORDER BY created_at DESC LIMIT 20;
```

---

### 2. DBChat：NL2SQL 质量回收

#### 现状

NL2SQL 有最天然的可验证信号——SQL 对错立即知晓。但每次查询结果只存在日志中，哪些自然语言模式反复生成错误 SQL 完全没有回收。

#### PG 表结构

```sql
CREATE TABLE nl2sql_outcomes (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    natural_language TEXT NOT NULL,        -- 用户原始自然语言
    generated_sql TEXT,
    execution_status VARCHAR(50),          -- success | syntax_error | permission_denied | timeout | empty_result
    row_count INTEGER,                     -- 返回行数
    error_message TEXT,
    user_feedback VARCHAR(50),             -- correct | incorrect_too_few | incorrect_wrong_table | null（未提供）
    data_source_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_nl2sql_status ON nl2sql_outcomes(execution_status);
CREATE INDEX idx_nl2sql_feedback ON nl2sql_outcomes(user_feedback);
```

**隐式失败信号**：用户看到查询结果后立即重新表述查询 = 上一条查询不对。可通过分析同一 session 内连续查询的时间间隔和语义相似度推导。

#### 分析查询

```sql
-- 哪些自然语言模式反复生成错误 SQL？
SELECT
    substring(natural_language from '(查询|统计|列出|计算).*') as pattern,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE execution_status != 'success') as errors,
    ROUND(AVG(CASE WHEN execution_status != 'success' THEN 1.0 ELSE 0.0 END)*100,1) as error_rate
FROM nl2sql_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY pattern
HAVING COUNT(*) >= 5
ORDER BY error_rate DESC;

-- 用户隐式否定率（重新查询 = 上一条不对）
SELECT
    COUNT(*) FILTER (WHERE user_feedback LIKE 'incorrect%') as implicit_failures,
    COUNT(*) as total,
    ROUND(AVG(CASE WHEN user_feedback LIKE 'incorrect%' THEN 1.0 ELSE 0.0 END)*100,1) as pct_implicit_fail
FROM nl2sql_outcomes
WHERE created_at > NOW() - INTERVAL '7 days';
```

---

### 3. OMS：元数据健康回收

#### 现状

OMS 的 CRUD 日志是最丰富的信号源。每次操作的成败直接反映元数据质量（字段名正确性、数据类型定义、约束设计）。

#### PG 表结构

```sql
CREATE TABLE oms_operation_outcomes (
    id BIGSERIAL PRIMARY KEY,
    object_type VARCHAR(255) NOT NULL,     -- api_name of object type
    operation VARCHAR(10) NOT NULL,        -- GET | POST | PUT | DELETE
    status_code INTEGER NOT NULL,
    status_detail VARCHAR(50),             -- success | idempotent_409 | no_rows_matched | validation_error
    error_message TEXT,
    field_errors JSONB,                    -- 哪些字段导致验证失败
    data_source_id BIGINT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_oms_op_type ON oms_operation_outcomes(object_type, operation);
CREATE INDEX idx_oms_op_status ON oms_operation_outcomes(status_code);
```

#### 分析查询

```sql
-- 哪些对象类型的操作失败率最高？
SELECT object_type, operation,
    COUNT(*) as total,
    COUNT(*) FILTER (WHERE status_code >= 400) as failures,
    ROUND(AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END)*100,1) as failure_rate
FROM oms_operation_outcomes
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY object_type, operation
HAVING COUNT(*) >= 10
ORDER BY failure_rate DESC;

-- 哪些字段反复验证失败？（元数据设计缺陷定位）
SELECT object_type,
    je.value as field_name,
    COUNT(*) as failure_count
FROM oms_operation_outcomes,
    jsonb_array_elements(field_errors->'fields') je
WHERE field_errors IS NOT NULL
GROUP BY object_type, je.value
ORDER BY failure_count DESC LIMIT 20;
```

---

## Meta-Critic 层：Loom 自分析场景

用 Loom 自己分析自己的信号——这就是自进化的 OntoX 版本。

```yaml
# scenario: system-health-meta-critic
who_am_i:
  name: System Health Meta-Critic
  description: 定期分析四驱执行信号，识别系统薄弱环节

i_can_do:
  - does:
      type: data_batch
      data_source_id: <<loom_pg>>
      sql: |
        SELECT 'loom' as service, scenario_id as entity_id,
               failure_rate, common_fail_step
        FROM (
          SELECT scenario_id,
            ROUND(AVG(CASE WHEN status='failed' THEN 1.0 ELSE 0.0 END)*100,1) as failure_rate,
            MODE() WITHIN GROUP (ORDER BY first_failure_step) as common_fail_step
          FROM loom_scenario_outcomes
          WHERE created_at > NOW() - INTERVAL '7 days'
          GROUP BY scenario_id HAVING COUNT(*) >= 3
        ) loom_fail
        WHERE failure_rate > 30
        UNION ALL
        SELECT 'dbchat', pattern, error_rate, null
        FROM (
          SELECT substring(natural_language from '(查询|统计).*') as pattern,
            ROUND(AVG(CASE WHEN execution_status != 'success' THEN 1.0 ELSE 0.0 END)*100,1) as error_rate
          FROM nl2sql_outcomes
          WHERE created_at > NOW() - INTERVAL '7 days'
          GROUP BY pattern HAVING COUNT(*) >= 5
        ) dbchat_fail
        WHERE error_rate > 40
        UNION ALL
        SELECT 'oms', object_type||'/'||operation, failure_rate, null
        FROM (
          SELECT object_type||'/'||operation as entity_id,
            ROUND(AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END)*100,1) as failure_rate
          FROM oms_operation_outcomes
          WHERE created_at > NOW() - INTERVAL '7 days'
          GROUP BY object_type, operation HAVING COUNT(*) >= 10
        ) oms_fail
        WHERE failure_rate > 20
    output:
      what: weak_points

  - does:
      type: llm_chat
      messages: |
        分析以下系统薄弱点，按严重程度排序，给出具体修复建议：
        {{weak_points}}
    output:
      what: improvement_suggestions

  - does:
      type: data_batch
      sql: |
        INSERT INTO system_improvement_log
        (service, entity_id, issue, suggestion, created_at)
        VALUES (:service, :entity_id, :issue, :suggestion, NOW())
```

---

## 实施优先级

| 优先级 | 组件 | 改动量 | ROI |
|---|---|---|---|
| P0 | Loom post-execution collector | 1 新模块 + 2 新表 | 最高——激活已有死代码 |
| P1 | DBChat NL2SQL 回收 | 1 新表 + 接入点 | 高——最天然可验证信号 |
| P2 | OMS 操作回收 | 1 新表 + middleware | 中——已有日志需结构化 |
| P3 | Meta-Critic Loom 场景 | 1 个场景 YAML | 高——闭环完成 |

---

## 不做什么

| 不做 | 理由 |
|---|---|
| 改 Loom 引擎核心 (_run_serial) | 风险太高，post-execution collector 不侵入引擎就能收集信号 |
| 实时信号分析 | OntoX 是请求驱动不是事件驱动，批量分析足够 |
| 自动修改 OMS 元数据 | 元数据是 SSOT，自动修改风险极高 |
| 复杂 ML 异常检测 | SQL 聚合 + 阈值已覆盖核心场景 |

---

## 验证标准

每个子系统的信号回收必须满足：
1. 信号采集率 > 95%（不漏记任何调用结果）
2. 分析结果可复现（相同数据 → 相同报告）
3. 不侵入主流程性能（收集开销 < 调用本身的 1%）
4. 闭环可验证（信号 → 改进 → 改进后的信号变化可见）
