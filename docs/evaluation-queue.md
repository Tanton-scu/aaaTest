# 异步 Candidate Evaluation Queue

> 历史阶段说明：本文件保留早期 SQLite v1 设计记录，不代表当前 1.0.0 契约。当前 MySQL/SQLite、external Worker、evaluation-v2 identity、lease fencing、FailureClassifier 与 Candidate clone 规则请以 [`EVALUATION_QUEUE.md`](EVALUATION_QUEUE.md) 为准。

## 状态与职责

`EvaluationJob` 表示 intent/scheduling/retry，状态为 `PENDING/RUNNING/RETRY_WAIT/SUCCESS/DEAD/CANCELLED`；`EvaluationResult` 是 objective、trajectory、best configuration 和 used budget 的实验事实。Job 重试不会覆盖已成功 Result。

v1 是 SQLite-backed 单 worker，不引入外部 broker。claim 使用短事务和 lease；事务提交后才运行长 benchmark，因此 benchmark 期间不持有 DB lock。

## Logical identity 与预算

idempotency key = SHA-256(`task_id|candidate_id|seed|budget|contract_version`)；数据库有 UNIQUE 约束。enqueue 在 `BEGIN IMMEDIATE` 中按顺序执行：

1. 先查唯一 key，重复则返回现有 job，不再次预留；
2. 检查 `total_budget - consumed - reserved`；
3. 插入 job 并增加 `reserved_evaluations`。

成功 final commit 将预留移除并按 `result.used_budget` 结算 consumed；`DEAD/CANCELLED` 释放预留。Result 实际用量不得超过 job 预留。预算归零时新 logical submission 抛出中文 `BudgetExhaustedError`，重复提交仍返回原 job。

## 四个失败问题的精确答案

### Benchmark timeout

执行适配器抛出 `EvaluationTimeoutError`；worker 视为 transient，按 `2^(attempt-1)` 秒回退到 `RETRY_WAIT`。达到 `max_attempts` 后进入 `DEAD` 并释放预算。v1 Queue 定义处理契约；真实 candidate 子进程硬超时在阶段 14 的 executor adapter 实现，当前 Fake Harness 注入同一异常。

### Worker 在 benchmark 后、final status 前崩溃

Job 保持 `RUNNING`，结果和预算均未提交。lease 过期后 recovery 置回 `PENDING`（多次过期达上限则 `DEAD`），另一个/重启后的单 worker 可重新执行。Benchmark 计算可能发生两次，这是 at-least-once execution；但唯一 logical result 只有一条，预算只在成功 final commit 结算一次。项目不声称 exactly-once。

### 同 candidate/seed 提交两次

SQLite UNIQUE idempotency key 返回同一 job，产生 `DUPLICATE_EVALUATION_IGNORED`，不重复预留、评价或结算。不是 Python “先查再插”的唯一保障。

### 预算为零

enqueue 的同一 SQLite 写事务检查 consumed + reserved；不足时不插 job、不改变 reserved。已预留 job 可继续结算或失败释放。

## 错误分类

- `CandidateInvalidError`：deterministic permanent，直接 `DEAD`，candidate 标记 `INVALID`，不盲目重试。
- `EvaluationTimeoutError` / `TransientEvaluationError`：有界 retry。
- 未分类异常：默认 permanent，避免无限重试。
- lease crash：恢复为可 claim；超过最大 attempts 后 dead letter。

## Harness

```powershell
$env:PYTHONPATH='src'
python -m prievo_agent.cli.queue_demo
```

确定性报告覆盖 retry、唯一 job、预算阻止、invalid/timeout dead letter、lease recovery、one logical result/one charge。

## 限制

当前只有一个 worker loop；SQLite 适合本地 demo，不声称分布式吞吐或 exactly-once。若多进程 benchmark 吞吐成为真实瓶颈，可在相同 Queue Port 后评估 broker。
