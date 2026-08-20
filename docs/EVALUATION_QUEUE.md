# 持久化 Evaluation Queue

## 1. 设计目标与语义

Candidate Evaluation 被建模为 durable `EvaluationJob`，而不是 PriEvO Core 内的一次无记录函数调用。系统追求的是：

- 一个 logical evaluation 只有一个有效 Job/Result；
- 并发提交不能超预算；
- 多 Worker 不能同时拥有同一个 Job；
- Worker 崩溃后 Job 可以重新调度；
- 旧 owner 的迟到结果不能覆盖新 owner；
- 基础设施瞬时故障重跑同一 Candidate，Candidate 缺陷则进入 Repair 生成新版本。

当前执行语义是 **at-least-once benchmark execution + idempotent logical commit**，不是 exactly-once execution。Worker 可能在 benchmark 完成、数据库提交前崩溃，恢复后 benchmark 会再执行；数据库唯一键、owner fencing 和唯一 Result 使逻辑结果只提交一次。

核心入口：

- 领域对象：`src/prievo_agent/domain/models.py`；
- 提交/Worker：`src/prievo_agent/runtime/evaluation_queue.py`；
- Runtime 等待器：`src/prievo_agent/runtime/evaluation_driver.py`；
- 失败分类：`src/prievo_agent/runtime/failure_classifier.py`；
- MySQL 事务：`src/prievo_agent/infrastructure/mysql_store.py`；
- SQLite Demo 事务：`src/prievo_agent/infrastructure/sqlite_store.py`；
- 独立 Worker：`src/prievo_agent/cli/evaluation_worker.py`。

## 2. 数据模型与状态机

`EvaluationJob` 保存 `run_id/task_id/candidate_id`、seed、budget、幂等键、attempt/max_attempts、available time、owner token、lease expiry、result/error。`EvaluationResult` 保存 objective、完整 trajectory、best configuration 和实际 used budget。候选代码不复制进 Job，而由 `candidate_id` 回表读取。

```text
PENDING --claim--> RUNNING --success/fenced commit--> SUCCESS
                         \--retryable failure------> RETRY_WAIT --到期 claim--> RUNNING
                         \--non-retry/max attempts-> DEAD
PENDING/RETRY_WAIT --run cancel--------------------> CANCELLED
RUNNING --lease expiry--> RETRY_WAIT 或 DEAD
```

状态的数据库更新和 Run 预算账本在同一短事务内结算。耗时 benchmark 永远不在数据库事务内运行。

## 3. Logical Evaluation 幂等

`EvaluationQueueService.submit()` 使用 `evaluation-v2` 物料规范化为 JSON 后取 SHA-256：

```json
{
  "material_version": "evaluation-v2",
  "task_id": "...",
  "candidate_id": "...",
  "seed": 2024,
  "budget": 10,
  "dataset_digest": "...",
  "evaluator_version": "...",
  "evaluation_parameters_version": "..."
}
```

Run ID 通过 Job 归属和 Candidate 归属校验参与数据库一致性；稳定 Job ID 和唯一 `idempotency_key` 阻止重复 logical evaluation。普通 Candidate 只允许绑定一个 logical identity：相同 identity 重放返回原 Job；Dataset 内容、evaluator 实现、seed、预算或必要参数版本任一变化，必须先 clone 新 Candidate，再提交新的 identity。Final Optimization 已按 seed 建立 immutable clone。SQLite 的 Candidate 唯一索引和 MySQL V006 在数据库层兜底，避免出现“第二 Job 成功、读取旧 Result、Candidate 却被新 objective 覆写”的分裂事实。

`enqueue_job()` 的语义不是“先查再插”：

1. 进入短事务并锁定 Run；
2. 查询唯一 key，已存在则直接返回已有 Job；
3. 锁定并校验 Candidate/Task/Run 归属；若 Candidate 已绑定不同 identity 或已有无法对账的 Result，事务内拒绝并要求 clone；
4. 计算 `available = total_budget - consumed - reserved`；
5. 不足则拒绝；足够则插入 Job 并增加 `reserved`；
6. 提交事务。

重复提交不会增加 reservation，也不会生成第二个 Result；它会记录 `DUPLICATE_EVALUATION_IGNORED`。MySQL 用行锁和唯一约束作为并发防线，SQLite Demo 用 `BEGIN IMMEDIATE`。

## 4. Budget Reservation

Run 账本分成：

```text
total_budget
consumed_evaluations
reserved_evaluations
available = total - consumed - reserved
```

预算在 submit 时预留，在成功 commit 时把 reservation 转成实际 consumed；永久失败或取消时释放 reservation；可重试失败仍保留 reservation，因为同一 logical job 还会执行。这样两个提交者即使同时看到“最后一份预算”，也必须在锁住同一 Run 行后串行决定。

成功事务同时完成：

- 校验当前 `RUNNING + owner token`；
- 插入唯一 EvaluationResult；
- 更新 Candidate objective/status/result artifact；
- Job 转 `SUCCESS` 并关联 result；
- `reserved -= job.budget`；
- `consumed += result.used_budget`。

若 evaluator 报告的 used budget 不满足 Job 契约，final commit 会失败而不会默默污染预算。`tests/test_evaluation_queue.py` 和 Harness `duplicate_evaluation_submit`、`budget_contention` 场景验证重复提交和预算不超卖；这些 Harness 场景基于隔离 SQLite，证明事务语义，不是 MySQL 性能数据。

## 5. MySQL Claim、Lease 与 Fencing

Full 模式的 Job ownership 完全在 MySQL。`claim_next_job()` 在事务中选择可执行的 `PENDING/RETRY_WAIT` Job，排除已暂停/取消 Run，并使用 `FOR UPDATE SKIP LOCKED` 避免多个 Worker 阻塞或认领同一行。claim 后写入：

- `status=RUNNING`；
- attempt 自增；
- 本进程唯一 `worker_id` owner token；
- `lease_expires_at`。

claim 返回后，Worker 会立即追加 `EVALUATION_STARTED` durable Event，记录 Job、Candidate、worker owner token、attempt 与 lease expiry；它服务于审计时间线，不是新的队列事实源。

Worker ID 由 prefix、hostname、PID 与随机 nonce 组成，代表一次进程 incarnation。成功、失败与续租均要求 owner token 匹配；stale sweep 改写状态或新 Worker 接管后，旧 Worker 的后续写入会被 fencing 拒绝。`tests/test_evaluation_lease_fencing.py` 使用可控时钟验证“旧 owner 的迟到成功/失败都不能覆盖新 owner”。

当前 `EvaluationWorker` 在 evaluator 前和 evaluator 后续租，也暴露 `renew_lease()`，但独立 Worker **没有后台 heartbeat**。CLI 强制：

```text
lease_seconds > evaluation_timeout_seconds
```

Compose 默认 lease 30 秒、benchmark timeout 10 秒。若 benchmark 超过 lease，后置续租失败，当前结果不会提交，等待 stale recovery/requeue。这是 owner fencing，不是周期 heartbeat。

## 6. 独立 Worker 与 benchmark 边界

Full 模式：

```text
App / PriEvO Runtime                  Evaluation Worker
submit durable Job                    claim from MySQL
poll durable status          <---->   run benchmark subprocess outside tx
consume durable Result                fenced success/failure transaction
```

`DurableEvaluationJobDriver(mode="external")` 永不调用 App 内 evaluator。ready Job 也被视为等待独立 Worker 的正常状态，并按有上限的 polling delay 等待，不会 busy-spin。Demo 的 `inline` 模式才调用注入的 `EvaluationWorker.run_once()`。

Worker 的 evaluator 是 `src/prievo_agent/algorithm/executable_dataset_evaluator.py`。它把 Job seed/budget 覆盖到 Task 副本，保证 multi-seed Final Optimization 不会偷偷使用 Task 默认 seed。候选在 `python -I` 子进程执行，父 Worker 监督 timeout、退出码和协议结果；POSIX 尽力设置 rlimit。该层不是强安全沙箱，生产环境仍需无凭据、禁网、资源限制的容器/VM。

`tests/test_external_evaluation_runtime.py` 证明 Full/external path 中 App evaluator 调用为 0，独立 Worker 消费演化与 Final Optimization Job；`tests/test_evaluation_worker_service.py` 覆盖 CLI settings、无 heartbeat 的 lease 校验、bounded polling、周期 stale sweep、SIGTERM/health/Compose 合同。

## 7. Retry、Repair 与 Dead Letter

`FailureClassifier` 只根据明确异常类型、error code、return code 和 stderr evidence 决策，不让 LLM 猜测基础设施状态：

| FailureType | 动作 | 含义 |
| --- | --- | --- |
| `NETWORK`、`WORKER_CRASH`、`TRANSIENT_INFRA` | Backend `RETRY` | 同一 Candidate、同一 logical Job 重新执行 |
| `SYNTAX_ERROR`、`RUNTIME_ERROR`、`INTERFACE_ERROR` | Agent `REPAIR` | 原 Candidate immutable，新建修复版本再评价 |
| `ALGORITHM_TIMEOUT`、`ALGORITHM_OOM`、`LOGIC_FAILURE` | Agent `REPAIR` | 算法自身问题，不伪装成基础设施重试 |
| `UNKNOWN` | `DEAD` | 证据不足时保守终止 |

Backend retry 使用 `2 ** (attempts - 1)` 秒的有界调度时间，Job 进入 `RETRY_WAIT`；达到 `max_attempts` 后转 `DEAD` 并释放 reservation。Runtime 根据 terminal failure 生成 `CANDIDATE_FAILURE`，可配置 Repair workflow 生成新 Candidate lineage；Repair 不修改原 Candidate。

证据包括 `tests/test_failure_classifier.py`、`tests/test_runtime_retry_scheduler.py`、`tests/test_repair_workflow.py`，以及 Harness 的 `evaluation_transient_retry`、`candidate_syntax_failure`、`candidate_timeout`、`candidate_oom`、`repair_success`、`repair_beyond_limit`。

## 8. Worker crash 与 stale recovery

典型故障窗口：

```text
claim -> benchmark 已完成 -> Worker crash -> 未提交 result
```

Job 会暂留 `RUNNING`。当 `lease_expires_at < now` 时：

- 还有 attempt 时转为可调度状态；
- attempt 已耗尽时转 `DEAD` 并释放 reservation；
- 旧 owner 即使恢复也无法提交；
- 新 Worker 可能再次执行 benchmark，因此只承诺 logical commit 幂等。

恢复入口有三处：独立 Worker 周期 stale sweep、API 启动时 `RecoveryManager`、Runtime 恢复入口针对当前 Run 的 sweep。`tests/test_evaluation_queue.py` 验证 benchmark 后 crash 再执行、最终只有一个 Result 且预算只结算一次；Harness `worker_crash`、`lease_expire_recovery` 给出结构化 Trace/assertion。

## 9. Cancel 与 Pause 的 Queue 语义

- Pause 是 cooperative。已暂停或 pause-requested Run 不再被 claim；Runtime 在一致安全点进入 `PAUSED`。
- Cancel 事务会取消该 Run 的 `PENDING/RETRY_WAIT` Job 并一次性释放 reservation。
- 正在 `RUNNING` 的 Job 不由另一个线程粗暴覆盖；活 owner 可以结算，失活 owner 由 lease recovery 按取消状态释放。
- Evaluation driver 每轮 polling 都调用 Runtime control check，避免等待重试时忽略 cancel。

`tests/test_pause_resume_reconciliation.py` 覆盖 paused Run 不被 claim、真实评价期间请求 pause、resume、取消 pending job 与预算只释放一次；`tests/test_runtime_cancellation_race.py` 验证完成回调不能把已取消 Run 覆写为完成。

## 10. Final Optimization 复用同一 Queue

`src/prievo_agent/runtime/final_optimization.py` 不建立旁路 evaluator。每个稳定 seed clone 都是独立 Candidate，每个 trial 都按自己的 seed/budget 生成幂等 Job，使用同一 reservation、lease、retry、fencing 和 external worker。Report identity 允许进程恢复时复用已有成功 trial 和已有 report。

这是对参考工程未实现阶段的工程补全，不是论文原始可执行代码。`tests/test_final_optimization.py` 覆盖多 seed 真实执行、独立预算、瞬时重试不重复扣费和 failed report recovery。

## 11. 如何查看证据

运行工程 Harness：

```powershell
$env:PYTHONPATH='src'
python scripts/agent_harness.py
```

重点查看 `reports/agent_harness.json` 中以下 scenario：

- `evaluation_transient_retry`；
- `worker_crash`；
- `lease_expire_recovery`；
- `duplicate_evaluation_submit`；
- `budget_contention`；
- pause / resume / cancel。

真实 Run 可通过 `GET /api/runs/{run_id}/events` 或 `/api/runs/{run_id}/trace` 查看 `EVALUATION_SUBMITTED`、claim/start、retry/dead、`CANDIDATE_EVALUATED` 与关联 Job/Candidate/Artifact ID。数据库才是最终状态；Redis 消息不能作为完成证据。

## 12. 当前限制与准确措辞

可以准确说：

> 通过 MySQL 事务、完整幂等物料、预算预留、`SKIP LOCKED` lease claim、owner fencing 和 stale requeue 实现持久化异步评价；Full 模式由独立 Worker 执行，Redis 仅做通知/缓存。

不能说：

- “exactly once benchmark execution”；真实语义是 at-least-once execution、idempotent commit。
- “已有周期 heartbeat”；当前只有 benchmark 前后续租。
- “Redis Queue 是事实源”；事实源是 MySQL。
- “SQLite Harness 证明 MySQL 高并发吞吐”；它只证明语义。
- “子进程等于强沙箱”；它只是风险降低层。

MySQL 的 `SKIP LOCKED` EvaluationJob claim 路径和 owner fencing 已存在；当前仓库没有单独留存两个真实 MySQL Evaluation Worker 同时争抢最后预算的端到端压力报告。`tests/test_mysql_agent_tasks.py` 证明的是 AgentTask 的 MySQL 并发 claim，不能移作 EvaluationJob 证据。下一步应使用两个独立连接/Worker、同一 Run 最后预算、barrier 同步提交与 claim，并断言唯一 Job/Result、预算守恒和无脏 owner commit。
