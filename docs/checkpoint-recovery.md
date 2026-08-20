# Checkpoint、控制请求与跨进程恢复

## 两层 Durable State

系统不会等到 Checkpoint 才保存工作：LLM Draft、Candidate、EvaluationJob、
EvaluationResult、Agent Artifact 和 Selection 都会在各自事务边界立即落库。这样即使
进程在两次 Checkpoint 之间退出，恢复也不会重新调用已成功的 LLM，也不会重复结算
已有 Result。

Run-level Checkpoint 只描述一致性安全点，保存：

- Run、Dataset、schema/code/package 版本；
- 当前 population 的稳定 Candidate ID；
- Candidate → EvaluationResult ID；
- consumed/reserved/remaining budget；
- `algorithm_cursor`：下一代、下一个 operator index 和本代已完成批次引用；
- RNG/模型 adapter 可重放状态、Prior/Agent Artifact 引用。

Checkpoint 不复制 Candidate code 或 Result payload，也不保存任意绝对路径。较大 JSON
作为 `CHECKPOINT` Artifact 保存，MySQL/SQLite `checkpoints` 表只保存 metadata 和
artifact ID；读 Artifact 时复验 SHA-256。

## 安全点和 cooperative pause

Runtime 在 initial population、每个 operator 的 `P+P -> P` selection、每代 selection
完成后保存安全 Checkpoint。API 的 `POST /api/runs/{id}/pause` 只原子设置
`pause_requested` 并记录 `RUN_PAUSE_REQUESTED`；此时 Run 仍是 `RUNNING`。正在执行的
evaluation 会先通过 owner fencing 结算，Runtime 随后：

1. 停止生成新 Candidate/认领新 Job；
2. 确认最近安全 Checkpoint 与 runtime cursor 已提交；
3. 清理 Runtime owner lease；
4. 迁移为 `PAUSED`，记录带 checkpoint ID 的 `RUN_PAUSED`。

`POST /resume` 把 `PAUSED -> RUNNING`，清理 pause flag，并调度新的 Runtime owner。
它先加载最新有效 Checkpoint，再查询 checkpoint 后 Candidate/Job/Result 并做
reconciliation，最后从 `algorithm_cursor` 的下一个 operator 继续。

## Cancel 与并发所有权

`POST /cancel` 在一个数据库事务内：

- durable 标记 `cancel_requested` 和 `CANCELLED`；
- 将 `PENDING/RETRY_WAIT` EvaluationJob 与 `PENDING` AgentTask 置 `CANCELLED`；
- 对每个刚取消 Job 的 reservation 恰好释放一次；
- 清除 Runtime owner lease，并记录 cancel reason/event。

Job enqueue/claim、AgentTask claim 和 Runtime 入口都会读取 durable cancel flag，因而
多进程不能在取消后继续制造工作。已经 RUNNING 的 evaluator 采用 cooperative settle：
当前 owner 可按 fencing 规则结算；若进程丢失，则 RecoveryManager 在 lease 到期后把
它置 CANCELLED 并释放尚未结算的 reservation。

每个 Run 还有 `runtime_owner_id + runtime_lease_expires_at`。同一时间只有一个未过期
owner 可以推进算法；重复 startup sweep 的输家收到 `RuntimeLeaseConflict`，不会把健康
Run 误标为 FAILED。

## Reconciliation

恢复入口先回收 stale EvaluationJob，再按 durable facts 对账：

- Result 已存在、Checkpoint 尚未引用：同步 Candidate 并直接复用；
- Candidate 已保存、Job 尚未创建：以稳定 Candidate ID 建立幂等 Job，不重调 LLM；
- Job 已存在：保留原 reservation，不重复提交；
- expired RUNNING Job：按 retry/max-attempt policy 重新排队或进入终态；
- 超过有界 claim timeout（默认 300 秒）的 orphan `CLAIMED` AgentTask：启动
  sweep 恢复为可幂等派发；仍在 timeout 内的活跃任务不会被第二实例误抢；
- expired Runtime owner：清 lease 后重新调度 active Run。

Redis 只承担通知和 hot cache；Run/Job/Result/Checkpoint/Artifact metadata 的事实源始终
是 MySQL（Full Mode）或 SQLite（Demo/Test）。

## 可验证证据

```powershell
$env:PYTHONPATH='src'
python -m unittest tests.test_pause_resume_reconciliation
python -m unittest tests.test_checkpoint_recovery
python -m unittest tests.test_recovery_manager
```

关键用例包含：evaluation 期间 pause→安全 PAUSED→resume 完成、pending cancel 预算恰好
释放、Candidate 已存/Job 未建的补齐、Result 已存/checkpoint 未更新的复用、启动时 orphan
AgentTask/Run owner 恢复，以及独立 OS 进程 hard-exit 后与不中断执行结果一致。

Full Mode 另由 `test_mysql_run_control.py` 在 MySQL 8.4 验证 V004 migration、Runtime
lease、pause request、原子 cancel 和 claim guard。V001～V003 checksum 未修改。
