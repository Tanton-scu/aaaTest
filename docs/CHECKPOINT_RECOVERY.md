# Checkpoint 与 Crash Recovery

## 1. 恢复模型：细粒度事实 + 一致快照

本项目没有把 `generation=4` 当作完整恢复状态，也不会在每次 Candidate 变化时复制整个世界。恢复由两层持久化共同完成：

1. **Fine-grained durable facts**：CandidateDraft/Artifact、Candidate、EvaluationJob、EvaluationResult、AgentTask、AgentMemory、ToolCall、Event 在各自完成时立即提交。
2. **Run-level consistent Checkpoint**：只在 Population 与算法 cursor 一致的安全边界保存稳定引用和可重放 Core state。

Checkpoint 是恢复 baseline，不是唯一事实源。崩溃可能发生在 Checkpoint 之后，此时恢复必须查询最新 Candidate/Job/Result 并 reconciliation，而不是机械回滚并重放 LLM。

核心源码：

- Runtime：`src/prievo_agent/runtime/persistent_runtime.py`；
- 生命周期：`src/prievo_agent/runtime/lifecycle.py`；
- 状态机：`src/prievo_agent/runtime/state_machine.py`；
- 启动恢复：`src/prievo_agent/application/recovery_manager.py`；
- Store：`src/prievo_agent/infrastructure/mysql_store.py` 与 `sqlite_store.py`；
- 生成持久化：`src/prievo_agent/application/generation_workflow.py`；
- 跨进程 Harness：`src/prievo_agent/runtime/checkpoint_harness.py`。

## 2. 什么会立即持久化

| 工作边界 | durable fact | 为什么不能等到 Checkpoint |
| --- | --- | --- |
| LLM 完成一次生成 | `CANDIDATE_DRAFT` Artifact、完成的 AgentTask | LLM 非确定；恢复后重调可能得到另一份代码 |
| Draft materialize | `Candidate(status=CREATED)` 与 lineage refs | 下一次 LLM 失败也不能丢前一候选 |
| 候选可执行代码 | `CANDIDATE_CODE` Artifact、Candidate ref | Job 只引用稳定 Candidate ID |
| 评价提交 | EvaluationJob + budget reservation | Worker 可以跨进程认领 |
| 评价成功 | EvaluationResult + result Artifact + Candidate + budget | Checkpoint 尚未更新时仍可复用 |
| Agent 工作 | AgentTask、输出 Artifact、Event | Coordinator 从持久事实补缺，不靠内存队列 |
| Tool / Research | ToolCall、Evidence / Explanation Artifact、Memory | 恢复后 Evidence 仍能进入 resume prompt |
| Selection | retained Population、累计 offspring ID 与 selection Event | Operator Checkpoint 保留代初 P；Generation Checkpoint 建立在统一 `5P -> P` 后 |

`tests/test_generation_crash_recovery.py` 注入“第一个生成成功、第二次 LLM 失败”，证明第一个 Draft/Candidate 已保留，恢复不会重调已经完成的第一次模型调用。

## 3. Checkpoint schema v4

`PersistentEvolutionRuntime._save_checkpoint()` 先读取 durable budget，再写两个内容寻址 Artifact：轻量 `POPULATION_SNAPSHOT` 与完整 `CHECKPOINT` JSON；数据库 Checkpoint table 只保存 metadata 和 Artifact ref。快照包含：

```text
schema_version / run_id / dataset_id / dataset_digest
generation
population_ids / population_artifact_id
evaluation_references (candidate_id -> result_id)
consumed_budget / reserved_budget / remaining_budget
algorithm_cursor
core_state
strategy(total_generations, population_size, parent_count,
         faithful_mode, evaluation_mode, selection_cadence)
code_version / package_version / model_adapter
prior_refs
reason
```

它不复制 Candidate code、trajectory 或 Evidence 正文；这些由稳定 Candidate/Result/Artifact ID 回表。内容 SHA-256 参与 Checkpoint ID，Store 保存 metadata 后还必须以当前 runtime owner 完成 `runtime_cursor_artifact_id` 的 fenced 更新，否则 Runtime 报 `RuntimeLeaseLost`。

## 4. 当前真实安全点

当前实现触发 Checkpoint 的位置是：

1. 初始 population 全部评价并选回 P 后：`reason=initial_population`；
2. 每个 operator 生成 P 个 offspring并评价、累计 offspring refs，但尚未 selection：`reason=operator_batch_boundary`；
3. 一代四个 operator 完成后的 generation boundary：`reason=generation_selection_boundary`。

Operator checkpoint 的 population 始终是代初 retained P；cursor 至少保存 `next_generation`、`next_operator_index`、已累计 `generation_offspring_ids`、`completed_operator_count` 与 selection strategy，因此恢复后不会重跑已完成批次，也不会让后续 operator 错把同代 offspring 当 parent。四批结束后统一执行 `5P -> P`，Generation checkpoint 才保存下一代 retained P，并将 cursor 推到下一代第一个 operator。

Pause 复用最近已提交的安全 Checkpoint：候选边界发现 `pause_requested` 时，如果已有快照就暂停；快照之后已经完成的 Draft/Candidate/Result 不删除，resume 时 reconciliation 复用。初始 population 尚无快照时不会伪造安全点，而是继续到首个一致边界。

当前尚未实现可配置的“每 N 个 Evaluation、每 T 秒、graceful shutdown”CheckpointPolicy，也没有为任意中间语句制造快照。项目说明不能把这些建议项当成现成功能。

## 5. Restore 校验

`_restore_checkpoint()` 在导入 Core state 前校验：

- schema version；
- Run ID 与 generation identity；
- payload、metadata、当前 Run 的 Dataset ID；
- Dataset 内容 digest；
- Checkpoint budget 不得领先 durable Run；
- Runtime strategy 中 total generations；
- code version；
- payload 与 metadata 的 population refs 一致。

校验通过后按 `population_ids` 回表读取 Candidate，再强制读取每个 Candidate 的 EvaluationResult 填充 objective。Result 是 Checkpoint 外的事实，不从快照复制。当前只读取“最新 Checkpoint”；若最新 Artifact 损坏或版本不兼容会明确失败，尚未实现自动回退到上一有效快照。

## 6. Reconciliation

Runtime 每次取得 owner lease 后先执行：

1. 回收当前 Run 中 lease 过期的 EvaluationJob；
2. 枚举 durable Candidate、Job、Result；
3. Candidate 已有 Result：同步 Candidate 为 evaluated 并复用，不重新 benchmark；
4. Candidate 没 Result、也没 Job：用当前 Dataset/evaluator/parameters identity 创建 Job；
5. Candidate 已有非终态/终态 Job：保持该 Job，交给 queue 恢复或读取结果；
6. 记录 `RUN_RECONCILED` 及复用/补建数量。

Final Optimization Candidate 带 `creation_type=FINAL_OPTIMIZATION`，通用演化 reconciliation 会跳过它，避免按 Task 默认 seed/budget 错建 Job；其多 seed trial 和 report 由 `FinalOptimizationService` 自己以稳定 identity 恢复。

典型恢复表：

| 崩溃时 durable facts | 恢复动作 | 不做什么 |
| --- | --- | --- |
| Candidate + Result，Checkpoint 未更新 | 复用 Result，更新 in-memory population/objective | 不重跑 benchmark |
| Candidate + RUNNING Job，lease 已过期 | sweep 后 requeue/dead | 不让旧 owner 迟到覆盖 |
| Candidate 已保存但无 Job | 用稳定 identity 补建 Job | 不重调 LLM |
| Draft/AgentTask 已完成，Candidate 尚未 materialize | workflow 从输出 Artifact 恢复并 materialize | 不重复完成同一 AgentTask |
| Research Evidence 已提交 | resume Generation Context 重用 Evidence ref/content | 不修改 Original Prior |
| Final Optimization 部分 seed 成功 | 复用成功 trial，继续缺失 seed，重建/复用 report | 不改写原最终 Candidate |

`tests/test_pause_resume_reconciliation.py` 明确覆盖 Result 复用和 Candidate 无 Job 时补建；`tests/test_generation_crash_recovery.py` 覆盖 LLM 生成边界。不是每一个 Artifact/数据库语句之间的 crash seam 都有独立进程测试，不能把“关键 seam 覆盖”说成“穷举所有指令级故障窗口”。

## 7. Runtime owner lease

Run 与 EvaluationJob 使用不同 lease：

- Evaluation lease 保护某个 benchmark Job；
- Runtime lease 保护整个 PriEvO orchestration 的唯一推进者。

`PersistentEvolutionRuntime.execute()` 先以 `runtime_owner_id` claim Run；输家抛出 `RuntimeLeaseConflict` 而不推进。当前 owner 在执行入口、评价 polling/control check、Checkpoint 前与安全点续租。Checkpoint metadata 写入后，cursor 更新还检查 owner token，防止过期 Runtime 把新 owner 的游标倒退。

当前 runtime lease 默认 600 秒，没有后台 heartbeat。只要 Runtime 持续经过显式推进/轮询点就会续租；若某个未来同步步骤能阻塞超过窗口，则需要把它拆成可轮询工作或增加后台 heartbeat。API 单实例执行池目前 `max_workers=1`，多实例启动竞争依赖数据库 lease，而非进程内锁。

## 8. 启动恢复与常驻 sweep 边界

FastAPI lifespan 调用 `RunApplicationFacade.recover_startup()`，内部 `RecoveryManager.recover()` 以数据库为唯一输入：

1. 回收 stale EvaluationJob；
2. 回收超过 300 秒仍为 CLAIMED 的 orphan AgentTask；
3. 清理过期 Run runtime owner lease；
4. 找出未取消的 PENDING/RUNNING Run；
5. `DurableAgentCoordinator.reconcile()` 补齐缺失 AgentTask；
6. 写 `RECOVERY_SWEEP_COMPLETED`；
7. 按稳定 Run ID 重新调度。

独立 Evaluation Worker 另有周期 stale-job sweep。当前没有单独常驻的通用 RecoveryManager 服务去周期回收 AgentTask/Run；这两类一般恢复依赖 App 启动以及可调用的 coordinator/recovery sweep。因此不能概括为所有资源都有后台守护线程。

## 9. Pause / Resume / Cancel / Crash

### Pause

API 只写 durable request 与 `RUN_PAUSE_REQUESTED`。Runtime 停止创建新工作，在最近一致 Checkpoint 转 `PAUSED`，清空 runtime owner 并写 `RUN_PAUSED(checkpoint_id)`。评价期间的 pause 请求会在 polling 中被看到；没有安全快照时先等待当前必要工作形成快照。

### Resume

状态机执行 `PAUSED -> RUNNING`，清理控制请求并重新调度。新 Runtime claim owner、restore 最新 Checkpoint、reconcile 快照之后的 durable facts，再从 cursor 推进。

### Cancel

Store 在一个事务内设置 durable cancel、取消 PENDING/RETRY_WAIT EvaluationJob、取消 PENDING AgentTask、精确释放 reservation、清空 runtime owner；Runtime/driver 的 control check 阻止继续创建工作。RUNNING Job 由当前 owner settle，或 lease 过期后按取消语义回收。`tests/test_runtime_cancellation_race.py` 保证后台完成回调不会覆盖 `CANCELLED`。

### Crash

Crash 不是 PAUSED。数据库中的 Run 可能仍为 `RUNNING`，但 owner lease 终会过期；启动 Recovery 会清 owner并重新调度。恢复后的新 Runtime 先验证 Checkpoint，再 reconciliation，不调用“恢复 LLM”自由决定算法路径。

## 10. 跨进程恢复证据

`tests/test_checkpoint_recovery.py` 启动真实子进程，在代 Checkpoint 后模拟硬退出，再用同一 SQLite/Artifact 目录启动第二个进程。它与不中断基线比较：

- 最终 selected Candidate digest 一致；
- 27 个逻辑 EvaluationResult；
- consumed budget 为 81；
- duplicate evaluation 为 0；
- recovery trace 与 Checkpoint/Run 事件存在。

这是确定性小数据集上的工程恢复验证，不是算法性能 Benchmark。Harness 报告 `reports/agent_harness.json` 的 `run_crash` 与 `checkpoint_recovery` 场景提供 expected/actual trace 和断言证据。

其他直接测试：

| 测试 | 证明内容 |
| --- | --- |
| `tests/test_generation_crash_recovery.py` | 成功 LLM output 立即持久化，恢复不重放 |
| `tests/test_pause_resume_reconciliation.py` | pause safe point、resume、Result reuse、missing Job |
| `tests/test_recovery_manager.py` | stale Job、orphan AgentTask/Run owner、active Run 调度 |
| `tests/test_evaluation_lease_fencing.py` | 旧 Evaluation owner 迟到提交被拒绝 |
| `tests/test_external_evaluation_runtime.py` | external wait 期间控制检查与独立 Worker |
| `tests/test_runtime_retry_scheduler.py` | retry 等待非 busy-spin，等待期间可取消 |

## 11. Trace 定位方法

对真实 Run 查询：

```text
GET /api/runs/{run_id}
GET /api/runs/{run_id}/events
GET /api/runs/{run_id}/artifacts
GET /api/runs/{run_id}/trace
GET /api/runs/{run_id}/candidates
```

重点关联 `RUN_STARTED/RUN_RECOVERED/RUN_RECONCILED`、`CANDIDATE_DRAFT_MATERIALIZED`、`EVALUATION_SUBMITTED`、`CANDIDATE_EVALUATED`、`CHECKPOINT_SAVED`、`RUN_PAUSE_REQUESTED/RUN_PAUSED/RUN_RESUMED`。Trace 是 durable facts 的因果查询，不是另一套状态。

## 12. 能力边界

推荐：

> 我把恢复拆成 Candidate/Job/Result/Artifact 的细粒度事实与 Run 级一致 Checkpoint；Checkpoint 只存稳定 ID、预算和算法 cursor，启动后校验版本并与 Checkpoint 之后事实对账，因此已完成 LLM 生成和评价不会无条件重放。

需要主动说明：

- 当前 safe points 是初始 population、每个 operator 的“已评价并累计但未选择”边界，以及统一 `5P -> P` 后的代边界；
- 没有 N-evaluation/T-time/graceful policy；
- Run lease 没有后台 heartbeat；
- 通用 AgentTask/Run recovery 主要发生在 App 启动，只有 Evaluation stale sweep 在独立 Worker 周期运行；
- 最新 Checkpoint 损坏时目前明确失败，不自动回退上一快照；
- Final Optimization 是工程补全，但使用同一 durable recovery 原语。
