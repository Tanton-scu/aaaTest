# Backend Runtime

## 1. 文档边界

本文描述的是当前源码已经成立的 Backend Runtime，而不是目标架构草图。生产形态指 `PRIEVO_MODE=full`：FastAPI、MySQL、Redis 和独立 Evaluation Worker 由 `docker-compose.yml` 编排；`PRIEVO_MODE=demo` 则使用 SQLite 和进程内评价，便于无外部依赖演示。两种模式复用同一套 PriEvO Core、Run 状态机、EvaluationJob、Checkpoint、Agent/Node 与 Trace 语义，但并发和部署证据不能混用。

最重要的事实边界如下：

| 主题 | Full | Demo / Test |
| --- | --- | --- |
| durable facts | MySQL 8.4，`MySQLRuntimeStore` | SQLite，`SQLiteRuntimeStore` |
| Evaluation 执行 | `external`，App 只提交/轮询，独立 Worker claim | `inline`，Runtime 可在同进程驱动 Worker |
| Redis | 事件唤醒、hot status、Run 内近期记忆缓存 | 可禁用；数据库轮询/持久记忆仍可工作 |
| Artifact 内容 | 共享 `/var/lib/prievo` volume；MySQL 保存 metadata/ref | 本地 Artifact 目录 |
| LLM | 三项配置齐全时走 OpenAI-compatible adapter | 三项全空时走 deterministic FakeLLM |

Redis 从来不是 Job ownership、Run status、lease、budget、Event 或完整 Memory 的事实源；Redis 故障只会降低唤醒和近期缓存能力。

## 2. 真实调用链

```text
POST /api/runs (202)
  -> RunApplicationFacade.create_run
  -> OptimizationTask + Run + RUN_CREATED 写入数据库
  -> 后台调度 PersistentEvolutionRuntime
  -> runtime owner lease claim
  -> PriEvO Core 决定 FLA / strategy / parent / operator / selection
  -> 需要语义推理时进入 3 Agent + 2 Node 工作流
  -> CandidateDraft 与 Candidate 立即持久化
  -> EvaluationQueueService.submit（事务内去重 + 预算预留）
  -> Full: App 轮询 durable Job；Worker 服务 claim 并在子进程执行 Candidate
  -> EvaluationResult + Candidate + Job + budget 原子结算
  -> 每个一致性安全点写 Checkpoint metadata + snapshot Artifact
  -> FinalSelection；如唯一最优则不调用 LLM，精确并列才调用 FinalSelectionNode
  -> 工程补全 Final Optimization（稳定 clone、多 seed、同一 EvaluationQueue）
  -> RUN_COMPLETED
```

组合根在 `src/prievo_agent/infrastructure/local_runtime.py`。算法与 Runtime 的装配入口是 `src/prievo_agent/algorithm/prievo_engine.py`；应用入口是 `src/prievo_agent/application/run_facade.py`；长任务推进在 `src/prievo_agent/runtime/persistent_runtime.py`。Coordinator 不决定 PriEvO 的下一步，只为持久化 Artifact 中缺失的语义工作派生 AgentTask。

## 3. Run 状态与控制面

`src/prievo_agent/domain/models.py` 定义六个终态/运行态：

```text
PENDING -> RUNNING -> COMPLETED
                 \-> FAILED
                 \-> CANCELLED
RUNNING -> PAUSED -> RUNNING
```

允许迁移集中在 `src/prievo_agent/runtime/state_machine.py`，生命周期副作用集中在 `src/prievo_agent/runtime/lifecycle.py`。API 不直接伪造暂停完成：

- `POST /api/runs/{run_id}/pause` 只写 `pause_requested`，返回 202；Run 可能暂时仍为 `RUNNING`。
- Runtime 在已有一致 Checkpoint 的安全点确认暂停并转为 `PAUSED`；如果初始种群尚未形成首个 Checkpoint，会先完成必要评价再暂停。
- `resume` 清理控制请求、转回 `RUNNING` 并重新调度。
- `cancel` 以数据库事务设置取消状态，取消可取消的 EvaluationJob / AgentTask 并释放相应 reservation；已经被活 Worker 持有的评价由 owner 结算或在 lease 过期后回收。

`PersistentEvolutionRuntime` 还对 Run 使用独立的 `runtime_owner_id + runtime_lease_expires_at`。多个 API 实例在启动恢复时可以同时发现同一 Run，但只有 claim 成功者能推进；写 Checkpoint、等待评价和控制安全点都会续租并做 fencing。当前 Run lease 没有后台 heartbeat 线程，续租发生在 Runtime 的显式推进/轮询边界；默认窗口为 600 秒。

## 4. 异步执行与独立 Worker

Full 模式设置 `evaluation_execution_mode=external`。`src/prievo_agent/runtime/evaluation_driver.py` 在该模式只观察 Job 状态并有界等待，不调用 App 进程里的 evaluator。独立入口为：

```powershell
python -m prievo_agent.cli.evaluation_worker --root /var/lib/prievo
```

Compose 中 `worker` 与 `app` 使用同一 MySQL 和同一 Artifact volume，但 Worker 不需要 LLM 凭据。其处理边界是：

1. 在短数据库事务内 claim Job，并写入本进程唯一的 owner token 与 lease；
2. 事务外用 `ExecutableDatasetEvaluator` 启动候选子进程；
3. 成功或失败后，以 `job_id + worker_id + RUNNING` 条件做 fenced final commit；
4. 空队列使用有上限的指数退避，周期调用 stale-job sweep；
5. 收到 SIGTERM 后停止领取新任务，并等待当前 benchmark 收尾。

Worker 当前只在 benchmark 前后续租，没有 benchmark 期间的后台 heartbeat。CLI 因此强制 `EVALUATION_LEASE_SECONDS > EVALUATION_TIMEOUT_SECONDS`（Compose 默认 30 秒与 10 秒）。这能覆盖当前有界 benchmark，却不应被描述成“周期 heartbeat”；如果未来允许超过 lease 的长评价，应加入独立续租线程/进程并补 fencing 压测。

`tests/test_external_evaluation_runtime.py` 的关键断言是 App 侧 evaluator 调用数为 0、独立 Worker 实际消费演化和 Final Optimization Job；`tests/test_evaluation_worker_service.py` 覆盖配置、退避、stale sweep、丢 lease、健康检查和 Compose 合同。

## 5. 数据、事务与缓存边界

Full 模式的 MySQL schema 由 `src/prievo_agent/infrastructure/mysql_store.py` 内嵌迁移管理：

- V001：Task、Run、Candidate、EvaluationJob/Result、Event、Artifact、Checkpoint 等核心事实；
- V002：Agent Memory；
- V003：durable AgentTask；
- V004：pause/cancel、runtime cursor 与 runtime owner lease；
- V005：AgentTask claim token 与 lease fencing；
- V006：`evaluation_jobs(candidate_id)` 唯一约束，固定“一 Candidate 一 logical evaluation”，重评必须 clone。

SQLite adapter 在 `src/prievo_agent/infrastructure/sqlite_store.py` 中直接创建兼容 schema，并通过 `BEGIN IMMEDIATE` 串行化本地写事务；它是 Demo/Test adapter，不是 MySQL 多 Worker 生产并发证据。MySQL 的 enqueue 会锁 Run 行，claim 使用 `FOR UPDATE SKIP LOCKED`，最终结算也在短事务内完成。详细语义见 `docs/EVALUATION_QUEUE.md`。

State、Event、Artifact、Trace 的分工是：

- State：Run/Candidate/Job/Task 当前状态；
- Event：带 Run 内单调 sequence 的已发生事实；
- Artifact：内容寻址的结构化产物，数据库保存 metadata，内容落 Artifact Store；
- Trace：`src/prievo_agent/application/agent_trace.py` 将 Event、AgentTask、ToolCall、Artifact 引用重建为因果视图。

系统是持久状态 + 审计事件，不宣称完整 Event Sourcing。当前 Artifact metadata API 会返回 adapter 保存的 `uri`，本地部署中可能是绝对路径；公网部署前应改为 opaque ref 或受控下载 URL。

## 6. Candidate 执行与安全边界

真实评价入口为 `src/prievo_agent/algorithm/executable_dataset_evaluator.py`，候选子进程 worker 为 `src/prievo_agent/security/heuristic_worker.py`。它使用 `python -I`、最小环境变量、临时工作目录、wall timeout，并在 POSIX 平台尽力设置 CPU、地址空间、文件大小和文件描述符 rlimit。评价结果还校验 trajectory、best configuration 与 budget。

这是一层风险降低，不是强安全沙箱：Windows 上没有等价 rlimit，Python 隔离也不能替代容器/VM/seccomp。生产环境运行不可信候选应将 Worker 放在无凭据、只读文件系统、禁网、限 CPU/内存/PID 的独立容器或微虚机中。

## 7. 预算与 Final Optimization

创建 API Run 时，`RunApplicationFacade` 预先校验总预算至少覆盖演化和 Final Optimization：

```text
evolution = candidate_budget * population_size * (1 + 4 * generations)
final_opt = seed_count * (2 * candidate_budget)
```

当前默认 Final Optimization seeds 为 `1009`、`2027`，每个 seed 的预算为单候选预算的 2 倍。例如 `population_size=2`、`generations=1`、`candidate_budget=3`，最低总预算为 `30 + 12 = 42`。

`src/prievo_agent/runtime/final_optimization.py` 是工程补全：参考 PriEvO 可执行源码没有落地该阶段，只有说明性意图。实现会对最终选中 Candidate 建立不可变 stable clone，每个 seed 使用独立 Candidate/Job/Result，仍走相同 queue、budget ledger、真实 evaluator 与 external worker；最终生成 `FINAL_OPTIMIZATION_REPORT`，包含逐 seed trajectory、均值/中位数/标准差及 best seed/config。它不会改写原选中 Candidate。失败 trial 会形成可恢复的 FAILED report，而不是用成功日志掩盖。

## 8. API、SSE 与观测

`src/prievo_agent/api/app.py` 暴露：

- `GET /api/health`、`GET /api/datasets`；
- `POST /api/runs` 与 Run list/detail；
- Run 的 events、artifacts/content、metrics、candidates；
- `/agent-trace` 与兼容别名 `/trace`；
- pause/resume/cancel；
- `/events/stream` SSE；
- `/` 简单 Dashboard。

SSE 的游标依据数据库 Event sequence；Redis Pub/Sub 只负责提前唤醒。Redis 不可用时仍轮询数据库，因此不会丢 durable event。FastAPI lifespan 在启动时调用 `RecoveryManager`：回收 stale EvaluationJob、超时 CLAIMED AgentTask 和过期 Run owner，reconcile active Run 的缺失 AgentTask，并重新调度 PENDING/RUNNING Run；运行期间另按 `COORDINATOR_SWEEP_SECONDS` 周期执行 durable missing-work sweep。

当前 `RunApplicationFacade` 每个 App 实例的执行池为 `max_workers=1`，因此单 App 内不会并行推进多个 Run；横向实例依赖 runtime lease 防重复。这是正确性优先的当前实现，不应包装成高吞吐调度器。

## 9. 可验证证据

| 能力 | 源码 | 测试 / 报告 |
| --- | --- | --- |
| Full 外部评价链 | `local_runtime.py`、`evaluation_driver.py`、`cli/evaluation_worker.py` | `tests/test_external_evaluation_runtime.py` |
| API + SSE + 生命周期 | `api/app.py`、`run_facade.py`、`lifecycle.py` | `tests/test_api_sse.py`、`tests/test_runtime_cancellation_race.py` |
| MySQL Run 控制/lease | `mysql_store.py` V004 | `tests/test_mysql_run_control.py`（需 `DATABASE_URL`） |
| 启动恢复 | `application/recovery_manager.py` | `tests/test_recovery_manager.py` |
| 子进程评价 | `executable_dataset_evaluator.py`、`heuristic_worker.py` | `tests/test_executable_dataset_evaluator.py`、`tests/test_candidate_security.py` |
| Final Optimization | `runtime/final_optimization.py` | `tests/test_final_optimization.py`、`tests/test_external_evaluation_runtime.py` |
| Trace/SSE | `application/agent_trace.py`、`api/app.py` | `tests/test_agent_trace.py`、`tests/test_observability_contract.py`、`tests/test_api_sse.py` |
| 综合故障路径 | `runtime/agent_harness.py` | `reports/agent_harness.json`：24/24 场景、116/116 断言 |

Harness 使用隔离 SQLite 和 Fake 边界，验证的是工程语义和真实产品调用路径，不是 MySQL 性能 benchmark。MySQL 测试若环境未提供 `DATABASE_URL` 会跳过；对外说明结果时必须同时说明运行环境。

## 10. 当前诚实限制

1. Evaluation 与 Run lease 都没有后台 heartbeat；当前以 timeout 小于 lease 和显式安全点续租保证。
2. AgentTask 有事件触发 reconcile、启动恢复和 30 秒周期 missing-work sweep；Run lease 的 orphan 回收仍主要发生在启动恢复，Evaluation Worker 另有周期 stale sweep。
3. Checkpoint 已覆盖真实一致性边界，但尚无“每 N 次评价、每 T 秒、graceful shutdown”可配置 policy。
4. SQLite 是 Demo/Test 适配器；不能用 SQLite Harness 数字声称 MySQL 多 Worker 吞吐。
5. MySQL `SKIP LOCKED` 与 owner fencing 已实现；当前没有单独记录“两个真实 MySQL Evaluation Worker 同时争抢最后一份预算”的端到端压测报告。
6. Candidate subprocess 是风险降低边界，不是恶意代码强隔离。
7. Artifact API 的本地 URI 和鉴权仍需在公网部署前收紧。
