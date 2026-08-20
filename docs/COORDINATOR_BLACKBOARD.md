# Durable Coordinator 与 Blackboard

本文说明当前多 Agent 的协作控制面。核心结论是：Coordinator 发现“已有事实之后还缺什么 Agent 产物”，Blackboard 提供只读引用视图，Dispatcher 执行任务；三者都不拥有 PriEvO 算法状态。

## 1. 事实、任务与投影

系统把三个概念分开持久化：

- `ArtifactMetadata`：不可变产物的 ID、Run、kind、media type、size、digest 和 URI；内容通过 Store 的 Artifact API 单独读取；
- `AgentTask`：Agent 工作项，保存 task type、required capability、输入/输出 Artifact refs、状态、领取者、attempt 和幂等键；
- `Event`：按 Run 单调序列记录事实发生顺序及关联 ref。

`application/blackboard.py::Blackboard` 是由这三类 durable 记录重建的不可变投影，不是第四个事实源。应用重启后用同一个 SQLite/MySQL Store 调用 `Blackboard.from_store(store, run_id)`，即可得到相同的协作现场。

## 2. Missing-work 规则

`application/durable_agent_coordinator.py::MISSING_WORK_RULES` 当前只有以下五条：

| 已有输入 Artifact | 若缺少的对应输出 | 创建 task type | required capability |
| --- | --- | --- | --- |
| `TOP5_CANDIDATES` | `SIMILARITY_DECISION` | `SEMANTIC_SIMILARITY_SELECTION` | `SEMANTIC_SIMILARITY` |
| `GENERATION_REQUEST` | `CANDIDATE_DRAFT` 或 `KNOWLEDGE_GAP` | `HEURISTIC_GENERATION` | `HEURISTIC_GENERATION` |
| `KNOWLEDGE_GAP` | 同时存在 `PRIOR_EXPLANATION` 与 `LITERATURE_EVIDENCE` | `PRIOR_RESEARCH` | `PRIOR_RESEARCH` |
| `CANDIDATE_FAILURE` | `REPAIR_DECISION` 或 `REPAIRED_CANDIDATE_DRAFT` | `CANDIDATE_REPAIR` | `CANDIDATE_REPAIR` |
| `FINAL_TIE` | `FINAL_SELECTION_DECISION` | `FINAL_SELECTION` | `FINAL_SELECTION` |

同 kind 只有一个输入时，规则兼容早期未写 source ref 的 Artifact；同 kind 多输入时，Coordinator 会读取候选输出 JSON 并要求其中真实引用对应输入，避免一个输出错误地满足多个请求。

研究后的 `GENERATION_RESUME_REQUEST -> HEURISTIC_GENERATION_RESUME` 由 `application/generation_workflow.py::_GenerationResumeCoordinator` 处理，而非上表的通用规则。它同样创建 durable task，并要求 resume 输出是 resumed `CANDIDATE_DRAFT` 或 `KNOWLEDGE_GAP`。

这些规则只表达 Agent 工作依赖，不表达算法依赖。例如“某算子需要生成 P 个 offspring”“四批 4P 与 retained P 如何统一保留 P 个”不属于 missing-work rules，而由 `core/evolution.py` 和 `runtime/persistent_runtime.py` 执行。

## 3. Reconcile 的确定性与幂等

`DurableAgentCoordinator.reconcile(run_id)` 的步骤是：

1. 从 Store 读取并稳定排序当前 Run 的 Artifact metadata；
2. 按 input kind 遍历所有潜在输入；
3. 检查是否已有与该输入对应的可接受输出；
4. 检查是否已有相同 idempotency key 的 task；
5. 仅在两者都不存在时创建 `PENDING` task 并写 `AGENT_TASK_CREATED` Event。

任务 identity 不是随机数。source digest 由 `input_kind|input_artifact_id` 的 SHA-256 截断得到，idempotency key 为 `missing-work:v1:<task_type>:<digest>`；task ID 再由 `run_id|idempotency_key` 确定。相同 Run/输入重复 reconcile 会得到相同 identity。

内存中的 `known_tasks` 是第一道去重，SQLite/MySQL `add_agent_task` 的唯一约束是并发 reconcile 的最终防线。`sweep(active_run_ids)` 只是按稳定 Run ID 顺序批量 reconcile，不增加新语义。

## 4. Coordinator 明确不做什么

Coordinator 不执行以下动作：

- 不调用 LLM、Skill、RAG 或 evaluator；
- 不读取或修改 Run status、control request、generation、budget；
- 不创建、评价、选择或修复 Candidate；
- 不决定 PriEvO strategy、parent、operator schedule、population 或 final winner；
- 不把 Blackboard 当消息总线或自由规划空间。

因此 Coordinator 的“missing work”不会改变论文算法。它只能把 workflow 已经写出的输入事实转换成恰好一次的可恢复 AgentTask。

## 5. Blackboard 的可见内容

Blackboard 的三个集合如下：

| 投影 | 字段 | 故意排除 |
| --- | --- | --- |
| `AgentTaskView` | task identity/type/capability、input/output refs、status、claimed_by、attempts/error/timestamps | handler 对象、任意可写方法 |
| `BlackboardEventView` | sequence、event type/message、从 payload 提取的 task/artifact refs、occurred_at | 原始任意 payload |
| `ArtifactRef` | id/run/kind/media type/size/digest/uri | Artifact 内容读取接口 |

投影类和内部索引均不可变；`open_tasks()`、`tasks_by_type()`、`events_by_type()`、`artifacts_by_kind()` 只返回查询结果。任何状态变更必须先提交 Store，再重新构建 Blackboard。

Blackboard 不包含 Population、Candidate code、Fitness trajectory、Prior 内容或预算账本。handler 如需读取获准输入，会先通过 `artifact_by_ref` 核验 metadata，再由其 workflow 显式调用 `store.artifact_content(ref)`；这一步仍受输入 kind、schema 和 Context Policy 约束。

## 6. Registry 与 Dispatcher

`agents/registry.py::AgentRegistry` 对 capability 采用一对一注册：

- 同一 capability 不能注册两个 Agent；
- 同一 Agent name 也不能重复；
- 不做竞价、评分、动态投票或 fallback 到“最像”的 Agent；
- 未注册 capability 会明确报错。

`application/agent_dispatcher.py::AgentTaskDispatcher` 执行单个 task：

```text
get AgentTask
  -> 非 PENDING：返回 executed=false
  -> Registry.registration_for(capability)
  -> Store.claim_agent_task(task, agent_name)
  -> 写 AGENT_TASK_CLAIMED
  -> Blackboard.from_store
  -> handler.handle(task, board)
  -> artifact_writer 持久化并校验同 Run refs
  -> Store.complete_agent_task
  -> 写 AGENT_TASK_COMPLETED
```

若并发 Dispatcher 已先领取，后到者刷新 durable task 并返回，不重复执行 handler。handler、schema 或 Artifact writer 失败时，Store 先增加 attempts 并结算为重新 `PENDING` 或最终 `FAILED`，随后写 `AGENT_TASK_FAILED`。

`dispatch_with_retries` 只对 timeout、connection、`LLMTimeoutError`、JSON decode 和 `Malformed*` 模型协议错误同步 redrive；上限来自 durable `AgentTask.max_attempts`，默认 3。普通领域或编程错误不会被无限吞掉。路由缺失保留 `PENDING`，等待 capability 注册恢复。

Dispatcher 自身不读取或修改 Run、budget、Candidate、Population 或 PriEvO strategy。Artifact writer 也必须先写入同一 durable Store；伪造的、未持久化的或其他 Run 的 output ref 会使 task 失败。

## 7. 状态机与恢复

`AgentTaskStatus` 包含：

```text
PENDING -> CLAIMED -> COMPLETED
                  \-> PENDING   （仍有重试次数）
                  \-> FAILED    （重试耗尽）
PENDING/CLAIMED -> CANCELLED     （Store 控制路径）
```

应用启动时，`application/recovery_manager.py::RecoveryManager` 会：

1. 回收超时 `CLAIMED` 的孤儿 AgentTask；
2. 处理陈旧 EvaluationJob 和 runtime lease；
3. 对活动 Run 重新执行 Coordinator reconcile；
4. 把需要继续的 Run 交回运行调度。

正常低延迟路径仍在关键 Artifact COMMIT 后立即 `reconcile(run_id)`。此外 FastAPI lifespan 以 `COORDINATOR_SWEEP_SECONDS`（默认 30 秒）调用 `RunApplicationFacade.reconcile_active_agent_tasks()`：只扫描 PENDING/RUNNING Run，从 durable facts 补 missing task，并幂等调度出现新任务的 Run。它覆盖“COMMIT 成功但 reconcile 回调丢失、服务进程仍存活”的窗口，不维护第二套内存队列。

恢复依赖 durable facts，而非旧进程的 Python 对象或 Redis。任务已完成且 output refs 完整时不会重跑；输入存在、输出和 task 都缺失时才补 task。

## 8. 与 reference 和工程增强的关系

reference PriEvO executable 没有 AgentTask/Blackboard/Coordinator，它在一个进程内同步调用 LLM 和 evaluator。当前系统保持其语义节点，但增加持久边界：Top-5 后相似性、每个 Core generation request、KnowledgeGap、candidate failure 和 exact final tie 都可独立追踪、重试、恢复。

这属于 Backend/Agent 工程增强，不是把 reference 算法重写为 multi-agent voting。特别是：

- Operator schedule 仍由 `core/schedule.py` 决定；
- Parent sampling 与 generation-level `5P -> P` 仍由 `core/evolution.py` / Runtime 决定；
- Candidate 真实评价仍由 EvaluationJob/Worker 决定；
- Coordinator 只能判断 Agent Artifact 是否缺失。

## 9. 可验证证据

- 规则、幂等和并发去重：`tests/test_durable_agent_coordinator.py`；
- 一对一 Registry 与 refs-only Blackboard、Store 重开重建：`tests/test_agent_registry_blackboard.py`；
- claim、完成、失败、bounded redrive 和重复 dispatch：`tests/test_agent_dispatcher.py`；
- SQLite/MySQL task 持久化：`tests/test_sqlite_agent_tasks.py`、`tests/test_mysql_agent_tasks.py`；
- 启动回收与 reconcile：`tests/test_recovery_manager.py`；
- 周期 missing-work 兜底：`tests/test_periodic_coordinator_sweep.py`；
- 端到端任务基数与幂等：`tests/test_agent_harness.py`、`reports/agent_harness.json`。
