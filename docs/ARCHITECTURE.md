# PriEvO-Agent 总体架构

本文描述当前 `src/` 产品主链，而不是早期审计时的目标图。事实入口是 `src/prievo_agent/algorithm/prievo_engine.py::PriEvOEngine`。旧版 `Evolution/Research/Review` 边界 Agent 及同步 Coordinator 已删除，产品中只保留围绕真实 PriEvO 语义节点设计的五类 Agent。

## 1. 系统定位

PriEvO-Agent 是一个以 PriEvO 为算法核心的长任务后端。系统先根据目标 Dataset 的 Fitness Landscape 构造 instance-specific prior，再由固定演化调度驱动五种 heuristic generation strategy，真实执行候选 `run_tuners`，按 fitness/diversity 规则完成代级 `5P -> P`，并用持久任务、评价队列、恢复、Memory、RAG 和 Trace 包装原始算法。

这里的“多 Agent”不是让一个 Supervisor LLM 自由规划算法。PriEvO Core 始终拥有 generation、strategy、parent、population、selection 和 budget 语义；Agent 只处理 Core 或持久事实已经表明需要语义推理的节点。

## 2. 分层与所有权

```text
FastAPI / Dashboard / CLI
        |
RunApplicationFacade ---- startup RecoveryManager
        |
PriEvOEngine
   +---- Landscape / Numeric Top-5 / Prior Extraction
   +---- PriEvoEvolutionCore -----------------------------+
   |       schedule / parent / population selection       |
   |                                                       |
   +---- Durable Agent Workflows                           |
   |       Artifact -> Coordinator -> AgentTask            |
   |       -> Registry -> Dispatcher -> Agent -> Artifact  |
   |                                                       |
   +---- PersistentEvolutionRuntime                        |
           Candidate -> EvaluationJob -> Worker            |
           -> supervised subprocess -> EvaluationResult    |
           -> four operator batches -> generation 5P -> P  |
           -> operator/generation Checkpoint               |
           -> Final Selection -> Final Optimization         |
                                                           |
MySQL / SQLite durable truth <----> Artifact Store         |
        ^                              ^                   |
        +-- Redis notification + run-local recent cache ---+
```

| 层 | 当前职责 | 明确不拥有的内容 | 主要源码 |
| --- | --- | --- | --- |
| API / Application | 创建、查询、控制和调度 Run；启动恢复 | 算法选择、SQL 细节 | `src/prievo_agent/api/app.py`、`application/run_facade.py` |
| Core | strategy schedule、parent selection、Candidate identity、early/late population selection | HTTP、SQL、Redis、Agent 路由 | `core/evolution.py`、`core/schedule.py`、`core/selection.py` |
| Agent | 五类受约束语义任务与结构化输出 | Run 状态、预算、Job ownership、population | `agents/similarity.py` 等五个产品 Agent |
| Coordinator / Dispatcher | 从缺失 Artifact 派生唯一 AgentTask；按 capability 领取、执行、结算 | 决定 PriEvO 下一步 | `application/durable_agent_coordinator.py`、`agent_dispatcher.py` |
| Runtime | 长任务 cursor、安全点、评价等待、恢复、final phase | 自由更改 Agent 结论 | `runtime/persistent_runtime.py` |
| Evaluation | 幂等提交、预算预留、lease/fencing、真实候选子进程 | Agent repair 决策 | `runtime/evaluation_queue.py`、`algorithm/executable_dataset_evaluator.py` |
| Infrastructure | MySQL/SQLite/Redis、Artifact、LLM、Skill、RAG adapter | 领域决策 | `infrastructure/` |

`domain/models.py` 定义 Run、Candidate、EvaluationJob、AgentTask 等稳定事实；具体数据库和模型 SDK 不进入 Core。

## 3. 两种部署模式

### Demo Mode

`PRIEVO_MODE=demo` 使用 SQLite、文件 Artifact Store、确定性 FakeLLM（未配置完整 LLM 三元组时）和 inline Evaluation Worker。该模式便于本地演示与 Harness，不依赖 MySQL、Redis 或外部模型。

### Full Mode

`PRIEVO_MODE=full` 使用 MySQL 作为状态和 ownership 的事实源，Redis 只承担 Event 通知与近期 Agent Memory 缓存。App 以 `evaluation_execution_mode=external` 提交并观察 EvaluationJob，不在编排线程 claim；`docker-compose.yml` 中独立 `worker` 服务通过 `cli/evaluation_worker.py` 领取并执行任务。

Full Mode 当前仍由每个 App 实例内的 `ThreadPoolExecutor(max_workers=1)` 串行推进本实例调度到的 Run；多 App 实例之间由 durable runtime lease 防止同一 Run 双重推进。独立 Evaluation Worker 可横向增加副本，ownership 仍由 MySQL claim/fencing 决定。

LLM 配置位于环境变量 `LLM_API_ENDPOINT`、`LLM_API_KEY`、`LLM_MODEL`。三项全空时使用确定性 FakeLLM；产品 composition root 对“只填一部分”直接报错，不会假装真实 API 已配置。

## 4. 持久事实与缓存边界

| 数据 | 事实源 | 恢复/缓存语义 |
| --- | --- | --- |
| Run、控制请求、runtime lease/cursor | MySQL / SQLite | `RunStateMachine` 集中守卫状态迁移 |
| Candidate、EvaluationJob、EvaluationResult | MySQL / SQLite | 幂等 key、预算 reservation、lease token 防重复结算 |
| AgentTask、Event、ToolCall、Artifact metadata | MySQL / SQLite | Blackboard 与 Trace 随时从记录重建 |
| Candidate code、Prompt、Evidence、Checkpoint payload | Artifact Store | 内容 digest 校验；领域记录引用 artifact ID |
| 近期 Agent Memory | Redis | 可丢失；miss/故障时只从同 Run、同 scope 的持久历史回温 |
| Event 唤醒 | Redis | 数据库 Event sequence 仍是 SSE cursor 和事实源 |

Blackboard 不是数据库替代品，Checkpoint 也不是 Artifact/Event 的全集。四者的区别详见 `docs/COORDINATOR_BLACKBOARD.md`。

## 5. 三类算法事实必须分开表述

### 5.1 来自 reference executable 的核心机制

- Dataset -> 采样/FLA -> numeric Top-5 -> LLM semantic 1～3 -> repository prior；
- i1=Synthesize（0 parent）、e1=Imitate（2）、e2=Recombine（2）、m1=Revise（有效 1）、m2=Fine-tune（有效 1）；
- 每个 active operator 生成 `population_size=P` 个 offspring；reference executable 每批立即 `P + P -> P`，下一 operator 可从更新后的 population 选择 parent；
- early selection 使用 fitness tier 与 operator diversity，late selection 使用 objective 优先、operator count tie-break；
- 最终先检查完整评价与有效代码行；唯一 exact minimum 直接选，exact tie 才调用 LLM。

### 5.2 deliberate faithful 决策

这些行为保留论文/用户明确语义，但不是对 reference 默认 Python 行序或 RNG 的逐字节 replay：

- early strategy 顺序固定为 `i1,e1,e2,m1`；reference 默认 operator list 的可执行顺序实际是 `e1,e2,m1,i1`；
- 四个 active strategy 都从同一个代初 retained P 取 parent，累计 4P 后统一执行一次 `5P -> P`；这是用户明确指定的论文/产品语义，故意不同于 reference executable 的四次 `2P -> P`；
- schedule 与 population manager 使用同一 `is_early_generation()` 边界；`G=1` 也执行 early `i1/e1/e2/m1`，修正阶段语义错位；
- m1/m2 只抽取并传入一个有效 parent；reference 会“抽 2 用 1”，所以 parent 语义一致、RNG 消耗不一致；
- engineering mode 将 final 的 20-budget 条件推广为当前 `candidate_budget`，无合格项使用显式 stable fallback；faithful mode 固定要求 20 点 trajectory、空合格集抛错，并保留 reference population tie 顺序。

### 5.3 明确的 Agent / Backend 工程增强

- durable AgentTask、Capability Registry、Blackboard projection、missing-work reconciliation；
- MySQL 事务预算、幂等 EvaluationJob、lease/fencing、多 Worker、受监督候选子进程；
- cooperative Pause/Resume/Cancel、runtime lease、ref-based checkpoint、startup recovery；
- KnowledgeGap 驱动的 PriorResearch、Literature Hybrid RAG、同 Run Memory；
- FailureClassifier 后的 Repair 新 Candidate version；
- Trace 与 24 场景 Agent Engineering Harness；
- 多 seed Final Optimization：默认 2 个 seed，每个使用 `2 * candidate_budget`，以稳定 clone 复用同一评价队列和预算账本。

最后一项是对 README/论文式 Stage 4 的工程补全；`references/prievo/` 可执行 Python 没有该阶段。代码和报告均写入 `engineering_extension=true`、`reference_executable_missing=true`，不得描述成 reference 原仓已有实现。

## 6. `research_faithful_mode` 的准确语义

`RESEARCH_FAITHFUL_MODE=true` 是“隔离可选 Research/Repair 对 population 的影响”，不是整个系统的字节级 reference replay：

- SimilarityAgent、HeuristicGenerationAgent、FinalSelectionAgent 仍在原生语义节点工作；
- Engine 不装配 PriorResearch workflow；若 Generation 模型仍返回 KnowledgeGap，Artifact 会保留，但没有研究恢复路径，Runtime 显式失败，避免静默注入额外知识；
- Candidate failure 仍可形成 Repair 审计和新版本，但 repaired Candidate 被标为 INVALID 且不得进入 population；
- Final Optimization 目前仍会运行，它是独立工程补全，不受该开关关闭。

因此该开关应被解释为 Research/Repair behavior isolation，而不是“完全等同 reference executable”。

## 7. 当前真实调用链

```text
POST /api/runs
  -> RunApplicationFacade.create_run（总预算含 Final Optimization reserve）
  -> PriEvOEngine.prepare
       -> LandscapeAnalysisService
       -> PriorRetrievalService.retrieve_numeric(top_k=5)
       -> DurableSimilarityWorkflow -> SimilarityAgent
       -> repository Prior extraction
  -> PersistentEvolutionRuntime.execute
       -> prior seed evaluation / i1 fill
       -> for generation
            -> for active operator
                 -> DurableGenerationWorkflow -> HeuristicGenerationAgent
                    -> optional KnowledgeGap -> PriorResearch -> resume
                 -> Candidate durable materialization
                 -> EvaluationJob -> worker -> supervised benchmark
                 -> optional FailureClassifier -> RepairAgent -> new Candidate
                 -> operator batch durable boundary（暂不选择）
            -> retained P + four batches 4P
            -> one early/late 5P -> P selection
            -> generation-boundary checkpoint
       -> DurableFinalSelectionWorkflow
            -> unique/fallback direct，或 exact tie -> FinalSelectionAgent
       -> FinalOptimizationService（2 seeds × 2B）
       -> RUN_COMPLETED
```

## 8. 可核验证据

- 算法与 operator-batch：`tests/test_prievo_core.py`、`tests/test_v5_prievo_engine.py`；
- 五 Agent 产品路径：`tests/test_similarity_workflow.py`、`test_generation_workflow.py`、`test_prior_research_workflow.py`、`test_repair_workflow.py`、`test_final_selection_workflow.py`；
- 真实执行：`tests/test_executable_dataset_evaluator.py`；
- Memory/RAG：`tests/test_generation_workflow_memory.py`、`test_hybrid_literature_rag.py`、`test_rag_evaluation.py`；
- 协调与恢复：`tests/test_durable_agent_coordinator.py`、`test_pause_resume_reconciliation.py`、`test_recovery_manager.py`；
- 独立 Worker：`tests/test_evaluation_worker_service.py`；
- Final Optimization：`tests/test_final_optimization.py`；
- 综合故障路径：`tests/test_agent_harness.py` 与 `reports/agent_harness.json`。

上述证据证明工程调用链，不等价于大规模算法效果或生产吞吐 Benchmark。
