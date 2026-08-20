# PriEvO-Agent 五 Agent 架构

本文只描述当前 `src/prievo_agent/algorithm/prievo_engine.py` composition root 实际装配的五类 Agent。早期的 `Evolution/Research/Review` 边界 Agent 与同步 Coordinator 已删除，避免形成两套互相矛盾的产品叙事。

## 1. 设计边界

PriEvO-Agent 采用“确定性 Core 决策、Agent 处理语义工作”的边界：

- PriEvO Core 决定 generation、active operator、parent、每算子 offspring 数、代级 `5P -> P` selection、评价预算和最终候选资格；
- Coordinator 只从持久 Artifact 发现缺失工作并创建 `AgentTask`；
- Dispatcher 根据 capability 找到唯一 Agent handler，负责领取、有限重试和回写 Artifact；
- Agent 只能读取其输入 Artifact 和 Context Policy 放行的字段，不能直接修改 Run、Population、Candidate、EvaluationJob 或预算；
- Agent 输出先成为可审计 Artifact，Application Workflow 再校验并转换成 Candidate 等领域对象。

这不是由 LLM 自由规划步骤的 Supervisor 架构。五类 capability 固定在 `src/prievo_agent/domain/models.py::AgentCapability`，其注册和路由分别位于 `agents/registry.py` 与 `application/agent_dispatcher.py`。

## 2. 通用持久执行链

```text
Core / Workflow 写输入 Artifact
        |
DurableAgentCoordinator.reconcile(run_id)
        |  按 input kind + source artifact 构造幂等任务
AgentTask(PENDING)
        |
AgentTaskDispatcher.dispatch(task_id)
        |  claim -> Blackboard.from_store -> Registry.resolve
Agent handler
        |  Skill + 白名单 Context + LLM/Tool
输出 Artifact(s)
        |
AgentTask(COMPLETED, output_artifact_refs)
        |
Workflow 校验 artifact kind/source ref/schema 后消费
```

`AgentTask` 只保存引用和执行状态。Prompt、decision、evidence、draft 都是 Artifact；Blackboard 也只投影 metadata/ref，不把 Artifact 内容复制成共享可写状态。该结构使任务可在 SQLite 或 MySQL 重开后恢复，而不是依赖进程内对话历史。

## 3. 五类 Agent 总表

| Agent | 触发 Artifact / Task | Capability | 当前 Skill | 可见 Context | 主要产物 | 禁止边界 |
| --- | --- | --- | --- | --- | --- | --- |
| SimilarityAgent | `TOP5_CANDIDATES` / `SEMANTIC_SIMILARITY_SELECTION` | `SEMANTIC_SIMILARITY` | `semantic_similarity_selection` | target 的 8 个 FLA metrics、指标语义、numeric Top-5/rank | `SIMILARITY_PROMPT`、`SIMILARITY_DECISION` | 不看 Population、Memory、RAG，不生成 Prior |
| HeuristicGenerationAgent | `GENERATION_REQUEST` / `HEURISTIC_GENERATION`；研究后另有 resume task | `HEURISTIC_GENERATION` | 按 i1/e1/e2/m1/m2 映射 | 任务、原 Prior、父代、最近 3 步 lineage、最多 6 条同 Run generation memory（含已测 fitness/trajectory）、最多 5 条 evidence | `GENERATION_PROMPT` 或 `GENERATION_RESUME_PROMPT`；`CANDIDATE_DRAFT` 或 `KNOWLEDGE_GAP` | 不选 operator/parent，不看整个 Population，不把 Prompt 猜测当 fitness |
| PriorResearchAgent | `KNOWLEDGE_GAP` / `PRIOR_RESEARCH` | `PRIOR_RESEARCH` | `prior_explanation` + `literature_evidence_review` | 当前 gap、原 Prior slice/ref、landscape、strategy、parent summary、同 Run research memory、evidence | query/explanation prompt、`LITERATURE_EVIDENCE`、`PRIOR_EXPLANATION` | 不改原 Prior，不产生 Candidate，不在无证据时编造结论 |
| RepairAgent | `CANDIDATE_FAILURE` / `CANDIDATE_REPAIR` | `CANDIDATE_REPAIR` | `candidate_failure_diagnosis` + `candidate_code_repair` | 失败 Candidate、分类事实、inspection refs、attempt/budget、同 Run repair/failure history | diagnosis/repair prompt、`REPAIR_DECISION`、可选 `REPAIRED_CANDIDATE_DRAFT` | infrastructure failure 不触发；不覆盖原 Candidate，不绕过评价；不可修复是合法业务结果 |
| FinalSelectionAgent | `FINAL_TIE` / `FINAL_SELECTION` | `FINAL_SELECTION` | `final_heuristic_audit` | 仅完全并列的 candidate C/D/O/F/T 与输出 schema | `FINAL_SELECTION_PROMPT`、`FINAL_SELECTION_DECISION` | 不看非并列候选、Memory 或 RAG，不改 qualification/fitness |

## 4. SimilarityAgent

### 触发与输入

`application/similarity_workflow.py::DurableSimilarityWorkflow` 先把 numeric distance 产生的 Top-5 写为 `TOP5_CANDIDATES`。Coordinator 据此创建 `SEMANTIC_SIMILARITY_SELECTION`；handler 从 Blackboard 核验唯一输入的 kind，再读取内容。

`agents/similarity.py::SimilarityAgent` 的 Prompt 包含目标 landscape、八个 FLA 指标及含义、五个候选和 numeric rank。Agent 先经过 `SimilarityContextPolicy` 白名单与 ContextBuilder，再构造最终 JSON-only Prompt；decision 保留 context metadata。

### 输出约束

输出必须选择原 Top-5 中恰好 1–3 个不同 Dataset ID，每个选择都要给出覆盖八个指标的证据。Agent 的 schema/allowlist 校验拒绝越界 ID、重复 ID、数量错误或缺失指标；有效输出写成 `SIMILARITY_DECISION`，随后 repository workflow 才据此抽取 instance-specific prior。

SimilarityAgent 只负责 reference executable 中“numeric Top-5 后的 semantic selection”这一语义节点。它没有权力更改距离计算或扩大候选池。

## 5. HeuristicGenerationAgent

### Strategy 与父代契约

Skill 映射由 `agents/heuristic_generation.py` 固定：

| Strategy | Skill | 父代数 | 语义 |
| --- | --- | ---: | --- |
| `i1` | `synthesize` | 0 | 从 immutable prior 合成新 heuristic |
| `e1` | `imitate` | 2 | 以父代信息进行 imitation |
| `e2` | `recombine` | 2 | 重组两个父代 |
| `m1` | `revise` | 1 | 修订一个父代 |
| `m2` | `fine_tune` | 1 | 细调一个父代 |

Core 在 `core/evolution.py` 选定 strategy、父代和稳定 generation sequence 后，`application/generation_workflow.py::DurableGenerationWorkflow` 才写 `GENERATION_REQUEST`。Agent 能看到父代的 code、description、fitness、trajectory 和 operators，但看不到整个 Population，也不能自行换 strategy。

每个 generation 的每个 active operator 都调用该 workflow `P` 次；Agent 产出的仍只是 draft，不能把 Prompt 中的猜测写成已测 fitness。Workflow 校验代码、description、operators、strategy/parent contract 后，才创建稳定 ID 的 `Candidate` 和后续 `EvaluationJob`。

### KnowledgeGap 与 resume

Generation 输出二选一：

- `CANDIDATE_DRAFT`：直接进入 Candidate 创建与真实评价；
- `KNOWLEDGE_GAP`：说明缺失知识，交给 PriorResearchAgent。

研究完成后，Workflow 写 `GENERATION_RESUME_REQUEST`，使用专门的 resume coordinator 创建 `HEURISTIC_GENERATION_RESUME`，再次由同一 capability 执行，Prompt/输出分别为 `GENERATION_RESUME_PROMPT` 与新的 draft 或 gap。resume 是持久因果链的一部分，不是对第一次 LLM 调用的内存内追加消息。

`RESEARCH_FAITHFUL_MODE=true` 时 Engine 不装配 PriorResearchWorkflow；初次 Generation、Similarity 和 Final Selection 仍工作。若 Generation 返回 KnowledgeGap，gap 会持久化，但因没有 research resolution，Runtime 明确失败，而不是悄悄改变算法路径或假装研究已完成。

### 同 Run working memory

成功 draft 会落 `GENERATION_SUMMARY`；真实 EvaluationResult 结算后另写 `GENERATION_EVALUATION`，保留已测 fitness、trajectory、parents 和 refs，再尽力回温 Redis/InMemory recent cache。后续 generation 通过 `load_with_fallback(run_id, "generation", store)` 只读取同 Run、同 scope 记录；缓存为空或不可用时回源 SQLite/MySQL。Memory 是生成上下文辅助，不参与 Core selection，事实权威仍是 Candidate/Result。

## 6. PriorResearchAgent

`application/prior_research_workflow.py` 只在 `KNOWLEDGE_GAP` 后运行。handler 会回溯 gap 指向的 generation request，并构造 `ResearchContextPolicy` 允许的最小上下文。query 阶段加载 `prior_explanation`，证据审阅/解释阶段加载 `literature_evidence_review`；两者的 name/version/digest/ref 都进入 Prompt、Artifact、Memory 与恢复证据来源。

执行分为两个显式模型步骤：

1. 根据 gap 与上下文生成受限检索 query；
2. Literature Tool 经治理层执行 Hybrid retrieval；只有存在 evidence item 时，模型才生成 evidence-grounded explanation。

每次工具调用写 `ToolCallRecord`。证据写入 `LITERATURE_EVIDENCE`，解释写入 `PRIOR_EXPLANATION`；query Prompt 和 explanation Prompt 也分别持久化。检索为空时跳过 explanation 模型调用，写出明确的 `EMPTY` 结果，不允许把模型常识伪装成文献证据。

“研究”只为当前 gap 解释原 Prior 的适用性；原 Prior 的 digest/ref 保持不变。query、命中文献/chunk、evidence status 和 Skill refs 写入同 Run research scope；Redis miss 时从 durable history 回温。

## 7. RepairAgent

`runtime/failure_classifier.py` 先区分 infrastructure failure 与 candidate failure。前者只按评价队列策略 retry/dead-letter；只有可归因于 candidate code 的错误才由 `application/repair_workflow.py` 写 `CANDIDATE_FAILURE` 并触发 Agent。

RepairAgent 的 Diagnose 阶段使用 `candidate_failure_diagnosis`，Repair 阶段单独使用 `candidate_code_repair`。两阶段各自保留 Skill/context/prompt provenance；只有 diagnosis 明确 repairable 且 attempt/budget guard 允许时才输出 `REPAIRED_CANDIDATE_DRAFT`。Workflow 施加代码和 lineage guard，创建新 ID（原 candidate root 加 repair 次数后缀）的 `Candidate`，再走与普通 candidate 相同的验证、预算预留和真实评价。`repairable=false` 会完成 task、持久 decision/memory 并跳过该 INVALID Candidate，不作为 Runtime 系统异常。

原 Candidate 不会被原地覆盖。若研究忠实模式开启，Repair 的过程 Artifact 仍可审计，但 Runtime 把 repaired candidate 标记为 `INVALID` 并阻止其进入 Population。因此该模式隔离的是 Agent Research/Repair 工程增强，不等于关闭所有 LLM。

Repair history 只来自同 Run repair scope；Redis/SQLite/MySQL 回源后按 candidate family 过滤并受 ContextPolicy 数量上限约束。History 超阈值时才经 `history_summary` 滚动压缩。

## 8. FinalSelectionAgent

`application/final_selection_workflow.py` 先由确定性代码执行资格判定。engineering mode 允许当前 `candidate_budget` 完整条件并对空集做显式 stable fallback；faithful mode 固定要求 20 点 trajectory 和至少 50 行有效代码，空集直接失败。若有唯一最优 fitness，直接选择，不创建 AgentTask、不调用 LLM。

只有多个合格候选拥有完全相同最优 fitness 时，Workflow 才写 `FINAL_TIE`。FinalSelectionAgent 仅看到 tied candidates 的 code、description、operators、fitness 和 trajectory，以及 `final_heuristic_audit` Skill；输出还必须包含结构化 structure/operator comparison。engineering mode 异常时按稳定 ID 回退，faithful mode 保留 reference population tie 顺序并回退原序第一项。

这保持了 reference 的核心语义：唯一 best 不经 LLM，exact tie 才进行代码审查。当前工程差异是资格长度随 `candidate_budget` 参数化，且无合格候选不再像 reference executable 那样抛出 `ValueError`。

## 9. 重试、失败与幂等

- Coordinator 的 task ID/idempotency key 由 task type 与 source Artifact 确定，不因进程重启变化；
- Store 的唯一约束处理并发 reconcile；
- Dispatcher 领取 task 后持有 claim token + lease，完成/失败必须同时满足 owner/token/未过期/Run active fencing；orphan 后旧 Dispatcher 不能结算，重复 dispatch 已完成任务不会再次调用模型；
- transient、timeout、JSON/malformed output 进入有上限的 redrive，`max_attempts` 默认 3；耗尽后任务为 `FAILED`；
- Agent 重试不能增加 PriEvO population 或跳过 Candidate 的真实评价；
- RecoveryManager 会回收孤儿 AgentTask，并重新运行 missing-work reconcile。

LLM adapter 位于 `infrastructure/llm_adapter.py`；环境变量为 `LLM_API_ENDPOINT/LLM_API_KEY/LLM_MODEL`。三项全空时 composition 使用 `FakeLLM`，部分配置会拒绝启动，仓库不预置真实 API 密钥。

## 10. 与 reference 机制及工程增强的关系

| 类别 | 当前处理 |
| --- | --- |
| reference 原机制 | numeric Top-5 后语义选 1–3；Prior seed + i1 initial population；四 operator 各生成 `P`；fitness/diversity selection；unique best / exact tie final selection |
| deliberate product/faithful 差异 | early operator 顺序显式为 `i1,e1,e2,m1`；四批基于代初 P、统一 `5P -> P`（reference executable 为顺序 `2P -> P`）；m1/m2 直接抽一个父代；final faithful mode 固定 20 点/空集失败/保留 tie 原序 |
| Agent/Backend 工程增强 | durable AgentTask/Artifact、KnowledgeGap Research、Repair、同 Run Memory、Hybrid RAG、恢复、Trace、外部 Evaluation Worker、最终双 seed optimization |

Coordinator、Context、Memory、RAG 和 Trace 只包装并审计这些语义节点。它们没有把 PriEvO 的 operator schedule、parent sampling、selection 或 budget 改交给 Agent。

## 11. 可验证证据

- 五 Agent/schema：`tests/test_similarity_agent.py`、`tests/test_heuristic_generation_agent.py`、`tests/test_prior_research_agent.py`、`tests/test_repair_agent.py`、`tests/test_final_selection_agent.py`；
- 五 workflow：`tests/test_similarity_workflow.py`、`tests/test_generation_workflow.py`、`tests/test_prior_research_workflow.py`、`tests/test_repair_workflow.py`、`tests/test_final_selection_workflow.py`；
- Memory/Context：`tests/test_generation_workflow_memory.py`、`tests/test_run_local_agent_memory.py`、`tests/test_agent_context_policies.py`；
- 调度与恢复：`tests/test_durable_agent_coordinator.py`、`tests/test_agent_dispatcher.py`、`tests/test_recovery_manager.py`；
- 端到端受控场景：`tests/test_agent_harness.py` 与 `reports/agent_harness.json`。
