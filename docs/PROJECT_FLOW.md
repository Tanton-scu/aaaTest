# PriEvO-Agent 项目执行流

本文从一个真实 API Run 出发，说明输入、调用者、LLM 边界、持久产物和下一步；这里描述的是已经工程化后的当前产品链。

## 1. 参数与预算

`POST /api/runs` 的四个核心参数分别是：

- `generations=G`：外层 heuristic evolution 代数；
- `population_size=P`：保留 population 大小，也是每个 active operator 生成的 offspring 数；
- `candidate_budget=B`：每个 heuristic 内部 `evaluate` 可消费的有效新配置数；
- `random_seed`：FLA sampling 与 Core parent RNG 的运行 seed。

无 Candidate failure 的名义评价上限为：

```text
Evolution = B * P * (1 + 4G)
Final Optimization = 2 seeds * (2B) = 4B
API minimum total budget = B * [P * (1 + 4G) + 4]
```

默认 `G=4,P=10,B=20` 时，Evolution 最多评价 170 个 heuristic、消费 3400 个有效配置预算；Final Optimization 再预留 80，总预算为 3480。重复映射到同一 Dataset 配置不会增加 `used_budget`。Repair 新版本会消费正常预算，因此异常运行可能在后续阶段遇到权威账本不足，不会绕过预算。

## 2. 总调用链

```text
Create Run
 -> Dataset / FLA
 -> Numeric Top-5
 -> SimilarityAgent 选择 1~3
 -> Original Prior
 -> Prior seeds + i1 补齐初始 P
 -> [Generation 1..G]
      -> [4 active operators]
           -> 每个 operator 生成 P
           -> 真实评价本批 P
           -> 当前 P + 本批 P
           -> 立即筛回 P
 -> Final qualification / exact tie
 -> Final Optimization（2 seeds × 2B）
 -> Final report / Run Completed
```

“一代生成 40 个”与“逐 operator 筛选”同时成立：当 `P=10` 时四个 active operator 共生成 40 个，但不是先堆成 40 再与原 10 做一次 `50 -> 10`。真实时序是四次 `10 + 10 -> 10`；后一个 operator 的 parents 来自前一批刚更新的 population。

## 3. Phase A：创建与启动 Run

1. `api/app.py` 校验请求并调用 `RunApplicationFacade.create_run()`。
2. Facade 计算包含 Evolution 与 Final Optimization 的最小总预算，原子保存 `OptimizationTask`、`Run(PENDING)` 和 `RUN_CREATED`。
3. App 的后台 executor 调用 `LocalRuntimeComposition.execute()`；Full Mode 启动时还会由 `RecoveryManager` 调度未完成 orphan Run。
4. `PriEvOEngine.run()` 先调用 `prepare()`，之后才进入持久 Runtime。

该阶段不调用 LLM。HTTP 返回 202 不等待演化完成。

## 4. Phase B：Dataset、采样与 FLA

调用链：

```text
DatasetRegistry.load(dataset_id)
 -> LandscapeAnalysisService.analyze
 -> DatasetLandscapeSampler.sample
 -> exact / nearest Dataset mapping
 -> ReferencePflaccoAnalyzer（8 metrics）
```

输入是 Dataset 的参数枚举空间、objective table、Dataset digest 和 task seed。采样大小为 `min(100, search_space_size)`；每条样本都记录原配置、映射配置以及 exact/nearest provenance。

输出：

- `LANDSCAPE_SAMPLE` Artifact；
- `LANDSCAPE_SAMPLED` Event；
- 8 个 FLA 指标：FDC、FBD、PLO、Skewness、Kurtosis、CL、MIE、NBC。

已安装 research extras 时现场调用 `pflacco` adapter。依赖不可用时，只允许按目标 Dataset 精确 ID 读取 recorded profile，并在 Artifact/Event 中写明 `fla_source` 和 `fallback_reason`；未知 Dataset 不会被伪造指标替代。

## 5. Phase C：Numeric Top-5、SimilarityAgent 与 Original Prior

### 5.1 数值检索

`PriorRetrievalService.retrieve_numeric(target, top_k=5)` 对 8 个指标执行 reference-compatible 定向、归一化和距离排序。它是 deterministic Core 能力，不调用 LLM。

产物 `TOP5_CANDIDATES` 只包含 target profile、metric semantics 和 numeric candidates；随后 `SIMILARITY_CANDIDATES_READY` 触发 missing-work reconcile。

### 5.2 语义筛选

```text
TOP5_CANDIDATES
 -> SEMANTIC_SIMILARITY_SELECTION AgentTask
 -> AgentRegistry
 -> SimilarityAgent
 -> SIMILARITY_PROMPT + SIMILARITY_DECISION
```

SimilarityAgent 调用一次结构化 LLM，只能从 Top-5 exact ID allowlist 中选 1～3 个，且 `metric_evidence` 必须覆盖所选实例的全部 8 个指标。它无 Memory、无 RAG，也不能生成 Prior。

### 5.3 Prior 提取

`PriorRetrievalService.extract()` 根据 Decision 中的实例 ID，从 `resources/prior_knowledge/` 的结构化 repository 读取真实 optimizer/operator/code evidence，形成 immutable `InstanceSpecificPrior`。`INSTANCE_SPECIFIC_PRIOR` Artifact 保存 selected IDs、evidence version 和上游 refs。

## 6. Phase D：初始 population

1. `PriEvoEvolutionCore.propose_prior_seeds()` 按 code 去重并执行静态安全兼容审计，只物化最多 P 个 `SUPPORTED` historical heuristic；`UNSUPPORTED` 条目仍保留为 immutable Prompt/Evidence，并写独立兼容性 Artifact/Event；
2. Runtime 把每个 prior seed 送入正式 EvaluationJob；无效 seed 会留下失败/Repair 审计并被跳过；
3. 若有效数量不足 P，使用 i1/Synthesize 逐个生成、持久化并评价，直到补齐；
4. 使用 early fitness-tier/diversity manager 筛为初始 P；
5. 保存 `initial_population` Checkpoint，cursor 指向 generation 1/operator 0。

每个成功 CandidateDraft 在下一次 LLM 调用前先成为 durable Artifact/Candidate，因此第 N 次模型调用崩溃不会抹掉前 N-1 个输出。

## 7. Phase E：前期与后期 operator schedule

`core/schedule.py` 当前固定：

| 阶段 | Active operators | 正式 Strategy | 有效 Parent 数 |
| --- | --- | --- | ---: |
| `generation <= G // 2` | `i1,e1,e2,m1` | Synthesize、Imitate、Recombine、Revise | 0、2、2、1 |
| `generation > G // 2` | `e1,e2,m1,m2` | Imitate、Recombine、Revise、Fine-tune | 2、2、1、1 |

Core 使用 reference rank-weighted、有放回 parent selection。e1/e2 可抽到同一 Candidate 两次；m1/m2 当前只抽取一个有效 parent，因此保留一元策略语义，但不 replay reference “抽 2 用 1”的 RNG 消耗。

前期顺序 `i1,e1,e2,m1` 采用论文/用户明确语义；reference 默认 executable 实际顺序是 `e1,e2,m1,i1`。这是 deliberate faithful decision，而不是未披露的逐行复刻。

## 8. Phase F：单个 GenerationTask

对每个 operator 的每个 sequence，Runtime 依次执行：

```text
Core 固定 strategy + parents
 -> GENERATION_REQUEST Artifact
 -> HEURISTIC_GENERATION AgentTask
 -> HeuristicGenerationAgent 加载 strategy SKILL.md
 -> GENERATION_PROMPT
 -> CandidateDraft 或 KnowledgeGap
```

Generation Context 包含：

- task / `run_tuners(file,budget,seed,maxlives)` / injected `evaluate` contract；
- immutable Original Prior slice 与 refs；
- 当前 strategy Skill body、version、digest；
- selected Parent 的 C/D/F/T/O、used budget 与最近 3 步 lineage；
- 最多 6 条同 Run generation memory；
- 最多 5 条已有 Research evidence；
- 严格输出 schema。

它禁止整个 population、跨 Run memory，以及由模型生成 fitness/trajectory。模型只可输出：

- `CandidateDraft(code, description, operators, generation_note)`；或
- 真正算法证据不足时的 `KnowledgeGap`。

CandidateDraft 会产生 `CANDIDATE_DRAFT`、generation memory 和 `CANDIDATE_DRAFT_MATERIALIZED`；Core 再生成稳定 Candidate ID。Skill、prompt、prior、context refs 被写入 lineage。

## 9. Phase G：KnowledgeGap -> Research -> 同一 Generation Resume

只在非 faithful mode 且 Generation 返回 KnowledgeGap 时发生：

```text
KNOWLEDGE_GAP
 -> 唯一 PRIOR_RESEARCH AgentTask
 -> PriorResearchAgent.formulate_query
 -> ToolGovernanceGateway -> Hybrid Literature RAG
 -> LITERATURE_EVIDENCE + PRIOR_EXPLANATION
 -> GENERATION_RESUME_REQUEST
 -> 唯一 HEURISTIC_GENERATION_RESUME AgentTask
 -> RESUMED_CANDIDATE_DRAFT
```

Research 保存 Original Prior digest，并只新增 annotation/evidence。RAG 为空时写 `status=EMPTY`，不调用模型编造 explanation；恢复请求仍以 best-effort policy 继续一次。恢复后若再次返回 KnowledgeGap，工作流显式失败，不创建无界研究循环。

`RESEARCH_FAITHFUL_MODE=true` 时 Engine 不装配 Research workflow；KnowledgeGap 会保留但不能改变算法上下文，Run 随后显式失败。

## 10. Phase H：真实 Candidate Evaluation

Candidate 先保存 code Artifact，再提交 EvaluationJob。幂等 material 包含 run/task、candidate、dataset digest、seed、budget、evaluator version 和参数版本；提交事务先预留 budget。

Full Mode：

```text
App 等待 durable Job
 -> 独立 worker MySQL claim
 -> owner token / lease
 -> DatasetEvaluator
 -> python -I 受监督子进程
 -> Candidate run_tuners
 -> injected evaluate exact/nearest mapping
 -> EvaluationResult / Artifact / budget settlement
```

Job 的 `seed`、`budget` 是单次评价事实源，Worker 会覆盖 Task 默认值后传给 evaluator。父进程控制 timeout、最小环境和资源限制；trajectory、best configuration、objective 和 `used_budget` 来自真实执行。

FailureClassifier 严格分流：

- transient infrastructure -> 同 Candidate/Job retry/requeue；
- syntax/runtime/interface/algorithm timeout/OOM/logic failure -> Candidate repair path；
- terminal/no budget -> 明确 DEAD/INVALID，不伪造结果。

## 11. Phase I：Repair 新版本

Candidate failure 先持久化为 `CANDIDATE_FAILURE`，再派生唯一 `CANDIDATE_REPAIR` task。RepairAgent 执行 Diagnose -> Repair，两阶段模型输出分别受 schema、attempt 和 budget guard 约束。

成功时创建形如 `root-R1` 的新 Candidate：

- `creation_type=REPAIR`；
- `repair_parent_id` 指向原件；
- 原 Candidate 和失败 Artifact 不被覆盖；
- 新版本提交新的 EvaluationJob。

faithful mode 下 repaired version 只保留审计并标为 INVALID，不进入 population。

## 12. Phase J：四批累计后统一 `5P -> P`

每代开始时固定一份 retained population `P`。四个 active operator 依次执行，但都只能从这同一份 `P` 选择 parent。一个 operator 的 P 个 offspring 全部评价完成后：

1. 写 `OPERATOR_BATCH_COMPLETED`，记录本批与累计 offspring 数；
2. 把已评价 Candidate refs 加入 generation cursor；
3. 保存 ref-only operator-boundary Checkpoint；
4. 到安全点响应 pause/cancel，再进入下一个 operator；
5. 不在本批做 selection，后续 operator 仍看到代初 retained `P`。

四批完成后得到 `4P` offspring，与 retained `P` 合并为 `5P`。前期只调用一次 `select_population_early()`，后期只调用一次 `select_population_late()`，写单个 `OPERATOR_SELECTION_COMPLETED`/`GENERATION_SELECTION_COMPLETED`，再保存 generation-boundary Checkpoint。这是用户指定的批语义；reference executable 的逐批 `2P -> P` 只作为审计对照，不冒充当前产品行为。

## 13. Phase K：Final Selection

Final workflow 先确定性投影每个 Candidate 的 C/D/O/F/T，并执行：

1. 有效非注释代码行至少 50；
2. `used_budget == candidate_budget` 或 trajectory 长度完整；
3. objective 为有限数值；
4. 在合格集中找最小 objective 的 exact equality tie group。

分支：

- 唯一 best：零 LLM，直接保存 `FINAL_SELECTION_DECISION`；
- 多个 exact ties：创建 `FINAL_TIE` 和唯一 FinalSelectionTask，Prompt 只含 tied C/D/O/F/T 与 `final_heuristic_audit` Skill；
- 模型输出不合法：稳定回退到 tied IDs 排序第一项，并记录 degraded reason；
- 无合格项：engineering mode 使用有限 objective 的 stable fallback；faithful mode 与 reference 一样直接失败，不静默降级。

## 14. Phase L：Final Optimization 工程补全

选出的 heuristic code 保持不可变。`FinalOptimizationService` 为默认 seeds `1009,2027` 各建一个稳定 clone，每个用 `2B` 预算走同一 EvaluationJob/Worker/账本。报告包含每 seed trajectory、objective、best configuration、均值/中位数/总体标准差、聚合轨迹与最终最佳配置。

产物为 `FINAL_OPTIMIZATION_REPORT`、`FINAL_OPTIMIZATION_COMPLETED` 和 `FINAL_CONFIGURATION_SELECTED`。报告明确声明该阶段是工程补全，因为 reference executable 没有实际实现 README 所述 Stage 4。

## 15. Pause、Resume、Cancel 与崩溃恢复

- Pause：API 只持久化 `pause_requested`；Runtime 在已有一致 Checkpoint 的 Candidate/operator/selection 边界停止创建新工作并转为 PAUSED。
- Resume：PAUSED -> RUNNING，重新 claim runtime lease，恢复最新 Checkpoint，再对 Candidate/Job/Result 做 reconciliation。
- Cancel：事务内标记 CANCELLED，取消未 claim EvaluationJob/PENDING AgentTask并释放对应 reservation；Runtime/Worker 的 owner fencing 拒绝迟到写入。
- Process restart：FastAPI lifespan 调用 `RecoveryManager`，回收 stale EvaluationJob、orphan AgentTask/runtime lease，重做 missing-work reconcile 并调度 PENDING/RUNNING Run。

## 16. 推荐的源码验证顺序

1. `tests/test_landscape_analysis.py`、`test_similarity_workflow.py`；
2. `tests/test_heuristic_generation_agent.py`、`test_generation_workflow.py`、`test_generation_crash_recovery.py`；
3. `tests/test_executable_dataset_evaluator.py`、`test_failure_classifier.py`、`test_repair_workflow.py`；
4. `tests/test_prievo_core.py`、`test_v5_prievo_engine.py`；
5. `tests/test_final_selection_workflow.py`、`test_final_optimization.py`；
6. `tests/test_pause_resume_reconciliation.py`、`test_recovery_manager.py`；
7. `tests/test_agent_harness.py` 与 `reports/agent_harness.json`。
