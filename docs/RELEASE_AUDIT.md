# PriEvO-Agent 1.0.0 发布审计

审计日期：2026-08-20（Asia/Shanghai）  
审计规格：工作区根目录 `prompt.md`，SHA-256 `B2EC862073300CE1903AC93CA9CC61941D63A1FA04FC6769DB8410D6B08421BE`  
项目版本：`1.0.0`

## 1. 最终结论

当前项目已经形成一条自洽、可运行、可恢复、可追踪的 PriEvO + Backend + Agent 产品链，核心能力都有源码与测试/Harness 证据，不再依赖旧三 Agent 或 README 声明。Full Mode 架构是 FastAPI + MySQL + Redis + 独立 Evaluation Worker；Demo/Test 是 SQLite + inline evaluator。两者共享同一 PriEvO Core、Agent workflows、Job/Checkpoint/Trace 语义。

本轮删除了旧 `EvolutionAgent/ResearchAgent/ReviewAgent`、同步 `AgentCoordinator`、旧 Reflection/Literature Harness 及对应测试，避免两套互相矛盾的调用链。RepairAgent 现在真实经过 `CandidateInspectionTool -> ToolGovernanceGateway`；LiteratureSearchTool 真实位于 KnowledgeGap 主链。ToolCall 持久记录 caller、reason、input metadata、start/end、duration、status 与 failure。

结论不是“生产无缺陷”。项目仍是单租户 Engineering Prototype；默认 vector 不是语义 embedding，候选子进程不是强安全沙箱，Worker 没有后台 heartbeat，RAG 评测集很小，最新 Docker/MySQL 回归受本轮宿主权限限制。以下边界全部在 README/专项文档中显式说明。

## 2. 这是否还是 PriEvO？

是。算法控制权仍在确定性 Core，而非 Agent Coordinator：

1. 目标 Dataset 经固定 seed sampling、exact/nearest mapping 与八项 FLA；
2. 八指标 numeric Top-5 与 LLM semantic 1～3 selection 分层；
3. selected instances 只作为 allowlist，Original Prior 由结构化 repository 提取；
4. initial population 使用 prior seed，不足由 i1 补齐；
5. early `i1/e1/e2/m1`、late `e1/e2/m1/m2`；parent 数为 0/2/2/1/1；
6. 四个 strategy 都从代初 retained P 选 parent、各生成 P 个 Candidate；评价完整 4P 后，与 retained P 统一执行一次 `5P -> P`；
7. early fitness/diversity 与 late fitness-first selection 来自 reference 规则；
8. 资格过滤后 unique best 零 LLM，objective 精确等优才触发 FinalSelectionAgent。

真实 Candidate evaluator 会执行 `run_tuners(file,budget,seed,maxlives)`，注入权威 `evaluate`，处理 Dataset exact/nearest mapping、duplicate 不耗预算、trajectory 与返回 best 一致性。旧 code-hash shuffle 已退出产品路径。

## 3. 哪些是论文/reference 原始机制？

- Dataset→sampling/FLA→numeric Top-5→semantic selection→prior；
- C/D/F/T/O Individual 信息；
- Synthesize/Imitate/Recombine/Revise/Fine-tune 的语义与 parent 约束；
- 前后期 operator 集合、rank-based parent selection；
- early diversity 与 late fitness population management；
- final qualification、unique direct、exact-tie LLM 与稳定 fallback。

两个 deliberate 差异必须主动说明：

- early 顺序采用论文/用户明确的 `i1,e1,e2,m1`；reference executable 因配置列表实际按 `e1,e2,m1,i1` 运行；
- schedule 与 selection 在 midpoint 使用统一边界，修复 reference executable 的一代错位。
- 用户明确要求整代四批统一 `5P -> P`；reference executable 实际是逐批四次 `2P -> P`，Audit 与 Checkpoint cadence version 显式区分二者。

## 4. 哪些是 Agent 工程增强？

- Similarity、HeuristicGeneration、PriorResearch、Repair、FinalSelection 五类领域 Agent；
- durable missing-work Coordinator、只读 Blackboard、capability Registry、Dispatcher；
- 五个独立 Context Policy、版本化 Strategy Skills、Prompt/Skill/context refs；
- Redis/MySQL Run-local generation/research/repair memory；
- KnowledgeGap→Literature RAG→PriorExplanation/Evidence→Generation resume；
- FailureClassifier→Diagnose→Repair 新 Candidate version；
- ScriptedFakeLLM、有界 redrive、Agent Trace 与 Engineering Harness。

这些增强不拥有 generation、parents、population、selection 或 budget。

## 5. 哪些是 Backend 工程增强？

- MySQL/SQLite durable Run/Candidate/Job/Result/Event/Artifact/Checkpoint/AgentTask/Memory/ToolCall；
- evaluation-v2 幂等 material、短事务预算 reservation/settlement；
- `FOR UPDATE SKIP LOCKED` claim、lease、renew、owner fencing、stale recovery；
- 独立 Evaluation Worker 与受监督 Candidate subprocess；
- Run owner lease/cursor、cooperative Pause/Resume/Cancel、startup RecoveryManager；
- operator safe-point Checkpoint 与 post-checkpoint reconciliation；
- FastAPI/SSE/Dashboard/metrics/trace；
- 选中 heuristic 的独立两个 seed、两倍预算 Final Optimization。

Final Optimization 是 reference README 意图的工程补全；reference executable 没有对应执行调用，不能称为原始源码复现。

## 6. 哪些功能默认开启？

| 功能 | 默认 |
| --- | --- |
| Compose `PRIEVO_MODE=full` | MySQL + Redis + external Worker |
| LLM 三项为空 | deterministic FakeLLM；链路不被短路 |
| LLM 三项完整 | OpenAI-compatible real adapter |
| `RESEARCH_FAITHFUL_MODE=false` | KnowledgeGap Research 与 Repair 可影响后续 generation/population |
| Hybrid RAG | KnowledgeGap 时按需启用；无 gap 零调用 |
| Vector backend | deterministic token hashing，明确 `production_semantic=false` |
| Final tie Agent | 只在 exact tie；unique best 零调用 |
| Final Optimization | 两个独立 seed，每 seed `2 × candidate_budget` |

## 7. faithful mode 如何工作？

`RESEARCH_FAITHFUL_MODE=true` 时 Engine 不装配 PriorResearch workflow，因此 Literature Evidence 不会改变 Generation Context；若 Generation 只返回 KnowledgeGap，gap 会持久化且 Run 明确失败，不伪造 Candidate。Candidate failure 仍可产生 Repair diagnosis/draft 审计，但 repaired Candidate 被标为 INVALID，并由 `FAITHFUL_REPAIR_EXCLUDED` 记录，不能进入 population。Similarity/Generation/Final 等原生语义节点仍运行。

`tests/test_faithful_mode.py` 注入一次动态 Candidate failure，证明 repaired version 存在但不进入 Checkpoint population；该小预算 fixture 因 reference qualification 要求 trajectory=20 且有效代码行不少于 50，会明确抛出 `NoQualifiedFinalCandidateError`，不会用 engineering fallback 伪造完成。

## 8. 系统断电或 API 进程退出如何恢复？

CandidateDraft、Candidate、EvaluationJob、Result、AgentTask 与 Artifact 在各自成功时立即提交；Checkpoint 只记录一致 population refs、Core RNG/state、operator cursor、Dataset/Prior/version refs。启动时 `RecoveryManager`：

1. 回收过期 Evaluation lease；
2. 回收超时 Runtime owner；
3. 只把超过 orphan timeout 的 CLAIMED AgentTask 重新入队；
4. 对 active Run 执行 missing-work sweep；
5. Runtime 从最新一致 Checkpoint 加载 population IDs，并查询 Checkpoint 后的 Candidate/Job/Result 做 reconciliation。

因此 LLM Draft 已完成但 Candidate 尚未 materialize 时可从 Artifact 恢复；Result 已提交但 Checkpoint 尚未更新时直接复用，不重复收费。

## 9. Worker 崩溃如何恢复？

Job 的 `lease_expires_at` 表示 owner 有效期。Worker 崩溃后 Job 暂时保持 RUNNING；periodic/startup stale sweep 在 lease 到期后将其重新入队，或在 attempts 耗尽后 dead-letter。新 Worker claim 后，旧 Worker 即使迟到也因 `status=RUNNING AND worker_id=owner` fencing 条件不命中而不能提交结果、修改 Candidate 或释放预算。

物理 benchmark 可能执行两次，逻辑 Result/预算只能结算一次。

## 10. Candidate timeout、OOM 与语法/接口错误如何处理？

受监督子进程设置 wall timeout、POSIX CPU/address-space/file/no-file limits，并验证 AST/import/interface。FailureClassifier 把 `SYNTAX_ERROR/RUNTIME_ERROR/INTERFACE_ERROR/ALGORITHM_TIMEOUT/ALGORITHM_OOM/LOGIC_FAILURE` 路由为 Repair，而不是原样重跑坏代码。原 Candidate 保持 INVALID；RepairAgent 先 Diagnose，再产生带 `repair_parent_id/attempt` 的新 Candidate version，重新走独立评价。

这不是强安全沙箱：Windows 只保证 wall timeout 降级，生产不可信多租户应增加容器/microVM 隔离。

## 11. LLM timeout 或 malformed output 如何处理？

LLM 调用属于 AgentTask。Dispatcher 只对 transient provider error 与 malformed protocol 做 durable、有界 redrive；每次增加 attempts，写 `AGENT_TASK_FAILED` 和 `AGENT_TASK_REDRIVE_SCHEDULED`。transient-once 可在同次 workflow 调用恢复；持续 malformed 恰好到 `max_attempts` 后 Task FAILED，不 busy-loop。已完成 Draft/Task 会按 idempotency/Artifact 恢复，不重调。

真实 provider timeout 不会被误分类成 Candidate evaluation retry；API 后台捕获未恢复异常并把 Run 转为 FAILED。

## 12. Redis 挂了如何处理？

Redis 只承担通知/近期缓存。写失败记录 warning 后继续；读 miss 或连接失败时，`load_with_fallback` 只按同一个 `run_id + scope` 从 MySQL/SQLite AgentHistory 读取，并尝试回温 Redis。Run A memory 永远不会自动进入 Run B Prompt。SSE 在 Full Mode 可退回数据库轮询等待，durable Event 不回滚。

## 13. 同一 Evaluation 重复提交如何处理？

logical identity 使用 `task/candidate/seed/budget/dataset_digest/evaluator_version/evaluation_parameters_version` 的 canonical material。MySQL/SQLite 对 idempotency key 建唯一约束；提交事务先查 existing，重复调用返回同一个 Job，不重复 reservation。同一 Candidate 若请求不同 Dataset、evaluator、seed、budget 或参数 identity，会在预算预留前被事务拒绝；需要重评时必须像 Final Optimization 一样先 clone Candidate。SQLite Candidate 唯一索引与 MySQL V006 同时兜底该不变量。

## 14. 两个 Worker 同时抢 Job 如何处理？

MySQL claim 使用短事务与 `FOR UPDATE SKIP LOCKED`；两个连接只能有一个把 Job 转为 RUNNING 并写入自己的 worker ID/lease。后续 renew、success、failure 都携带 owner token 并执行条件更新。SQLite 测试模拟双连接；真实 MySQL 专项在配置 `DATABASE_URL` 时运行。

## 15. 两个任务同时消耗最后预算如何处理？

`enqueue_job` 在同一事务中锁住 Run 行，计算 `total - consumed - reserved`，预算足够才插入 Job并增加 reservation。第二个事务必须在第一个提交后重新观察 reservation，因此不能依据旧 remaining 超卖。成功时 reservation 转 consumed；permanent failure/cancel 释放；retry 保留原 reservation。

## 16. KnowledgeGap 如何触发 Research？

HeuristicGenerationAgent 合法输出只有 `CandidateDraft` 或 `KnowledgeGap`。KnowledgeGap 先成为 Artifact；Coordinator missing-work rule 发现没有对应 `PRIOR_EXPLANATION + LITERATURE_EVIDENCE`，派生唯一 `PRIOR_RESEARCH` Task。Registry 路由 PriorResearchAgent，模型先构造有界 query，Gateway 授权 LiteratureSearchTool，随后 evidence/explanation 持久化。重复 event/reconcile 使用稳定 task/idempotency，不重复研究。

## 17. Research Evidence 如何进入 Generation Prompt？

Research 完成后创建 `GENERATION_RESUME_REQUEST`，引用原始 stable generation request、immutable Prior digest、explanation/evidence/tool refs。Generation Context Policy 只放行有界 Evidence；resume Prompt 明确包含 evidence content/provenance 和 Original Prior 旁路 annotation。一次 resume 后若模型仍返回 KnowledgeGap，workflow 明确失败，避免无界 Research loop。

测试同时比较研究前后 Prior canonical JSON/digest，证明 evidence 不能覆盖 Original Prior。

## 18. Repair 与 Retry 的区别？

| Retry | Repair |
| --- | --- |
| network、worker crash、transient infrastructure | candidate syntax/runtime/interface/timeout/OOM/logic |
| 同一个 Candidate、同一个 logical Job | 新 Candidate ID/version/lineage |
| 保留 reservation，延迟后重新 claim | 原 Job dead并释放 reservation，新 version 重新 submit |
| 不调用 RepairAgent | Diagnose→CandidateInspectionTool→RepairAgent |

FailureClassifier 是唯一分流入口；Agent 不能把 infrastructure failure 解释成代码修复。

## 19. Blackboard、Memory、Checkpoint、Artifact 的区别？

- **Artifact**：不可变、内容寻址的具体产物，如 Prompt、Decision、Draft、Evidence、Result、Checkpoint payload；
- **Blackboard**：从 durable Task/Event/Artifact metadata 重建的只读协作投影，只放 refs，不是事实源；
- **Memory**：同一 Run、同一 Agent scope 的有界历史摘要，用于后续 Prompt；Redis 是 cache，MySQL 是 durable history；
- **Checkpoint**：算法一致安全点的 population refs、Core/RNG/cursor/version baseline，用于恢复；不替代细粒度 facts。

四者不能互相替代，也不保存跨 Run 自动学习。

## 20. Harness 如何验证真实 Agent Path？

`AgentEngineeringHarness` 只替换不可控边界：供应商 LLM、外部文献、生产 Redis 和大 Dataset；仍运行产品 Runtime、Coordinator、Blackboard、Registry、ContextPolicy、SQLite durable Store、Artifact、Tool Gateway、Queue、Checkpoint 与 Trace。ScriptedFakeLLM 按 route FIFO，可注入 KnowledgeGap、transient、malformed，并记录完整 Prompt/call。

2026-08-20 重新生成的 `reports/agent_harness.json` 为 24/24 scenarios、116/116 assertions。它逐场景保存 expected/actual trace、assertions、Task/Artifact/Tool evidence，不只检查最终 Candidate 存在。

## 21. HitRate@K、MRR@K、Recall@K 如何计算？

- `HitRate@K`：每个 query 的 Top-K 是否至少命中一个 gold chunk，取宏平均；
- `MRR@K`：第一个 gold chunk 的倒数排名，未命中为 0，取宏平均；
- `Recall@K`：Top-K 命中的不同 gold chunks 数 / 该 query gold chunks 总数，取宏平均。

Gold 使用精确 `paper_id + chunk_id`，不使用标题/关键词模糊命中。当前 K=3、5 cases 的 BM25/Hybrid/Hybrid+Rerank MRR 为 0.9，Vector 为 0.8；语料只有 3 papers/6 chunks，不能外推生产效果。

## 22. 六条最终验收

| 验收问题 | 结论 | 核心证据 |
| --- | --- | --- |
| PriEvO Core 是否保留算法语义 | 通过，有 deliberate differences | Core/FLA/Prior/五 Skill/代级 5P selection/final tests |
| Evaluation transaction/idempotency/reservation/lease/requeue | 通过 | MySQL Store、Queue/lease/external Worker tests |
| Run durable state/checkpoint/reconciliation/control/recovery | 通过 | process recovery、pause/reconcile/recovery tests |
| 五 Agent 是否进入正确真实路径 | 通过，按条件触发 | Engine workflows、Agent mainline、Harness traces |
| Context/Redis/MySQL/RAG 是否进入 Prompt | 通过，strict run-local | context/memory/research tests 与 prompt assertions |
| Harness 是否覆盖 FakeLLM/fixture/fault/trace/RAG eval | 通过 | 24/24、116/116；独立 RAG report |

## 23. 最终测试与环境事实

- 本机可执行的非 HTTP 全量：230 tests 全部通过，其中 6 个因未配置真实 MySQL/optional `pflacco` 环境而跳过，0 failure/error；
- Agent Harness：24/24 scenarios、116/116 assertions，报告于 2026-08-20 重写；
- Original Prior：31 条完成静态执行契约审计，21 supported、10 unsupported；unsupported 仍作为 Prompt/Evidence，但不物化为 initial seed；
- RAG eval：5/5 cases 完成，报告于 2026-08-13 重写；
- Compose 静态配置：`docker compose config --quiet` 通过，声明 mysql/redis/app/worker；
- 本轮最新 Docker image、HTTP 与真实 MySQL 再跑被宿主 Docker named-pipe 权限/平台审批额度拒绝。此前共享 Runtime 合并前的容器 API 3/3、MySQL V004/AgentTask 集成和 Worker 9/9 均通过，但不冒充本轮最新镜像结果。

该环境限制不改变纯源码/SQLite/Harness 回归结论，但在获得 Docker daemon 后仍应执行：

```powershell
docker compose up --build -d
docker compose exec -T app python -m unittest discover -s tests -v
docker compose exec -T app python scripts/full_mode_smoke.py
```

## 24. 发布结论

项目已经覆盖 PriEvO 算法真实性、durable evaluation 并发一致性、crash recovery、五 Agent 受约束编排、Context/Memory/RAG 与 Harness。发布说明同时保留 exactly-once、沙箱、vector、RAG corpus、heartbeat 和 reference differences 的真实边界，不把未实现能力写成已完成能力。
