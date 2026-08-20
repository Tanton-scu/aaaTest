# CHANGELOG

所有面向人的变更说明使用中文；版本号仅在相应 Exit Gate 与验证完成后提升。

## Unreleased

### 1.0.0 公开发布复审 — 2026-08-20

- 按用户指定语义固定整代四批共享代初 parent pool，评价 `4P` 后统一执行一次 `5P -> P`，并把 cadence 纳入 Checkpoint 兼容指纹。
- 补齐 Run/Evaluation/AgentTask 的 lease、owner fencing、terminal cleanup 与 evaluation identity 校验；旧 Result 只有完整 dataset/evaluator/seed/budget identity 一致时才能恢复复用。
- 新增 Original Prior 静态兼容矩阵：31 条中 21 条可进入受控 evaluator，10 条仅保留为 immutable Prompt/Evidence；报告位于 `reports/prior_compatibility.json`。
- 非 HTTP 回归 230 tests 全部通过（6 environment skip）；Agent Harness 24/24 场景、116/116 断言。
- 公开仓库忽略 `.env`、数据库、运行 Artifact、缓存、snapshot、论文 PDF 与压缩包；敏感信息扫描未发现高置信度凭据。

### 1.0.0 — 2026-08-13

- 完成 PriEvO、Backend、Agent、Context/Memory/RAG、恢复与 Harness 六类能力验收，版本提升为 1.0.0；重写 README、架构与发布审计文档。
- Final Optimization 已接入 `RUN_COMPLETED` 前：对选中 heuristic 创建两个独立 seed clone、每 seed 两倍预算，复用 durable Queue/ledger 并生成可恢复 configuration/report；API 和 Dashboard 总预算同步包含该阶段。
- 完成 cooperative Pause/Resume/Cancel、Run owner lease/cursor、operator safe-point Checkpoint、startup RecoveryManager 与 post-checkpoint reconciliation；增加真实进程退出、评价中暂停、取消竞态与 orphan work 测试。
- Generation Memory 已进入产品 Prompt：Redis 按 `run + agent scope` 缓存，同 Run MySQL/SQLite history 回源并回温，跨 Run/scope 严格拒绝。
- Literature 主链升级为统一 PDF section/chunk schema、BM25 + deterministic vector + fusion + rerank + same-section neighbor；重新生成 5-case RAG 指标报告并明确 hashing vector/小语料限制。
- 删除旧 `EvolutionAgent/ResearchAgent/ReviewAgent`、同步 Coordinator 与旧 Reflection/Literature Harness；产品只保留五类 PriEvO Agent 与 durable workflows。
- 将 `CandidateInspectionTool` 真实接入 RepairAgent，并将 `LiteratureSearchTool` 固化在 PriorResearch 主链；Gateway 持久记录 caller/reason/input/start/end/duration/status/failure 和真实 ToolCall ref。
- 修复 faithful mode 在可替代 prior seed 失败时错误终止整个 Run：Repair 草案仍审计，但标记 INVALID 并排除 population；新增专门回归测试。
- 2026-08-13 重新生成 Agent Harness 报告：24/24 scenarios、112/112 assertions；当前本机非 HTTP 全量 182 项为 178 pass、4 environment skip、0 failure/error。
- 新增 `docs/README.md` 权威文档入口，并为旧 0.6/V5 报告增加历史标识；最终检查 114 份 Markdown 无断链、无 U+FFFD/已知乱码。
- 将旧 `build/`/`*.egg-info` 缓存加入 Docker ignore；平台拒绝物理清理后保留为可再生、非事实源的历史构建缓存。

- Full Mode 新增独立 Evaluation Worker 服务：App 对普通进化评测与 Final Optimization 都只提交并有界轮询 MySQL durable Job，Demo 保持 inline；补齐 stale recovery、owner fencing、优雅停止、有界空闲退避和只读健康检查。
- App 与独立 Worker 现在共享 `EVALUATION_TIMEOUT_SECONDS`，保证 logical evaluation identity 与实际 benchmark timeout 一致；明确当前只有 benchmark 前后续租，不虚构后台 heartbeat。
- 新增 external 双连接端到端测试，严格断言 App evaluator 零调用，并覆盖独立 Worker 推进、Final Optimization、轮询期间 Runtime lease/control 检查与 checkpoint 上的 pause。

### 2026-08-12

- 接受根目录最新 `prompt.md` 为最高优先级规格，废止 0.6.0 对旧 Agent prompt 的“全部完成”结论。
- 完成当前 PriEvO-Agent、`references/prievo/` 与 `references/mindbridge/` 的新一轮只读源码审计。
- 完成 Architecture Audit，明确五类 PriEvO Agent、durable Coordinator/Blackboard、run-local Memory、真实 Candidate evaluator、Hybrid Literature RAG 和 Engineering Harness 目标。
- 记录当前测试环境限制：本机 Python 3.9 缺项目依赖，Docker named pipe 无访问权限；未沿用旧报告数字冒充本轮结果。
- 完成全部七份审计文档，Gate 0 通过；开始 Gate 1 durable AgentTask/Registry/Blackboard/Coordinator 实现。
- 新增 AgentCapability、AgentTaskStatus、AgentTask 领域模型及 SQLite durable task 表/幂等领取/完成/重试语义；新增 2 项 SQLite 测试并通过。
- 复验现有 Docker Compose 基线：旧镜像 app、mysql、redis 均 healthy；新代码仍待重建容器验证。
- 新增 MySQL V003 `agent_tasks`、一对一 capability Registry、durable Blackboard projection 与五类 missing-work Coordinator/sweep；12 项相邻专项测试通过。
- MySQL 双连接 claim 集成测试发现只读连接可能保留 REPEATABLE READ snapshot；修复 `_one/_all` 读事务结束语义后定向测试通过。
- 新增 10 个 PriEvO/Agent Skills：semantic similarity、五 generation strategy、prior explanation、candidate repair、final audit、history summary；SkillRegistry 当前可加载共 13 个版本化 Skill。
- 完成 durable AgentTask Dispatcher、五类 missing-work reconcile/sweep、只读 Blackboard 与一对一 capability Registry；Gate 1 Exit Gate 通过。
- SimilarityAgent 已接入产品 `prepare()`：numeric Top-5 经唯一 durable AgentTask 产生严格 1～3 allowlist decision，original prior 仍由结构化 repository 提取，不由 LLM 编造。
- Memory key 改为 Run + agent scope 隔离，并支持 Redis miss/不可用时只从 MySQL 同 Run history 回温；新增五类白名单 ContextPolicy。
- 新增 FailureClassifier，明确 infrastructure retry 与 syntax/runtime/interface/algorithm timeout/OOM/logic repair 的边界；算法超时不再原样重试。
- 将 evolution selection cadence 改为 reference executable 的每 operator `2P -> P`，一代四次选择；新增 `OPERATOR_SELECTION_COMPLETED` 可审计事件。
- 新增 HeuristicGenerationAgent：五策略分别执行 synthesize/imitate/recombine/revise/fine_tune Skill，严格校验 parent 数量、C/D/F/T/O、operator 继承/差异及 m2 结构不变约束。
- Generation 已走 `GENERATION_REQUEST -> AgentTask -> GENERATION_PROMPT -> CANDIDATE_DRAFT -> Candidate` durable 主链；第 2 次 LLM 故障恢复测试证明先前 Draft/Candidate 不丢失且不重调。
- 新增纯领域 PriorResearchAgent 与 RepairAgent：前者保留 literature provenance 和 immutable original prior，后者按 FailureClassifier 执行 Diagnose→Repair 并创建新版本链；产品 runtime 接线仍在后续 Gate 完成。
- 产品 DatasetEvaluator 现通过受监督 `python -I` 子进程真实执行 Candidate，注入 reference-compatible `evaluate`，验证 unique budget、nearest mapping、duplicate、trajectory 与 return；旧 hash-shuffle 已完全退出产品路径。
- EvaluationJob 幂等 material 升级为包含 Dataset digest、Evaluator/参数版本的 `evaluation-v2`；真实 retry delay 可取消等待且不 busy-spin。
- SQLite/MySQL 增加 lease owner fencing 与条件续租，陈旧 Worker 无法结算或释放预算；所有 terminal DEAD Candidate 统一转为 INVALID。
- Repair workflow 已接入 Runtime：candidate failure 产生 durable task、Diagnosis/Repair prompts 和新 Candidate version，原 Candidate 不覆盖；支持 `research_faithful_mode` 禁止 repaired draft 进入 population。
- FinalSelectionAgent 已接入产品终点：资格过滤与 unique best 走确定性逻辑，exact tie 才调用一次模型并走 strict allowlist；旧三 Agent 已从 Engine 组合根移除。
- 新增可编排 `ScriptedFakeLLM`：按 Agent/Skill FIFO 返回、完整 Prompt/调用记录、KnowledgeGap/malformed/transient 故障注入及可恢复状态序列化。
- AgentTask Dispatcher 增加 durable `max_attempts` 有界 redrive；transient-once 可在同一次工作流调用自动恢复，持续 malformed 恰好耗尽上限进入 FAILED，并保留每次重驱 Trace。
- Agent Engineering Harness 扩为 24 个真实协作/故障场景、112 个 Routing/Context/Artifact/Tool/State/Trace 断言；当时的 Docker 容器实测 24/24 场景、112/112 断言通过，报告落于 `reports/agent_harness.json`。
