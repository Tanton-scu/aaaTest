# 组件边界与领域对象

## 对象所有权与生命周期

| 对象 | 语义 | 所有者 | 生命周期/不变量 |
|---|---|---|---|
| `OptimizationTask` | 不随单次执行改变的目标实例、objective、预算模板和数据引用 | application/task service | 创建后版本化只读；可派生多个 Run |
| `Run` | 一次可恢复的优化执行，含 task/current state/counters | runtime coordinator | `PENDING` 到终态；状态只通过 state machine 改变 |
| `Candidate` | `H_i=<C_i,D_i,F_i,T_i,O_i>` 的持久元数据与 lineage | PriEvO core 产生，runtime 持久化 | 生成后 code artifact 不变；最多绑定一个 logical evaluation/result；重评先 clone，selected 状态可变 |
| `EvaluationJob` | 对 candidate 在 task/seed/budget 下评价的 durable work item | runtime queue | `PENDING→RUNNING→SUCCESS/RETRY_WAIT/DEAD/CANCELLED`；idempotency key 唯一 |
| `EvaluationResult` | objective、trajectory、best configuration、usage | evaluator 产生，runtime 接收 | candidate 只有一个逻辑成功结果；重复 final commit 不重复计费 |
| `Event` | 人类/机器可读的 append-only 事实 | runtime/application service | sequence/run_id/type/message/payload；用于解释，不回放成 current state |
| `ArtifactMetadata` | 大 payload 的 URI、media type、size、digest、kind | artifact store + runtime catalog | 内容寻址/校验；关系库只存 metadata |
| `CheckpointMetadata` | 可恢复点的 run version、generation、population artifact、digest | runtime checkpoint service | 仅在一致边界发布；恢复前校验 digest 和 schema version |
| `InstanceSpecificPrior` | landscape query、matched instances、optimizer/operator empirical evidence | PriEvO prior service | 绑定 task feature version；数值检索结果确定，可选 LLM refinement 留 trace |
| `LiteratureEvidence` | 文献 chunk、source/citation、score、query | optional literature tool | 只辅助 generation/reflection；不得进入 prior similarity 或冒充 empirical evidence |

## 核心职责

### `core/`

- 计算/接收 landscape features、选择结构化 prior、建立 initial population。
- 根据 evolution schedule 产生 generation intent，调用 `LLMPort` 获取结构化 candidate proposal。
- 执行 parent/population selection、预算无关的算法规则、lineage 构造。
- 接收 `EvaluationResult` 后推进纯算法状态。

core 不 claim job、不写数据库、不发 SSE、不知道 SQLAlchemy model 或文件路径。

### `runtime/`

- Run 状态机、预算预留/结算、job 幂等创建、claim/lease/retry/dead letter。
- 把 core 的“需要评价这些候选”翻译为 durable jobs；收齐结果后调用 core 选择下一代。
- checkpoint 边界、恢复、取消、事件记录和 artifact metadata。
- 确定性控制；LLM 不决定状态迁移、预算、claim 或重试。

### `application/`

- 用例边界：create run、resume/cancel、query timeline/artifact/metric，以及 governed tool/prior use case。
- CLI/API 共享同一服务；不放 PriEvO 算法规则。

## Ports 取舍

| Port | 是否保留 | 替代实现/测试价值 |
|---|---|---|
| `LLMPort` | 是 | deterministic fake、OpenAI-compatible；隔离费用/网络/结构化响应 |
| `CandidateEvaluator` | 是 | deterministic fixture、local process、未来 remote benchmark |
| `PriorRepository` | 是 | in-memory fixture、CSV/JSON knowledge base；保持 prior 与 RAG 分离 |
| `ArtifactStore` | 是 | in-memory、filesystem、未来 object storage；大 payload 不进 DB |
| `RunRepository` | 是 | in-memory state-machine test、SQL adapter |
| `CandidateRepository` | 是 | in-memory/core integration、SQL adapter |
| `EvaluationRepository` | 是 | in-memory queue test、SQL durable jobs/results |
| `CheckpointStore` | 是 | in-memory/failure test、filesystem；虽可复用 ArtifactStore 底层，但有原子发布/恢复语义 |
| `LiteratureSearchPort` | 是（optional capability） | no-op、local BM25、未来 remote/vector；core 通过 tool facade 可选调用 |

V1 实际使用一个 `RuntimeStore` protocol 和 SQLite adapter，尚未拆成 repository/UoW。Job enqueue/claim/finalize 与预算在显式 SQLite 事务内；一般 Run state 与 Event 仍是相邻提交，这一限制在架构复审中记录，不假装已经具备 Unit of Work。

## 禁止跨界

- API 不直接 session.query 或调用 core。
- infrastructure 不把 ORM entity 泄漏成 domain object。
- evaluator 不直接修改 Run/Candidate；只返回 result envelope。
- literature evidence 不参与 `InstanceSpecificPrior` 的 landscape distance/rank。
- checkpoint 不替代数据库 current state；数据库状态与 checkpoint digest/version 互相校验。
