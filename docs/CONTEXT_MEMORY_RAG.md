# Agent Context、Memory 与 Literature RAG

本文描述当前产品实际注入 Agent Prompt 的信息路径。三者职责不同：Context Policy 决定字段白名单和字符预算；Agent Memory 提供同 Run、按 generation/research/repair 隔离的近期与摘要历史；RAG 只在 KnowledgeGap 的 PriorResearch 路径检索本地文献。它们都不替代 SQLite/MySQL、Artifact 或评价结果这些事实源。

## 1. Context Policy：每个 Agent 独立白名单

`src/prievo_agent/agents/context_policies.py` 定义五个 Policy。未知 payload 字段不会进入 Prompt，只会在 metadata 的 `excluded_keys` 中留下可审计记录。

| Policy | 允许字段 | 列表上限 | 明确没有的上下文 |
| --- | --- | ---: | --- |
| `similarity` | target landscape、八个 FLA metrics、metric semantics、numeric Top-5/rank、Skill、schema | Top-5 各 5 | Memory、RAG、Population |
| `generation` | task、immutable prior、strategy Skill、parents、lineage、同 Run memory、evidence、schema | lineage 最近 3；memory 最近 6；evidence 最近 5 | whole Population、跨 Run long-term memory |
| `research` | current gap、prior slice、landscape、strategy、parent summary、Skill、research history、evidence、schema | history/evidence 最近 5 | Generation/Repair 无关历史、可写 Prior |
| `repair` | failed candidate、classified failure、Skill、repair/failure history、schema | 两类 history 最近 3 | 其他 Candidate、Population、RAG |
| `final_selection` | tied candidates、审查 Skill、schema | 只含并列集合 | 非并列 Candidate、Memory、RAG |

Policy 把信息分入 `pinned`、`recent` 和 `evidence` channel，再统一交给 `agents/context.py::AgentContextBuilder`。框架始终传入空 `long_term_memory`，所以旧 Store 中按 Dataset 查询 memory 的接口不会自动把其他 Run 的内容注入 Prompt。

五个产品 Agent 均通过 `ContextPolicyFramework`。SimilarityAgent 也先执行 `build("similarity", payload)`，再把该白名单文本封装成最终 semantic-selection Prompt；其 decision 保存 policy metadata，产品测试断言 Memory/RAG/Population sentinel 不会进入文本。

## 2. Context 构建与压缩

`AgentContextBuilder` 默认 `max_chars=6000`，最小允许值为 800。渲染顺序是：

1. `Pinned Agent Context`；
2. `Literature evidence`；
3. `Recent Agent Working Memory`；
4. `Relevant Long-Term Agent Memory`。

若完整文本未超限，builder 附加 `compression: applied=false`。若超限：

- pinned section 全量保留，即使 pinned 自身超过预算也不静默截断；
- optional item 按上述顺序逐条尝试，超限项被省略；
- metadata 记录 `chars_before`、`chars_after`、`max_chars`、`compression_applied` 和 `omitted_items`；
- Policy 额外记录 included/excluded keys、各 section item_count，以及被列表上限裁掉的条数。

超过历史阈值时，`application/history_compactor.py::HistoryCompactor` 才加载 `history_summary` Skill。它保留 Recent Window，并用“上一版摘要 + 新到达的有界批次”滚动更新早期历史；水位线未达到时不调用 LLM，而是把少量未覆盖事实原样带入 Context。Prompt 有 12k 字符硬上限，摘要记录 covered memory IDs、source refs、Skill/prompt/summary Artifact refs。模型/协议失败只写 `HISTORY_SUMMARY_FAILED` 并降级为已有摘要+事实窗口，不伪造 summary。`AgentContextBuilder` 的字符裁剪仍是摘要之后的独立最后防线。

## 3. Agent Memory 的事实模型

### 3.1 Durable record 与 cache

`domain/models.py::AgentMemory` 在 SQLite/MySQL 中保存：ID、Run、Dataset、memory type、subject、content、evidence Artifact ref 和时间。`infrastructure/agent_memory.py` 提供三类近期 cache adapter：

- `NullAgentWorkingMemory`：Demo 默认，不保留进程内/Redis cache，但仍能通过 Store fallback 读取 durable memory；
- `InMemoryAgentWorkingMemory`：测试适配器，语义与有界 Redis list 相同；
- `RedisAgentWorkingMemory`：Full Mode 的有界、带 TTL 同 Run cache。

Redis key 为 `prievo:run:{run_id}:agent:{scope}:recent`，scope 只能是 `generation`、`research`、`repair`。默认 generation 还镜像旧 key `prievo:agent:{run_id}:working_memory`，仅用于数据迁移；research/repair 从不写旧 key。

Redis 是可丢失的加速层：

```text
load_with_fallback(run_id, scope)
  -> 读取 Redis/InMemory 同 run + 同 scope recent
  -> cache 为空/不可用：查询 Store.agent_memories_for_run
  -> 尽力回温相同 cache key
  -> 返回 durable 记录
```

Redis 故障只写 warning，Run 继续依赖 SQLite/MySQL；Redis 中的值不会覆盖 durable memory、Artifact、Candidate 或 EvaluationResult。

### 3.2 当前产品真正使用的 Memory

三个 scope 都已进入 Engine 注入的同一 Redis/MySQL 分层路径：

- Generation：Draft 成功后写 `GENERATION_SUMMARY`；真实评价结算后再写 `GENERATION_EVALUATION`，包含 candidate/parents、operators、fitness、used budget、trajectory 与 Draft/Evaluation refs。后续 request 读取最多 6 条，并通过 Store 重建 parent 最近 3 步祖先链；Memory 不参与 Core parent/selection。
- Research：每次 KnowledgeGap 研究写当前 gap、受治理 query、命中文献/chunk refs、evidence status/summary、Original Prior 和两个 Research Skill refs；下一次同 Run Research Prompt 可读取该 scope 历史，其他 Run 不可见。
- Repair：Diagnose/Repair 后写失败类型、原/新 Candidate、diagnosis/decision/draft refs 与 `REPAIRED_DRAFT_CREATED` 或 `NOT_REPAIRABLE` outcome；下一次同一 candidate family 可见最近 repair history。

三类 Workflow 都调用 `HistoryCompactor`，仅超过阈值才生成 `*_HISTORY_SUMMARY`。Redis flush/unavailable 时只从 MySQL/SQLite 的相同 run+scope 回温；Similarity/Final 永远不读 Memory。

Store 的 `agent_memories_for_dataset` 等旧 API 仍保留兼容性和测试，但当前 Context framework 禁止自动跨 Run 注入，因此它不是当前“长期记忆召回”产品能力。

## 4. Literature corpus 与离线 ingestion

产品 RAG 入口是 `infrastructure/literature_hybrid.py::LocalHybridLiteratureRAG`。Engine 当前合并读取：

- `data/literature/corpus.json`：仓库维护的 curated corpus；
- `data/literature/pdf_corpus.json`：离线脚本生成的 PDF corpus。

`scripts/ingest_papers.py` 将 PDF 解析成统一 paper/section/page/chunk schema，并写入 `pdf_corpus.json`。`pypdf` 是可选依赖；某个坏 PDF 会被明确跳过，不会伪造正文。运行期不联网抓论文，也不会让 Agent 任意浏览文件系统。

每个 chunk 的可追溯字段包括 paper ID/title/authors/year/identifier/source path、primary 标记、section、page range、chunk ID/index 和相邻 chunk refs。

## 5. Hybrid retrieval 的真实算法

一次 `retrieve` 的执行顺序是：

```text
LiteratureQuery
  -> canonical chunks 的 BM25 raw score
  -> query/chunk hashing-vector cosine raw score
  -> 每个 query 内分别 min-max normalization
  -> 0.58 * BM25 + 0.42 * Vector
  -> 可选 deterministic rerank bonus
  -> 稳定排序；每个 paper-section 最多选一个 anchor
  -> 同 paper + 同 section 的 anchor ±1 neighbor expansion
```

rerank bonus 来自 algorithm name exact overlap、title/section token overlap 和 primary-source 小额 bonus，不调用另一个 LLM。排序相同时以 `chunk_id` 稳定打破。

默认 VectorPort 是 256 维 `DeterministicHashingVectorizer`：对 tokens 和相邻 bigrams 做 SHA-256 bucket hashing、词频对数缩放和 L2 normalization。它是可复验的 lexical vector，不理解语义，元数据明确标记：

```json
{
  "backend": "deterministic-token-hashing-v1",
  "production_semantic_embedding": false
}
```

代码提供可替换 `VectorPort` 协议，但当前仓库没有真实 embedding provider adapter。因此“Hybrid RAG”属实，“已接入生产语义向量模型/向量数据库”不属实。

## 6. RAG 只在哪条产品链出现

RAG 只在 Generation 返回 `KNOWLEDGE_GAP` 后由 `PriorResearchAgent` 使用：

```text
KNOWLEDGE_GAP Artifact
  -> ResearchContextPolicy 构造 query Prompt
  -> 模型生成有界 query（空 query 回退到 gap summary）
  -> ToolGovernanceGateway 授权 PriorResearchAgent 的 read-only literature_search
  -> LocalHybridLiteratureRAG.retrieve(hybrid, rerank=true, top_k<=5)
  -> ToolCallRecord + evidence provenance
  -> LITERATURE_EVIDENCE Artifact
  -> 有证据才调用 explanation model
  -> PRIOR_EXPLANATION Artifact
  -> GENERATION_RESUME_REQUEST
```

Tool policy 对 caller、reason、输入 metadata、每 Run 次数、side effect 和 cost class 做约束；产品 `LiteratureSearchTool` 只允许 `PriorResearchAgent` 调用 `literature_search`，上限 16；`CandidateInspectionTool` 只允许 `RepairAgent` 检查本次失败 Candidate，上限 12。两者均为 `READ_ONLY/LOCAL_LOW`。caller、reason、started/finished timestamp、duration、completed/failed/denied 状态会写入 durable ToolCall/Event，工具返回的真实 `tool_call_ref` 继续进入 Evidence 或 Repair Context。

检索适配器把每条 result 的 BM25 raw/normalized、vector raw/normalized、fusion、rerank bonus、final score、retrieval mode、vector backend/semantic flag、paper/section/chunk/page、neighbors 和 source path 放进 evidence provenance。Generation resume 只能把这个证据包作为有界 annotation 加到 immutable Original Prior 旁边；不会覆盖 Prior digest。

检索返回空列表时，evidence status 为 `EMPTY`，包含 query/tool refs 和“不允许编造”的显式消息；PriorResearchAgent 不调用 explanation model，并保留 Original Prior。一次 bounded research 后 Generation 若仍返回 KnowledgeGap，workflow 失败，避免无界 research loop。

Similarity、Repair、FinalSelection 和 Core selection 不调用 RAG；Repair 只调用本地只读的 CandidateInspectionTool。

## 7. 固定 RAG 评测结果与限制

`reports/rag_eval.json` 生成于 `2026-08-13T07:48:15.562794+00:00`：curated corpus 3 篇论文、6 个 chunk，5 个 gold query case，`K=3`。该报告评测 `data/literature/corpus.json`，不包含运行时合并的 `pdf_corpus.json`。

| 配置 | Hit@3 | MRR@3 | Recall@3 |
| --- | ---: | ---: | ---: |
| BM25 | 1.0 | 0.9 | 1.0 |
| Vector | 1.0 | 0.8 | 1.0 |
| Hybrid | 1.0 | 0.9 | 1.0 |
| Hybrid + deterministic rerank | 1.0 | 0.9 | 1.0 |

这些数字只能证明固定小语料和人工 gold case 上的可复验 ranking 行为。限制包括：语料极小、gold 未做多人一致性标注、hashing vector 仍是 lexical、评测只覆盖 retrieval ranking 而不覆盖下游生成答案的事实正确性。报告没有显示 Hybrid 在该小集合上优于 BM25，也不能外推为生产效果。

## 8. 与 reference 机制和工程增强的关系

reference PriEvO 的 instance-specific prior 来自 FLA numeric/semantic selection 和 repository extraction，不依赖 Literature RAG。当前系统仍先完成同样的 Prior 构造；Memory/RAG 只包装后续 Agent 工程分支：

- Generation memory 减少同一 Run 内重复、失去上下文的生成，不改变 operator/selection；
- PriorResearch 在 Agent 明确报告 KnowledgeGap 时提供有来源 annotation，不改 Original Prior；
- Context Policy 限制每类 Agent 的可见范围，避免把 Blackboard、全 Population 或跨 Run 数据倾倒进 Prompt。

`RESEARCH_FAITHFUL_MODE=true` 会隔离 PriorResearch/RAG 和 Repair 的 population 影响；它不会关闭 Similarity、Generation、Final Selection 或 Context 白名单，也不代表 reference executable 的逐字复刻。

## 9. 可验证证据

- Context 构造/滚动摘要：`tests/test_agent_context.py`、`tests/test_agent_context_policies.py`、`tests/test_history_compactor.py`；
- 三 scope durable memory、cache fallback 与跨 Run/scope 隔离：`tests/test_generation_workflow_memory.py`、`tests/test_prior_research_workflow.py`、`tests/test_repair_workflow.py`、`tests/test_run_local_agent_memory.py`、`tests/test_mysql_agent_memory.py`；
- RAG schema、BM25、Hybrid、provenance 和 neighbor expansion：`tests/test_literature_rag.py`、`tests/test_hybrid_literature_rag.py`；
- 固定评测重跑：`tests/test_rag_evaluation.py`、`src/prievo_agent/rag_eval/evaluator.py`、`reports/rag_eval.json`；
- KnowledgeGap、空 evidence、Prior immutable 和 resume：`tests/test_prior_research_agent.py`、`tests/test_prior_research_workflow.py`、`tests/test_generation_workflow.py`。
