# Architecture

PriEvO-Agent 当前定位是：PriEvO 算法核心 + 长任务后端 Runtime + 受约束 LLM Agent/Node 层。

## 主调用链

```text
FastAPI / CLI
  -> RunApplicationFacade
  -> PriEvOEngine
      -> Landscape / Prior Retrieval
      -> SimilaritySelectionNode
      -> PersistentEvolutionRuntime
          -> EvolutionPlannerAgent -> GenerationPlan
          -> HeuristicGenerationAgent + operator Skill
          -> Candidate durable materialization
          -> EvaluationJob -> Worker -> benchmark subprocess
          -> RepairAgent（仅 candidate failure）
          -> generation 5P -> P selection
      -> FinalSelectionNode（仅 exact tie）
```

## Agent / Node 边界

- `SimilaritySelectionNode`：numeric Top-5 后的语义筛选 Node，不进入 AgentTask 自由调度。
- `EvolutionPlannerAgent`：只输出 `GenerationPlan`，Runtime 保留实际 parent selection 权限。
- `HeuristicGenerationAgent`：接收 plan strategy 与 skill，生成 `CandidateDraft` 或 `KnowledgeGap`，不得覆盖 planner strategy。
- `RepairAgent`：只处理 candidate failure，不处理 infra failure。
- `FinalSelectionNode`：仅 final exact tie 时调用；unique best 不调用 LLM。

旧 `SimilarityAgent`、`FinalSelectionAgent` 名称仍作为兼容 facade 保留，避免旧测试和已持久化任务失效。

## 数据与持久化

- MySQL/SQLite 是 Run、Candidate、EvaluationJob、Artifact metadata、Agent memory、Trace 的事实源。
- Redis 只做 event notification 与 run-local recent memory cache，清空后可从 MySQL/SQLite 回温。
- Candidate 的 `generation`、`plan_id`、`generation_strategy`、`code_digest`、`selected_parent_ids`、`prior_refs` 已提升为一等字段。
- `generation_plans` 保存 Planner 决策；`trace_records` 保存 Agent/Node/工具可审计跨度。

## PriEvO 算法语义

- early operators：`i1,e1,e2,m1`
- late operators：`e1,e2,m1,m2`
- 每个 active operator 生成 `population_size=P` 个 offspring。
- 每代将 retained P 与 4P offspring 合并后统一执行一次 `5P -> P`。
- early selection 偏 fitness + operator diversity；late selection objective first。

## RAG

RAG 路径为 PDF/curated corpus -> chunk -> BM25 ranking + embedding ranking -> weighted RRF -> deterministic rerank -> neighbor expansion -> provenance evidence。外部文档内容必须视作 `UNTRUSTED EXTERNAL CONTENT`，只能作为证据进入上下文，不能覆盖系统/开发者/算法规则。
