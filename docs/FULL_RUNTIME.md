# 完整运行链路

本文只描述当前 GitHub-ready 版本保留的完整后端模式。

## 1. App、Worker 和 Docker 的关系

`app` 不是 Docker 专有概念，它就是 FastAPI 后端进程。可以直接运行：

```powershell
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

`worker` 也不是 Docker 专有概念，它是另一个 Python 进程，专门消费 MySQL 里的 EvaluationJob：

```powershell
python -m prievo_agent.cli.evaluation_worker --root .prievo-runtime
```

Docker 在本项目里只用于快速启动 MySQL 和 Redis：

```powershell
docker compose up -d
```

如果本机已经安装并启动 MySQL/Redis，也可以不用 Docker，只要 `.env` 里的 `DATABASE_URL` 和 `REDIS_URL` 指向真实服务即可。

## 2. 用户点击创建任务后的调用链

前端 Dashboard 位于 `src/prievo_agent/api/dashboard.html`。点击创建任务时，它向后端发送：

```http
POST /api/runs
```

FastAPI 路由位于 `src/prievo_agent/api/app.py`，它把请求交给 `RunApplicationFacade.create_run()`。

`create_run()` 会：

1. 从 `DatasetRegistry` 校验 dataset；
2. 根据 `generations × 4 × population_size × candidate_budget` 计算演化预算；
3. 创建 `OptimizationTask` 和 `Run`；
4. 调用 MySQL Store 写入 `tasks`、`runs`、`events`；
5. 追加 `RUN_CREATED` 事件；
6. 把 `run_id` 放入后台 scheduler；
7. HTTP 立即返回 `202 Accepted`。

HTTP 立即返回的原因是：一次 PriEvO run 可能持续较久，API 请求线程不能一直阻塞。真正的演化由后台线程推进，前端通过 SSE 和轮询看进度。

## 3. 后台线程、Run lease 与恢复

`RunApplicationFacade` 内部有一个 `ThreadPoolExecutor(max_workers=1)`。它不是为每个 candidate 开进程，而是在当前 app 进程里开一个后台执行线程，用于推进 Run。

如果 app 重启后发现上一次 run 没跑完，不会修改旧进程 id；旧进程已经不存在。恢复逻辑是：

1. MySQL 里保存了 Run 状态、candidate、checkpoint、EvaluationJob、AgentTask；
2. 新 app 启动时执行 startup recovery；
3. 过期的 run owner lease 会被释放；
4. 新 app 重新 claim 这个 run 的 lease；
5. Runtime 从 checkpoint / durable facts 继续补齐未完成工作。

保存的不是操作系统进程 id，而是业务层的 `runtime_owner_id` 和 `runtime_lease_expires_at`。这些字段用于判断当前是否还有活着的 runtime owner。

## 4. 为什么会有多个 app 实例

单机开发时通常只有一个 app 实例，也就是一条：

```powershell
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

但完整后端项目需要考虑：

- 误开了两个 API 进程；
- 部署时开了多个副本；
- 服务重启过程中旧实例未完全退出，新实例已经启动；
- 多台机器同时连接同一个 MySQL。

Run lease 的意义就是防止这些情况下两个 app 同时推进同一个 Run。

## 5. PriEvO 主流程

`PriEvOEngine` 先完成 dataset prior 准备：

1. FLA / landscape sampling；
2. numeric top-k similar instances；
3. SimilaritySelectionNode 语义精选；
4. 生成 `INSTANCE_SPECIFIC_PRIOR` artifact。

之后进入 `PersistentEvolutionRuntime`：

1. 初始化 original prior seeds；
2. 如果初始 population 不够，用 `i1` 补齐；
3. early generation 使用 `i1/e1/e2/m1`；
4. late generation 使用 `e1/e2/m1/m2`；
5. 每个 operator 生成 `population_size` 个候选；
6. 每个候选保存 `CANDIDATE_CODE` artifact；
7. 创建 durable EvaluationJob；
8. 等待独立 worker 评估；
9. 根据 fitness 和 diversity 选择保留 population；
10. 保存 checkpoint；
11. 完成 final selection / final optimization；
12. 保存 `FINAL_HEURISTIC` artifact；
13. 写入 `RUN_COMPLETED`。

## 6. Agent 可用工具的职责

工具不是独立 Agent，而是由后端 workflow 在明确条件下调用的受治理能力。当前主要有三类：

```text
literature_search
  由 GenerationAgent 返回 KnowledgeGap 后触发，用于检索文献证据。

candidate_inspection
  按 candidate_id 查看同 Run 内个体详情、代码摘要、父代、评估结果和失败 job。

candidate_code_audit
  对候选代码做 AST / run_tuners 入口 / 禁止 import / 危险 builtin 静态审查。
```

每次工具调用都会经过 `ToolGovernanceGateway`，检查 caller allowlist、调用原因、scope 预算，并写入 tool call audit 与 durable event。

文献检索不是一个独立“先验研究主流程”。它是算法生成阶段可用的工具。

更准确的链路是：

```text
HeuristicGenerationAgent 生成候选
→ 如果发现当前 prior / evidence 不足，返回 KnowledgeGap
→ Generation workflow 调用 literature search tool
→ ToolGovernanceGateway 检查 caller、scope、预算和只读权限
→ LocalHybridLiteratureRAG 检索 corpus
→ 检索 Evidence 回到 generation context
→ HeuristicGenerationAgent 再生成候选算法
```

这样设计的重点是：RAG 是工具，不是额外抢主导权的 Agent。

## 7. RAG 检索与 BGE rerank

默认 RAG 是可复现的轻量版本：

```text
BM25 recall
+ deterministic lexical hashing vector recall
+ weighted RRF fusion
+ deterministic rerank bonus
```

启用 BGE 后是生产级两阶段检索：

```text
BM25 recall
+ BGE-M3 embedding recall
+ weighted RRF fusion 得到粗召回候选
+ BGE-Reranker cross-encoder 只重排 Top-N
```

配置位于 `.env`：

```env
RAG_EMBEDDING_BACKEND=bge
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_QUERY_INSTRUCTION=为这个查询生成表示以用于检索相关文献：
RAG_RERANKER_BACKEND=bge
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANK_CANDIDATE_POOL=40
```

实现落点：

- `src/prievo_agent/knowledge/literature/retrieval/hybrid.py`：负责 chunk、BM25、vector recall、RRF、候选池、provenance；
- `src/prievo_agent/knowledge/literature/retrieval/bge.py`：负责 FlagEmbedding adapter；
- query embedding 调用 `encode_query()`，会加 `RAG_QUERY_INSTRUCTION`；
- document chunk 调用 `encode_document()`，不加 query 前缀；
- reranker 调用 `FlagReranker.compute_score([[query, chunk], ...])`；
- reranker 只接收粗召回 Top-N，不对全库打分。

## 8. Blackboard 与 Coordinator

Blackboard 是只读投影，不是数据库。它从 MySQL Store 读取：

- AgentTask；
- Event；
- Artifact metadata。

它不直接保存 candidate code、population 或大段文献内容。Agent 如果需要大对象，通过 artifact ref 再读取。

Coordinator 的职责是 missing-work reconciliation：

```text
读取 durable facts
→ 判断缺少哪些 AgentTask
→ 用 idempotency key 防重
→ 创建缺失任务
```

Dispatcher 的职责是真正执行任务：

```text
claim AgentTask
→ 重建 Blackboard
→ 调用对应 handler
→ complete/fail AgentTask
```

## 9. Evaluation Worker

候选算法不在 API 进程里直接跑。Runtime 只创建 EvaluationJob 并预留预算：

```text
runs.reserved_evaluations += candidate_budget
evaluation_jobs.status = PENDING
```

Worker 循环执行：

```text
claim job
→ 获得 worker_id 和 lease
→ 子进程中受控执行候选算法
→ 成功则写 EvaluationResult 并结算预算
→ 失败则进入 RETRY_WAIT 或 DEAD
```

如果 worker 崩溃，job lease 过期后会被 stale recovery 回收。

## 10. 异常与重试

- LLM 输出结构不合法：Agent parser 抛错，AgentTask fail，未超过上限则重试。
- 候选代码运行异常：Worker 记录 failure evidence，EvaluationJob 进入 retry 或 dead。
- Worker 崩溃：MySQL 中 job lease 过期后被回收。
- App 崩溃：新 app 启动后通过 startup recovery 重新 claim run。
- 用户暂停：只写 `pause_requested`，Runtime 在 safe point 停住。
- 用户取消：写 `cancel_requested`，并取消未完成 AgentTask / EvaluationJob。

## 11. MCP

当前 GitHub-ready 版本没有使用 MCP。原因是项目工具只服务于本系统内部 Agent Runtime，不需要暴露给其他 Agent 项目复用。工具治理由 `ToolGovernanceGateway` 直接完成，链路更短，也更容易在面试中解释。
