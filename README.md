# PriEvO-Agent

PriEvO-Agent 是一个面向算法自动生成的后端 + Agent 项目。它把 PriEvO 的启发式搜索流程工程化为可恢复、可审计、可观测的运行时：FastAPI 接收任务，MySQL 保存 durable facts，Redis 做事件通知和短期记忆缓存，独立 Evaluation Worker 消费候选算法评估任务。

## 核心能力

- PriEvO 主流程：前期 `i1/e1/e2/m1`，后期 `e1/e2/m1/m2`；每个算子生成 `population_size` 个候选，再按适应度和算子多样性选择保留种群。
- 多 Agent 编排：主链路是 3 个 Agent + 2 个 LLM Node；Coordinator 根据 MySQL 中的事实补偿缺失任务，Dispatcher 领取任务并重建 Blackboard。
- RAG 工具链：RAG 不是独立 Agent，而是 GenerationAgent 在 `KnowledgeGap` 时调用的只读 literature search tool。
- 企业级检索扩展：默认可快速启动；需要生产级 RAG 时，可启用 BGE-M3 embedding + BGE cross-encoder reranker。
- 独立评估 Worker：候选算法不在 API 进程里直接跑，而是写入 EvaluationJob，由 worker 领取、执行、结算。
- 可恢复运行时：Run、AgentTask、EvaluationJob 均有 lease/fencing/retry，服务重启后可从 MySQL 恢复。
- 可观测性：SSE 推送 durable events；`/api/runs/{run_id}/trace` 可从事件、任务、工具调用、artifact 引用重建执行链路。

## 项目结构

Agent / Node / Tool 的边界：

```text
3 个 Agent
├── HeuristicGenerationAgent：生成候选算法，必要时提出 KnowledgeGap
├── RepairAgent：修复候选代码和失败候选
└── FinalSelectionAgent：最终候选选择与审计

2 个 LLM Node
├── SimilaritySelectionNode：语义精选相似 prior instance
└── EvolutionPlannerNode：生成当前 operator batch 的计划与父代选择说明

Tool
├── LiteratureSearchTool / RAG：只读文献检索，不拥有自主目标
├── CandidateInspectionTool：按 candidate_id 查看同 Run 个体详情、父代、评估结果和失败 job
└── CandidateCodeAuditTool：对候选代码做 AST / 入口函数 / 禁止 import 与危险 builtin 审查
```

```text
assets/                  # 所有版本化运行资产
├── datasets/            # 内置优化数据集
├── literature/          # Literature RAG corpus 与评测集
├── prior/               # FLA、optimizer 与 operator 先验
└── skills/              # Agent Skill 指令
scripts/                 # 可直接执行的维护与 harness 入口
src/prievo_agent/        # 产品 Python package
├── agents/              # Agent/Node、上下文策略、工具治理与 registry
├── api/                 # FastAPI 路由与 Dashboard
├── application/         # 用例编排、工作流、记忆与可观测查询
├── cli/                 # API Server 与 Evaluation Worker 命令入口
├── devtools/            # 集中的测试支持与 engineering harness
│   ├── harness/         # Agent、checkpoint、queue、prior、lifecycle harness
│   └── rag_eval/        # Literature RAG 离线评测工具
├── domain/              # 跨功能域模型、持久化端口、事件与错误
├── evaluation/          # 数据集、候选执行、任务队列与最终优化
├── evolution/           # PriEvO 演化引擎、种群、调度、选择与序列化
├── infrastructure/      # 运行时装配、MySQL、Redis、LLM 等 adapter
│   └── local/           # FakeLLM 与 SQLite demo adapter
├── knowledge/           # literature RAG、prior knowledge 与 Agent skills
├── runtime/             # 持久化演化循环、租约生命周期与状态机
└── security/            # 候选代码校验与受控执行
tests/                   # pytest 测试；不承载产品实现
```

`scripts/` 不是另一套产品代码。它只放供开发者直接执行的仓库维护入口：

- `agent_harness.py`：运行跨 Agent、checkpoint、queue、prior 的工程验收场景；
- `audit_prior_compatibility.py`：重新生成或核对 prior candidate 兼容性报告；
- `ingest_papers.py`：把 PDF 文献整理进 `assets/literature/` corpus。

产品启动入口统一在 `prievo_agent.cli`；测试辅助实现统一在
`prievo_agent.devtools`，因此 `scripts/` 中不再承载可复用业务逻辑。

## 快速运行

1. 安装依赖：

```powershell
python -m pip install -e ".[research,papers,test]"
```

如果要启用 BGE embedding / reranker，再安装：

```powershell
python -m pip install -e ".[rag]"
```

或直接：

```powershell
python -m pip install FlagEmbedding
```

2. 准备配置：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少填入：

```env
LLM_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=你的方舟APIKey
LLM_MODEL=glm-5-2-260617
```

需要通过完整 Dashboard 离线体验时，可以改用确定性 FakeLLM，无需填写 API Key：

```env
RUNTIME_MODE=demo
LLM_BACKEND=fake
LLM_API_ENDPOINT=
LLM_API_KEY=
ARK_API_KEY=
LLM_MODEL=
```

Demo 模式使用本地 SQLite、FakeLLM 和 API 进程内候选评估，不需要 MySQL、Redis
或独立 Evaluation Worker。它适合本地体验；生产运行仍应使用 `RUNTIME_MODE=full`。

Demo 模式可以直接启动：

```powershell
$env:RUNTIME_MODE="demo"
$env:LLM_BACKEND="fake"
python -m prievo_agent.cli.api_server --root .prievo-demo-runtime
```

3. 启动 MySQL 和 Redis：

```powershell
docker compose up -d
```

这里 Docker 只负责本机基础设施。App 和 Worker 仍然用 Python 命令启动，方便调试和阅读日志。

4. 启动 API / Dashboard：

```powershell
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

5. 另开一个终端启动独立评估 Worker：

```powershell
python -m prievo_agent.cli.evaluation_worker --root .prievo-runtime
```

6. 打开：

```text
http://127.0.0.1:8000
```

## 启用 BGE RAG

默认配置使用轻量 hashing embedding，方便 clone 后快速启动。要切换为 BGE 全套，在 `.env` 中修改：

```env
RAG_EMBEDDING_BACKEND=bge
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_QUERY_INSTRUCTION=为这个查询生成表示以用于检索相关文献：
RAG_RERANKER_BACKEND=bge
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANK_CANDIDATE_POOL=40
```

实现细节：

- embedding 阶段会区分 `encode_query()` 和 `encode_document()`；
- query 会加 `RAG_QUERY_INSTRUCTION` 前缀；
- document chunk 不加前缀；
- reranker 是 cross-encoder，只对粗召回 Top-N 候选重排，不对全库逐条打分；
- 没有开启 `RAG_RERANKER_BACKEND=bge` 时，系统仍保留确定性的轻量 rerank bonus，便于测试复现。

CPU 环境可以设置：

```env
RAG_BGE_USE_FP16=false
RAG_BGE_DEVICE=cpu
```

## API

- `POST /api/runs`：创建 PriEvO run。
- `GET /api/runs/{run_id}`：查看 run 状态、预算、候选。
- `POST /api/runs/{run_id}/pause`：请求在 safe point 暂停。
- `POST /api/runs/{run_id}/resume`：从 checkpoint / durable facts 继续。
- `POST /api/runs/{run_id}/cancel`：取消 run，并取消未完成评估任务。
- `GET /api/runs/{run_id}/events/stream`：SSE 事件流。
- `GET /api/runs/{run_id}/artifacts`：查看产物索引。
- `GET /api/runs/{run_id}/trace`：查看 Agent / Node / Tool 因果追踪。

更详细的后台调用链见 [docs/FULL_RUNTIME.md](docs/FULL_RUNTIME.md)。
