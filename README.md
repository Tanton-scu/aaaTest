# PriEvO-Agent

PriEvO-Agent 是一个融合 PriEvO 算法、后端服务与 Agent Engineering 的自动启发式设计系统。它根据目标 Dataset 的 Fitness Landscape 检索 instance-specific prior，通过五种 LLM evolution strategy 生成可执行 heuristic，真实评价、筛选并恢复长任务；工程层使用 FastAPI、MySQL、Redis、独立 Evaluation Worker、五类 durable Agent、Skills、受治理 Tools、Literature RAG、Checkpoint 和 Trace。

当前版本：`1.0.0`。前端是随 FastAPI 提供的轻量控制台，重点能力在 Backend 与 Agent Runtime。

## 一键启动 Full Mode

需要 Docker Desktop/Compose：

```powershell
Copy-Item .env.example .env
# 可选：在 .env 同时填写 LLM_API_ENDPOINT、LLM_API_KEY、LLM_MODEL
docker compose up --build -d
docker compose ps
```

打开 `http://localhost:8000`。Compose 会启动四个服务：

- `app`：API、Dashboard、PriEvO Runtime 与 Agent workflows；
- `worker`：独立认领并执行 MySQL `EvaluationJob`；
- `mysql`：Run、Task、Candidate、Job、Result、Artifact metadata、AgentTask、Memory、ToolCall 的事实源；
- `redis`：SSE 唤醒与 Run-local 近期 Agent Memory 缓存，不是事实源。

Compose 中 app 与 worker 都固定为 Full Mode，避免一端 SQLite、另一端 MySQL 的拆分配置；本地 Demo 请使用下文的 Python 命令。

LLM 三项全部留空时会使用 deterministic FakeLLM，方便零密钥演示完整调用链；三项只填一部分会拒绝启动，避免误以为正在调用真实模型。填写兼容 Chat Completions 的 endpoint/key/model 后，Similarity、Generation、Research、Repair 和 Final tie 走真实模型 adapter。

## 在哪里设置代数和种群大小

Dashboard 的“运行参数”可直接设置：

- `generations`：外层演化代数，1～20；
- `population_size`：保留种群大小 `P`，也是每个 active strategy 生成的 offspring 数，2～40；
- `candidate_budget`：每个 heuristic 内最多消费的有效新配置数，1～200；
- `random_seed`：控制地形采样、parent selection 与 Core 随机序列；普通 Candidate 评价为与 reference 对齐使用固定 seed `101`，Final Optimization 使用独立固定 seed 集合。

也可以调用 API：

```powershell
$body = @{
  dataset_id      = 'xgboost-Covtype'
  generations     = 4
  population_size = 10
  candidate_budget = 20
  random_seed      = 2024
} | ConvertTo-Json

Invoke-RestMethod http://localhost:8000/api/runs `
  -Method Post -ContentType 'application/json' -Body $body
```

名义演化预算为 `B × P × (1 + 4G)`；独立 Final Optimization 默认再使用 `2 seeds × 2B = 4B`。例如 `G=4、P=10、B=20`，权威总预算是 `3400 + 80 = 3480`。失败 Job 会释放 reservation，不会把未完成评价伪装成已消费预算。

## 当前真实主链

```text
Dataset CSV
 -> deterministic sampling / exact-nearest mapping
 -> 8 FLA metrics
 -> numeric Top-5
 -> SimilarityAgent semantic 1~3 allowlist decision
 -> immutable Original Prior
 -> initial population + real Candidate evaluation
 -> four strategies: each generates P offspring from the same retained P
 -> evaluate 4P -> one generation-level (P + 4P) -> P selection
 -> qualification / unique best or exact-tie FinalSelectionAgent
 -> independent multi-seed Final Optimization
 -> final heuristic + configuration + report
```

前期 strategy 是 `i1/e1/e2/m1`，后期是 `e1/e2/m1/m2`。`i1` 无 parent，`e1/e2` 各 2 个，`m1/m2` 各 1 个。四个 strategy 都从同一个代初 retained `P` 选择 parent，各生成 `P` 个个体；四批共 `4P` 全部评价后，与 retained `P` 统一执行一次 `5P -> P`。这是用户明确指定的论文/产品批语义；只读 reference executable 实际逐 operator 执行 `2P -> P`，两者已在 Audit、Checkpoint 版本和测试中明确区分。

五类 Agent 分别是 Similarity、Heuristic Generation、Prior Research、Repair 和 Final Selection。PriEvO Core 决定 generation、strategy、parents、population、selection 和 budget；Agent 只处理语义工作，不能接管算法状态。`DurableAgentCoordinator` 从 Artifact 缺口派生幂等 `AgentTask`，Blackboard 只投影 refs，Registry 按 capability 唯一路由。

## Dataset 与 Literature PDF

合法 Dataset 位于 `resources/datasets/`，CSV 至少包含配置列和一个 `$<` 最小化目标列。服务启动后可用 `GET /api/datasets` 查看经过校验的列表；Dashboard 不接受任意文件路径。

Literature RAG 与 FLA Prior Repository 是两个独立数据源。要加入自己的论文：

1. 把 PDF 放入 `data/papers/`；
2. 执行：

```powershell
docker compose exec -T app python scripts/ingest_papers.py
```

脚本按 section/page 生成稳定 chunk 与同 section 邻接关系，写入宿主机持久化的 `data/literature/pdf_corpus.json`。单个损坏 PDF 会记录并跳过，不破坏其他论文。KnowledgeGap 出现时，产品 `LiteratureSearchTool` 才会经过 Gateway 调用 Hybrid BM25/Vector/Fusion/Rerank；原 PriEvO Prior 不会被 RAG 覆盖。

Linux 上的 bind mount 必须允许容器用户写入 `data/literature/`；若脚本明确报告权限错误，可在宿主机安装 `papers` extra 后运行 `python scripts/ingest_papers.py`，避免生成 root-owned 文件。

## 生命周期、Trace 与 API

控制台支持 Pause、Resume、Cancel、SSE timeline、Candidate 列表、fitness trajectory、Agent Trace、最终 heuristic/configuration/report。Pause 是 cooperative safe-point：请求先持久化，Runtime 在一致 Checkpoint 边界确认；Cancel 会原子取消未领取工作并释放预算。

常用 API：

- `GET /api/health`、`GET /api/datasets`；
- `POST /api/runs`、`GET /api/runs/{id}`；
- `POST /api/runs/{id}/pause|resume|cancel`；
- `GET /api/runs/{id}/events`、`GET /api/runs/{id}/events/stream`；
- `GET /api/runs/{id}/candidates`、`GET /api/runs/{id}/artifacts`；
- `GET /api/runs/{id}/trace`、`GET /api/runs/{id}/metrics`。

`/trace` 从 MySQL/SQLite 中的 Event、AgentTask、Artifact ref 和 ToolCall 动态投影，不保存第二份可变状态。可从 `Event -> Task -> AgentCall/ToolCall -> Artifact` 追踪 KnowledgeGap、Repair 或 Final tie 的因果链。

## Demo Mode 与验证

本地 Demo 使用 SQLite 和 inline evaluator，不要求 MySQL/Redis：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[research,papers,test]"
$env:PRIEVO_MODE='demo'
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

测试与机器报告：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python scripts/agent_harness.py
python -m prievo_agent.rag_eval --corpus data/literature/corpus.json `
  --cases data/literature/eval_cases.json --output reports/rag_eval.json `
  --markdown docs/RAG_EVALUATION.md -k 3
docker compose exec -T -e PRIEVO_MODE=demo -e DATABASE_URL= -e REDIS_URL= `
  -e LLM_API_ENDPOINT= -e LLM_API_KEY= -e LLM_MODEL= `
  app python -m unittest discover -s tests -v
docker compose exec -T app python scripts/full_mode_smoke.py
```

交付镜像包含 `tests/` 与 smoke 脚本。Full smoke 默认最多等待 600 秒；首个 EvaluationJob 在 30 秒内无人领取会直接给出 worker 诊断。可通过 `.env` 的 `FULL_MODE_SMOKE_TIMEOUT_SECONDS` 与 `FULL_MODE_SMOKE_WORKER_READY_TIMEOUT_SECONDS` 调整。

最近一次工作区可执行回归（2026-08-20）：非 HTTP 模块 230 项全部通过，其中 6 项因未配置真实 MySQL/FLA extras 而按环境跳过；没有 failure/error。Agent Harness 已重新生成 [机器报告](reports/agent_harness.json)：24/24 场景、116/116 结构化断言。Original Prior 的 31 条记录已生成 [兼容矩阵](reports/prior_compatibility.json)：21 条满足受控执行静态契约，10 条仅保留为 Prompt/Evidence。RAG 报告使用 3 篇论文、6 个 chunk、5 个 gold case，只是工程验证集，不是研究 Benchmark。最新 Docker/MySQL 复跑因当前执行环境无 Docker daemon 权限而未重复执行；详细边界见 [发布审计](docs/RELEASE_AUDIT.md)。

## Faithful Mode 与诚实边界

`RESEARCH_FAITHFUL_MODE=true` 时不装配 PriorResearch workflow；若模型只返回 KnowledgeGap，会持久化 gap 后显式失败。Repair 仍可产生诊断/草案审计，但 repaired Candidate 会标为 INVALID 并排除出 population。默认 `false` 展示完整 Agent 工程增强。早期 strategy 顺序采用论文/用户明确语义 `i1/e1/e2/m1`；reference executable 因配置列表实际为 `e1/e2/m1/i1`，该差异已在审计中显式记录。独立 Final Optimization 是对 reference README 意图的工程补全，reference 可执行源码本身没有实现。

受监督 `python -I` 子进程、资源限制和 import allowlist 只是风险降低，不是容器级强安全沙箱；默认 hashing vector 是可复验 lexical adapter，不是生产语义 embedding；项目是单租户 Engineering Prototype，不宣称商业级多租户平台或论文指标复现。

## 文档导航

- [文档入口与版本说明](docs/README.md)
- [总体架构](docs/ARCHITECTURE.md) / [项目执行流](docs/PROJECT_FLOW.md)
- [Backend Runtime](docs/BACKEND_RUNTIME.md) / [Evaluation Queue](docs/EVALUATION_QUEUE.md) / [Checkpoint Recovery](docs/CHECKPOINT_RECOVERY.md)
- [五 Agent](docs/AGENT_ARCHITECTURE.md) / [Coordinator 与 Blackboard](docs/COORDINATOR_BLACKBOARD.md)
- [Context、Memory 与 RAG](docs/CONTEXT_MEMORY_RAG.md) / [Run Trace](docs/RUN_TRACE.md)
- [Engineering Harness](docs/ENGINEERING_HARNESS.md) / [RAG 评测](docs/RAG_EVALUATION.md) / [Prior 执行兼容性](docs/PRIOR_COMPATIBILITY.md)
- [发布审计](docs/RELEASE_AUDIT.md)
