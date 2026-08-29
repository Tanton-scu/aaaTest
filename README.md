# PriEvO-Agent

PriEvO-Agent 是一个面向算法配置优化的后端 + Agent 工程项目。它把 PriEvO 风格的启发式演化流程封装成可恢复、可观测、可测试的服务：后端负责任务生命周期、候选算法评估、持久化与并发控制；Agent 负责相似实例选择、演化规划、候选生成、RAG 补充证据、修复与最终选择。

## 项目亮点

- 后端主链：FastAPI + SQLite demo mode，MySQL + Redis + 独立 Worker full mode。
- Agent 编排：SimilaritySelectionNode、EvolutionPlannerAgent、HeuristicGenerationAgent、PriorResearchAgent、RepairAgent、FinalSelectionNode。
- PriEvO 演化规则：early 使用 `i1/e1/e2/m1`，late 使用 `e1/e2/m1/m2`；每个 operator 生成 `population_size` 个候选，再执行 `5P -> P` 保留。
- RAG：本地 literature corpus 经 BM25 + hashing vector + weighted RRF + rerank 检索，证据带 provenance，不覆盖算法 prior。
- 工程可靠性：Run/Event/Artifact/Candidate/EvaluationJob/Checkpoint/Trace 全部持久化；支持 pause/resume/cancel、lease fencing、idempotency 与 crash recovery。
- 安全边界：候选代码执行在受控子进程中，带 timeout、import allowlist、资源限制和结构化校验。
- 可验证交付：包含 unit tests、agent harness、full mode smoke 脚本和 Docker Compose。

## 目录结构

```text
PriEvO-Agent/
├── src/prievo_agent/
│   ├── api/             # FastAPI API 与内置 dashboard
│   ├── application/     # workflow、trace、tool governance、run facade
│   ├── agents/          # Agent / Node 实现与上下文策略
│   ├── algorithm/       # PriEvO engine 与 dataset evaluator
│   ├── core/            # PriEvO schedule、selection、prior retrieval
│   ├── domain/          # 领域模型、事件、端口
│   ├── infrastructure/  # SQLite/MySQL/Redis/LLM/RAG/skill adapters
│   ├── runtime/         # durable runtime、queue、checkpoint、harness
│   └── security/        # 候选代码校验与隔离执行
├── skills/              # i1/e1/e2/m1/m2 与 repair/research/final skills
├── resources/           # 小型 dataset 与 prior knowledge
├── data/literature/     # 小型 curated RAG corpus
├── scripts/             # demo、harness、paper ingest、full smoke
├── tests/               # 可运行测试
└── docs/                # 必要工程说明
```

## 快速运行

推荐 Python 3.10+。Windows / Conda 示例：

```powershell
conda create -n prievo-agent python=3.10 -y
conda activate prievo-agent
cd PriEvO-Agent

python -m pip install -U pip
python -m pip install -e ".[research,papers,test]"
Copy-Item .env.example .env
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

打开：

```text
http://127.0.0.1:8000
```

`.env.example` 默认使用 `demo + fake evaluator`。如果没有配置 LLM，系统会自动使用 deterministic FakeLLM，适合快速演示完整后端流程。

建议第一次创建 Run 时使用小参数：

```text
dataset: brotli
generations: 1
population_size: 2
candidate_budget: 3
```

## 接入真实 LLM

项目使用 OpenAI-compatible Chat Completions adapter。以火山方舟 GLM-5.2 为例，在 `.env` 中填写：

```env
PRIEVO_MODE=demo
PRIEVO_EVALUATOR_MODE=fake

LLM_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=你的方舟 API Key
LLM_MODEL=glm-5-2-260617
LLM_TIMEOUT_SECONDS=1800
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.1
LLM_THINKING_TYPE=enabled
```

然后重启：

```powershell
python -m prievo_agent.cli.api_server --root .prievo-runtime-ark
```

`PRIEVO_EVALUATOR_MODE=fake` 只影响候选 benchmark，不会阻止真实 LLM 调用。也就是说可以先用真实模型生成/规划，但用 fake evaluator 低成本走通流程。

## 本地 RAG 论文导入

仓库默认只带小型 curated corpus。若要加入自己的 PDF：

```powershell
New-Item -ItemType Directory -Force data\papers
# 把 PDF 放入 data\papers\，可保留多级子目录
python scripts\ingest_papers.py --paper-root data\papers --output data\literature\pdf_corpus.json
```

运行时会自动合并读取：

- `data/literature/corpus.json`
- `data/literature/pdf_corpus.json`，如果存在

PDF 和生成的 `pdf_corpus.json` 默认不建议提交到 Git。

## 测试与 Harness

```powershell
python -m compileall src\prievo_agent scripts
python -m unittest discover -s tests -v
python scripts\agent_harness.py
python -m prievo_agent.rag_eval --corpus data/literature/corpus.json `
  --cases data/literature/eval_cases.json --output reports/rag_eval.json `
  --markdown docs/RAG_EVALUATION.md -k 3
```

Full mode 需要 Docker、MySQL、Redis、App、Worker：

```powershell
docker compose up --build
```

## 关键 API

- `GET /api/health`
- `GET /api/datasets`
- `POST /api/runs`
- `GET /api/runs/{id}`
- `POST /api/runs/{id}/pause`
- `POST /api/runs/{id}/resume`
- `POST /api/runs/{id}/cancel`
- `GET /api/runs/{id}/events`
- `GET /api/runs/{id}/events/stream`
- `GET /api/runs/{id}/artifacts`
- `GET /api/runs/{id}/trace`

## 边界说明

本项目是单租户工程原型，不宣称商业级多租户、高可用或强安全沙箱。默认 hashing vector 是可复验 lexical vector，不是生产语义 embedding。候选代码隔离执行降低风险，但不等价于容器级安全沙箱。
