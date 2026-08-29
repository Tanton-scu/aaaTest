# 使用火山方舟 GLM-5.2 运行 PriEvO-Agent

本文给出最短本地运行路径。项目当前 LLM adapter 使用 OpenAI-compatible Chat Completions，因此火山方舟 `/api/v3/chat/completions` 可以直接接入。

## 1. 推荐环境

项目声明 Python `>=3.10`。推荐 Conda，Windows 上更省心：

```powershell
conda create -n prievo-agent python=3.10 -y
conda activate prievo-agent
cd PriEvO-Agent
python -m pip install -U pip
python -m pip install -e ".[research,papers,test]"
```

不用 Conda 也可以：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[research,papers,test]"
```

## 2. 接入火山方舟 GLM-5.2

先复制本地环境文件：

```powershell
Copy-Item .env.example .env
```

然后只编辑本地 `.env`。不要把真实 API Key 写进 README、prompt、代码或提交到 GitHub；`.env` 已被 `.gitignore` 忽略。

你的 curl：

```bash
curl https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARK_API_KEY" \
  -d '{
    "model": "glm-5-2-260617",
    "messages": [
      {"role": "system", "content": "你是人工智能助手."},
      {"role": "user", "content": "你好"}
    ]
  }'
```

对应项目 `.env`：

```env
LLM_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=你的火山方舟 API Key
LLM_MODEL=glm-5-2-260617
LLM_TIMEOUT_SECONDS=1800
LLM_MAX_TOKENS=4096
LLM_TEMPERATURE=0.1
LLM_THINKING_TYPE=enabled
```

也可以用 `LLM_API_KEY` 代替 `ARK_API_KEY`。如果两者都设置，优先使用 `LLM_API_KEY`。

如果你已经在 PowerShell 里临时设置环境变量，也可以不写 `.env`：

```powershell
$env:LLM_API_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
$env:ARK_API_KEY = "你的火山方舟 API Key"
$env:LLM_MODEL = "glm-5-2-260617"
```

直接运行 `python -m prievo_agent.cli.api_server` 时，CLI 会默认读取当前目录的 `.env`。如果系统环境变量和 `.env` 同时存在，系统环境变量优先。也可以显式指定：

```powershell
python -m prievo_agent.cli.api_server --env-file .env --root .prievo-runtime-ark
```

## 3. 是否把 tools 传给模型？

火山方舟 Chat Completions 支持 OpenAI 风格的 `tools` / `tool_choice`，模型返回 `tool_calls` 后，由你的程序执行工具，再把 tool result 作为下一轮消息传回模型。

但 PriEvO-Agent 当前不把工具权限直接交给模型。项目采用 Runtime-side tool governance：

- 模型只输出结构化 JSON 决策。
- `LiteratureSearchTool`、`CandidateInspectionTool` 由后端 Gateway 按 caller、scope、次数、side effect 和 cost class 审批。
- 工具结果作为 Evidence/Context 回写给 Agent，而不是让模型自由调用外部函数。

这样可以保证 LLM 不拥有系统权限，工具调用由后端可审计地执行。如果未来要接入 provider-level function calling，应该新增独立 adapter，并继续保留后端 allowlist、参数校验、超时和结果大小限制。

## 4. 先不真实评估，只走完整流程

如果你只是想先看流程、页面、Agent/Plan/Trace，不想真实执行候选 benchmark，可以用 demo + fake evaluator：

```powershell
conda activate prievo-agent
cd PriEvO-Agent

$env:PRIEVO_MODE = "demo"
$env:PRIEVO_EVALUATOR_MODE = "fake"
$env:LLM_API_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
$env:ARK_API_KEY = "你的火山方舟 API Key"
$env:LLM_MODEL = "glm-5-2-260617"
$env:LLM_TIMEOUT_SECONDS = "1800"
$env:LLM_THINKING_TYPE = "enabled"

python -m prievo_agent.cli.api_server --root .prievo-runtime
```

打开：

```text
http://localhost:8000
```

然后在 Dashboard 创建 Run。建议第一次参数小一点：

- dataset：`xgboost-Covtype`
- generations：`1`
- population_size：`2`
- candidate_budget：`3`

这会走：

```text
SimilaritySelectionNode
-> EvolutionPlannerAgent
-> HeuristicGenerationAgent
-> Candidate GENERATED
-> fake EvaluationResult
-> 5P -> P
-> FinalSelectionNode/Final Optimization
-> Trace
```

`PRIEVO_EVALUATOR_MODE=fake` 只允许 `PRIEVO_MODE=demo`；Full Mode 会拒绝启动，避免把假评价带进 MySQL/Redis/Worker 部署。

## 5. 不用真实 LLM 的离线演示

如果你还没有 API Key，把 LLM 三项都留空即可：

```powershell
$env:PRIEVO_MODE = "demo"
$env:PRIEVO_EVALUATOR_MODE = "fake"
$env:LLM_API_ENDPOINT = ""
$env:LLM_API_KEY = ""
$env:ARK_API_KEY = ""
$env:LLM_MODEL = ""
python -m prievo_agent.cli.api_server --root .prievo-runtime
```

项目会使用 deterministic `FakeLLM`，方便零成本验证后端流程。

## 6. Full Mode

Full Mode 需要 Docker、MySQL、Redis、App、Worker：

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入 LLM_API_ENDPOINT / ARK_API_KEY / LLM_MODEL
docker compose up --build
```

当前设计中 Full Mode 强制真实 evaluator，不允许 `PRIEVO_EVALUATOR_MODE=fake`。

## 7. 本地论文 RAG 预处理

论文 PDF 的正确位置是 `data/papers/`。你也可以按来源分子目录，例如：

```text
data/papers/tuner_paper/algorithm_tuners/*.pdf
data/papers/tuner_paper/general_tuners/*.pdf
data/papers/tuner_paper/system_tuners/*.pdf
```

安装 PDF 依赖后执行：

```powershell
python -m pip install -e ".[papers]"
python scripts\ingest_papers.py --paper-root data\papers --output data\literature\pdf_corpus.json
```

当前预处理做的是：递归扫描 PDF、提取文本、识别 section、按页码切 chunk、写入相邻 chunk 引用和 provenance，然后生成 `data/literature/pdf_corpus.json`。

这里还没有预计算或落盘真实 semantic embedding。运行时的 Hybrid RAG 会加载 `corpus.json + pdf_corpus.json`，用 BM25 ranking + deterministic hashing vector ranking 做 weighted RRF，再做轻量 rerank。也就是说，当前“embedding/vector”是离线可复验的 lexical hashing adapter，不是调用火山/云端 embedding 模型。
