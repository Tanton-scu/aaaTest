# Run Trace 与 Agent 因果链

`application/agent_trace.py::AgentTraceQuery` 从 durable Run、Event、AgentTask、Artifact 和 ToolCall 投影一次 Run 的 Agent 协作链。它是只读查询，不保存第二份状态，也不声称实现完整 Event Sourcing。

## 1. 查询入口

FastAPI 提供两个等价路径：

- `GET /api/runs/{run_id}/trace`：规范路径；
- `GET /api/runs/{run_id}/agent-trace`：兼容旧调用方。

两者都调用 `application/run_facade.py` 中的 trace 查询，并最终执行 `AgentTraceQuery(store).trace(run_id)`。Dashboard 会读取规范路径，但界面只展示 summary、tasks 和 tool calls；完整 HTTP 响应还包含 steps、artifacts 和 causal edges。

## 2. 返回结构

| 字段 | 来源与含义 |
| --- | --- |
| `run_id/dataset_id/status` | durable Run 当前快照 |
| `steps` | 过滤后的 Agent/Tool/Research/Repair 相关 Event，保持 sequence |
| `tasks` | 当前 Run 的全部 AgentTask 状态、capability、attempt 和输入/输出 refs |
| `artifacts` | `AGENT_ARTIFACT_KINDS` 白名单内的 metadata 与 JSON payload |
| `tool_calls` | durable ToolCallRecord 的 caller/reason/request/response/status |
| `causal_edges` | 从 task refs 与 Event task ref 推导的有向边 |
| `summary` | task 总数/完成/失败、按 task type 和 Artifact kind 的计数、tool call 数 |

Task 和 ToolCall 来自事实表；Trace 不靠日志文本猜任务状态。Run 重开后，只要 Store 和 Artifact 内容仍在，查询可以重新构建。

## 3. Agent Artifact 白名单

Trace 当前展开以下协作面 Artifact：

- Similarity：`TOP5_CANDIDATES`、`SIMILARITY_PROMPT`、`SIMILARITY_DECISION`；
- Generation：`GENERATION_REQUEST`、`GENERATION_RESUME_REQUEST`、两类 prompt、`CANDIDATE_DRAFT`、`RESUMED_CANDIDATE_DRAFT`、`KNOWLEDGE_GAP`、`RESUMED_KNOWLEDGE_GAP`；
- Research：两类 research prompt、`LITERATURE_EVIDENCE`、`PRIOR_EXPLANATION`；
- Repair：`CANDIDATE_FAILURE`、diagnosis/repair prompt、`REPAIR_DECISION`、`REPAIRED_CANDIDATE_DRAFT`；
- Final：`FINAL_TIE`、`FINAL_SELECTION_PROMPT`、`FINAL_SELECTION_DECISION`。

Artifact 查询会尝试 UTF-8 解码并 `json.loads`。JSON Artifact 返回真实 payload，因此 draft payload 可能包含 Agent 生成的代码；plain-text Prompt 不是 JSON，会返回 `{"unavailable": true}`，但其 `artifact_id/kind/digest/size` 仍可用，完整 Prompt 需通过通用 Artifact 内容 API 读取。Artifact 丢失、解码失败或 JSON 失败也使用同一显式标记，不伪造内容。

Trace 不展开 `Candidate` 表、EvaluationJob/Result、Population snapshot、Checkpoint、`FINAL_HEURISTIC` 或 `FINAL_OPTIMIZATION_REPORT`。这些应分别通过 run candidates、events、artifacts 和具体 Artifact 内容接口查询；不能从 Agent Trace 不存在某字段推断算法步骤没有发生。

## 4. 因果边的精确定义

`causal_edges` 目前只有三类：

```text
Artifact --INPUT--> AgentTask
AgentTask --OUTPUT--> Artifact
Event --OBSERVES--> AgentTask
```

- `INPUT` 来自 `AgentTask.input_artifact_refs`；
- `OUTPUT` 来自 `AgentTask.output_artifact_refs`；
- `OBSERVES` 只在 Event payload 有 `agent_task_id` 或 `task_id` 时产生；
- 重复五元组会去重，遍历顺序保持稳定。

ToolCall 虽然单独列出，但当前没有 ToolCall node 到 Evidence/Task 的 graph edge；关联需通过 tool call 的 caller/reason、Event sequence 和 Evidence provenance 中的 logical tool-call ref 联合查看。Candidate materialization 和评价也不在 edge graph 中，只在相关 Event/Artifact 或其他 API 中观察。

`steps[].actor` 优先使用 Event payload 的 `agent`，其次 `caller`，否则显示 `DurableAgentCoordinator`。这个 fallback 是展示标签，不证明所有缺少 actor 字段的 Event 都由 Coordinator 发出；权威因果仍是 event type、sequence 和 refs。

## 5. 五条可观察路径

### 5.1 Similarity -> Prior

```text
TOP5_CANDIDATES
  --INPUT--> SEMANTIC_SIMILARITY_SELECTION task
  --OUTPUT--> SIMILARITY_PROMPT
  --OUTPUT--> SIMILARITY_DECISION
```

`INSTANCE_SPECIFIC_PRIOR` 不在 Agent Artifact 白名单，需要从通用 Artifact 列表验证。Trace 能证明 semantic task 使用哪个 Top-5 并输出哪个 decision，不能单独证明 repository extraction 内容。

### 5.2 普通 Generation

```text
GENERATION_REQUEST
  --INPUT--> HEURISTIC_GENERATION task
  --OUTPUT--> GENERATION_PROMPT
  --OUTPUT--> CANDIDATE_DRAFT
  -> CANDIDATE_DRAFT_MATERIALIZED Event
```

Candidate ID、真实 fitness 和 trajectory 应到 candidate/evaluation 查询核对。一个 generation/operator 会有多个独立 request/task；task count 反映实际调用基数，而不是只有一个抽象“GenerationAgent 节点”。

### 5.3 KnowledgeGap -> Research -> resume

```text
GENERATION_REQUEST
  -> HEURISTIC_GENERATION task
  -> KNOWLEDGE_GAP
  -> PRIOR_RESEARCH task
  -> PRIOR_RESEARCH_QUERY_PROMPT
  -> PRIOR_RESEARCH_EXPLANATION_PROMPT
  -> LITERATURE_EVIDENCE
  -> PRIOR_EXPLANATION
  -> GENERATION_RESUME_REQUEST
  -> HEURISTIC_GENERATION_RESUME task
  -> GENERATION_RESUME_PROMPT
  -> RESUMED_CANDIDATE_DRAFT
```

其中 Literature Tool 的 started/completed/failed/denied 记录在 steps/tool_calls；Original Prior 是否不变可比较 Generation request 与 Research/Resume Artifact 的 prior digest/ref。

### 5.4 Candidate failure -> Repair

```text
CANDIDATE_FAILURE
  --INPUT--> CANDIDATE_REPAIR task
  --OUTPUT--> REPAIR_DIAGNOSIS_PROMPT
  --OUTPUT--> REPAIR_DECISION
  [可修复] --OUTPUT--> REPAIR_PROMPT + REPAIRED_CANDIDATE_DRAFT
  -> REPAIRED_CANDIDATE_MATERIALIZED Event
```

Trace 可区分 candidate failure 与 repair task；infrastructure retry 不应出现 `CANDIDATE_FAILURE`。新 Candidate 的真实评价仍在 Evaluation Queue/Result 侧验证。

### 5.5 Final exact tie

```text
FINAL_TIE
  --INPUT--> FINAL_SELECTION task
  --OUTPUT--> FINAL_SELECTION_PROMPT
  --OUTPUT--> FINAL_SELECTION_DECISION
```

唯一 best 或无合格候选 stable fallback 不创建 `FINAL_TIE` task，因此“没有 FinalSelection AgentTask”可以是正确路径。最终双 seed optimization 发生在 final selection 之后，但其 report 不属于 Agent Trace。

## 6. 定位一次失败 Run

建议按 durable ID 逐层收窄：

1. 看 `status` 与 summary，确定失败任务数、task type 和 tool call 数；
2. 找到 `FAILED` task，检查 attempts/max/error 与 input refs；
3. 在 artifacts 中核对输入 payload、digest 和 source refs；plain-text prompt 用通用 Artifact API 打开；
4. 按 Event sequence 查看 claim、redrive、tool、materialization 的前后关系；
5. 用 causal edges 确认任务是否消费了期望输入、输出是否真实登记；
6. 若故障在 Candidate 执行、预算、lease、checkpoint 或 final optimization，转到 run candidates/events、EvaluationJob/Result 和相应 Artifact，而不是停留在 Agent Trace。

Trace 的目的是把“哪个持久事实触发哪个 AgentTask，任务产出哪些 Artifact”连起来；它不替代日志、指标或算法结果报告。

## 7. Agent Engineering Harness 与 Trace

`runtime/agent_harness.py` 使用产品 `PersistentEvolutionRuntime`、Core、Coordinator/Dispatcher/Blackboard、五 Agent、Context/Skill、SQLite/Artifact、Tool Governance、Evaluation Queue 和 `AgentTraceQuery`，只替换真实供应商 LLM、外部文献源、生产 Redis 和大规模 Dataset 等不可控边界。

每个场景在独立临时 SQLite/Artifact 目录中运行，使用 `infrastructure/scripted_fake_llm.py::ScriptedFakeLLM` 按 route FIFO 注入响应/错误并保留完整调用记录。场景报告的 `expected_trace/actual_trace` 是 Harness 定义的关键事件有序子序列，用于断言关键步骤不能乱序；它不是 HTTP Trace 响应的完整副本。

当前固定 `reports/agent_harness.json` 生成于 `2026-08-13T08:09:12.536193+00:00`：

- 24/24 场景通过，scenario pass rate 1.0；
- 116/116 个结构化断言通过；
- Harness 内部计时 6321 ms；
- 场景覆盖主链、KnowledgeGap/research/resume、exact tie、malformed/transient LLM、RAG empty、syntax/timeout/OOM、Repair、queue/lease/幂等/预算、Redis failure、crash/checkpoint、pause/resume/cancel 和未授权 Tool。

这些是隔离的小型工程验证数据，不是算法 Benchmark、真实 LLM 质量、生产网络、MySQL 吞吐或大规模并发结论。报告中的通过数只对报告记录的代码与 fixture 有效。

## 8. 与 reference 和工程增强的关系

reference executable 是进程内顺序控制流，没有 durable AgentTask、Artifact graph 或 ToolCall audit。当前 Trace 没有改变其 Prior、operator、selection 或 final semantics；它为语义节点增加可查询的因果证据。

Trace 中应区分三类证据：

- reference 原机制：Top-5 semantic decision、每个 operator 的 Generation、exact tie selection；
- deliberate faithful 差异：early schedule/midpoint、资格参数化、stable fallback 等应到 Core/Final workflow 与算法审计查看；
- Agent/Backend 工程增强：KnowledgeGap Research、Repair、durable task、ToolCall、恢复和 external evaluation 等会增加新的 Event/Artifact，但不成为 Core selection 的投票者。

## 9. 可验证证据

- Trace schema、Artifact 白名单、边构造：`src/prievo_agent/application/agent_trace.py`；
- API 两个路由与 Dashboard 消费：`src/prievo_agent/api/app.py`、`src/prievo_agent/api/dashboard.html`；
- 完整 Engine 路径和 refs 断言：`tests/test_agent_trace.py`；
- Harness 实现、报告和解释：`src/prievo_agent/runtime/agent_harness.py`、`tests/test_agent_harness.py`、`reports/agent_harness.json`、`docs/ENGINEERING_HARNESS.md`。
