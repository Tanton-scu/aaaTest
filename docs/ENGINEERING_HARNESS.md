# Agent Engineering Harness

## 目标

本项目的 Harness 不是把普通 Unit Test 换一个名字。它运行真实的 Agent 与后端协作链，并对路由、上下文、Artifact、工具调用、状态和 Trace 同时断言。Multi-Agent、故障注入、Trace 定位、崩溃恢复和并发一致性都能落到可重复执行的源码与报告，而不是演示日志。

Harness 只替换不可控边界：真实供应商 LLM、外部文献源、生产 Redis 和大规模 Dataset。下列组件保持为产品实现：

- `PersistentEvolutionRuntime` 与 PriEvO Core；
- `DurableAgentCoordinator`、`AgentTaskDispatcher`、`Blackboard`、`AgentRegistry`；
- 五类 Agent、ContextPolicy 和 SkillRegistry；
- SQLite durable facts、内容寻址 Artifact 和 `AgentTraceQuery`；
- `ToolGovernanceGateway`；
- Evaluation Queue、lease、fencing、budget reservation 和 checkpoint recovery；
- 候选代码隔离子进程及 syntax/timeout/OOM 分类。

## ScriptedFakeLLM

`src/prievo_agent/infrastructure/scripted_fake_llm.py` 提供可编排模型边界。响应按 route FIFO 消费：

| route | 对应能力 |
| --- | --- |
| `similarity` | SimilarityAgent |
| `generation:i1` … `generation:m2` | HeuristicGenerationAgent 的五个 Strategy Skill |
| `research:query` / `research:explain` | PriorResearchAgent 两阶段模型调用 |
| `repair:diagnose` / `repair:repair` | RepairAgent Diagnose/Repair |
| `final` | FinalSelectionAgent exact-tie |

每次调用记录顺序号、route、完整 Prompt、结构化请求、返回值或异常。队列元素可以是 JSON-like value、异常或 callable，因此能够注入 malformed structured output、transient provider failure、KnowledgeGap 以及按调用次数变化的行为。队列与调用记录支持序列化；恢复测试可以验证“已完成调用不重放”。

模型边界失败使用 `AgentTaskDispatcher.dispatch_with_retries` 有界重驱：

- 每次失败先写入 durable AgentTask attempts 和 `AGENT_TASK_FAILED`；
- 仅瞬时 provider 错误和 malformed model protocol 进入自动 redrive；
- 每次重驱写入 `AGENT_TASK_REDRIVE_SCHEDULED`；
- 上限完全取自 `AgentTask.max_attempts`，不存在无状态 busy loop；
- transient-once 在同一次工作流调用内自动成功；持续 malformed 恰好耗尽 3 次后进入 `FAILED`。

## 隔离策略

每个场景使用独立临时目录，其中包含独立 SQLite 数据库和 Artifact 目录。固定文献 backend 与固定 Dataset 不依赖网络。Redis 故障使用始终抛出连接异常的 client；验证 Redis 失败后从同一 Run、同一 scope 的 durable memory 回源，Redis 不承担 Source of Truth。

Checkpoint 场景启动真实子进程模拟硬退出，随后以同一数据库恢复。Evaluation Queue 场景使用可控时钟与 injected evaluator；实际走 enqueue、budget reservation、worker claim、lease recovery、fencing 与 final commit。

## 场景矩阵

当前报告固定执行 24 个场景：

| 类别 | 场景 |
| --- | --- |
| Agent 正常链 | Similarity→Prior、Generation→Candidate、KnowledgeGap→Research→Evidence→resume、Final exact tie |
| 模型与 RAG 故障 | 持续 malformed 终态失败、transient-once 自动 redrive、RAG empty 后有界恢复 |
| Candidate 故障 | syntax、algorithm timeout、OOM/abnormal exit |
| Repair | 新版本修复成功、超过 repair limit |
| Queue/一致性 | transient retry、worker crash、lease expire、duplicate submit、budget contention |
| Memory/幂等 | Redis unavailable、duplicate Runtime Event/Coordinator reconcile |
| 恢复 | Run crash、checkpoint recovery 与不中断结果对照 |
| 生命周期/安全 | pause、resume、cancel、tool unauthorized |

## Trace 与断言

每个场景输出 `expected_trace`、`actual_trace` 和细粒度 assertion。Trace 默认做有序子序列断言：允许真实链插入额外审计事件，但不允许关键因果步骤乱序。

KnowledgeGap 场景同时验证：

1. `HEURISTIC_GENERATION` 恰好 1 个；
2. `PRIOR_RESEARCH` 恰好 1 个；
3. `HEURISTIC_GENERATION_RESUME` 恰好 1 个；
4. Literature Tool durable call 恰好 1 个；
5. Original Prior 内容与 digest 不变；
6. Evidence 出现在恢复后的 Generation Context；
7. 重复进入同一 generation step 不重复调用模型或创建任务。

因此，如果最终 Candidate 虽然存在，但多创建了 ResearchTask、遗漏 Evidence 引用或越过 Original Prior，不会被判定为通过。

## 运行

安装项目依赖后直接运行：

```powershell
$env:PYTHONPATH='src'
python scripts/agent_harness.py
```

当前 `Dockerfile` 的 Python 3.11 镜像可以这样运行：

```powershell
docker compose exec -T app python scripts/agent_harness.py
```

默认报告写入 `reports/agent_harness.json`。希望收集全部失败而不是遇到失败抛出异常时：

```powershell
python scripts/agent_harness.py --collect-all
```

## 当前实测指标

执行 harness 后会生成 `reports/agent_harness.json`：

- Scenario：24；
- Scenario Pass：24；
- Scenario Pass Rate：100%；
- Trace/State/Context 等结构化断言：112；
- Assertion Pass：112；
- Harness 内部计时：6321 ms（不含命令启动开销）。

本次重跑使用宿主可用的 Python 环境；Harness 场景本身使用隔离 SQLite/Artifact/Fake 边界，不依赖 FastAPI、真实 MySQL 或外部网络。最新 Docker Full smoke 因宿主 Docker daemon 权限被拒未重跑，不能把本数字写成容器/MySQL 证据。

这些数据是小型工程验证集结果，不是算法研究 Benchmark，也不代表生产 LLM、真实网络或大规模 MySQL 并发性能。MySQL 并发与 RAG HitRate/MRR 由各自独立测试/评测报告验证；Harness 的职责是复现真实 Agent Path 和 Backend Reliability failure path。

## 报告定位

报告顶层包含 pass rate、隔离配置和全部 scenario。每个 scenario 都包含：

- `assertions`：期望、实际、是否通过和断言说明；
- `expected_trace` / `actual_trace`：关键因果链；
- `evidence`：Task ID、Artifact ID、模型调用记录或可靠性统计；
- `error`：场景意外异常，不会被吞掉。

定位问题时先查看失败 scenario，再按 Artifact/Task ID 查询 Agent Trace；不要只查看最终 Run 状态。
