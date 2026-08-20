# PriEvO-Agent 文档入口

这里集中保存当前 `1.0.0` 的产品架构、算法语义、运行时契约、可靠性设计与验证报告。

## 核心文档

- [总体架构](ARCHITECTURE.md) / [项目执行流](PROJECT_FLOW.md)
- [Backend Runtime](BACKEND_RUNTIME.md) / [Evaluation Queue](EVALUATION_QUEUE.md)
- [Checkpoint Recovery](CHECKPOINT_RECOVERY.md) / [独立 Evaluation Worker](EVALUATION_WORKER.md)
- [五 Agent 架构](AGENT_ARCHITECTURE.md) / [Coordinator 与 Blackboard](COORDINATOR_BLACKBOARD.md)
- [Context、Memory 与 RAG](CONTEXT_MEMORY_RAG.md) / [Run Trace](RUN_TRACE.md)
- [Engineering Harness](ENGINEERING_HARNESS.md) / [RAG 评测](RAG_EVALUATION.md)
- [Original Prior 兼容矩阵](PRIOR_COMPATIBILITY.md) / [发布审计](RELEASE_AUDIT.md)

## 架构决策

`decisions/` 保存 ADR，用于解释关键技术选择及其边界。判断当前行为时，以源码、测试和上述核心文档为准。
