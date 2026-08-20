# Research PriEvO 与 Engineering PriEvO-Agent 的差异

本项目是融合后端与 Agent Engineering 的可运行原型，不宣称 100% 复现论文实验环境。

| 维度 | Research PriEvO | Engineering PriEvO-Agent |
|---|---|---|
| 目标 | 研究自动启发式设计方法及 benchmark 表现 | 展示 dataset-driven 长任务、恢复、队列、Agent 协作和可观测性 |
| 数据与 evaluator | 论文指定 benchmark 与实验配置 | 内置真实 CSV 上的可重复 offline CTP evaluator；可替换真实 evaluator |
| FLA | 研究实现的完整 landscape pipeline | 保留可解释 numerical feature evidence，允许工程化简化 |
| Operator | 论文 prompt/实现细节 | 保留用户确认的 early `i1/e1/e2/m1`、late `e1/e2/m1/m2` 语义 |
| Generation | 每个 operator 生成 P，4P offspring 与 retained P 一次选择 | 已按该 5P -> P 语义实现和测试 |
| LLM | 研究配置 | OpenAI-compatible adapter；无配置时 deterministic FakeLLM |
| Evaluation | 论文 benchmark fitness | EvaluationJob、预算预留、重试、dead letter、受限候选执行 |
| Stability/final | 论文完整最终稳定性流程 | 可信的工程 final heuristic 选择与 artifact，不等同论文全部实验 |
| Agent | 研究算法 prompt/reflection | Agent 只给 Advice；Backend 仍控制预算、状态、选择和恢复 |
| Persistence | 实验脚本/结果 | MySQL 事实源、Redis 通知/working memory、checkpoint recovery |

可以准确表述为“以 PriEvO 研究思想为算法核心的工程化 Agent 系统”；不能表述为“论文官方完整复现”或“达到论文全部 benchmark 结果”。
