# ADR-0008：Instance Prior 使用 FLA + 结构化实验证据

- 状态：已接受
- 日期：2026-08-07

## 决策

Prior Retrieval 以 8 维 FLA profile 为输入，按 reference 的方向变换/min-max/等权欧氏距离取 numeric top-k；LLM 只可选精排 top-k。最终 evidence 是 historical instance → optimizer rank → operator/module/code，使用 typed object/CSV adapter 和 artifact trace。

Literature evidence 不参与该排名。v1 不引入 graph/vector database；结构化 repository 已足够。无真实 LLM 时使用明确标记的 numeric fallback/Fake refiner。

## 后果

PriEvO prior 可直接成为 initial population seeds，并可回答每个 optimizer/operator 的经验来源。完整 FLA 计算由 optional `PflaccoLandscapeAnalyzer` 提供；Harness 的 recorded metrics 不冒充在线计算。
