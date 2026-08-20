# ADR-0000：固定两份参考源码的角色与禁止等价关系

- 状态：已接受
- 日期：2026-08-07

## 背景

目标项目同时参考 PriEvO 研究源码与 MindBridge 工程源码。若直接合并，会把 population、agent、prior、RAG、evaluation 和 harness 混为一谈。

## 决策

PriEvO 是核心算法语义来源；保留 FLA、instance-specific prior、operator-level evolution、candidate evaluation、fitness/diversity selection 和 lineage。MindBridge 仅作为运行时、持久化、queue、tool、RAG、API 与 harness 的模式参考。目标生产代码只写入 `PriEvO-Agent/`，两份 reference 永久只读。

明确区分 MindBridge 工程模式与 PriEvO 算法语义，禁止把相似概念直接等价映射。Literature RAG 与
Prior Retrieval 使用独立模型和接口。

## 后果

- 需要重写工程外壳，不能复制任何一个仓库的目录树。
- 算法回归与系统 harness 分开测试。
- 技术引入以实际问题为依据，不设 Agent/MCP/RAG 数量指标。
