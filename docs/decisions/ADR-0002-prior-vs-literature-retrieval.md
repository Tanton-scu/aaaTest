# ADR-0002：Prior Retrieval 与 Literature RAG 永久分离

- 状态：已接受
- 日期：2026-08-07

## 决策

`InstanceSpecificPrior` 由 landscape metrics 查询结构化历史 instance/optimizer/operator empirical evidence；数值 top-k 是主路径，LLM 只可选精排。`LiteratureEvidence` 由自然语言 query 检索论文/文档 chunk，仅作为 generation/reflection tool。

两者使用不同 port、模型、artifact kind 和事件，不共享排名或把文档相似度当作实例相似度。

## 后果

v1 即使没有 literature index 也能完整运行 PriEvO。项目文档分别解释算法 prior 与工程 RAG，避免错误映射。
