# ADR-0004：以行为不变量增量迁移 PriEvO Core

- 状态：已接受
- 日期：2026-08-07

## 决策

不做 big-bang clean architecture rewrite。先迁移可 characterization 的 operator schedule、rank parent selection、fitness/diversity population management、candidate serialization 和 generation/evaluation ports；通过只读加载 reference 函数比较 fixture 结果。

松散 dict/operators 文本改为 typed Candidate/list/lineage，随机源改为实例私有；LLM prompt、FLA/prior、candidate execution 分阶段接入。无法保证 LLM stochastic trajectory 等价时，记录并测试结构、调度、选择和 lineage 不变量。

## 后果

ResearchRunner 与 persistent Runtime 已共享 `PriEvoEvolutionCore`，同时保留真实算法仍未完整迁移的诚实边界。后续迁移若改变已锁定语义，必须新增 ADR 和 regression 证据。
