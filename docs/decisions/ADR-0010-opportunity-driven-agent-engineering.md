# ADR-0010：只实现有真实调用路径的 Skill/Tool/Governance

- 状态：已接受
- 日期：2026-08-07

## 决策

实现 `literature_evidence_review` Skill、governed Literature Search Tool 与审计 Gateway。暂不实现 Multi-Agent registry/MCP；generation/reflection 保持职责分离的 service。Candidate evaluation 继续由专属 queue/port 治理，不伪装成 Agent。

## 后果

每个保留抽象均有实际代码路径、测试、artifact/event trace 和明确的工程理由。未来只有在外部 consumer 或独立自治职责带来可测价值时再加 MCP/Multi-Agent。
