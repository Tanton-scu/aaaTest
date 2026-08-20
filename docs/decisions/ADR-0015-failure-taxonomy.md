# ADR-0015：失败按重试与完整性语义分类

## 决策

- malformed structured output、candidate invalid、LLM timeout、prior repository unavailable、artifact integrity、evaluation transient/timeout 使用不同错误类型；
- candidate invalid/permanent benchmark 不重试，transient/timeout 有界重试；
- accepted Run 在启动前或运行中遇到未恢复错误均可转 `FAILED`；
- 外部边界允许捕获 broad exception 用于审计与重新抛出/状态迁移，但禁止静默 `pass`；
- SQLite transaction exception 保留 adapter 原始类型，同时必须验证 rollback 不变量。

## 理由

错误类型决定 retry、状态、预算和用户可见行为。把所有失败压成字符串或静默忽略会破坏恢复能力与可观测性。
