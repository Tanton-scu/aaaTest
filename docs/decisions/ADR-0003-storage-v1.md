# ADR-0003：v1 使用 SQLite + Filesystem Artifact Store

- 状态：已接受
- 日期：2026-08-07

## 决策

SQLite 保存 Run 当前状态、events、candidate/evaluation metadata、job lease 与 checkpoint metadata；filesystem 保存 code、prompt/response、trajectory、logs 和 checkpoint population 等大 payload。生产 adapter 通过 unit of work 保证 state/event/job 的事务一致性，artifact 用临时文件、digest 和原子 rename 发布。

v1 不引入 Redis、外部 broker、Neo4j、vector DB、MinIO。

阶段 15 校正：EvaluationJob enqueue/claim/finalize 与预算具有显式事务；一般 Run state/event、checkpoint metadata/event 尚为相邻提交，不应宣称已有完整 Unit of Work。current state 是恢复事实源，event 是解释历史；严格审计/多节点部署前需引入 transactional outbox 或等价 UoW。

## 原因与后果

该组合足以证明本地持久化、失败恢复、幂等与 trace，又避免无法解释的基础设施。它不是多节点高可用设计；若未来出现并发/共享存储的实测瓶颈，再在既有 ports 后替换实现。
