# ADR-0007：v1 使用 SQLite Durable Evaluation Queue

- 状态：已接受
- 日期：2026-08-07

## 决策

EvaluationJob 与 EvaluationResult 分离。SQLite UNIQUE idempotency key、事务预算预留/结算、短事务 claim + lease、有界 retry/dead letter 构成 v1 队列；单 worker 在事务外运行 benchmark。

执行语义是 at-least-once benchmark execution + idempotent logical result/budget commit，不声称 exactly-once。Candidate invalid 永不盲目重试；timeout/transient 有界重试。暂不引入 Redis/Celery/Kafka。

## 后果

worker crash 可能造成 benchmark 重算，但不会重复持久化结果或双重计费。SQLite 的单机写吞吐是明确限制；只有实测需要并发跨节点时才替换 adapter。
