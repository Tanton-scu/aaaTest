# ADR-0005：Current State、Event History 与 Artifact 分离

- 状态：已接受
- 日期：2026-08-07

## 决策

`Run.status` 是当前控制状态，只能经 `RunStateMachine` 改变；`Event` 是 append-only 的已发生事实，用于 timeline/harness；`ArtifactMetadata` 指向代码、结果、population 等大 payload。系统不采用 full Event Sourcing，恢复时数据库 current state 仍是 source of truth。

只有真实产生并被 inspection/API 消费的事件才写入。阶段 06 不伪造 `LANDSCAPE_ANALYZED`、`PRIOR_RETRIEVED`、`CHECKPOINT_SAVED`；相应能力实现时再增加。

## 后果

完成后的 Run 可完全通过 store 检查，不依赖 console；candidate/result/artifact 关联成为 harness 不变量。Artifact ID 同时包含 kind 与 digest，避免同内容不同语义覆盖 metadata。
