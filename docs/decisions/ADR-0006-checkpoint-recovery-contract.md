# ADR-0006：一致性安全点 Checkpoint 与 Reconciliation 契约

- 状态：已接受
- 日期：2026-08-07

## 决策

在 initial population、每个 operator selection 和 generation selection 的一致性边界保存 checkpoint；恢复从 `algorithm_cursor` 指向的下一批工作继续。Checkpoint 只记录 population/evaluation 的稳定 ID、预算账本、RNG/LLM 状态、strategy、schema/code hash，不复制 Candidate code 等大对象。

恢复不能只回放 checkpoint：先加载 baseline，再查询 checkpoint 后 Candidate、EvaluationJob、EvaluationResult、Artifact 并做 reconciliation。已存在 Result 的 Candidate 不重新评价；Candidate 已存但 Job 未建时直接创建幂等 Job，不重新调用 LLM。

Pause 是请求与确认分离的 cooperative protocol。API 只落 `pause_requested`；Runtime 在上述安全点写完 checkpoint 才迁移为 `PAUSED`。Cancel 原子取消未认领工作并按 Job reservation 恰好释放预算。每个 Run 使用 owner lease 防止两个 Runtime 并发推进。

验证必须跨独立 OS process，而非同一内存对象调用 `resume()`。SQLite current state + artifact digest 是恢复依据，event history 用于解释，不回放构造全部状态。

## 后果

确定性 Fake 路径可与不中断执行逐项对照；Full Mode 在 MySQL 8.4 以 V004 迁移保存 control/cursor/owner lease。修改 core code/strategy 会拒绝旧 checkpoint，未来需显式迁移而不能静默继续。Lease 和事务提供多实例互斥/恢复，但不宣称 Byzantine 容错或跨机强 sandbox。
