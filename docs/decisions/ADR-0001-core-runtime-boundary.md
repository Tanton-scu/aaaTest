# ADR-0001：PriEvO Core 与 Runtime 控制面分离

- 状态：已接受
- 日期：2026-08-07

## 决策

Core 负责 landscape/prior/evolution/population selection 等优化语义；Runtime 负责 Run 状态机、预算、durable job、lease/retry/dead letter、checkpoint/recovery 和 event。CLI 与 API 均经 Application 使用同一 Runtime/Core。

Core 不依赖 FastAPI、SQLAlchemy、SSE、MCP、浏览器和具体 LLM SDK；外部能力只经 domain ports。

阶段 15 复审后，具体 SQLite/FakeLLM/FakeEvaluator 的选择集中到 `LocalRuntimeComposition`。`RunApplicationFacade` 只接收 store factory 与 run executor；自动依赖测试禁止 domain/application 反向导入 infrastructure/API。

## 原因与后果

原 PriEvO 把算法、文件、并行、超时、LLM 和恢复耦合，无法安全长跑或复用。分离后同步 research runner 与 persistent runtime 可以共享算法，但需要显式 domain model、port 和 mapper。
