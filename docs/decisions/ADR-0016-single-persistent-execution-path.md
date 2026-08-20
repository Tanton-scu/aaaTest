# ADR-0016：产品与 Demo 统一到单一持久执行链

## 决策

删除早期同步 `RuntimeCoordinator` 与薄 `OptimizationApplicationService`。API 和中文 Demo 均经 `RunApplicationFacade` 调用 `PersistentEvolutionRuntime`；算法研究仍经 `ResearchRunner` 直接使用相同 PriEvO Core。

## 理由

旧 coordinator 重复演化/评价/artifact 逻辑，却不使用 durable queue/checkpoint。保留两条产品执行链会制造行为漂移，且无法用 5–7 个组件清晰解释。

## 后果

端到端产品不变量只需在 persistent path 维护。同步 research path 仍保留，但只承担算法 characterization，不冒充可恢复产品 Runtime。
