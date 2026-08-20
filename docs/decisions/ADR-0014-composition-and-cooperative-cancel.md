# ADR-0014：具体 adapter 集中装配，取消以持久状态协作

## 背景

阶段 15 实码复审发现：Application 直接构造 SQLite/Fake adapters，且后台 Runtime 可能用旧 `RUNNING` 对象覆盖并发写入的 `CANCELLED`。

## 决策

- `LocalRuntimeComposition` 作为本地 V1 composition root，向 Application 注入 store factory 与 run executor；
- Application/Domain 禁止导入 Infrastructure/API，由依赖测试守卫；
- Persistent Runtime 在初始评价、每代开始、每代评价后及最终发布前重新读取持久 Run；发现 `CANCELLED` 立即协作退出，不再保存旧状态。

## 后果

取消延迟最多覆盖一个正在执行的 candidate 评价批次，并非强制抢占；已开始的 logical evaluation 仍按队列事务结算。强制停止需要更强的 process worker 管理，V1 不作虚假承诺。
