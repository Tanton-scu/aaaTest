# ADR-0011：FastAPI 使用持久 Run 契约

## 决策

HTTP router 保持薄层：创建、查询、取消和恢复统一委托给 application facade。创建接口先写 SQLite，再异步执行并返回 202。SSE 从同一事件表按单调 sequence 读取，支持游标重连；Dashboard 只消费公开 API。

## 理由

该设计允许浏览器断开、服务重建和历史回放，不把算法循环、SQL 或进程内对象暴露为 HTTP 生命周期的一部分。

## 约束

当前单进程执行器是 V1 调度入口，不是分布式 worker 声明。硬退出恢复边界仍是 generation checkpoint；后续部署阶段再决定多进程领取策略。
