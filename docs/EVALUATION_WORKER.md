# Full Mode 独立 Evaluation Worker

## 结论

Full Mode 的 Candidate benchmark 已从 App/Runtime 进程中拆出。App 只向 MySQL 提交 durable `EvaluationJob` 并进行有界轮询，独立 `worker` 服务通过 owner token 和 lease claim Job、执行受监督 Candidate 子进程，再把结果和预算结算原子写回 MySQL。Demo Mode 保持单进程 inline Worker，便于本地一键演示。

这条边界同时覆盖普通进化评测和 Final Optimization；Full Mode 的产品终点不会退回 App 内执行 evaluator。

## 运行关系

```text
App / Runtime (external)
  ├─ submit EvaluationJob + reserve budget ──────┐
  ├─ bounded poll + renew Runtime lease          │ MySQL（事实源）
  └─ cancel / safe-checkpoint pause control      │
                                                  │
Evaluation Worker                                │
  ├─ recover stale lease                         │
  ├─ claim with owner fencing ───────────────────┘
  ├─ execute supervised Candidate subprocess
  └─ settle SUCCESS / RETRY_WAIT / DEAD + budget

Redis：仅发送可选通知，不保存 Job ownership、状态或预算事实。
共享 artifact volume：保存 Candidate code、evaluation artifact 与最终报告。
```

## 启动

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose 会启动 `mysql`、`redis`、`app` 和独立 `worker`。`app` 健康且 MySQL schema 就绪后，Worker 才开始 claim。

也可以单独运行诊断命令：

```powershell
python -m prievo_agent.cli.evaluation_worker --root /var/lib/prievo --print-config
python -m prievo_agent.cli.evaluation_worker --root /var/lib/prievo --health-check
python -m prievo_agent.cli.evaluation_worker --root /var/lib/prievo --once
```

`--print-config` 只输出脱敏投影，不显示数据库账号或密码；`--health-check` 只验证 MySQL 连接，不创建或修改 schema；`--once` 会先回收 stale Job，再最多处理一个当前可用 Job。

## 配置契约

| 环境变量 | 默认值 | 约束与作用 |
|---|---:|---|
| `PRIEVO_MODE` | `full` | 独立 Worker 只接受 `full` |
| `DATABASE_URL` | Compose 本地 MySQL | 必须为 `mysql+pymysql://` |
| `REDIS_URL` | Compose Redis | 可选通知通道，不是事实源 |
| `EVALUATION_TIMEOUT_SECONDS` | `10` | `(0, 300]`；App 与 Worker 必须相同 |
| `EVALUATION_LEASE_SECONDS` | `30` | 必须严格大于 timeout |
| `EVALUATION_WORKER_POLL_SECONDS` | `0.25` | 空闲初始 polling 间隔 |
| `EVALUATION_WORKER_MAX_POLL_SECONDS` | `2` | 空闲指数退避上限 |
| `EVALUATION_STALE_SWEEP_SECONDS` | `10` | stale lease 回收周期 |
| `EVALUATION_WORKER_ID_PREFIX` | `evaluation-worker` | 实例 ID 前缀；实际 ID 还包含 host、pid 与 nonce |

App 和 Worker 都读取 `EVALUATION_TIMEOUT_SECONDS`。该值会进入 App 的 `evaluation_parameters_version`，也是 Worker 的真实子进程 timeout，确保 logical Job identity 与执行语义一致。

## 调度与故障语义

- Worker 空闲 polling 使用指数退避并受最大值约束；不会零延迟 busy-spin。
- Runtime 等待 PENDING、RUNNING 或 RETRY_WAIT 时使用正数、有界 polling，并持续续租 Runtime owner lease。
- external 模式绝不调用 App 内的 `EvaluationWorker.run_once()`；独立进程可以横向扩展，由 MySQL `SKIP LOCKED` 和 owner fencing 隔离 claim。
- benchmark transient failure 进入有界 `RETRY_WAIT`；确定性 syntax/runtime/interface/OOM 等由 FailureClassifier 决定 repair 或终态。
- Worker 在 benchmark 后失去 lease 时，旧 owner 不能结算结果；stale recovery 允许新 Worker 接管。这是 at-least-once execution，不宣称 exactly-once computation。
- 收到 SIGTERM 后不再 claim 新 Job；当前受监督子进程会先完成或触发 timeout，然后连接才关闭。
- 当前 Worker 仅在 benchmark 前后续租，没有后台 heartbeat。因此强制 `lease > timeout`，文档和健康信息都不宣称存在周期 heartbeat。
- cancel 在每轮轮询前后检查；pause 只有在已有一致 checkpoint 时才确认。初始 population 尚无 checkpoint 时，会完成当前评价并在第一个安全点暂停，不伪造 PAUSED。

## 可验证证据

- `tests/test_external_evaluation_runtime.py`：使用两个独立 SQLite store/连接模拟 App 与 Worker 进程，断言 App evaluator 调用次数严格为 0；独立 Worker 推进普通评测与两个 Final Optimization seed 后 Run 成功。
- 同文件覆盖 external wait 的 sleep 后 pause：已有 checkpoint 时变为 `PAUSED`，Job 和 reservation 保留。
- `tests/test_evaluation_worker_service.py`：覆盖配置脱敏、lease/timeout 约束、唯一实例 ID、有界 polling、stale sweep、lost-lease fencing、`--once`、只读健康检查和 Compose 服务契约。
- `tests/test_runtime_retry_scheduler.py`、`tests/test_final_optimization.py`：验证 inline 兼容、有界 retry 和最终优化回归。

## 已知边界

Candidate 执行层的 `python -I`、wall timeout 和资源限制属于纵深防护，不等价于强安全沙箱。真实生产部署仍应把 Worker 放入低权限、禁网、只读根文件系统并设置 CPU/内存限制的独立容器或微虚拟机。当前实现没有 benchmark 期间后台 heartbeat；若未来允许单次 benchmark 超过 lease，必须先实现受 fencing 约束的周期续租，再放宽现有校验。
