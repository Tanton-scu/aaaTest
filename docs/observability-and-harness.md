# 可观测性与工程 Harness

## 三类信号

- Trace/Event history：Run 内发生过什么，是用户可见、持久化、可回放的业务事实；
- Log：后台执行器的低层诊断信息，以 `run_id` 关联，不作为恢复事实源；
- Metric：从 Run、Event、Candidate 与 EvaluationJob 派生的聚合视图，不维护独立可变计数器。

`GET /api/runs/{run_id}/metrics` 当前提供实际被 Dashboard 和验收测试消费的 run/generation duration、无效候选率、评价尝试/重试、literature 调用、预算、best fitness 与 improvement。没有接入尚无消费方的 Prometheus、Grafana 或 Jaeger。

相关事件在适用时携带 `generation`、`candidate_id`、`evaluation_job_id`、`tool_call_id`；`run_id` 是事件表的一级关联键。

## 自动化回归面

| 风险 | 确定性测试/Harness |
|---|---|
| 生命周期和 artifact | `test_v0_vertical_slice.py` |
| 幂等、预算、重试、dead-letter | `test_evaluation_queue.py` |
| 真实进程退出与 checkpoint 恢复 | `test_checkpoint_recovery.py` |
| PriEvO reference characterization | `test_prievo_core.py` |
| structured prior fixture | `test_prior_retrieval.py` |
| 固定 literature corpus | `test_literature_rag.py` |
| API 断连/重连与 SSE | `test_api_sse.py` |
| 指标契约 | `test_observability_contract.py` |

这些工程测试与候选 fitness benchmark、PriEvO 研究 benchmark 分离。
