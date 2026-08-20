"""在 Compose App 容器中验证 MySQL/Redis/独立 Worker 的真实产品链。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TERMINAL_RUN_STATUSES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
TERMINAL_JOB_STATUSES = frozenset({"SUCCESS", "DEAD", "CANCELLED"})


def _positive_seconds(name, default, *, maximum=3600.0):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("{} 必须是秒数".format(name)) from exc
    if not 0 < value <= maximum:
        raise ValueError("{} 必须在 (0, {}] 秒".format(name, maximum))
    return value


def _status_value(value):
    return str(getattr(value, "value", value))


def _job_snapshot(store, run_id):
    jobs = list(store.evaluation_jobs_for_run(run_id))
    return [
        {
            "job_id": item.id,
            "status": _status_value(item.status),
            "attempts": int(item.attempts),
            "worker_id": item.worker_id or "",
            "error_code": item.error_code or "",
        }
        for item in jobs
    ]


def wait_for_full_mode_run(
    facade,
    composition,
    run_id,
    *,
    timeout_seconds,
    worker_ready_timeout_seconds,
    poll_seconds=0.5,
    monotonic=time.monotonic,
    sleeper=time.sleep,
):
    """有截止时间地等待 Run，并在 Worker 从未领取首个 Job 时尽快失败。"""

    if worker_ready_timeout_seconds > timeout_seconds:
        raise ValueError("Worker ready timeout 不能大于 Full smoke 全局 timeout")
    deadline = monotonic() + timeout_seconds
    first_unclaimed_at = None
    worker_seen = False
    latest_run = None
    latest_jobs = []
    monitor_store = composition.open_store()
    try:
        while True:
            now = monotonic()
            latest_run = facade.get_run(run_id)
            latest_jobs = _job_snapshot(monitor_store, run_id)
            status = str(latest_run["status"])
            if status in TERMINAL_RUN_STATUSES:
                if status != "COMPLETED":
                    raise RuntimeError(
                        "Full smoke Run 提前进入终态 {}：jobs={}".format(
                            status, latest_jobs
                        )
                    )
                return latest_run

            if any(item["attempts"] > 0 for item in latest_jobs):
                worker_seen = True
            outstanding = [
                item
                for item in latest_jobs
                if item["status"] not in TERMINAL_JOB_STATUSES
            ]
            if outstanding and not worker_seen:
                if first_unclaimed_at is None:
                    first_unclaimed_at = now
                if now - first_unclaimed_at >= worker_ready_timeout_seconds:
                    raise RuntimeError(
                        "独立 Evaluation Worker 在 {:.1f} 秒内未领取首个 Job；"
                        "请检查 `docker compose ps worker` 与 worker 日志。jobs={}".format(
                            worker_ready_timeout_seconds, outstanding
                        )
                    )

            if now >= deadline:
                raise RuntimeError(
                    "Full Mode smoke 超过全局截止时间 {:.1f} 秒；run={} jobs={}".format(
                        timeout_seconds, latest_run, latest_jobs
                    )
                )
            sleeper(min(poll_seconds, max(0.001, deadline - now)))
    finally:
        monitor_store.close()


def _cancel_for_cleanup(facade, run_id):
    """超时/失败后尽力持久化取消，使等待中的 Runtime 能退出。"""

    if not run_id:
        return
    try:
        status = facade.get_run(run_id)["status"]
        if status in {"PENDING", "RUNNING", "PAUSED"}:
            facade.cancel(run_id)
    except Exception as exc:  # pragma: no cover - 只用于失败清理与诊断
        print("Full smoke 清理 Run 失败：{}".format(exc), file=sys.stderr)


def main():
    if os.getenv("PRIEVO_MODE") != "full":
        raise SystemExit("full_mode_smoke 必须在 PRIEVO_MODE=full 下执行")

    from prievo_agent.application.run_facade import RunApplicationFacade
    from prievo_agent.infrastructure.local_runtime import LocalRuntimeComposition

    timeout_seconds = _positive_seconds("FULL_MODE_SMOKE_TIMEOUT_SECONDS", 600)
    worker_ready_timeout_seconds = _positive_seconds(
        "FULL_MODE_SMOKE_WORKER_READY_TIMEOUT_SECONDS", 30
    )
    composition = LocalRuntimeComposition(Path("/var/lib/prievo"), PROJECT_ROOT)
    if composition.evaluation_execution_mode != "external":
        raise RuntimeError("Full Mode 必须由独立 Evaluation Worker 执行")
    facade = RunApplicationFacade(
        composition.open_store,
        composition.execute,
        True,
        0,
        composition.dataset_registry,
    )
    run_id = ""
    succeeded = False
    try:
        created = facade.create_run(
            "xgboost-Covtype",
            generations=1,
            population_size=2,
            candidate_budget=3,
            random_seed=11,
        )
        run_id = created["run_id"]
        run = wait_for_full_mode_run(
            facade,
            composition,
            run_id,
            timeout_seconds=timeout_seconds,
            worker_ready_timeout_seconds=worker_ready_timeout_seconds,
        )
        events = facade.events(run_id)
        if run["dataset_id"] != "xgboost-Covtype":
            raise RuntimeError("Dataset binding 失败")
        if run["budget"] != {
            "total": 42,
            "consumed": 42,
            "reserved": 0,
            "remaining": 0,
        }:
            raise RuntimeError("Evolution + Final Optimization 预算账本不正确")

        event_types = {item["event_type"] for item in events}
        required_events = {
            "LANDSCAPE_SAMPLED",
            "LANDSCAPE_ANALYZED",
            "SIMILARITY_CANDIDATES_READY",
            "CANDIDATE_DRAFT_MATERIALIZED",
            "OPERATOR_SELECTION_COMPLETED",
            "FINAL_OPTIMIZATION_COMPLETED",
            "FINAL_CONFIGURATION_SELECTED",
            "RUN_COMPLETED",
        }
        missing = required_events - event_types
        if missing:
            raise RuntimeError("缺少关键事件：{}".format(sorted(missing)))
        batches = [
            item for item in events
            if item["event_type"] == "OPERATOR_BATCH_COMPLETED"
        ]
        if len(batches) != 4 or any(
            item["payload"]["offspring_count"] != 2 for item in batches
        ):
            raise RuntimeError("每个 operator 必须生成 population_size 个 offspring")
        selections = [
            item for item in events
            if item["event_type"] == "OPERATOR_SELECTION_COMPLETED"
        ]
        if len(selections) != 1 or any(
            item["payload"]["combined_count"] != 10
            or item["payload"]["offspring_count"] != 8
            or item["payload"]["operator_batch_count"] != 4
            or len(item["payload"]["population_ids"]) != 2
            for item in selections
        ):
            raise RuntimeError("四个 operator 的 4P offspring 必须与 retained P 统一执行 5P -> P")

        store = composition.open_store()
        try:
            checkpoint = store.latest_checkpoint(run_id)
            artifacts = list(store.artifacts_for_run(run_id))
            memories = list(store.agent_memories_for_run(run_id, "generation", 20))
            tool_calls = list(store.tool_calls_for_run(run_id))
            jobs = list(store.evaluation_jobs_for_run(run_id))
        finally:
            store.close()
        if checkpoint.dataset_id != "xgboost-Covtype":
            raise RuntimeError("Checkpoint 的 Dataset binding 失败")
        artifact_kinds = {item.kind for item in artifacts}
        required_artifacts = {
            "LANDSCAPE_SAMPLE",
            "TOP5_CANDIDATES",
            "SIMILARITY_DECISION",
            "CANDIDATE_DRAFT",
            "FINAL_SELECTION_DECISION",
            "FINAL_HEURISTIC",
            "FINAL_OPTIMIZATION_REPORT",
        }
        if not required_artifacts.issubset(artifact_kinds):
            raise RuntimeError("缺少 Agent/Final 主链 Artifact")
        if not memories or any(item.run_id != run_id for item in memories):
            raise RuntimeError("MySQL Run-local Generation Memory 未工作")
        if any(
            item.tool_name not in {"candidate_inspection", "literature_search"}
            for item in tool_calls
        ):
            raise RuntimeError("存在未经产品策略声明的 ToolCall")
        if any(item.status != "COMPLETED" for item in tool_calls):
            raise RuntimeError("Full smoke 中 ToolCall 未正确结算")
        if len(jobs) < 12 or any(item.worker_id for item in jobs):
            # terminal Job 会清空 owner；至少 10 次 evolution success + 2 final seeds。
            raise RuntimeError("独立 Worker 的 Job 数量或 terminal owner 语义不正确")

        redis_key = "prievo:run:{}:agent:generation:recent".format(run_id)
        memory_count = composition.event_bus.client.llen(redis_key)
        if memory_count <= 0 or composition.event_bus.client.ttl(redis_key) <= 0:
            raise RuntimeError("Redis Run-local Generation Memory 未工作")
        trace = facade.agent_trace(run_id)
        summary = trace["summary"]
        if summary["task_counts_by_type"].get("SEMANTIC_SIMILARITY_SELECTION") != 1:
            raise RuntimeError("SimilarityTask cardinality 不正确")
        if summary["task_counts_by_type"].get("HEURISTIC_GENERATION") != 8:
            raise RuntimeError("GenerationTask cardinality 不正确")
        if summary["completed_task_count"] != summary["task_count"]:
            raise RuntimeError("AgentTask 没有全部完成")

        print(json.dumps({
            "message": "MySQL/Redis/独立 Worker/Agent 主链 Full Mode smoke 通过",
            "run_id": run["run_id"],
            "budget": run["budget"],
            "event_count": len(events),
            "evaluation_job_count": len(jobs),
            "working_memory_items": memory_count,
            "agent_trace": summary,
            "timeouts": {
                "global_seconds": timeout_seconds,
                "worker_ready_seconds": worker_ready_timeout_seconds,
            },
        }, ensure_ascii=False, indent=2))
        succeeded = True
        return 0
    finally:
        if not succeeded:
            _cancel_for_cleanup(facade, run_id)
        facade.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
