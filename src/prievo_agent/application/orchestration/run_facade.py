from __future__ import annotations

import threading
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import OptimizationTask, Run, RunStatus
from prievo_agent.domain.ports import RuntimeStore
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.state_machine import RunStateMachine
from prievo_agent.application.observability.metrics import RunMetricsQuery
from prievo_agent.application.observability.agent_trace import AgentTraceQuery
from .recovery_manager import RecoveryManager
from .durable_agent_coordinator import DurableAgentCoordinator
from prievo_agent.runtime.persistent_runtime import RuntimeLeaseConflict
from prievo_agent.evaluation.final_optimization import (
    required_final_optimization_budget,
)


logger = logging.getLogger(__name__)


class RunNotFound(KeyError):
    pass


class RunConflict(RuntimeError):
    pass


class RunApplicationFacade:
    """API/CLI 共用的用例 facade；HTTP router 不含 SQL/演化循环。"""

    def __init__(
        self,
        store_factory: Callable[[], RuntimeStore],
        run_executor: Callable[[str], None],
        auto_start: bool = True,
        start_delay_seconds: float = 0.0,
        dataset_registry=None,
    ) -> None:
        self.store_factory = store_factory
        self.run_executor = run_executor
        self.auto_start = auto_start
        self.start_delay_seconds = start_delay_seconds
        self.dataset_registry = dataset_registry
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prievo-run")
        # 线程首次创建可能显著抬高第一个 POST 的延迟；在应用初始化阶段预热。
        self.executor.submit(lambda: None).result()
        self._scheduled = set()
        self._lock = threading.Lock()
        store = self._store()
        store.close()

    def recover_startup(self) -> Dict[str, object]:
        """应用启动后恢复 orphan work；调用方控制是否自动执行。"""
        store = self._store()
        try:
            report = RecoveryManager().recover(store, self._schedule)
            return report.to_dict()
        finally:
            store.close()

    def reconcile_active_agent_tasks(self) -> Dict[str, object]:
        """周期兜底 missing-work；只从 durable facts 创建唯一 AgentTask。

        正常路径仍在关键 COMMIT 后立即 reconcile。这个入口覆盖“COMMIT 成功但
        回调丢失、服务进程仍存活”的窗口；新建任务对应的 Run 会被幂等调度，
        已在本进程执行的 Run 由 ``_scheduled`` 集合自然去重。
        """

        store = self._store()
        try:
            active_ids = tuple(
                sorted(
                    run.id
                    for run in store.list_runs()
                    if run.status in {RunStatus.PENDING, RunStatus.RUNNING}
                    and not run.cancel_requested
                )
            )
            created = DurableAgentCoordinator(store).sweep(active_ids)
            created_run_ids = tuple(sorted({task.run_id for task in created}))
        finally:
            store.close()
        for run_id in created_run_ids:
            self._schedule(run_id)
        return {
            "active_run_count": len(active_ids),
            "created_task_count": len(created),
            "created_task_ids": [task.id for task in created],
            "scheduled_run_ids": list(created_run_ids),
        }

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def create_run(
        self,
        dataset_id,
        evaluation_budget=None,
        total_budget=None,
        generations=4,
        population_size=10,
        candidate_budget=20,
        random_seed=2024,
    ) -> Dict[str, object]:
        if self.dataset_registry is None:
            # 旧 harness 仍走确定性 FakeEvaluator；产品 API 永远会提供 registry。
            name = dataset_id
            bound_dataset_id = ""
            candidate_budget = evaluation_budget or 3
            total_budget = total_budget or 60
            generations = 2
            population_size = 3
        else:
            dataset = self.dataset_registry.get(dataset_id)
            name = dataset.display_name
            bound_dataset_id = dataset.id
            # 每代 4 个 operator，各自生成 population_size 个 offspring。
            evolution_budget = (
                candidate_budget * population_size * (1 + 4 * generations)
            )
            final_optimization_budget = required_final_optimization_budget(
                candidate_budget
            )
            minimum_budget = evolution_budget + final_optimization_budget
            total_budget = max(minimum_budget, candidate_budget)
        identifier = uuid.uuid4().hex
        task = OptimizationTask(
            "task-{}".format(identifier),
            name,
            "minimize",
            candidate_budget,
            total_budget,
            dataset_id=bound_dataset_id,
            generations=generations,
            population_size=population_size,
            random_seed=random_seed,
        )
        run = Run("run-{}".format(identifier), task.id, dataset_id=bound_dataset_id)
        store = self._store()
        try:
            if hasattr(store, "create_task_run"):
                store.create_task_run(
                    task, run, EventType.RUN_CREATED.value, "运行已创建",
                    {"dataset_id": bound_dataset_id},
                )
            else:
                store.add_task(task)
                store.add_run(run)
                store.append_event(
                    run.id, EventType.RUN_CREATED.value, "运行已创建",
                    dataset_id=bound_dataset_id,
                )
        finally:
            store.close()
        # 直接从刚持久化的实体构造 202 响应，避免为同一请求再次开库读回。
        response = {
            "run_id": run.id,
            "task_id": task.id,
            "name": task.name,
            "dataset_id": run.dataset_id,
            "status": run.status.value,
            "generation": run.generation,
            "settings": {
                "generations": task.generations,
                "population_size": task.population_size,
                "candidate_budget": task.evaluation_budget,
                "random_seed": task.random_seed,
            },
            "budget": {
                "total": task.total_budget,
                "consumed": 0,
                "reserved": 0,
                "remaining": task.total_budget,
            },
            "best_candidate_id": None,
            "candidates": [],
        }
        if self.auto_start:
            self._schedule(run.id)
        return response

    def resume(self, run_id: str) -> Dict[str, object]:
        store = self._store()
        try:
            run = self._get_run(store, run_id)
            if run.status == RunStatus.PAUSED:
                RunLifecycleService(store, RunStateMachine()).start(run)
            elif run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
                raise RunConflict("当前状态不允许恢复：{}".format(run.status.value))
        except RuntimeError as exc:
            raise RunConflict(str(exc)) from exc
        finally:
            store.close()
        self._schedule(run_id)
        return self.get_run(run_id)

    def pause(self, run_id: str, reason: str = "用户请求暂停") -> Dict[str, object]:
        store = self._store()
        try:
            run = self._get_run(store, run_id)
            if run.status != RunStatus.RUNNING:
                raise RunConflict("当前状态不允许请求暂停：{}".format(run.status.value))
            RunLifecycleService(store, RunStateMachine()).request_pause(run, reason)
        except RuntimeError as exc:
            raise RunConflict(str(exc)) from exc
        finally:
            store.close()
        return self.get_run(run_id)

    def cancel(self, run_id: str) -> Dict[str, object]:
        store = self._store()
        try:
            run = self._get_run(store, run_id)
            if run.status not in {RunStatus.PENDING, RunStatus.RUNNING, RunStatus.PAUSED}:
                raise RunConflict("当前状态不允许取消：{}".format(run.status.value))
            RunLifecycleService(store, RunStateMachine()).cancel(run)
        except RuntimeError as exc:
            raise RunConflict(str(exc)) from exc
        finally:
            store.close()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> Dict[str, object]:
        store = self._store()
        try:
            run = self._get_run(store, run_id)
            task = store.get_task(run.task_id)
            candidates = list(store.candidates_for_run(run.id))
            return {
                "run_id": run.id,
                "task_id": run.task_id,
                "name": task.name,
                "dataset_id": run.dataset_id,
                "status": run.status.value,
                "generation": run.generation,
                "settings": {
                    "generations": task.generations,
                    "population_size": task.population_size,
                    "candidate_budget": task.evaluation_budget,
                    "random_seed": task.random_seed,
                },
                "budget": {
                    "total": task.total_budget,
                    "consumed": run.consumed_evaluations,
                    "reserved": run.reserved_evaluations,
                    "remaining": task.total_budget - run.consumed_evaluations - run.reserved_evaluations,
                },
                "best_candidate_id": run.best_candidate_id,
                "control": {
                    "pause_requested": run.pause_requested,
                    "cancel_requested": run.cancel_requested,
                    "reason": run.control_reason,
                    "runtime_cursor_artifact_id": run.runtime_cursor_artifact_id,
                    "runtime_owner_id": run.runtime_owner_id,
                    "runtime_lease_expires_at": (
                        run.runtime_lease_expires_at.isoformat()
                        if run.runtime_lease_expires_at else None
                    ),
                },
                "candidates": [
                    {
                        "candidate_id": item.id,
                        "status": item.status.value,
                        "objective": item.objective,
                        "operators": item.operators,
                        "lineage": item.lineage,
                        "code_artifact_id": item.code_artifact_id,
                        "evaluation_artifact_id": item.evaluation_artifact_id,
                        "trajectory": self._trajectory(store, item.id),
                    }
                    for item in candidates
                ],
            }
        finally:
            store.close()

    def list_runs(self) -> List[Dict[str, object]]:
        store = self._store()
        try:
            ids = [run.id for run in store.list_runs()]
        finally:
            store.close()
        return [self.get_run(identifier) for identifier in ids]

    def events(self, run_id: str, after: int = 0) -> List[Dict[str, object]]:
        store = self._store()
        try:
            self._get_run(store, run_id)
            return [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "message": event.message,
                    "payload": event.payload,
                    "occurred_at": event.occurred_at.isoformat(),
                }
                for event in store.events_for_run(run_id)
                if event.sequence > after
            ]
        finally:
            store.close()

    def artifacts(self, run_id: str) -> List[Dict[str, object]]:
        store = self._store()
        try:
            self._get_run(store, run_id)
            return [
                {
                    "artifact_id": item.id,
                    "kind": item.kind,
                    "media_type": item.media_type,
                    "size": item.size,
                    "digest": item.digest,
                    "uri": item.uri,
                }
                for item in store.artifacts_for_run(run_id)
            ]
        finally:
            store.close()

    def artifact_content(self, run_id, artifact_id):
        store = self._store()
        try:
            self._get_run(store, run_id)
            allowed = {item.id for item in store.artifacts_for_run(run_id)}
            if artifact_id not in allowed:
                raise RunNotFound(artifact_id)
            return store.artifact_content(artifact_id)
        finally:
            store.close()

    def metrics(self, run_id: str) -> Dict[str, object]:
        store = self._store()
        try:
            self._get_run(store, run_id)
            return RunMetricsQuery(store).snapshot(run_id)
        finally:
            store.close()

    def agent_trace(self, run_id: str) -> Dict[str, object]:
        store = self._store()
        try:
            self._get_run(store, run_id)
            return AgentTraceQuery(store).trace(run_id)
        finally:
            store.close()

    def candidates(self, run_id: str) -> Dict[str, object]:
        """返回候选投影；保持 Run detail 兼容，同时提供独立列表端点。"""
        detail = self.get_run(run_id)
        return {
            "run_id": run_id,
            "items": detail["candidates"],
        }

    def _schedule(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._scheduled:
                return
            self._scheduled.add(run_id)
        self.executor.submit(self._execute_background, run_id)

    def _execute_background(self, run_id: str) -> None:
        logger.info("运行后台执行开始", extra={"run_id": run_id})
        if self.start_delay_seconds:
            time.sleep(self.start_delay_seconds)
        store = self._store()
        try:
            run = store.get_run(run_id)
            if run.status == RunStatus.CANCELLED:
                return
            store.close()
            store = None
            self.run_executor(run_id)
            logger.info("运行后台执行结束", extra={"run_id": run_id})
        except RuntimeLeaseConflict:
            # 多 API 实例 startup sweep 同时调度时，durable runtime lease 的输家
            # 直接退出；不得把健康的 orphan Run 误标为 FAILED。
            logger.info("运行已由其他 Runtime owner 接管", extra={"run_id": run_id})
        except Exception as exc:
            logger.exception("运行后台执行失败", extra={"run_id": run_id})
            failure_store = self._store()
            try:
                run = failure_store.get_run(run_id)
                if run.status in {RunStatus.PENDING, RunStatus.RUNNING}:
                    # 失败迁移同样走 expected-state CAS。若控制面刚刚取消，Store
                    # 会拒绝 stale executor 把 CANCELLED 覆盖成 FAILED。
                    RunLifecycleService(failure_store, RunStateMachine()).fail(
                        run, str(exc)
                    )
            except Exception:
                logger.exception(
                    "后台失败后无法持久化 RUN_FAILED",
                    extra={"run_id": run_id},
                )
            finally:
                failure_store.close()
        finally:
            if store is not None:
                store.close()
            with self._lock:
                self._scheduled.discard(run_id)

    def _store(self) -> RuntimeStore:
        return self.store_factory()

    @staticmethod
    def _get_run(store: RuntimeStore, run_id: str) -> Run:
        try:
            return store.get_run(run_id)
        except KeyError as exc:
            raise RunNotFound(run_id) from exc

    @staticmethod
    def _trajectory(store: RuntimeStore, candidate_id: str):
        try:
            return store.result_for_candidate(candidate_id).trajectory
        except KeyError:
            return []
