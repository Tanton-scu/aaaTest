from dataclasses import dataclass
from datetime import datetime, timezone

from prievo_agent.domain.models import RunStatus

from .durable_agent_coordinator import DurableAgentCoordinator


@dataclass(frozen=True)
class RecoveryReport:
    stale_evaluation_jobs: int
    reconciled_agent_tasks: int
    scheduled_run_ids: tuple
    orphan_agent_tasks: int = 0
    orphan_runtime_leases: int = 0

    def to_dict(self):
        return {
            "stale_evaluation_jobs": self.stale_evaluation_jobs,
            "reconciled_agent_tasks": self.reconciled_agent_tasks,
            "scheduled_run_ids": list(self.scheduled_run_ids),
            "orphan_agent_tasks": self.orphan_agent_tasks,
            "orphan_runtime_leases": self.orphan_runtime_leases,
        }


class RecoveryManager:
    """从 durable facts 发现 orphan work；不维护第二套运行状态。"""

    def __init__(self, coordinator_factory=None, clock=None,
                 agent_task_orphan_seconds=300):
        self.coordinator_factory = coordinator_factory or DurableAgentCoordinator
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.agent_task_orphan_seconds = int(agent_task_orphan_seconds)
        if self.agent_task_orphan_seconds <= 0:
            raise ValueError("agent_task_orphan_seconds 必须大于 0")

    def recover(self, store, schedule):
        now = self.clock()
        stale_count = int(store.recover_stale_jobs(now))
        orphan_task_count = int(
            store.recover_orphan_agent_tasks(
                now, self.agent_task_orphan_seconds
            )
            if hasattr(store, "recover_orphan_agent_tasks") else 0
        )
        orphan_lease_count = int(
            store.recover_orphan_runs(now)
            if hasattr(store, "recover_orphan_runs") else 0
        )
        active = sorted(
            (
                run
                for run in store.list_runs()
                if run.status in {RunStatus.PENDING, RunStatus.RUNNING}
                and not getattr(run, "cancel_requested", False)
            ),
            key=lambda item: item.id,
        )
        coordinator = self.coordinator_factory(store)
        reconciled_count = 0
        for run in active:
            created = coordinator.reconcile(run.id)
            reconciled_count += len(created)
            store.append_event(
                run.id,
                "RECOVERY_SWEEP_COMPLETED",
                "启动恢复已对账 durable Job/AgentTask，并安排 orphan Run",
                stale_evaluation_jobs=stale_count,
                reconciled_agent_tasks=len(created),
                orphan_agent_tasks=orphan_task_count,
                orphan_runtime_leases=orphan_lease_count,
            )

        # 数据库连接可在外层关闭；schedule 仅接收稳定 Run ID。
        run_ids = tuple(run.id for run in active)
        for run_id in run_ids:
            schedule(run_id)
        return RecoveryReport(
            stale_count,
            reconciled_count,
            run_ids,
            orphan_task_count,
            orphan_lease_count,
        )
