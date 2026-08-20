from __future__ import annotations

from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import Run, RunStatus, utc_now
from prievo_agent.domain.ports import RuntimeStore

from .state_machine import RunStateMachine


class RunLifecycleService:
    def __init__(self, store: RuntimeStore, state_machine: RunStateMachine) -> None:
        self.store = store
        self.state_machine = state_machine

    def start(self, run: Run, owner_id: str = "", now=None) -> Run:
        was_paused = run.status == RunStatus.PAUSED
        # 状态判断与写入必须在 Store 的同一 CAS 中完成；不得用过期 Run 快照
        # 清除一个并发到达的 cancel/control/lease 事实。
        persisted = self.store.start_run(
            run.id, run.status, now or utc_now(), owner_id=owner_id
        )
        self.store.append_event(
            run.id,
            EventType.RUN_RESUMED.value if was_paused else EventType.RUN_STARTED.value,
            "运行已恢复" if was_paused else "运行已启动",
        )
        return persisted

    def request_pause(self, run: Run, reason: str = "用户请求暂停") -> Run:
        """持久化 cooperative pause 请求；此时 Run 仍为 RUNNING。

        真正的 RUNNING -> PAUSED 只能由 Runtime 在 checkpoint 安全点确认。
        """

        if run.status != RunStatus.RUNNING:
            raise ValueError("只有 RUNNING Run 可以请求暂停")
        persisted = self.store.request_run_pause(run.id, reason)
        self.store.append_event(
            run.id,
            EventType.RUN_PAUSE_REQUESTED.value,
            "已收到暂停请求，将在下一一致性安全点暂停",
            reason=reason,
        )
        return persisted

    def pause(
        self, run: Run, checkpoint_id: str = "", owner_id: str = "", now=None
    ) -> Run:
        """Runtime 在安全 checkpoint 已提交后确认暂停。"""

        if not owner_id:
            raise ValueError("确认暂停必须提供 Runtime owner fencing token")
        persisted = self.store.pause_run(run.id, owner_id, now or utc_now())
        self.store.append_event(
            run.id,
            EventType.RUN_PAUSED.value,
            "运行已在一致性安全点暂停",
            checkpoint_id=checkpoint_id,
        )
        return persisted

    def complete(self, run: Run, owner_id: str = "", now=None) -> Run:
        if not owner_id:
            raise ValueError("完成 Run 必须提供 Runtime owner fencing token")
        persisted = self.store.complete_run(
            run.id, owner_id, now or utc_now(), run.generation,
            run.best_candidate_id or "",
        )
        self.store.append_event(run.id, EventType.RUN_COMPLETED.value, "运行已完成")
        return persisted

    def fail(self, run: Run, reason: str, owner_id: str = "", now=None) -> Run:
        persisted = self.store.fail_run(
            run.id, reason, now or utc_now(), owner_id=owner_id
        )
        self.store.append_event(
            run.id, EventType.RUN_FAILED.value, "运行失败", reason=reason
        )
        return persisted

    def cancel(self, run: Run, reason: str = "用户请求取消") -> None:
        self.state_machine.transition(run, RunStatus.CANCELLED)
        run.cancel_requested = True
        run.control_reason = reason
        # Store 在一个事务内取消 PENDING/RETRY_WAIT job 与 PENDING AgentTask，
        # 并按 job reservation 精确释放预算。状态迁移意图仍只来自 StateMachine。
        persisted = self.store.request_run_cancel(run.id, reason)
        run.reserved_evaluations = persisted.reserved_evaluations
        run.runtime_owner_id = persisted.runtime_owner_id
        run.runtime_lease_expires_at = persisted.runtime_lease_expires_at
        self.store.append_event(
            run.id,
            EventType.RUN_CANCEL_REQUESTED.value,
            "已收到取消请求并停止认领新工作",
            reason=reason,
        )
        self.store.append_event(
            run.id,
            EventType.RUN_CANCELLED.value,
            "运行已取消",
            reason=reason,
        )
