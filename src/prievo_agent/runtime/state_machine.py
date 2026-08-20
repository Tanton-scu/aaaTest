from __future__ import annotations

from typing import Dict, Set

from prievo_agent.domain.models import Run, RunStatus, utc_now


class RunStateMachine:
    """Run 状态迁移的唯一守卫入口。"""

    _transitions: Dict[RunStatus, Set[RunStatus]] = {
        RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
        RunStatus.RUNNING: {
            RunStatus.PAUSED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        },
        RunStatus.PAUSED: {RunStatus.RUNNING, RunStatus.CANCELLED},
        RunStatus.COMPLETED: set(),
        RunStatus.FAILED: set(),
        RunStatus.CANCELLED: set(),
    }

    def transition(self, run: Run, target: RunStatus) -> None:
        if target not in self._transitions[run.status]:
            raise ValueError(
                "非法 Run 状态迁移：{} -> {}".format(run.status.value, target.value)
            )
        run.status = target
        run.updated_at = utc_now()
