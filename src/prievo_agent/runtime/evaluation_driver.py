"""Runtime 与 Final Optimization 共用的 durable EvaluationJob 推进器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from prievo_agent.domain.models import EvaluationJobStatus


INLINE_EVALUATION = "inline"
EXTERNAL_EVALUATION = "external"
EVALUATION_EXECUTION_MODES = frozenset(
    {INLINE_EVALUATION, EXTERNAL_EVALUATION}
)

_TERMINAL = frozenset(
    {
        EvaluationJobStatus.SUCCESS,
        EvaluationJobStatus.DEAD,
        EvaluationJobStatus.CANCELLED,
    }
)


class EvaluationDriverError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationDriveResult:
    completed: bool
    cancelled: bool
    polls: int
    inline_claim_attempts: int


def normalize_evaluation_execution_mode(value):
    mode = str(value or INLINE_EVALUATION).strip().lower()
    if mode not in EVALUATION_EXECUTION_MODES:
        raise ValueError(
            "evaluation_execution_mode 只能是 inline/external"
        )
    return mode


class DurableEvaluationJobDriver:
    """等待一组 durable Job 到终态，并按模式决定是否本进程 claim。

    ``inline`` 供 Demo/测试使用：调用注入的 ``EvaluationWorker.run_once``。
    ``external`` 供 Full Mode 使用：本进程绝不 claim，只以有界 polling 观察由
    独立 Worker 提交的事实。``control_check`` 在每轮与 sleep 后调用，可续
    Runtime lease、处理 cancel，并在已有一致 checkpoint 时响应 pause。
    """

    def __init__(
        self,
        store,
        worker,
        *,
        mode=INLINE_EVALUATION,
        clock,
        sleeper,
        max_poll_seconds=0.25,
    ):
        if max_poll_seconds <= 0:
            raise ValueError("max_poll_seconds 必须大于 0")
        self.store = store
        self.worker = worker
        self.mode = normalize_evaluation_execution_mode(mode)
        self.clock = clock
        self.sleeper = sleeper
        self.max_poll_seconds = float(max_poll_seconds)
        if self.mode == INLINE_EVALUATION and self.worker is None:
            raise ValueError("inline evaluation 必须注入 EvaluationWorker")

    def drive(
        self,
        run_id: str,
        job_ids: Sequence[str],
        *,
        control_check: Callable[[str], bool] | None = None,
    ) -> EvaluationDriveResult:
        identifiers = tuple(dict.fromkeys(str(item) for item in job_ids))
        if not identifiers:
            return EvaluationDriveResult(True, False, 0, 0)

        polls = 0
        claim_attempts = 0
        while True:
            if not self._continue(control_check, "评价轮询前"):
                return EvaluationDriveResult(False, True, polls, claim_attempts)
            jobs = self._jobs(identifiers)
            if all(job.status in _TERMINAL for job in jobs):
                return EvaluationDriveResult(True, False, polls, claim_attempts)

            progressed = None
            if self.mode == INLINE_EVALUATION:
                claim_attempts += 1
                progressed = self.worker.run_once()
                if progressed is not None:
                    continue

            delay = self._wait_delay(jobs)
            polls += 1
            self.sleeper(delay)
            if not self._continue(control_check, "评价轮询等待后"):
                return EvaluationDriveResult(False, True, polls, claim_attempts)
            if self.mode == INLINE_EVALUATION:
                # inline worker 已经探测过一次但 Job 尚未 ready；在同一个有界
                # wait 段内等到下一 progress point，避免每个 poll 都重复 claim。
                while True:
                    refreshed = self._jobs(identifiers)
                    if all(job.status in _TERMINAL for job in refreshed):
                        return EvaluationDriveResult(
                            True, False, polls, claim_attempts
                        )
                    if self._has_ready_job(refreshed):
                        break
                    next_delay = self._wait_delay(refreshed)
                    polls += 1
                    self.sleeper(next_delay)
                    if not self._continue(control_check, "评价轮询等待后"):
                        return EvaluationDriveResult(
                            False, True, polls, claim_attempts
                        )

    def _wait_delay(self, jobs):
        now = self.clock()
        deadlines = []
        pending_ready = []
        retry_ready = []
        for job in jobs:
            if job.status in _TERMINAL:
                continue
            if job.status == EvaluationJobStatus.PENDING:
                if job.available_at <= now:
                    pending_ready.append(job.id)
                else:
                    deadlines.append(job.available_at)
            elif job.status == EvaluationJobStatus.RETRY_WAIT:
                if job.available_at <= now:
                    retry_ready.append(job.id)
                else:
                    deadlines.append(job.available_at)
            elif job.status == EvaluationJobStatus.RUNNING:
                # external worker 的 RUNNING 是正常状态；最多等一个 bounded poll，
                # 不把 lease expiry 当作本进程可结算的许可。
                if job.lease_expires_at is not None:
                    deadlines.append(job.lease_expires_at)
            else:
                raise EvaluationDriverError(
                    "EvaluationJob 出现未知非终态：{} {}".format(
                        job.id, job.status.value
                    )
                )

        if self.mode == INLINE_EVALUATION:
            if pending_ready:
                raise EvaluationDriverError(
                    "evaluation queue 存在 PENDING job 但 inline worker 无法 claim：{}"
                    .format(", ".join(pending_ready))
                )
            if retry_ready:
                raise EvaluationDriverError(
                    "evaluation queue 存在已 ready 的 RETRY_WAIT job，"
                    "但 inline worker 仍无法 claim：{}".format(
                        ", ".join(retry_ready)
                    )
                )
            running = [
                job.id
                for job in jobs
                if job.status == EvaluationJobStatus.RUNNING
            ]
            if running:
                raise EvaluationDriverError(
                    "evaluation queue job 已处于 RUNNING，但 inline worker 无进展：{}"
                    .format(", ".join(running))
                )

        # external 的 ready Job 仍需给独立 Worker 一个有界 claim 窗口；不能把它
        # 当错误，也不能零延迟 busy-spin。
        if pending_ready or retry_ready or not deadlines:
            return self.max_poll_seconds
        remaining = (min(deadlines) - now).total_seconds()
        return max(0.001, min(self.max_poll_seconds, remaining))

    def _jobs(self, identifiers):
        return tuple(
            self.store.get_evaluation_job(identifier)
            for identifier in identifiers
        )

    def _has_ready_job(self, jobs):
        now = self.clock()
        return any(
            job.status in {
                EvaluationJobStatus.PENDING,
                EvaluationJobStatus.RETRY_WAIT,
            }
            and job.available_at <= now
            for job in jobs
        )

    @staticmethod
    def _continue(control_check, stage):
        if control_check is None:
            return True
        return control_check(stage) is not False


__all__ = [
    "DurableEvaluationJobDriver",
    "EVALUATION_EXECUTION_MODES",
    "EvaluationDriveResult",
    "EvaluationDriverError",
    "EXTERNAL_EVALUATION",
    "INLINE_EVALUATION",
    "normalize_evaluation_execution_mode",
]
