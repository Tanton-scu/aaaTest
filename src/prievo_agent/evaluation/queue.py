from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from prievo_agent.domain.errors import (
    CandidateInvalidError,
    EvaluationTimeoutError,
    TransientEvaluationError,
    WorkerCrashError,
)
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    Candidate,
    EvaluationJob,
    EvaluationJobStatus,
    EvaluationResult,
    OptimizationTask,
)
from prievo_agent.domain.ports import CandidateEvaluator, EvaluationQueueStore

from .failures import FailureAction, FailureClassifier


Clock = Callable[[], datetime]


EVALUATION_IDEMPOTENCY_MATERIAL_VERSION = "evaluation-v2"


def utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def evaluation_identity_material(
    task_id: str,
    candidate_id: str,
    seed: int,
    budget: int,
    dataset_digest: str = "",
    evaluator_version: str = "",
    evaluation_parameters_version: str = "parameters-v1",
) -> dict:
    """返回 logical evaluation 的 canonical、可审计 identity material。"""

    return {
        "material_version": EVALUATION_IDEMPOTENCY_MATERIAL_VERSION,
        "task_id": str(task_id),
        "candidate_id": str(candidate_id),
        "seed": int(seed),
        "budget": int(budget),
        "dataset_digest": str(dataset_digest or ""),
        "evaluator_version": str(evaluator_version or ""),
        "evaluation_parameters_version": str(
            evaluation_parameters_version or "parameters-v1"
        ),
    }


def evaluation_idempotency_key(material: dict) -> str:
    identity = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class WorkerCrashed(WorkerCrashError):
    pass


class EvaluationQueueService:
    def __init__(self, store: EvaluationQueueStore, clock: Clock = utc_clock) -> None:
        self.store = store
        self.clock = clock

    def submit(
        self,
        run_id: str,
        task_id: str,
        candidate_id: str,
        seed: int,
        budget: int,
        max_attempts: int = 3,
        dataset_digest: str = "",
        evaluator_version: str = "",
        evaluation_parameters_version: str = "parameters-v1",
    ) -> EvaluationJob:
        """提交一个由完整 evaluation semantics 标识的 logical job。

        新参数均有默认值，旧调用仍保持相同的参数位置与幂等行为。Dataset 内容、
        evaluator 实现或必要参数版本变化时，会形成新的 logical evaluation。
        """

        material, key = self.expected_identity(
            task_id=task_id,
            candidate_id=candidate_id,
            seed=seed,
            budget=budget,
            dataset_digest=dataset_digest,
            evaluator_version=evaluator_version,
            evaluation_parameters_version=evaluation_parameters_version,
        )
        job = EvaluationJob(
            id="eval-{}".format(key[:20]),
            run_id=run_id,
            task_id=task_id,
            candidate_id=candidate_id,
            seed=seed,
            budget=budget,
            idempotency_key=key,
            max_attempts=max_attempts,
            available_at=self.clock(),
        )
        persisted, created = self.store.enqueue_job(job)
        self.store.append_event(
            run_id,
            (
                EventType.EVALUATION_SUBMITTED.value
                if created
                else EventType.DUPLICATE_EVALUATION_IGNORED.value
            ),
            "候选评价已提交" if created else "重复评价提交已忽略",
            job_id=persisted.id,
            evaluation_job_id=persisted.id,
            candidate_id=candidate_id,
            idempotency_key=key,
            idempotency_material_version=EVALUATION_IDEMPOTENCY_MATERIAL_VERSION,
            idempotency_material=material,
        )
        return persisted

    @staticmethod
    def expected_identity(
        task_id: str,
        candidate_id: str,
        seed: int,
        budget: int,
        dataset_digest: str = "",
        evaluator_version: str = "",
        evaluation_parameters_version: str = "parameters-v1",
    ) -> tuple[dict, str]:
        """计算与 :meth:`submit` 完全相同的 material/key，不产生写操作。"""

        material = evaluation_identity_material(
            task_id=task_id,
            candidate_id=candidate_id,
            seed=seed,
            budget=budget,
            dataset_digest=dataset_digest,
            evaluator_version=evaluator_version,
            evaluation_parameters_version=evaluation_parameters_version,
        )
        return material, evaluation_idempotency_key(material)

    def assert_candidate_identity(
        self,
        run_id: str,
        task_id: str,
        candidate_id: str,
        seed: int,
        budget: int,
        dataset_digest: str = "",
        evaluator_version: str = "",
        evaluation_parameters_version: str = "parameters-v1",
    ) -> EvaluationJob:
        """验证已有 authoritative Result 可按当前 Runtime identity 复用。

        该只读检查专供 recovery/reconcile：Result 缺少成功 Job、Job identity
        不同或 Job/Result 关联分裂时都显式失败，绝不把旧 fitness 当成当前评价。
        """

        from prievo_agent.domain.errors import EvaluationIdentityConflictError

        candidate = self.store.candidate_by_id(candidate_id)
        if candidate.run_id != run_id:
            raise EvaluationIdentityConflictError(
                "Candidate 不属于当前 Run，拒绝复用 EvaluationResult"
            )
        try:
            result = self.store.result_for_candidate(candidate_id)
        except KeyError as exc:
            raise EvaluationIdentityConflictError(
                "Candidate 尚无 authoritative EvaluationResult，不能执行 identity 复用校验"
            ) from exc
        if result.run_id != run_id or result.candidate_id != candidate_id:
            raise EvaluationIdentityConflictError(
                "EvaluationResult 与 Candidate/Run 归属分裂"
            )
        jobs = [
            job for job in self.store.evaluation_jobs_for_run(run_id)
            if job.candidate_id == candidate_id
        ]
        if len(jobs) != 1:
            raise EvaluationIdentityConflictError(
                "authoritative EvaluationResult 必须恰好关联一个 logical EvaluationJob"
            )
        job = jobs[0]
        _, expected_key = self.expected_identity(
            task_id=task_id,
            candidate_id=candidate_id,
            seed=seed,
            budget=budget,
            dataset_digest=dataset_digest,
            evaluator_version=evaluator_version,
            evaluation_parameters_version=evaluation_parameters_version,
        )
        if (
            job.run_id != run_id
            or job.task_id != task_id
            or job.seed != int(seed)
            or job.budget != int(budget)
            or job.idempotency_key != expected_key
            or job.status != EvaluationJobStatus.SUCCESS
            or job.result_id != result.id
        ):
            raise EvaluationIdentityConflictError(
                "已有 EvaluationResult 的 logical identity 与当前 Runtime 不一致；重评必须 clone Candidate"
            )
        return job


class EvaluationWorker:
    """claim 后结束 DB 事务，再运行 benchmark，并以 owner token fencing 结算。

    当前实现会在 evaluator 前后各续租一次，并暴露 :meth:`renew_lease` 供外部
    heartbeat 调用。它没有启动后台 heartbeat 线程：若单次 evaluator 持续时间
    超过 lease，后置续租会失败，结果不会结算，必须等待 recovery/requeue。
    """

    def __init__(
        self,
        store: EvaluationQueueStore,
        evaluator: CandidateEvaluator,
        worker_id: str = "local-worker-1",
        lease_seconds: int = 30,
        clock: Clock = utc_clock,
        failure_classifier: FailureClassifier = None,
    ) -> None:
        self.store = store
        self.evaluator = evaluator
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.failure_classifier = failure_classifier or FailureClassifier()

    def run_once(self, crash_after_benchmark: bool = False) -> Optional[EvaluationJob]:
        job = self.store.claim_next_job(
            self.worker_id, self.clock(), self.lease_seconds
        )
        if job is None:
            return None
        owner_token = job.worker_id
        if not owner_token:
            raise WorkerCrashed("claim 未返回 evaluation lease owner token")
        self.store.append_event(
            job.run_id,
            EventType.EVALUATION_STARTED.value,
            "Worker 已领取候选评价",
            job_id=job.id,
            evaluation_job_id=job.id,
            candidate_id=job.candidate_id,
            worker_id=owner_token,
            attempt=job.attempts,
            lease_expires_at=(
                job.lease_expires_at.isoformat()
                if job.lease_expires_at is not None
                else None
            ),
        )
        self._require_lease(job.id, owner_token, "evaluator 前")
        candidate = self.store.candidate_by_id(job.candidate_id)
        # Job 是一次评价的事实源。独立 final optimization / multi-seed job 可以
        # 覆盖 Run 默认 seed/budget，evaluator 不应偷偷回读原 Task 默认值。
        task = replace(
            self.store.get_task(job.task_id),
            random_seed=job.seed,
            evaluation_budget=job.budget,
        )
        try:
            result = self.evaluator.evaluate(candidate, task)
            self._require_lease(job.id, owner_token, "evaluator 后")
            if crash_after_benchmark:
                raise WorkerCrashed("benchmark 已结束但 job 尚未 final commit")
            artifact = self.store.put_artifact(
                job.run_id,
                "EVALUATION_RESULT",
                json.dumps(
                    {
                        "candidate_id": result.candidate_id,
                        "objective": result.objective,
                        "trajectory": result.trajectory,
                        "best_configuration": result.best_configuration,
                        "used_budget": result.used_budget,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8"),
                "application/json",
            )
            completed = self.store.complete_job_success(
                job.id, owner_token, result, artifact.id, self.clock()
            )
            self.store.append_event(
                job.run_id,
                EventType.CANDIDATE_EVALUATED.value,
                "候选评价完成",
                job_id=job.id,
                evaluation_job_id=job.id,
                candidate_id=job.candidate_id,
                result_id=result.id,
                objective=result.objective,
                artifact_id=artifact.id,
            )
            return completed
        except WorkerCrashed:
            raise
        except Exception as exc:
            # evaluator、artifact 或 final commit 失败后，只有仍持有 lease 的 owner
            # 才能把 job 转入 retry/dead；失去 owner 时不得以 failure 覆写接管者。
            self._require_lease(job.id, owner_token, "失败结算前")
            decision = self.failure_classifier.classify(exception=exc)
            return self._fail(job, owner_token, decision, str(exc))

    def recover_stale(self) -> int:
        return self.store.recover_stale_jobs(self.clock())

    def renew_lease(self, job_id: str, owner_token: str = "") -> bool:
        """供同进程或外部 heartbeat 在 benchmark 执行期间安全续租。"""

        return self.store.renew_job_lease(
            job_id,
            owner_token or self.worker_id,
            self.clock(),
            self.lease_seconds,
        )

    def _require_lease(self, job_id: str, owner_token: str, stage: str) -> None:
        if not self.renew_lease(job_id, owner_token):
            raise WorkerCrashed(
                "evaluation lease 已在{}丢失，拒绝继续写入：job_id={} owner={}".format(
                    stage, job_id, owner_token
                )
            )

    def _fail(
        self, job: EvaluationJob, owner_token: str, decision, message: str
    ) -> EvaluationJob:
        delay = 2 ** max(job.attempts - 1, 0)
        updated = self.store.complete_job_failure(
            job.id,
            owner_token,
            decision.failure_type.value,
            message,
            decision.action == FailureAction.RETRY,
            self.clock() + timedelta(seconds=delay),
            self.clock(),
        )
        if updated.status == EvaluationJobStatus.RETRY_WAIT:
            event_type = EventType.EVALUATION_RETRY_SCHEDULED
            human = "评价暂时失败，已安排重试"
        else:
            event_type = EventType.EVALUATION_DEAD_LETTERED
            human = "评价进入 dead letter"
        self.store.append_event(
            job.run_id,
            event_type.value,
            human,
            job_id=job.id,
            evaluation_job_id=job.id,
            candidate_id=job.candidate_id,
            error_code=decision.failure_type.value,
            failure_type=decision.failure_type.value,
            failure_action=decision.action.value,
            attempts=updated.attempts,
        )
        return updated


class InjectedFailureEvaluator:
    """按 candidate_id 注入 timeout/transient/invalid/permanent 的确定性 Harness evaluator。"""

    def __init__(
        self,
        delegate: CandidateEvaluator,
        scripts: Dict[str, Sequence[str]],
    ) -> None:
        self.delegate = delegate
        self.scripts = {key: list(values) for key, values in scripts.items()}
        self.calls: Dict[str, int] = {}

    def evaluate(
        self, candidate: Candidate, task: OptimizationTask
    ) -> EvaluationResult:
        index = self.calls.get(candidate.id, 0)
        self.calls[candidate.id] = index + 1
        actions = self.scripts.get(candidate.id, ["success"])
        action = actions[min(index, len(actions) - 1)]
        if action == "timeout":
            raise EvaluationTimeoutError("deterministic benchmark timeout")
        if action == "transient":
            raise TransientEvaluationError("deterministic transient failure")
        if action == "invalid":
            raise CandidateInvalidError("deterministic candidate invalid")
        if action == "permanent":
            raise RuntimeError("deterministic permanent failure")
        return self.delegate.evaluate(candidate, task)
