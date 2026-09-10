"""可恢复的多随机种子 Final Optimization 工程补全。

reference PriEvO 没有提供一条可直接执行、可恢复的 Final Optimization 主链。
本模块因此是产品工程新增：它复用正式 EvaluationQueue/Worker 和同一份持久预算
账本，但不声称该流程已经存在于 reference 实现中。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from prievo_agent.domain.models import (
    Candidate,
    CandidateStatus,
    EvaluationJob,
    EvaluationJobStatus,
    EvaluationResult,
    OptimizationTask,
    Run,
)

from .queue import EvaluationQueueService, EvaluationWorker, utc_clock
from .driver import (
    DurableEvaluationJobDriver,
    INLINE_EVALUATION,
    normalize_evaluation_execution_mode,
)


FINAL_OPTIMIZATION_SCHEMA_VERSION = "final-optimization-v1"
DEFAULT_FINAL_OPTIMIZATION_SEEDS = (1009, 2027)
DEFAULT_FINAL_BUDGET_MULTIPLIER = 2
FINAL_OPTIMIZATION_DISCLOSURE_CN = (
    "工程补全：reference PriEvO 未提供可执行的 Final Optimization 主链；"
    "本流程是产品化新增，不冒充 reference 已有实现。"
)
_TERMINAL_JOB_STATUSES = {
    EvaluationJobStatus.SUCCESS,
    EvaluationJobStatus.DEAD,
    EvaluationJobStatus.CANCELLED,
}


class FinalOptimizationError(RuntimeError):
    """Final Optimization 的显式基础异常。"""


class FinalOptimizationBudgetError(FinalOptimizationError):
    """权威预算账本无法为全部缺失 trial 原子式预留足够预算。"""


class FinalOptimizationFailed(FinalOptimizationError):
    """至少一个最终优化 Candidate clone 已明确失败。"""

    def __init__(self, report: "FinalOptimizationReport") -> None:
        self.report = report
        failed = [trial for trial in report.trials if trial.status != "SUCCESS"]
        detail = ", ".join(
            "seed={} error={}".format(trial.seed, trial.error_code or trial.status)
            for trial in failed
        )
        super().__init__("Final Optimization 未完成全部随机种子：{}".format(detail))


@dataclass(frozen=True)
class FinalOptimizationTrial:
    seed: int
    candidate_id: str
    job_id: str
    status: str
    result_id: str = ""
    objective: Optional[float] = None
    trajectory: Tuple[float, ...] = ()
    best_configuration: Optional[Dict[str, Any]] = None
    used_budget: int = 0
    error_code: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "candidate_id": self.candidate_id,
            "job_id": self.job_id,
            "status": self.status,
            "result_id": self.result_id,
            "objective": self.objective,
            "trajectory": list(self.trajectory),
            "best_configuration": dict(self.best_configuration or {}),
            "used_budget": self.used_budget,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FinalOptimizationTrial":
        return cls(
            seed=int(payload["seed"]),
            candidate_id=str(payload["candidate_id"]),
            job_id=str(payload["job_id"]),
            status=str(payload["status"]),
            result_id=str(payload.get("result_id", "")),
            objective=(
                None
                if payload.get("objective") is None
                else float(payload["objective"])
            ),
            trajectory=tuple(float(item) for item in payload.get("trajectory", [])),
            best_configuration=dict(payload.get("best_configuration", {})),
            used_budget=int(payload.get("used_budget", 0)),
            error_code=str(payload.get("error_code", "")),
            error_message=str(payload.get("error_message", "")),
        )


@dataclass(frozen=True)
class FinalOptimizationReport:
    report_identity: str
    run_id: str
    task_id: str
    source_candidate_id: str
    dataset_digest: str
    evaluator_version: str
    evaluation_parameters_version: str
    objective_direction: str
    evolution_candidate_budget: int
    final_budget_per_seed: int
    seeds: Tuple[int, ...]
    status: str
    trials: Tuple[FinalOptimizationTrial, ...]
    aggregate_trajectory: Tuple[Dict[str, Any], ...]
    best_seed: Optional[int]
    best_objective: Optional[float]
    best_configuration: Dict[str, Any]
    objective_mean: Optional[float]
    objective_median: Optional[float]
    objective_population_stddev: Optional[float]
    total_used_budget: int
    disclosure_cn: str = FINAL_OPTIMIZATION_DISCLOSURE_CN
    schema_version: str = FINAL_OPTIMIZATION_SCHEMA_VERSION
    artifact_id: str = ""

    @property
    def successful_seed_count(self) -> int:
        return sum(trial.status == "SUCCESS" for trial in self.trials)

    def to_dict(self, *, include_artifact_id: bool = True) -> Dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "report_identity": self.report_identity,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "source_candidate_id": self.source_candidate_id,
            "dataset_digest": self.dataset_digest,
            "evaluator_version": self.evaluator_version,
            "evaluation_parameters_version": self.evaluation_parameters_version,
            "objective_direction": self.objective_direction,
            "evolution_candidate_budget": self.evolution_candidate_budget,
            "final_budget_per_seed": self.final_budget_per_seed,
            "seeds": list(self.seeds),
            "status": self.status,
            "trials": [trial.to_dict() for trial in self.trials],
            "aggregate_trajectory": [dict(item) for item in self.aggregate_trajectory],
            "successful_seed_count": self.successful_seed_count,
            "best_seed": self.best_seed,
            "best_objective": self.best_objective,
            "best_configuration": dict(self.best_configuration),
            "objective_mean": self.objective_mean,
            "objective_median": self.objective_median,
            "objective_population_stddev": self.objective_population_stddev,
            "total_used_budget": self.total_used_budget,
            "disclosure_cn": self.disclosure_cn,
            "engineering_extension": True,
            "reference_executable_missing": True,
        }
        if include_artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload

    @classmethod
    def from_dict(
        cls, payload: Dict[str, Any], *, artifact_id: str = ""
    ) -> "FinalOptimizationReport":
        return cls(
            report_identity=str(payload["report_identity"]),
            run_id=str(payload["run_id"]),
            task_id=str(payload["task_id"]),
            source_candidate_id=str(payload["source_candidate_id"]),
            dataset_digest=str(payload["dataset_digest"]),
            evaluator_version=str(payload["evaluator_version"]),
            evaluation_parameters_version=str(
                payload["evaluation_parameters_version"]
            ),
            objective_direction=str(payload["objective_direction"]),
            evolution_candidate_budget=int(payload["evolution_candidate_budget"]),
            final_budget_per_seed=int(payload["final_budget_per_seed"]),
            seeds=tuple(int(item) for item in payload["seeds"]),
            status=str(payload["status"]),
            trials=tuple(
                FinalOptimizationTrial.from_dict(item)
                for item in payload.get("trials", [])
            ),
            aggregate_trajectory=tuple(
                dict(item) for item in payload.get("aggregate_trajectory", [])
            ),
            best_seed=(
                None if payload.get("best_seed") is None else int(payload["best_seed"])
            ),
            best_objective=(
                None
                if payload.get("best_objective") is None
                else float(payload["best_objective"])
            ),
            best_configuration=dict(payload.get("best_configuration", {})),
            objective_mean=(
                None
                if payload.get("objective_mean") is None
                else float(payload["objective_mean"])
            ),
            objective_median=(
                None
                if payload.get("objective_median") is None
                else float(payload["objective_median"])
            ),
            objective_population_stddev=(
                None
                if payload.get("objective_population_stddev") is None
                else float(payload["objective_population_stddev"])
            ),
            total_used_budget=int(payload.get("total_used_budget", 0)),
            disclosure_cn=str(
                payload.get("disclosure_cn", FINAL_OPTIMIZATION_DISCLOSURE_CN)
            ),
            schema_version=str(
                payload.get("schema_version", FINAL_OPTIMIZATION_SCHEMA_VERSION)
            ),
            artifact_id=artifact_id or str(payload.get("artifact_id", "")),
        )


class FinalOptimizationService:
    """在同一 durable store/预算账本上执行独立的多 seed 最终优化。

    每个 seed 使用一个稳定 Candidate clone。这样 store 当前“一 Candidate 一 Result”
    的事实模型仍成立，原进化 Candidate 的 fitness、状态和 artifact 永远不会被覆盖。
    """

    def __init__(
        self,
        store,
        evaluator,
        *,
        seeds: Sequence[int] = DEFAULT_FINAL_OPTIMIZATION_SEEDS,
        final_budget_per_seed: Optional[int] = None,
        evaluator_version: str = "",
        evaluation_parameters_version: str = "",
        max_attempts: int = 3,
        worker_id: str = "final-optimization-worker",
        lease_seconds: int = 30,
        clock: Callable[[], datetime] = utc_clock,
        sleeper: Callable[[float], None] = time.sleep,
        max_poll_seconds: float = 0.25,
        evaluation_execution_mode: str = INLINE_EVALUATION,
    ) -> None:
        normalized_seeds = tuple(sorted(_validate_seeds(seeds)))
        if final_budget_per_seed is not None:
            _positive_int(final_budget_per_seed, "final_budget_per_seed")
        _positive_int(max_attempts, "max_attempts")
        if max_poll_seconds <= 0:
            raise ValueError("max_poll_seconds 必须大于 0")
        self.store = store
        self.evaluator = evaluator
        self.seeds = normalized_seeds
        self.configured_final_budget = final_budget_per_seed
        self.evaluator_version = evaluator_version.strip() or _evaluator_version(
            evaluator
        )
        declared_parameters = (
            evaluation_parameters_version.strip()
            or _declared_evaluation_parameters_version(evaluator)
        )
        semantic_parameters = _evaluator_semantic_parameters(evaluator)
        self.evaluation_parameters_version = "{}:{}:{}".format(
            FINAL_OPTIMIZATION_SCHEMA_VERSION,
            declared_parameters or "default",
            semantic_parameters,
        )
        self.max_attempts = max_attempts
        self.clock = clock
        self.sleeper = sleeper
        self.max_poll_seconds = float(max_poll_seconds)
        self.queue = EvaluationQueueService(store, clock=clock)
        self.worker = EvaluationWorker(
            store,
            evaluator,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            clock=clock,
        )
        self.evaluation_execution_mode = normalize_evaluation_execution_mode(
            evaluation_execution_mode
        )
        self.evaluation_driver = DurableEvaluationJobDriver(
            store,
            self.worker,
            mode=self.evaluation_execution_mode,
            clock=clock,
            sleeper=sleeper,
            max_poll_seconds=self.max_poll_seconds,
        )

    def optimize(
        self,
        run: Run,
        task: OptimizationTask,
        selected: Candidate,
        dataset_digest: str,
        control_check=None,
    ) -> FinalOptimizationReport:
        """执行或恢复一次稳定的多 seed Final Optimization。"""

        self._validate_inputs(run, task, selected, dataset_digest)
        final_budget = (
            self.configured_final_budget
            or task.evaluation_budget * DEFAULT_FINAL_BUDGET_MULTIPLIER
        )
        if final_budget <= task.evaluation_budget:
            raise ValueError(
                "final_budget_per_seed 必须严格大于 evolution candidate_budget"
            )
        direction = _objective_direction(task.objective)
        identity = _report_identity(
            run,
            task,
            selected,
            str(dataset_digest),
            self.seeds,
            final_budget,
            self.evaluator_version,
            self.evaluation_parameters_version,
        )
        recovered = self._recover_report(run.id, identity)
        if recovered is not None:
            if recovered.status != "SUCCESS":
                raise FinalOptimizationFailed(recovered)
            return recovered

        clones = tuple(
            self._ensure_clone(
                run,
                task,
                selected,
                str(dataset_digest),
                seed,
                final_budget,
                identity,
            )
            for seed in self.seeds
        )
        existing_jobs = self._jobs_by_candidate(run.id)
        missing_job_count = sum(
            clone.id not in existing_jobs for clone in clones
        )
        self._preflight_budget(run.id, task, missing_job_count * final_budget)

        jobs = []
        for seed, clone in zip(self.seeds, clones):
            existing = existing_jobs.get(clone.id)
            if existing is not None:
                self._validate_existing_job(existing, seed, final_budget)
                jobs.append(existing)
                continue
            jobs.append(
                self.queue.submit(
                    run.id,
                    task.id,
                    clone.id,
                    seed=seed,
                    budget=final_budget,
                    max_attempts=self.max_attempts,
                    dataset_digest=str(dataset_digest),
                    evaluator_version=self.evaluator_version,
                    evaluation_parameters_version=(
                        self.evaluation_parameters_version
                    ),
                )
            )

        self._drive_jobs(
            run.id,
            tuple(job.id for job in jobs),
            control_check=control_check,
        )
        terminal_jobs = tuple(self.store.get_evaluation_job(job.id) for job in jobs)
        trials = tuple(
            self._trial(job) for job in sorted(terminal_jobs, key=lambda item: item.seed)
        )
        report = _build_report(
            identity=identity,
            run=run,
            task=task,
            selected=selected,
            dataset_digest=str(dataset_digest),
            evaluator_version=self.evaluator_version,
            evaluation_parameters_version=self.evaluation_parameters_version,
            direction=direction,
            final_budget=final_budget,
            seeds=self.seeds,
            trials=trials,
        )
        report = self._persist_report(report)
        if report.status != "SUCCESS":
            raise FinalOptimizationFailed(report)
        return report

    def _validate_inputs(self, run, task, selected, dataset_digest) -> None:
        if run.task_id != task.id:
            raise ValueError("Run.task_id 与 OptimizationTask.id 不一致")
        if selected.run_id != run.id:
            raise ValueError("selected Candidate 不属于当前 Run")
        if not isinstance(dataset_digest, str) or not dataset_digest.strip():
            raise ValueError("dataset_digest 不能为空")
        persisted_run = self.store.get_run(run.id)
        persisted_task = self.store.get_task(task.id)
        persisted_candidate = self.store.candidate_by_id(selected.id)
        if persisted_run.task_id != task.id or persisted_task.id != task.id:
            raise ValueError("durable store 中的 Run/Task 关联不一致")
        if persisted_candidate.run_id != run.id:
            raise ValueError("durable store 中的 selected Candidate 归属不一致")
        if persisted_candidate.code != selected.code:
            raise ValueError("selected Candidate code 与 durable store 不一致")
        if persisted_candidate.status != CandidateStatus.EVALUATED:
            raise ValueError("Final Optimization 只能接收已评价的 selected Candidate")

    def _ensure_clone(
        self,
        run,
        task,
        selected,
        dataset_digest,
        seed,
        final_budget,
        report_identity,
    ) -> Candidate:
        material = {
            "schema_version": FINAL_OPTIMIZATION_SCHEMA_VERSION,
            "run_id": run.id,
            "task_id": task.id,
            "source_candidate_id": selected.id,
            "source_code_digest": _sha256(selected.code.encode("utf-8")),
            "dataset_digest": dataset_digest,
            "seed": seed,
            "budget": final_budget,
            "evaluator_version": self.evaluator_version,
            "evaluation_parameters_version": self.evaluation_parameters_version,
        }
        clone_id = "final-opt-{}".format(_json_digest(material)[:24])
        try:
            existing = self.store.candidate_by_id(clone_id)
        except KeyError:
            existing = None
        if existing is not None:
            if (
                existing.run_id != run.id
                or existing.code != selected.code
                or existing.lineage.get("source_candidate_id") != selected.id
                or int(existing.lineage.get("final_seed", -1)) != seed
                or int(existing.lineage.get("final_budget", -1)) != final_budget
            ):
                raise FinalOptimizationError(
                    "稳定 Final Optimization Candidate id 发生语义碰撞：{}".format(
                        clone_id
                    )
                )
            return existing

        code_artifact = self.store.put_artifact(
            run.id,
            "FINAL_OPTIMIZATION_CANDIDATE_CODE",
            selected.code.encode("utf-8"),
            "text/x-python",
        )
        clone = Candidate(
            id=clone_id,
            run_id=run.id,
            code=selected.code,
            description=(
                "Final Optimization 工程补全 clone：source={} seed={} budget={}".format(
                    selected.id, seed, final_budget
                )
            ),
            operators=list(selected.operators),
            lineage={
                "creation_type": "FINAL_OPTIMIZATION",
                "engineering_extension": True,
                "reference_executable_missing": True,
                "source_candidate_id": selected.id,
                "final_seed": seed,
                "final_budget": final_budget,
                "dataset_digest": dataset_digest,
                "evaluator_version": self.evaluator_version,
                "evaluation_parameters_version": self.evaluation_parameters_version,
                "report_identity": report_identity,
            },
            status=CandidateStatus.VALIDATED,
            code_artifact_id=code_artifact.id,
        )
        self.store.add_candidate(clone)
        return clone

    def _jobs_by_candidate(self, run_id: str) -> Dict[str, EvaluationJob]:
        result = {}
        for job in self.store.evaluation_jobs_for_run(run_id):
            if not job.candidate_id.startswith("final-opt-"):
                continue
            prior = result.get(job.candidate_id)
            if prior is not None and prior.id != job.id:
                raise FinalOptimizationError(
                    "一个 Final Optimization clone 存在多个 logical job：{}".format(
                        job.candidate_id
                    )
                )
            result[job.candidate_id] = job
        return result

    @staticmethod
    def _validate_existing_job(job, seed, budget) -> None:
        if job.seed != seed or job.budget != budget:
            raise FinalOptimizationError(
                "已存在 Final Optimization job 的 seed/budget 与稳定身份不一致"
            )

    def _preflight_budget(self, run_id, task, needed_budget) -> None:
        if needed_budget <= 0:
            return
        refreshed = self.store.get_run(run_id)
        available = (
            task.total_budget
            - refreshed.consumed_evaluations
            - refreshed.reserved_evaluations
        )
        if needed_budget > available:
            raise FinalOptimizationBudgetError(
                "Final Optimization 预算不足：缺失 trial 需预留 {}，权威账本仅剩 {}"
                .format(needed_budget, available)
            )

    def _drive_jobs(
        self, run_id: str, job_ids: Tuple[str, ...], control_check=None
    ) -> None:
        if self.evaluation_execution_mode == INLINE_EVALUATION:
            self.worker.recover_stale()
        result = self.evaluation_driver.drive(
            run_id, job_ids, control_check=control_check
        )
        if result.cancelled:
            raise FinalOptimizationError(
                "Run 已取消，Final Optimization 停止等待"
            )

    def _bounded_wait(self, jobs: Iterable[EvaluationJob]) -> None:
        now = self.clock()
        deadlines = []
        ready_unclaimed = []
        for job in jobs:
            if job.status in _TERMINAL_JOB_STATUSES:
                continue
            if job.status in (
                EvaluationJobStatus.PENDING,
                EvaluationJobStatus.RETRY_WAIT,
            ):
                if job.available_at <= now:
                    ready_unclaimed.append(job.id)
                else:
                    deadlines.append(job.available_at)
            elif job.status == EvaluationJobStatus.RUNNING:
                if job.lease_expires_at is None or job.lease_expires_at <= now:
                    deadlines.append(now)
                else:
                    deadlines.append(job.lease_expires_at)
        if ready_unclaimed:
            raise FinalOptimizationError(
                "Final Optimization job 已 ready 但 worker 无法 claim：{}".format(
                    ", ".join(ready_unclaimed)
                )
            )
        if not deadlines:
            raise FinalOptimizationError("Final Optimization queue 无可等待的进度点")
        delay = max(0.001, (min(deadlines) - now).total_seconds())
        self.sleeper(min(delay, self.max_poll_seconds))

    def _trial(self, job: EvaluationJob) -> FinalOptimizationTrial:
        if job.status == EvaluationJobStatus.SUCCESS:
            try:
                result = self.store.result_for_candidate(job.candidate_id)
            except KeyError as exc:
                raise FinalOptimizationError(
                    "SUCCESS job 缺少 EvaluationResult：{}".format(job.id)
                ) from exc
            _validate_result(job, result)
            return FinalOptimizationTrial(
                seed=job.seed,
                candidate_id=job.candidate_id,
                job_id=job.id,
                status="SUCCESS",
                result_id=result.id,
                objective=float(result.objective),
                trajectory=tuple(float(item) for item in result.trajectory),
                best_configuration=dict(result.best_configuration),
                used_budget=int(result.used_budget),
            )
        return FinalOptimizationTrial(
            seed=job.seed,
            candidate_id=job.candidate_id,
            job_id=job.id,
            status=job.status.value,
            error_code=str(job.error_code or job.status.value),
            error_message=str(job.error_message or "EvaluationJob 未成功"),
        )

    def _recover_report(
        self, run_id: str, report_identity: str
    ) -> Optional[FinalOptimizationReport]:
        for artifact in self.store.artifacts_for_run(run_id):
            if artifact.kind != "FINAL_OPTIMIZATION_REPORT":
                continue
            payload = json.loads(self.store.artifact_content(artifact.id).decode("utf-8"))
            if payload.get("report_identity") != report_identity:
                continue
            report = FinalOptimizationReport.from_dict(
                payload, artifact_id=artifact.id
            )
            self._ensure_report_event(report)
            return report
        return None

    def _persist_report(
        self, report: FinalOptimizationReport
    ) -> FinalOptimizationReport:
        payload = report.to_dict(include_artifact_id=False)
        content = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        artifact = self.store.put_artifact(
            report.run_id,
            "FINAL_OPTIMIZATION_REPORT",
            content,
            "application/json",
        )
        persisted = replace(report, artifact_id=artifact.id)
        self._ensure_report_event(persisted)
        return persisted

    def _ensure_report_event(self, report: FinalOptimizationReport) -> None:
        event_type = (
            "FINAL_OPTIMIZATION_COMPLETED"
            if report.status == "SUCCESS"
            else "FINAL_OPTIMIZATION_FAILED"
        )
        for event in self.store.events_for_run(report.run_id):
            if (
                event.event_type == event_type
                and event.payload.get("report_identity") == report.report_identity
            ):
                return
        self.store.append_event(
            report.run_id,
            event_type,
            (
                "Final Optimization 多随机种子评估完成"
                if report.status == "SUCCESS"
                else "Final Optimization Candidate clone 评估失败，报告已持久化"
            ),
            report_identity=report.report_identity,
            report_artifact_id=report.artifact_id,
            source_candidate_id=report.source_candidate_id,
            status=report.status,
            seeds=list(report.seeds),
            final_budget_per_seed=report.final_budget_per_seed,
            total_used_budget=report.total_used_budget,
            engineering_extension=True,
            reference_executable_missing=True,
        )


def _build_report(
    *,
    identity,
    run,
    task,
    selected,
    dataset_digest,
    evaluator_version,
    evaluation_parameters_version,
    direction,
    final_budget,
    seeds,
    trials,
) -> FinalOptimizationReport:
    successful = tuple(trial for trial in trials if trial.status == "SUCCESS")
    objectives = [float(trial.objective) for trial in successful]
    best_trial = None
    if successful:
        selector = min if direction == "minimize" else max
        best_trial = selector(successful, key=lambda item: float(item.objective))
    status = "SUCCESS" if len(successful) == len(seeds) else "FAILED"
    return FinalOptimizationReport(
        report_identity=identity,
        run_id=run.id,
        task_id=task.id,
        source_candidate_id=selected.id,
        dataset_digest=dataset_digest,
        evaluator_version=evaluator_version,
        evaluation_parameters_version=evaluation_parameters_version,
        objective_direction=direction,
        evolution_candidate_budget=task.evaluation_budget,
        final_budget_per_seed=final_budget,
        seeds=tuple(seeds),
        status=status,
        trials=tuple(trials),
        aggregate_trajectory=_aggregate_trajectory(successful, direction),
        best_seed=(best_trial.seed if best_trial is not None else None),
        best_objective=(
            float(best_trial.objective) if best_trial is not None else None
        ),
        best_configuration=(
            dict(best_trial.best_configuration or {}) if best_trial is not None else {}
        ),
        objective_mean=(statistics.fmean(objectives) if objectives else None),
        objective_median=(statistics.median(objectives) if objectives else None),
        objective_population_stddev=(
            statistics.pstdev(objectives) if objectives else None
        ),
        total_used_budget=sum(trial.used_budget for trial in successful),
    )


def _aggregate_trajectory(successful, direction) -> Tuple[Dict[str, Any], ...]:
    if not successful:
        return ()
    length = max(len(trial.trajectory) for trial in successful)
    output = []
    for index in range(length):
        values = [
            float(trial.trajectory[index])
            for trial in successful
            if index < len(trial.trajectory)
        ]
        output.append(
            {
                "evaluation": index + 1,
                "seed_count": len(values),
                "mean": statistics.fmean(values),
                "best": min(values) if direction == "minimize" else max(values),
                "worst": max(values) if direction == "minimize" else min(values),
            }
        )
    return tuple(output)


def _validate_result(job: EvaluationJob, result: EvaluationResult) -> None:
    if result.candidate_id != job.candidate_id or result.run_id != job.run_id:
        raise FinalOptimizationError("EvaluationResult 与 Final Optimization job 不一致")
    if result.used_budget < 0 or result.used_budget > job.budget:
        raise FinalOptimizationError("EvaluationResult.used_budget 越过 job budget")
    if not math.isfinite(float(result.objective)):
        raise FinalOptimizationError("Final Optimization objective 必须为有限数值")
    if any(not math.isfinite(float(item)) for item in result.trajectory):
        raise FinalOptimizationError("Final Optimization trajectory 包含非有限数值")


def _report_identity(
    run,
    task,
    selected,
    dataset_digest,
    seeds,
    final_budget,
    evaluator_version,
    evaluation_parameters_version,
) -> str:
    material = {
        "schema_version": FINAL_OPTIMIZATION_SCHEMA_VERSION,
        "run_id": run.id,
        "task_id": task.id,
        "source_candidate_id": selected.id,
        "source_code_digest": _sha256(selected.code.encode("utf-8")),
        "dataset_digest": dataset_digest,
        "seeds": list(seeds),
        "evolution_candidate_budget": task.evaluation_budget,
        "final_budget_per_seed": final_budget,
        "evaluator_version": evaluator_version,
        "evaluation_parameters_version": evaluation_parameters_version,
    }
    return "final-report-{}".format(_json_digest(material)[:24])


def _validate_seeds(seeds: Sequence[int]) -> Tuple[int, ...]:
    values = tuple(seeds)
    if len(values) < 2:
        raise ValueError("Final Optimization 至少需要 2 个随机种子")
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Final Optimization seed 必须是整数")
    if len(set(values)) != len(values):
        raise ValueError("Final Optimization seeds 不得重复")
    return values


def required_final_optimization_budget(
    candidate_budget: int,
    *,
    seeds: Sequence[int] = DEFAULT_FINAL_OPTIMIZATION_SEEDS,
    multiplier: int = DEFAULT_FINAL_BUDGET_MULTIPLIER,
) -> int:
    """返回默认产品配置在权威账本中必须预留的 Final Optimization 预算。"""

    _positive_int(candidate_budget, "candidate_budget")
    _positive_int(multiplier, "multiplier")
    normalized = _validate_seeds(seeds)
    if multiplier <= 1:
        raise ValueError("Final Optimization multiplier 必须严格大于 1")
    return len(normalized) * int(candidate_budget) * int(multiplier)


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} 必须是正整数".format(name))


def _objective_direction(value: str) -> str:
    normalized = str(value).strip().casefold()
    if "max" in normalized:
        return "maximize"
    if "min" in normalized:
        return "minimize"
    raise ValueError("OptimizationTask.objective 必须明确包含 minimize 或 maximize")


def _evaluator_version(evaluator) -> str:
    declared = getattr(evaluator, "version", None)
    if declared is not None and str(declared).strip():
        return str(declared).strip()
    evaluator_type = type(evaluator)
    return "{}.{}".format(evaluator_type.__module__, evaluator_type.__qualname__)


def _declared_evaluation_parameters_version(evaluator) -> str:
    for attribute in (
        "evaluation_parameters_version",
        "parameters_version",
        "parameter_version",
    ):
        declared = getattr(evaluator, attribute, None)
        if declared is not None and str(declared).strip():
            return str(declared).strip()
    return ""


def _evaluator_semantic_parameters(evaluator) -> str:
    """指纹化会改变 benchmark 结果/隔离边界的真实参数。

    ``evaluation_parameters_version`` 是人工声明，不能代替 timeout、memory、
    max_lives 等实际值。对测试/监控 wrapper 只向内读取一层 delegate；不读取
    cache、calls 等运行态字段。
    """

    sources = [evaluator]
    delegate = getattr(evaluator, "delegate", None)
    if delegate is not None and delegate is not evaluator:
        sources.append(delegate)
    material = {}
    for attribute in ("timeout_seconds", "memory_limit_mb", "max_lives"):
        for source in sources:
            value = getattr(source, attribute, None)
            if isinstance(value, (str, int, float, bool)) and not isinstance(
                value, type(None)
            ):
                material[attribute] = value
                break
    if not material:
        return "semantics-default"
    return "semantics-{}".format(_json_digest(material)[:16])


def _json_digest(value: Dict[str, Any]) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "FINAL_OPTIMIZATION_DISCLOSURE_CN",
    "FinalOptimizationBudgetError",
    "FinalOptimizationError",
    "FinalOptimizationFailed",
    "FinalOptimizationReport",
    "FinalOptimizationService",
    "FinalOptimizationTrial",
]
