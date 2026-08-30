from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Dict, Tuple

from prievo_agent.domain.errors import BudgetExhaustedError
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    Candidate,
    EvaluationJobStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.testing.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore

from .evaluation_queue import (
    EvaluationQueueService,
    EvaluationWorker,
    InjectedFailureEvaluator,
    WorkerCrashed,
)


class ManualClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


@dataclass(frozen=True)
class QueueHarnessReport:
    retry_attempts: int
    duplicate_jobs_created: int
    budget_overrun_blocked: bool
    invalid_status: str
    timeout_status: str
    crash_recovered_status: str
    crash_benchmark_calls: int
    crash_logical_results: int
    crash_budget_charged: int
    concurrent_budget_winners: int

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class QueueReliabilityHarness:
    def run(self, root: Path) -> QueueHarnessReport:
        root = Path(root)
        retry_attempts, duplicate_count, budget_blocked = self._retry_duplicate_budget(
            root / "retry"
        )
        invalid_status = self._terminal_failure(root / "invalid", "invalid", 3)
        timeout_status = self._terminal_failure(root / "timeout", "timeout", 2)
        crash_status, crash_calls, crash_results, crash_budget = self._crash_recovery(
            root / "crash"
        )
        concurrent_winners = self._concurrent_budget_contention(
            root / "concurrent-budget"
        )
        return QueueHarnessReport(
            retry_attempts,
            duplicate_count,
            budget_blocked,
            invalid_status,
            timeout_status,
            crash_status,
            crash_calls,
            crash_results,
            crash_budget,
            concurrent_winners,
        )

    def _setup(
        self, root: Path, run_id: str, total_budget: int
    ) -> Tuple[SQLiteRuntimeStore, OptimizationTask, Run]:
        root.mkdir(parents=True, exist_ok=True)
        store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        task = OptimizationTask(
            "task-{}".format(run_id), "队列 Harness", "minimize", 3, total_budget
        )
        run = Run(run_id, task.id, status=RunStatus.RUNNING)
        store.add_task(task)
        store.add_run(run)
        store.append_event(run.id, EventType.RUN_CREATED.value, "运行已创建")
        return store, task, run

    def _candidate(
        self, store: SQLiteRuntimeStore, run: Run, identifier: str
    ) -> Candidate:
        candidate = Candidate(
            identifier,
            run.id,
            "def run_tuners(): return {!r}".format(identifier),
            identifier,
            ["Harness Operator"],
            {"operator": "harness", "generation": 0},
        )
        artifact = store.put_artifact(
            run.id, "CANDIDATE_CODE", candidate.code.encode("utf-8"), "text/x-python"
        )
        candidate.code_artifact_id = artifact.id
        store.add_candidate(candidate)
        return candidate

    def _retry_duplicate_budget(self, root: Path) -> Tuple[int, int, bool]:
        store, task, run = self._setup(root, "run-retry", 6)
        clock = ManualClock()
        transient = self._candidate(store, run, "candidate-transient")
        success = self._candidate(store, run, "candidate-success")
        overflow = self._candidate(store, run, "candidate-overflow")
        evaluator = InjectedFailureEvaluator(
            FakeEvaluator(), {transient.id: ["transient", "success"]}
        )
        queue = EvaluationQueueService(store, clock)
        worker = EvaluationWorker(store, evaluator, clock=clock)
        first = queue.submit(run.id, task.id, transient.id, 101, 3)
        duplicate = queue.submit(run.id, task.id, transient.id, 101, 3)
        if first.id != duplicate.id:
            raise AssertionError("重复提交未返回同一 logical job")
        queue.submit(run.id, task.id, success.id, 101, 3)
        blocked = False
        try:
            queue.submit(run.id, task.id, overflow.id, 101, 3)
        except BudgetExhaustedError:
            blocked = True

        for _ in range(2):
            worker.run_once()
        clock.advance(2)
        while worker.run_once() is not None:
            pass
        completed = store.get_evaluation_job(first.id)
        events = list(store.events_for_run(run.id))
        duplicates = sum(
            1
            for event in events
            if event.event_type == EventType.DUPLICATE_EVALUATION_IGNORED.value
        )
        result = (completed.attempts, duplicates, blocked)
        store.close()
        return result

    def _concurrent_budget_contention(self, root: Path) -> int:
        """两个独立连接同时争抢最后一份预算，事务上只能有一个赢家。"""

        store, task, run = self._setup(root, "run-concurrent-budget", 3)
        candidates = [
            self._candidate(store, run, "candidate-concurrent-{}".format(index))
            for index in range(2)
        ]
        database_path = store.database_path
        artifact_root = store.artifact_root
        store.close()

        barrier = threading.Barrier(3)
        outcomes = []
        lock = threading.Lock()

        def submit(candidate):
            contender = SQLiteRuntimeStore(
                database_path, artifact_root, ensure_schema=False
            )
            try:
                barrier.wait(timeout=5)
                EvaluationQueueService(contender).submit(
                    run.id, task.id, candidate.id, 101, 3
                )
                outcome = "reserved"
            except BudgetExhaustedError:
                outcome = "blocked"
            finally:
                contender.close()
            with lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=submit, args=(candidate,))
            for candidate in candidates
        ]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=10)
            if thread.is_alive():
                raise AssertionError("并发预算 Harness 线程未结束")

        verify = SQLiteRuntimeStore(
            database_path, artifact_root, ensure_schema=False
        )
        try:
            winners = outcomes.count("reserved")
            if outcomes.count("blocked") != 1 or winners != 1:
                raise AssertionError(
                    "并发预算预留应恰好一个成功、一个拒绝：{}".format(outcomes)
                )
            if verify.get_run(run.id).reserved_evaluations != 3:
                raise AssertionError("并发预算预留后的 durable reserved 不等于 3")
            if len(list(verify.evaluation_jobs_for_run(run.id))) != 1:
                raise AssertionError("并发预算争用创建了错误数量的 EvaluationJob")
            return winners
        finally:
            verify.close()

    def _terminal_failure(self, root: Path, action: str, max_attempts: int) -> str:
        store, task, run = self._setup(root, "run-{}".format(action), 3)
        clock = ManualClock()
        candidate = self._candidate(store, run, "candidate-{}".format(action))
        evaluator = InjectedFailureEvaluator(FakeEvaluator(), {candidate.id: [action]})
        queue = EvaluationQueueService(store, clock)
        worker = EvaluationWorker(store, evaluator, clock=clock)
        job = queue.submit(
            run.id, task.id, candidate.id, 101, 3, max_attempts=max_attempts
        )
        while True:
            worker.run_once()
            current = store.get_evaluation_job(job.id)
            if current.status == EvaluationJobStatus.DEAD:
                break
            clock.advance(4)
        if store.get_run(run.id).reserved_evaluations != 0:
            raise AssertionError("dead letter 未释放预算预留")
        store.close()
        return current.status.value

    def _crash_recovery(self, root: Path) -> Tuple[str, int, int, int]:
        store, task, run = self._setup(root, "run-crash", 3)
        clock = ManualClock()
        candidate = self._candidate(store, run, "candidate-crash")
        evaluator = InjectedFailureEvaluator(FakeEvaluator(), {candidate.id: ["success"]})
        queue = EvaluationQueueService(store, clock)
        worker = EvaluationWorker(store, evaluator, lease_seconds=30, clock=clock)
        job = queue.submit(run.id, task.id, candidate.id, 101, 3)
        try:
            worker.run_once(crash_after_benchmark=True)
        except WorkerCrashed:
            pass
        if store.get_evaluation_job(job.id).status != EvaluationJobStatus.RUNNING:
            raise AssertionError("worker crash 后 job 应保持 RUNNING 到 lease 过期")
        clock.advance(31)
        if worker.recover_stale() != 1:
            raise AssertionError("未恢复过期 lease")
        worker.run_once()
        current = store.get_evaluation_job(job.id)
        logical_results = 1 if store.result_for_candidate(candidate.id) else 0
        charged = store.get_run(run.id).consumed_evaluations
        result = (
            current.status.value,
            evaluator.calls[candidate.id],
            logical_results,
            charged,
        )
        store.close()
        return result
