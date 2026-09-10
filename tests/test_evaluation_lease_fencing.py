from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.domain.models import (
    Candidate,
    CandidateStatus,
    EvaluationJobStatus,
    EvaluationResult,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore
from prievo_agent.evaluation.queue import (
    EvaluationQueueService,
    EvaluationWorker,
)


class ManualClock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current

    def advance(self, seconds):
        self.current += timedelta(seconds=seconds)


class RecordingSQLiteStore(SQLiteRuntimeStore):
    def __init__(self, database_path, artifact_root):
        super().__init__(database_path, artifact_root)
        self.renewals = []

    def renew_job_lease(self, job_id, worker_id, now, lease_seconds):
        self.renewals.append((job_id, worker_id, now, lease_seconds))
        return super().renew_job_lease(job_id, worker_id, now, lease_seconds)


class AdvancingEvaluator:
    def __init__(self, clock):
        self.clock = clock

    def evaluate(self, candidate, task):
        self.clock.advance(3)
        return EvaluationResult(
            id="result-worker-renewal",
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            objective=0.4,
            trajectory=[0.8, 0.4],
            best_configuration={"source": "worker-renewal-test"},
            used_budget=2,
        )


class TaskOverrideEvaluator:
    def __init__(self):
        self.received = []

    def evaluate(self, candidate, task):
        self.received.append((task.random_seed, task.evaluation_budget))
        return EvaluationResult(
            id="result-task-override",
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            objective=0.3,
            trajectory=[0.3],
            best_configuration={"seed": task.random_seed},
            used_budget=task.evaluation_budget,
        )


class EvaluationLeaseFencingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="prievo-evaluation-fencing-"
        )
        root = Path(self.temporary.name)
        self.store = SQLiteRuntimeStore(
            root / "state.sqlite3", root / "artifacts"
        )
        self.now = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
        self.task = OptimizationTask(
            "task-fencing", "lease fencing", "minimize", 3, 10
        )
        self.run = Run("run-fencing", self.task.id, status=RunStatus.RUNNING)
        self.candidate = Candidate(
            "candidate-fencing",
            self.run.id,
            "def run_tuners(file, budget, seed, maxlives): return 1",
            "lease fencing fixture",
            ["fixture"],
            {},
        )
        self.store.add_task(self.task)
        self.store.add_run(self.run)
        self.store.add_candidate(self.candidate)
        self.job = EvaluationQueueService(
            self.store, clock=lambda: self.now
        ).submit(
            self.run.id,
            self.task.id,
            self.candidate.id,
            seed=7,
            budget=3,
            max_attempts=3,
        )

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def _result(self, identifier, objective):
        return EvaluationResult(
            id=identifier,
            run_id=self.run.id,
            candidate_id=self.candidate.id,
            objective=objective,
            trajectory=[objective],
            best_configuration={"result": identifier},
            used_budget=2,
        )

    def test_expired_owner_cannot_settle_after_another_worker_claims(self):
        claimed_a = self.store.claim_next_job("worker-a", self.now, 10)
        self.assertEqual("worker-a", claimed_a.worker_id)
        self.assertEqual(EvaluationJobStatus.RUNNING, claimed_a.status)

        # 当前 owner 可在过期前续租；错误 owner 与已经过期的 owner 都不能续租。
        self.assertTrue(
            self.store.renew_job_lease(
                self.job.id, "worker-a", self.now + timedelta(seconds=5), 10
            )
        )
        self.assertFalse(
            self.store.renew_job_lease(
                self.job.id, "worker-x", self.now + timedelta(seconds=6), 10
            )
        )
        expired_at = self.now + timedelta(seconds=16)
        self.assertFalse(
            self.store.renew_job_lease(
                self.job.id, "worker-a", expired_at, 10
            )
        )

        self.assertEqual(1, self.store.recover_stale_jobs(expired_at))
        claimed_b = self.store.claim_next_job("worker-b", expired_at, 10)
        self.assertEqual("worker-b", claimed_b.worker_id)
        self.assertEqual(2, claimed_b.attempts)

        stale_artifact = self.store.put_artifact(
            self.run.id, "EVALUATION_RESULT", b"stale-a", "application/json"
        )
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.complete_job_success(
                self.job.id,
                "worker-a",
                self._result("result-stale-a", 0.01),
                stale_artifact.id,
                expired_at,
            )
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.complete_job_failure(
                self.job.id,
                "worker-a",
                "STALE_FAILURE",
                "worker-a 已过期",
                False,
                expired_at,
                expired_at,
            )

        # A 的 success/failure 均不得写 result、改 candidate 或释放预算。
        self.assertEqual(
            0,
            self.store.connection.execute(
                "SELECT COUNT(*) FROM evaluation_results"
            ).fetchone()[0],
        )
        current = self.store.get_evaluation_job(self.job.id)
        self.assertEqual(EvaluationJobStatus.RUNNING, current.status)
        self.assertEqual("worker-b", current.worker_id)
        run = self.store.get_run(self.run.id)
        self.assertEqual(3, run.reserved_evaluations)
        self.assertEqual(0, run.consumed_evaluations)
        candidate = self.store.candidate_by_id(self.candidate.id)
        self.assertEqual(CandidateStatus.EVALUATING, candidate.status)
        self.assertIsNone(candidate.objective)

        self.assertFalse(
            self.store.renew_job_lease(
                self.job.id, "worker-a", expired_at + timedelta(seconds=1), 10
            )
        )
        self.assertTrue(
            self.store.renew_job_lease(
                self.job.id, "worker-b", expired_at + timedelta(seconds=1), 10
            )
        )

        winner_artifact = self.store.put_artifact(
            self.run.id, "EVALUATION_RESULT", b"winner-b", "application/json"
        )
        completed = self.store.complete_job_success(
            self.job.id,
            "worker-b",
            self._result("result-winner-b", 0.25),
            winner_artifact.id,
            expired_at + timedelta(seconds=2),
        )
        self.assertEqual(EvaluationJobStatus.SUCCESS, completed.status)
        self.assertEqual("result-winner-b", completed.result_id)
        self.assertIsNone(completed.worker_id)
        self.assertFalse(
            self.store.renew_job_lease(
                self.job.id, "worker-b", expired_at + timedelta(seconds=2), 10
            )
        )
        with self.assertRaisesRegex(RuntimeError, "lease"):
            self.store.complete_job_success(
                self.job.id,
                "worker-b",
                self._result("result-duplicate-b", 0.20),
                winner_artifact.id,
                expired_at + timedelta(seconds=2),
            )

        self.assertEqual(
            ["result-winner-b"],
            [
                row["id"]
                for row in self.store.connection.execute(
                    "SELECT id FROM evaluation_results ORDER BY id"
                ).fetchall()
            ],
        )
        final_run = self.store.get_run(self.run.id)
        self.assertEqual(0, final_run.reserved_evaluations)
        self.assertEqual(2, final_run.consumed_evaluations)
        self.assertEqual(
            0.25, self.store.candidate_by_id(self.candidate.id).objective
        )

    def test_worker_renews_before_and_after_evaluator(self):
        self.store.close()
        root = Path(self.temporary.name)
        self.store = RecordingSQLiteStore(
            root / "worker.sqlite3", root / "worker-artifacts"
        )
        clock = ManualClock(self.now)
        task = OptimizationTask(
            "task-worker-renewal", "worker renewal", "minimize", 3, 10
        )
        run = Run("run-worker-renewal", task.id, status=RunStatus.RUNNING)
        candidate = Candidate(
            "candidate-worker-renewal",
            run.id,
            "def run_tuners(file, budget, seed, maxlives): return 1",
            "worker renewal fixture",
            ["fixture"],
            {},
        )
        self.store.add_task(task)
        self.store.add_run(run)
        self.store.add_candidate(candidate)
        EvaluationQueueService(self.store, clock=clock).submit(
            run.id, task.id, candidate.id, seed=9, budget=3
        )

        completed = EvaluationWorker(
            self.store,
            AdvancingEvaluator(clock),
            worker_id="worker-heartbeat",
            lease_seconds=10,
            clock=clock,
        ).run_once()

        self.assertEqual(EvaluationJobStatus.SUCCESS, completed.status)
        self.assertEqual(2, len(self.store.renewals))
        self.assertEqual(
            ["worker-heartbeat", "worker-heartbeat"],
            [renewal[1] for renewal in self.store.renewals],
        )
        self.assertEqual(self.now, self.store.renewals[0][2])
        self.assertEqual(
            self.now + timedelta(seconds=3), self.store.renewals[1][2]
        )

    def test_any_terminal_failure_marks_candidate_invalid(self):
        claimed = self.store.claim_next_job("worker-interface", self.now, 10)
        self.assertEqual(EvaluationJobStatus.RUNNING, claimed.status)

        completed = self.store.complete_job_failure(
            self.job.id,
            "worker-interface",
            "INTERFACE_ERROR",
            "candidate 返回接口不符合约定",
            False,
            self.now,
            self.now,
        )

        self.assertEqual(EvaluationJobStatus.DEAD, completed.status)
        self.assertEqual(
            CandidateStatus.INVALID,
            self.store.candidate_by_id(self.candidate.id).status,
        )
        run = self.store.get_run(self.run.id)
        self.assertEqual(0, run.reserved_evaluations)
        self.assertEqual(0, run.consumed_evaluations)

    def test_worker_uses_job_seed_and_budget_instead_of_task_defaults(self):
        # setUp 的 Task defaults 是 seed=2024/budget=3；另建 job 以证明 worker
        # 把 logical EvaluationJob 的语义传进 evaluator。
        evaluator = TaskOverrideEvaluator()
        worker = EvaluationWorker(
            self.store,
            evaluator,
            worker_id="worker-task-override",
            clock=lambda: self.now,
        )
        # 当前 job seed=7，budget=3；先取消它，创建 budget 不同的新 logical job。
        claimed = self.store.claim_next_job("cancel-owner", self.now, 10)
        self.store.complete_job_failure(
            claimed.id, "cancel-owner", "FIXTURE", "fixture", False,
            self.now, self.now,
        )
        candidate = Candidate(
            "candidate-task-override", self.run.id,
            "def run_tuners(file, budget, seed, maxlives): return 1",
            "task override", ["fixture"], {},
        )
        self.store.add_candidate(candidate)
        EvaluationQueueService(self.store, clock=lambda: self.now).submit(
            self.run.id, self.task.id, candidate.id, seed=9876, budget=2
        )

        completed = worker.run_once()

        self.assertEqual(EvaluationJobStatus.SUCCESS, completed.status)
        self.assertEqual([(9876, 2)], evaluator.received)
        self.assertEqual(
            9876,
            self.store.result_for_candidate(candidate.id).best_configuration["seed"],
        )


if __name__ == "__main__":
    unittest.main()
