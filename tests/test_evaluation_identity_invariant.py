from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.domain.errors import EvaluationIdentityConflictError
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


class _RecordingEvaluator:
    version = "identity-evaluator-v1"
    evaluation_parameters_version = "identity-params-v1"

    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, task):
        self.calls += 1
        return EvaluationResult(
            "result-{}".format(candidate.id),
            candidate.run_id,
            candidate.id,
            0.25,
            [0.8, 0.25],
            {"seed": task.random_seed},
            2,
        )


class EvaluationIdentityInvariantTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prievo-eval-identity-")
        root = Path(self.temp.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
        self.now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        self.task = OptimizationTask(
            "task-identity", "identity", "minimize", 3, 12
        )
        self.run = Run(
            "run-identity", self.task.id, status=RunStatus.RUNNING
        )
        self.candidate = Candidate(
            "candidate-identity",
            self.run.id,
            "def run_tuners(file, budget, seed, maxlives): return 1",
            "identity fixture",
            ["fixture"],
            {},
        )
        self.store.add_task(self.task)
        self.store.add_run(self.run)
        self.store.add_candidate(self.candidate)
        self.queue = EvaluationQueueService(self.store, clock=lambda: self.now)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def _submit(self, **overrides):
        values = {
            "seed": 11,
            "budget": 3,
            "dataset_digest": "dataset-a",
            "evaluator_version": "evaluator-a",
            "evaluation_parameters_version": "params-a",
        }
        values.update(overrides)
        return self.queue.submit(
            self.run.id, self.task.id, self.candidate.id, **values
        )

    def _assert_unchanged(self, expected_job, expected_run, expected_candidate):
        jobs = list(self.store.evaluation_jobs_for_run(self.run.id))
        self.assertEqual([expected_job.id], [job.id for job in jobs])
        current_run = self.store.get_run(self.run.id)
        self.assertEqual(
            (expected_run.consumed_evaluations, expected_run.reserved_evaluations),
            (current_run.consumed_evaluations, current_run.reserved_evaluations),
        )
        current_candidate = self.store.candidate_by_id(self.candidate.id)
        self.assertEqual(expected_candidate.status, current_candidate.status)
        self.assertEqual(expected_candidate.objective, current_candidate.objective)
        self.assertEqual(
            expected_candidate.evaluation_artifact_id,
            current_candidate.evaluation_artifact_id,
        )

    def test_success_same_identity_replay_and_changed_identity_are_safe(self):
        baseline = self._submit()
        duplicate_before_success = self._submit()
        self.assertEqual(baseline.id, duplicate_before_success.id)
        evaluator = _RecordingEvaluator()
        completed = EvaluationWorker(
            self.store,
            evaluator,
            worker_id="identity-worker",
            clock=lambda: self.now,
        ).run_once()
        self.assertEqual(EvaluationJobStatus.SUCCESS, completed.status)
        self.assertEqual(1, evaluator.calls)

        duplicate_after_success = self._submit()
        self.assertEqual(baseline.id, duplicate_after_success.id)
        authoritative = self.store.result_for_candidate(self.candidate.id)
        self.assertEqual(
            baseline.id,
            self.queue.assert_candidate_identity(
                self.run.id,
                self.task.id,
                self.candidate.id,
                seed=11,
                budget=3,
                dataset_digest="dataset-a",
                evaluator_version="evaluator-a",
                evaluation_parameters_version="params-a",
            ).id,
        )
        expected_run = self.store.get_run(self.run.id)
        expected_candidate = self.store.candidate_by_id(self.candidate.id)
        self.assertEqual(2, expected_run.consumed_evaluations)
        self.assertEqual(0, expected_run.reserved_evaluations)

        changes = (
            {"dataset_digest": "dataset-b"},
            {"evaluator_version": "evaluator-b"},
            {"evaluation_parameters_version": "params-b"},
            {"seed": 12},
            {"budget": 2},
        )
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaises(EvaluationIdentityConflictError):
                    self._submit(**change)
                expected = {
                    "seed": 11,
                    "budget": 3,
                    "dataset_digest": "dataset-a",
                    "evaluator_version": "evaluator-a",
                    "evaluation_parameters_version": "params-a",
                }
                expected.update(change)
                with self.assertRaises(EvaluationIdentityConflictError):
                    self.queue.assert_candidate_identity(
                        self.run.id,
                        self.task.id,
                        self.candidate.id,
                        **expected,
                    )
                self._assert_unchanged(
                    baseline, expected_run, expected_candidate
                )
                self.assertEqual(
                    authoritative,
                    self.store.result_for_candidate(self.candidate.id),
                )

        started = [
            event for event in self.store.events_for_run(self.run.id)
            if event.event_type == "EVALUATION_STARTED"
        ]
        self.assertEqual(1, len(started))
        self.assertEqual(baseline.id, started[0].payload["job_id"])
        self.assertEqual(self.candidate.id, started[0].payload["candidate_id"])
        self.assertEqual("identity-worker", started[0].payload["worker_id"])
        self.assertEqual(1, started[0].payload["attempt"])
        self.assertTrue(started[0].payload["lease_expires_at"])

    def test_conflict_is_rejected_before_budget_reservation(self):
        baseline = self._submit()
        expected_run = self.store.get_run(self.run.id)
        expected_candidate = self.store.candidate_by_id(self.candidate.id)
        with self.assertRaises(EvaluationIdentityConflictError):
            self._submit(dataset_digest="changed-before-worker")
        self._assert_unchanged(baseline, expected_run, expected_candidate)
        with self.assertRaises(KeyError):
            self.store.result_for_candidate(self.candidate.id)

    def test_identity_isolated_by_clone_candidate(self):
        baseline = self._submit()
        clone = Candidate(
            "candidate-identity-clone",
            self.run.id,
            self.candidate.code,
            "explicit reevaluation clone",
            list(self.candidate.operators),
            {"creation_type": "REEVALUATION_CLONE",
             "source_candidate_id": self.candidate.id},
        )
        self.store.add_candidate(clone)
        clone_job = self.queue.submit(
            self.run.id,
            self.task.id,
            clone.id,
            seed=12,
            budget=3,
            dataset_digest="dataset-b",
            evaluator_version="evaluator-b",
            evaluation_parameters_version="params-b",
        )
        self.assertNotEqual(baseline.id, clone_job.id)
        self.assertEqual(2, len(list(self.store.evaluation_jobs_for_run(self.run.id))))
        self.assertEqual(6, self.store.get_run(self.run.id).reserved_evaluations)

    def test_reconcile_identity_check_rejects_result_without_job(self):
        legacy = Candidate(
            "candidate-legacy-result",
            self.run.id,
            self.candidate.code,
            "legacy fixture",
            [],
            {},
        )
        self.store.add_candidate(legacy)
        self.store.add_result(EvaluationResult(
            "result-legacy", self.run.id, legacy.id, 0.4, [0.4], {}, 1
        ))
        with self.assertRaisesRegex(
            EvaluationIdentityConflictError, "恰好关联一个"
        ):
            self.queue.assert_candidate_identity(
                self.run.id,
                self.task.id,
                legacy.id,
                seed=11,
                budget=3,
                dataset_digest="dataset-a",
                evaluator_version="evaluator-a",
                evaluation_parameters_version="params-a",
            )

    def test_cancelled_running_job_rejects_late_settlement_then_sweeps_once(self):
        baseline = self._submit()
        claimed = self.store.claim_next_job("late-worker", self.now, 10)
        self.assertEqual(baseline.id, claimed.id)
        self.store.request_run_cancel(self.run.id, "cancel during benchmark")
        cancelled = self.store.get_run(self.run.id)
        self.assertEqual(RunStatus.CANCELLED, cancelled.status)
        self.assertEqual(3, cancelled.reserved_evaluations)

        artifact = self.store.put_artifact(
            self.run.id, "EVALUATION_RESULT", b"late", "application/json"
        )
        result = EvaluationResult(
            "late-result", self.run.id, self.candidate.id,
            0.01, [0.01], {}, 1,
        )
        late_now = self.now + timedelta(seconds=1)
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.store.complete_job_success(
                baseline.id, "late-worker", result, artifact.id, late_now
            )
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.store.complete_job_failure(
                baseline.id, "late-worker", "LATE", "late", False,
                late_now, late_now,
            )
        self.assertEqual(3, self.store.get_run(self.run.id).reserved_evaluations)
        self.assertEqual(
            1,
            self.store.recover_stale_jobs(
                self.now + timedelta(seconds=11), run_id=self.run.id
            ),
        )
        reconciled = self.store.get_run(self.run.id)
        self.assertEqual(0, reconciled.reserved_evaluations)
        self.assertEqual(0, reconciled.consumed_evaluations)
        self.assertEqual(
            EvaluationJobStatus.CANCELLED,
            self.store.get_evaluation_job(baseline.id).status,
        )
        self.assertEqual(
            0,
            self.store.recover_stale_jobs(
                self.now + timedelta(seconds=12), run_id=self.run.id
            ),
        )


if __name__ == "__main__":
    unittest.main()
