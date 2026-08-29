from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    EvaluationIdentityConflictError,
    TransientEvaluationError,
)
from prievo_agent.domain.models import (
    Candidate,
    CheckpointMetadata,
    EvaluationJobStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import EvaluationQueueService
from prievo_agent.runtime.persistent_runtime import (
    CooperativePause,
    PersistentEvolutionRuntime,
)


class _ManualClock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.sleep_calls = []
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleep_calls.append(float(seconds))
        self.now += timedelta(seconds=float(seconds))
        if self.on_sleep is not None:
            self.on_sleep()


class _TransientOnceEvaluator:
    version = "transient-fixture-v1"
    parameters_version = "fixture-parameters-v1"

    def __init__(self):
        self.delegate = FakeEvaluator()
        self.calls = {}

    def evaluate(self, candidate, task):
        count = self.calls.get(candidate.id, 0) + 1
        self.calls[candidate.id] = count
        if count == 1:
            raise TransientEvaluationError("fixture transient infrastructure")
        return self.delegate.evaluate(candidate, task)


class _AlwaysFailureEvaluator:
    version = "algorithm-failure-fixture-v1"

    def __init__(self, exception):
        self.exception = exception
        self.calls = 0

    def evaluate(self, candidate, task):
        self.calls += 1
        raise self.exception


class _NeverClaimWorker:
    def run_once(self):
        return None


class _FirstAttemptThenNeverClaimWorker:
    def __init__(self, delegate):
        self.delegate = delegate
        self.calls = 0

    def run_once(self):
        self.calls += 1
        if self.calls == 1:
            return self.delegate.run_once()
        return None


def _runtime(root, evaluator, clock, suffix="main"):
    store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
    task = OptimizationTask(
        "task-" + suffix,
        "retry scheduler " + suffix,
        "minimize",
        evaluation_budget=3,
        total_budget=3,
        dataset_id="fixture-dataset",
        generations=0,
        population_size=1,
        random_seed=7,
    )
    run = Run("run-" + suffix, task.id, dataset_id=task.dataset_id)
    store.add_task(task)
    store.add_run(run)
    runtime = PersistentEvolutionRuntime(
        store,
        evaluator,
        PriEvoEvolutionCore(FakeLLM(), population_size=1, seed=7),
        total_generations=0,
        dataset_digest="dataset-digest-v1",
        clock=clock,
        sleeper=clock.sleep,
        retry_wait_poll_seconds=0.2,
    )
    return store, task, run, runtime


class RuntimeRetrySchedulerTest(unittest.TestCase):
    def test_transient_retry_waits_without_spin_then_mainline_succeeds_once(self):
        with tempfile.TemporaryDirectory(prefix="prievo-runtime-retry-") as directory:
            clock = _ManualClock()
            evaluator = _TransientOnceEvaluator()
            store, task, run, runtime = _runtime(
                Path(directory), evaluator, clock, "transient"
            )
            try:
                completed = runtime.execute(run.id)
                jobs = list(store.evaluation_jobs_for_run(run.id))
                candidates = list(store.candidates_for_run(run.id))
                result_count = store.connection.execute(
                    "SELECT COUNT(*) AS count FROM evaluation_results WHERE run_id=?",
                    (run.id,),
                ).fetchone()["count"]
                candidate_count = store.connection.execute(
                    "SELECT COUNT(*) AS count FROM candidates WHERE run_id=?",
                    (run.id,),
                ).fetchone()["count"]
                events = list(store.events_for_run(run.id))

                self.assertEqual(RunStatus.COMPLETED, completed.status)
                self.assertEqual(1, len(jobs))
                self.assertEqual(EvaluationJobStatus.SUCCESS, jobs[0].status)
                self.assertEqual(2, jobs[0].attempts)
                self.assertEqual(1, candidate_count)
                self.assertEqual(1, result_count)
                self.assertEqual(1, len(candidates))
                self.assertEqual(2, evaluator.calls[candidates[0].id])
                self.assertEqual(3, completed.consumed_evaluations)
                self.assertEqual(0, completed.reserved_evaluations)
                self.assertTrue(clock.sleep_calls)
                self.assertTrue(all(value > 0 for value in clock.sleep_calls))
                self.assertLessEqual(max(clock.sleep_calls), 0.2)
                self.assertEqual(
                    1,
                    sum(
                        event.event_type == "EVALUATION_RETRY_SCHEDULED"
                        for event in events
                    ),
                )
            finally:
                store.close()

    def test_candidate_identity_rejects_dataset_evaluator_and_parameter_changes(self):
        with tempfile.TemporaryDirectory(prefix="prievo-identity-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            clock = _ManualClock()
            task = OptimizationTask("task-id", "identity", "minimize", 1, 10)
            run = Run("run-id", task.id)
            candidate = Candidate(
                "candidate-id",
                run.id,
                "def run_tuners(file, budget, seed, maxlives): return 1",
                "identity fixture",
                ["fixture"],
                {},
            )
            store.add_task(task)
            store.add_run(run)
            run = store.start_run(run.id, RunStatus.PENDING, clock())
            store.add_candidate(candidate)
            queue = EvaluationQueueService(store, clock=clock)
            try:
                baseline = queue.submit(
                    run.id, task.id, candidate.id, 11, 1,
                    dataset_digest="dataset-a",
                    evaluator_version="evaluator-a",
                    evaluation_parameters_version="params-a",
                )
                duplicate = queue.submit(
                    run.id, task.id, candidate.id, 11, 1,
                    dataset_digest="dataset-a",
                    evaluator_version="evaluator-a",
                    evaluation_parameters_version="params-a",
                )
                for changed in (
                    {"dataset_digest": "dataset-b"},
                    {"evaluator_version": "evaluator-b"},
                    {"evaluation_parameters_version": "params-b"},
                ):
                    values = {
                        "dataset_digest": "dataset-a",
                        "evaluator_version": "evaluator-a",
                        "evaluation_parameters_version": "params-a",
                    }
                    values.update(changed)
                    with self.assertRaises(EvaluationIdentityConflictError):
                        queue.submit(
                            run.id, task.id, candidate.id, 11, 1, **values
                        )
                events = list(store.events_for_run(run.id))

                self.assertEqual(baseline.id, duplicate.id)
                self.assertTrue(
                    all(
                        event.payload["idempotency_material_version"]
                        == "evaluation-v2"
                        for event in events
                    )
                )
                submitted = [
                    event
                    for event in events
                    if event.event_type == "EVALUATION_SUBMITTED"
                ]
                self.assertEqual(1, len(submitted))
                self.assertEqual(
                    "dataset-a",
                    submitted[0].payload["idempotency_material"]["dataset_digest"],
                )
                self.assertEqual(1, store.get_run(run.id).reserved_evaluations)
                self.assertEqual(1, len(list(store.evaluation_jobs_for_run(run.id))))
            finally:
                store.close()

    def test_algorithm_timeout_and_oom_never_enter_backend_retry_wait(self):
        failures = [
            AlgorithmTimeoutError("candidate loop timeout"),
            AlgorithmOOMError("candidate OOM"),
        ]
        for index, failure in enumerate(failures):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory(
                prefix="prievo-algorithm-failure-"
            ) as directory:
                clock = _ManualClock()
                evaluator = _AlwaysFailureEvaluator(failure)
                store, task, run, runtime = _runtime(
                    Path(directory), evaluator, clock, "algorithm-{}".format(index)
                )
                try:
                    with self.assertRaisesRegex(RuntimeError, "候选评价未成功"):
                        runtime.execute(run.id)
                    jobs = list(store.evaluation_jobs_for_run(run.id))
                    self.assertEqual(1, len(jobs))
                    self.assertEqual(EvaluationJobStatus.DEAD, jobs[0].status)
                    self.assertEqual(1, jobs[0].attempts)
                    self.assertEqual(1, evaluator.calls)
                    self.assertEqual([], clock.sleep_calls)
                    current = store.get_run(run.id)
                    self.assertEqual(0, current.consumed_evaluations)
                    self.assertEqual(0, current.reserved_evaluations)
                finally:
                    store.close()

    def test_pending_without_claim_is_an_explicit_scheduler_error(self):
        with tempfile.TemporaryDirectory(prefix="prievo-no-claim-") as directory:
            clock = _ManualClock()
            store, task, run, runtime = _runtime(
                Path(directory), FakeEvaluator(), clock, "no-claim"
            )
            runtime.worker = _NeverClaimWorker()
            try:
                with self.assertRaisesRegex(RuntimeError, "PENDING job"):
                    runtime.execute(run.id)
                self.assertEqual([], clock.sleep_calls)
            finally:
                store.close()

    def test_ready_retry_without_claim_fails_instead_of_busy_spinning(self):
        with tempfile.TemporaryDirectory(prefix="prievo-ready-no-claim-") as directory:
            clock = _ManualClock()
            store, task, run, runtime = _runtime(
                Path(directory), _TransientOnceEvaluator(), clock, "ready-no-claim"
            )
            worker = _FirstAttemptThenNeverClaimWorker(runtime.worker)
            runtime.worker = worker
            try:
                with self.assertRaisesRegex(RuntimeError, "已 ready"):
                    runtime.execute(run.id)
                self.assertEqual(3, worker.calls)
                self.assertGreater(len(clock.sleep_calls), 0)
                self.assertLessEqual(sum(clock.sleep_calls), 1.000001)
            finally:
                store.close()

    def test_cancel_is_observed_during_retry_delay(self):
        with tempfile.TemporaryDirectory(prefix="prievo-retry-cancel-") as directory:
            clock = _ManualClock()
            evaluator = _TransientOnceEvaluator()
            store, task, run, runtime = _runtime(
                Path(directory), evaluator, clock, "cancel"
            )

            def cancel_after_first_sleep():
                if len(clock.sleep_calls) != 1:
                    return
                store.request_run_cancel(run.id, "retry wait fixture cancel")

            clock.on_sleep = cancel_after_first_sleep
            try:
                completed = runtime.execute(run.id)
                self.assertEqual(RunStatus.CANCELLED, completed.status)
                self.assertEqual(1, len(clock.sleep_calls))
                self.assertEqual(1, sum(evaluator.calls.values()))
                job = list(store.evaluation_jobs_for_run(run.id))[0]
                self.assertEqual(EvaluationJobStatus.CANCELLED, job.status)
            finally:
                store.close()

    def test_pause_is_observed_after_retry_wait_sleep_at_existing_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="prievo-retry-pause-") as directory:
            clock = _ManualClock()
            evaluator = _TransientOnceEvaluator()
            store, task, run, runtime = _runtime(
                Path(directory), evaluator, clock, "pause"
            )
            store.start_run(run.id, RunStatus.PENDING, clock())
            self.assertTrue(
                store.claim_run_lease(
                    run.id, runtime.runtime_owner_id, clock(),
                    runtime.runtime_lease_seconds,
                )
            )
            candidate = Candidate(
                "candidate-retry-pause",
                run.id,
                "def run_tuners(file, budget, seed, maxlives): return 1",
                "retry pause fixture",
                ["Fixture"],
                {},
            )
            store.add_candidate(candidate)
            artifact = store.put_artifact(
                run.id, "CHECKPOINT", b'{"fixture":"retry-pause"}',
                "application/json",
            )
            store.add_checkpoint(
                CheckpointMetadata(
                    "checkpoint-retry-pause",
                    run.id,
                    0,
                    [candidate.id],
                    0,
                    3,
                    artifact.id,
                    3,
                    "fixture-code",
                )
            )
            job = runtime.queue.submit(
                run.id,
                task.id,
                candidate.id,
                seed=7,
                budget=task.evaluation_budget,
            )
            retrying = runtime.worker.run_once()
            self.assertEqual(EvaluationJobStatus.RETRY_WAIT, retrying.status)

            def pause_after_first_sleep():
                if len(clock.sleep_calls) == 1:
                    store.request_run_pause(
                        run.id, "fixture pause during retry wait"
                    )

            clock.on_sleep = pause_after_first_sleep
            try:
                with self.assertRaises(CooperativePause):
                    runtime._wait_for_retry_or_cancel(run.id, [job.id])
                paused = store.get_run(run.id)
                retained = store.get_evaluation_job(job.id)
                self.assertEqual(RunStatus.PAUSED, paused.status)
                self.assertEqual(1, len(clock.sleep_calls))
                self.assertEqual(EvaluationJobStatus.RETRY_WAIT, retained.status)
                self.assertEqual(task.evaluation_budget, paused.reserved_evaluations)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
