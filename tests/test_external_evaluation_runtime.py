from __future__ import annotations

import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.domain.models import (
    Candidate,
    CheckpointMetadata,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import (
    EvaluationQueueService,
    EvaluationWorker,
)
from prievo_agent.runtime.final_optimization import FinalOptimizationService
from prievo_agent.runtime.persistent_runtime import (
    CooperativePause,
    PersistentEvolutionRuntime,
)


class _AppEvaluatorMustNotRun:
    version = "external-runtime-fixture-v1"
    evaluation_parameters_version = "fixture-parameters-v1"

    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, task):
        self.calls += 1
        raise AssertionError("external mode 不得在 App Runtime 内执行 evaluator")


class _IndependentEvaluator(FakeEvaluator):
    version = "external-runtime-fixture-v1"
    evaluation_parameters_version = "fixture-parameters-v1"

    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, task):
        self.calls += 1
        return super().evaluate(candidate, task)


class _RecordingSleeper:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(float(seconds))
        time.sleep(seconds)


class ExternalEvaluationRuntimeTest(unittest.TestCase):
    def test_external_runtime_and_final_optimization_never_claim_inline(self):
        with tempfile.TemporaryDirectory(prefix="prievo-external-worker-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            setup = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask(
                "task-external",
                "external worker",
                "minimize",
                evaluation_budget=1,
                total_budget=5,
                generations=0,
                population_size=1,
                random_seed=17,
            )
            run = Run("run-external", task.id)
            setup.add_task(task)
            setup.add_run(run)
            setup.close()

            stop = threading.Event()
            worker_ready = threading.Event()
            worker_result = {}

            def independent_worker():
                store = SQLiteRuntimeStore(database, artifacts)
                evaluator = _IndependentEvaluator()
                worker = EvaluationWorker(
                    store,
                    evaluator,
                    worker_id="independent-worker-fixture",
                    lease_seconds=10,
                )
                worker_result["evaluator"] = evaluator
                worker_ready.set()
                try:
                    while not stop.is_set():
                        worker.recover_stale()
                        if worker.run_once() is None:
                            stop.wait(0.005)
                finally:
                    store.close()

            worker_thread = threading.Thread(target=independent_worker)
            worker_thread.start()
            self.assertTrue(worker_ready.wait(2))
            app_store = SQLiteRuntimeStore(database, artifacts)
            app_evaluator = _AppEvaluatorMustNotRun()
            sleeper = _RecordingSleeper()
            final_service = FinalOptimizationService(
                app_store,
                app_evaluator,
                seeds=(101, 202),
                final_budget_per_seed=2,
                evaluation_execution_mode="external",
                sleeper=sleeper,
                max_poll_seconds=0.02,
            )
            runtime = PersistentEvolutionRuntime(
                app_store,
                app_evaluator,
                PriEvoEvolutionCore(FakeLLM(), population_size=1, seed=17),
                total_generations=0,
                dataset_digest="external-dataset-digest",
                final_optimization_service=final_service,
                evaluation_execution_mode="external",
                sleeper=sleeper,
                retry_wait_poll_seconds=0.02,
            )
            try:
                completed = runtime.execute(run.id)
            finally:
                stop.set()
                worker_thread.join(timeout=3)
                app_store.close()

            self.assertFalse(worker_thread.is_alive())
            self.assertEqual(RunStatus.COMPLETED, completed.status)
            self.assertEqual(0, app_evaluator.calls)
            # evolution 一次 + Final Optimization 两个 seed。
            self.assertEqual(3, worker_result["evaluator"].calls)
            self.assertTrue(sleeper.calls)
            self.assertTrue(all(value > 0 for value in sleeper.calls))
            self.assertLessEqual(max(sleeper.calls), 0.02)

            verify = SQLiteRuntimeStore(database, artifacts)
            try:
                jobs = list(verify.evaluation_jobs_for_run(run.id))
                self.assertEqual(3, len(jobs))
                self.assertTrue(all(job.status.value == "SUCCESS" for job in jobs))
                self.assertEqual(5, verify.get_run(run.id).consumed_evaluations)
                reports = [
                    item
                    for item in verify.artifacts_for_run(run.id)
                    if item.kind == "FINAL_OPTIMIZATION_REPORT"
                ]
                self.assertEqual(1, len(reports))
            finally:
                verify.close()

    def test_pause_requested_during_external_wait_is_confirmed_after_sleep_at_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="prievo-external-pause-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            task = OptimizationTask("task-pause", "pause", "minimize", 1, 2)
            run = Run("run-pause", task.id, status=RunStatus.RUNNING)
            candidate = Candidate(
                "candidate-pause",
                run.id,
                "def run_tuners(file, budget, seed, maxlives): return 1",
                "pause fixture",
                ["Fixture"],
                {},
            )
            store.add_task(task)
            store.add_run(run)
            store.add_candidate(candidate)
            checkpoint_artifact = store.put_artifact(
                run.id,
                "CHECKPOINT",
                b'{"fixture":"safe"}',
                "application/json",
            )
            store.add_checkpoint(
                CheckpointMetadata(
                    "checkpoint-pause",
                    run.id,
                    0,
                    [candidate.id],
                    0,
                    2,
                    checkpoint_artifact.id,
                    3,
                    "fixture-code",
                )
            )
            job = EvaluationQueueService(store).submit(
                run.id, task.id, candidate.id, seed=7, budget=1
            )
            runtime = PersistentEvolutionRuntime(
                store,
                _AppEvaluatorMustNotRun(),
                PriEvoEvolutionCore(FakeLLM(), population_size=1),
                total_generations=0,
                evaluation_execution_mode="external",
                retry_wait_poll_seconds=0.01,
                runtime_owner_id="runtime-pause-owner",
            )
            self.assertTrue(
                store.claim_run_lease(
                    run.id,
                    runtime.runtime_owner_id,
                    datetime.now(timezone.utc),
                    30,
                )
            )
            sleeps = []

            def request_pause_during_sleep(seconds):
                sleeps.append(seconds)
                store.request_run_pause(run.id, "fixture pause during external wait")

            runtime.evaluation_driver.sleeper = request_pause_during_sleep
            try:
                with self.assertRaises(CooperativePause):
                    runtime.evaluation_driver.drive(
                        run.id,
                        [job.id],
                        control_check=lambda stage: runtime._evaluation_wait_control(
                            run.id, stage
                        ),
                    )
                paused = store.get_run(run.id)
                pending = store.get_evaluation_job(job.id)
            finally:
                store.close()

            self.assertEqual(1, len(sleeps))
            self.assertEqual(RunStatus.PAUSED, paused.status)
            self.assertFalse(paused.pause_requested)
            self.assertEqual("PENDING", pending.status.value)
            self.assertEqual(1, paused.reserved_evaluations)

    def test_initial_external_wait_does_not_fake_pause_without_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="prievo-external-initial-pause-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            app_store = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask(
                "task-initial-pause", "initial pause", "minimize", 1, 1
            )
            run = Run(
                "run-initial-pause", task.id, status=RunStatus.RUNNING
            )
            candidate = Candidate(
                "candidate-initial-pause",
                run.id,
                "def run_tuners(file, budget, seed, maxlives): return 1",
                "initial pause fixture",
                ["Fixture"],
                {},
            )
            app_store.add_task(task)
            app_store.add_run(run)
            app_store.add_candidate(candidate)
            job = EvaluationQueueService(app_store).submit(
                run.id, task.id, candidate.id, seed=7, budget=1
            )
            app_evaluator = _AppEvaluatorMustNotRun()
            runtime = PersistentEvolutionRuntime(
                app_store,
                app_evaluator,
                PriEvoEvolutionCore(FakeLLM(), population_size=1),
                total_generations=0,
                evaluation_execution_mode="external",
                retry_wait_poll_seconds=0.01,
                runtime_owner_id="runtime-initial-pause-owner",
            )
            self.assertTrue(
                app_store.claim_run_lease(
                    run.id,
                    runtime.runtime_owner_id,
                    datetime.now(timezone.utc),
                    30,
                )
            )
            independent = _IndependentEvaluator()
            sleeps = []

            def request_pause_then_let_worker_finish(seconds):
                sleeps.append(float(seconds))
                app_store.request_run_pause(
                    run.id, "fixture pause before first checkpoint"
                )
                worker_store = SQLiteRuntimeStore(database, artifacts)
                try:
                    completed_job = EvaluationWorker(
                        worker_store,
                        independent,
                        worker_id="initial-pause-independent-worker",
                    ).run_once()
                    self.assertEqual("SUCCESS", completed_job.status.value)
                finally:
                    worker_store.close()

            runtime.evaluation_driver.sleeper = request_pause_then_let_worker_finish
            try:
                result = runtime.evaluation_driver.drive(
                    run.id,
                    [job.id],
                    control_check=lambda stage: runtime._evaluation_wait_control(
                        run.id, stage
                    ),
                )
                current = app_store.get_run(run.id)
            finally:
                app_store.close()

            self.assertTrue(result.completed)
            self.assertEqual(1, len(sleeps))
            self.assertEqual(0, app_evaluator.calls)
            self.assertEqual(1, independent.calls)
            self.assertEqual(RunStatus.RUNNING, current.status)
            self.assertTrue(current.pause_requested)
            self.assertEqual(1, current.consumed_evaluations)
            self.assertEqual(0, current.reserved_evaluations)


if __name__ == "__main__":
    unittest.main()
