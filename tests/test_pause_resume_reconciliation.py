from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    Candidate,
    CheckpointMetadata,
    EvaluationJobStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.testing.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import EvaluationQueueService, EvaluationWorker
from prievo_agent.runtime.lifecycle import RunLifecycleService
from prievo_agent.runtime.persistent_runtime import PersistentEvolutionRuntime
from prievo_agent.runtime.state_machine import RunStateMachine


class SignallingEvaluator(FakeEvaluator):
    def __init__(self, entered, release):
        self.entered = entered
        self.release = release

    def evaluate(self, candidate, task):
        self.entered.set()
        if not self.release.wait(timeout=3):
            raise RuntimeError("测试未释放 evaluator")
        return super().evaluate(candidate, task)


class CountingEvaluator(FakeEvaluator):
    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate, task):
        self.calls += 1
        return super().evaluate(candidate, task)


class PauseResumeReconciliationTest(unittest.TestCase):
    def test_paused_run_cannot_claim_pending_evaluation_or_agent_task(self):
        with tempfile.TemporaryDirectory(prefix="prievo-paused-claim-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-paused", "paused", "minimize", 1, 3)
                run = Run("run-paused", task.id, status=RunStatus.RUNNING)
                store.add_task(task)
                store.add_run(run)
                owner = "runtime-paused"
                self.assertTrue(
                    store.claim_run_lease(run.id, owner, task.created_at, 30)
                )
                candidate = Candidate(
                    "candidate-paused", run.id,
                    "def run_tuners(file, budget, seed, maxlives): return 1",
                    "paused", [], {},
                )
                store.add_candidate(candidate)
                EvaluationQueueService(store).submit(
                    run.id, task.id, candidate.id, 1, 1
                )
                agent_task = AgentTask(
                    "agent-task-paused", run.id, "HEURISTIC_GENERATION",
                    AgentCapability.HEURISTIC_GENERATION, "paused-agent",
                )
                store.add_agent_task(agent_task)
                store.pause_run(
                    run.id, owner, task.created_at + timedelta(seconds=1)
                )

                self.assertIsNone(
                    store.claim_next_job(
                        "worker-paused", task.created_at + timedelta(seconds=2), 30
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "非 active"):
                    store.claim_agent_task(agent_task.id, "GenerationAgent")
                self.assertEqual(
                    EvaluationJobStatus.PENDING,
                    list(store.evaluation_jobs_for_run(run.id))[0].status,
                )
                self.assertEqual(
                    AgentTaskStatus.PENDING,
                    store.get_agent_task(agent_task.id).status,
                )
            finally:
                store.close()

    def test_pause_during_evaluation_reaches_safe_checkpoint_then_resumes(self):
        with tempfile.TemporaryDirectory(prefix="prievo-pause-resume-") as directory:
            root = Path(directory)
            database = root / "state.sqlite3"
            artifacts = root / "artifacts"
            setup = SQLiteRuntimeStore(database, artifacts)
            task = OptimizationTask(
                "task-pause", "pause", "minimize", 1, 10,
                generations=1, population_size=2,
            )
            run = Run("run-pause", task.id)
            setup.add_task(task)
            setup.add_run(run)
            setup.close()

            entered = threading.Event()
            release = threading.Event()
            errors = []

            def execute_until_pause():
                worker_store = SQLiteRuntimeStore(database, artifacts)
                try:
                    PersistentEvolutionRuntime(
                        worker_store,
                        SignallingEvaluator(entered, release),
                        PriEvoEvolutionCore(FakeLLM(), population_size=2),
                        total_generations=1,
                    ).execute(run.id)
                except Exception as exc:  # pragma: no cover - failure diagnostics
                    errors.append(exc)
                finally:
                    worker_store.close()

            thread = threading.Thread(target=execute_until_pause)
            thread.start()
            self.assertTrue(entered.wait(timeout=2), "未进入真实 evaluator")
            control = SQLiteRuntimeStore(database, artifacts)
            try:
                running = control.get_run(run.id)
                self.assertEqual(RunStatus.RUNNING, running.status)
                requested = RunLifecycleService(
                    control, RunStateMachine()
                ).request_pause(running, "测试：评价期间暂停")
                self.assertTrue(requested.pause_requested)
                # cooperative：请求落库时仍在 RUNNING，必须等待 evaluator settle。
                self.assertEqual(RunStatus.RUNNING, requested.status)
            finally:
                control.close()
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual([], errors)

            paused_store = SQLiteRuntimeStore(database, artifacts)
            try:
                paused = paused_store.get_run(run.id)
                self.assertEqual(RunStatus.PAUSED, paused.status)
                self.assertFalse(paused.pause_requested)
                checkpoint = paused_store.latest_checkpoint(run.id)
                self.assertEqual(
                    checkpoint.artifact_id, paused.runtime_cursor_artifact_id
                )
                payload = json.loads(
                    paused_store.artifact_content(checkpoint.artifact_id).decode("utf-8")
                )
                self.assertEqual(
                    {"phase", "next_generation", "next_operator_index"},
                    set(payload["algorithm_cursor"]),
                )
                self.assertIn("population_ids", payload)
                self.assertNotIn("population", payload)
                self.assertNotIn("code", payload)
                event_types = [
                    event.event_type for event in paused_store.events_for_run(run.id)
                ]
                self.assertIn("RUN_PAUSE_REQUESTED", event_types)
                self.assertIn("CHECKPOINT_SAVED", event_types)
                self.assertIn("RUN_PAUSED", event_types)
                RunLifecycleService(
                    paused_store, RunStateMachine()
                ).start(paused)
            finally:
                paused_store.close()

            resumed_store = SQLiteRuntimeStore(database, artifacts)
            try:
                completed = PersistentEvolutionRuntime(
                    resumed_store,
                    FakeEvaluator(),
                    PriEvoEvolutionCore(FakeLLM(), population_size=2),
                    total_generations=1,
                ).execute(run.id)
                self.assertEqual(RunStatus.COMPLETED, completed.status)
                self.assertEqual(10, completed.consumed_evaluations)
                self.assertEqual(
                    10,
                    resumed_store.connection.execute(
                        "SELECT COUNT(*) FROM evaluation_results WHERE run_id=?",
                        (run.id,),
                    ).fetchone()[0],
                )
                self.assertIn(
                    "RUN_RECOVERED",
                    [event.event_type for event in resumed_store.events_for_run(run.id)],
                )
            finally:
                resumed_store.close()

    def test_cancel_atomically_cancels_pending_work_and_releases_budget_once(self):
        with tempfile.TemporaryDirectory(prefix="prievo-cancel-pending-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-cancel", "cancel", "minimize", 2, 5)
                run = Run("run-cancel", task.id, status=RunStatus.RUNNING)
                store.add_task(task)
                store.add_run(run)
                queue = EvaluationQueueService(store)
                for index in range(2):
                    candidate = Candidate(
                        "candidate-cancel-{}".format(index), run.id,
                        "def run_tuners(file, budget, seed, maxlives): return {}".format(index),
                        "pending", [], {},
                    )
                    store.add_candidate(candidate)
                    queue.submit(run.id, task.id, candidate.id, index, 2)
                agent_task = AgentTask(
                    "agent-task-cancel", run.id, "HEURISTIC_GENERATION",
                    AgentCapability.HEURISTIC_GENERATION, "cancel-agent-task",
                )
                store.add_agent_task(agent_task)
                self.assertEqual(4, store.get_run(run.id).reserved_evaluations)

                RunLifecycleService(store, RunStateMachine()).cancel(
                    run, "测试取消"
                )

                cancelled = store.get_run(run.id)
                self.assertEqual(RunStatus.CANCELLED, cancelled.status)
                self.assertTrue(cancelled.cancel_requested)
                self.assertEqual(0, cancelled.reserved_evaluations)
                self.assertEqual(
                    {EvaluationJobStatus.CANCELLED},
                    {job.status for job in store.evaluation_jobs_for_run(run.id)},
                )
                self.assertEqual(
                    AgentTaskStatus.CANCELLED,
                    store.get_agent_task(agent_task.id).status,
                )
                self.assertIsNone(store.claim_next_job("late-worker", task.created_at, 30))
                self.assertEqual(0, store.get_run(run.id).reserved_evaluations)
            finally:
                store.close()

    def test_reconcile_reuses_post_checkpoint_result_and_creates_missing_job(self):
        with tempfile.TemporaryDirectory(prefix="prievo-reconcile-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-reconcile", "reconcile", "minimize", 1, 8)
                run = Run("run-reconcile", task.id, status=RunStatus.RUNNING)
                store.add_task(task)
                store.add_run(run)
                completed_candidate = Candidate(
                    "candidate-result-after-checkpoint", run.id,
                    "def run_tuners(file, budget, seed, maxlives): return 1",
                    "after checkpoint", [], {},
                )
                store.add_candidate(completed_candidate)
                checkpoint_artifact = store.put_artifact(
                    run.id, "CHECKPOINT", b'{"baseline":"before-result"}',
                    "application/json",
                )
                store.add_checkpoint(
                    CheckpointMetadata(
                        "checkpoint-before-result", run.id, 0,
                        [completed_candidate.id], 0, 8, checkpoint_artifact.id,
                        3, "test-code-version",
                    )
                )
                evaluator = CountingEvaluator()
                runtime = PersistentEvolutionRuntime(
                    store, evaluator,
                    PriEvoEvolutionCore(FakeLLM(), population_size=2),
                    total_generations=1,
                )
                queue = EvaluationQueueService(store)
                queue.submit(
                    run.id,
                    task.id,
                    completed_candidate.id,
                    101,
                    1,
                    dataset_digest=runtime.dataset_digest,
                    evaluator_version=runtime.evaluator_version,
                    evaluation_parameters_version=(
                        runtime.evaluation_parameters_version
                    ),
                )
                EvaluationWorker(store, evaluator).run_once()
                self.assertEqual(1, evaluator.calls)

                candidate_without_job = Candidate(
                    "candidate-without-job", run.id,
                    "def run_tuners(file, budget, seed, maxlives): return 2",
                    "durable Candidate before Job", [], {"evaluation_seed": 77},
                )
                store.add_candidate(candidate_without_job)
                report = runtime._reconcile_durable_evaluations(run, task)

                self.assertEqual(
                    {"reused_results": 1, "evaluation_jobs_created": 1}, report
                )
                self.assertEqual(1, evaluator.calls, "reconcile 不得重复执行已有 Result")
                jobs = list(store.evaluation_jobs_for_run(run.id))
                missing_jobs = [
                    job for job in jobs if job.candidate_id == candidate_without_job.id
                ]
                self.assertEqual(1, len(missing_jobs))
                self.assertEqual(77, missing_jobs[0].seed)
                self.assertEqual(
                    checkpoint_artifact.id,
                    store.latest_checkpoint(run.id).artifact_id,
                )
                reconciled = [
                    event for event in store.events_for_run(run.id)
                    if event.event_type == "RUN_RECONCILED"
                ]
                self.assertEqual(1, len(reconciled))
                self.assertEqual(1, reconciled[0].payload["reused_results"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
