import tempfile
import unittest
import sqlite3
from pathlib import Path

from prievo_agent.application.orchestration.run_facade import RunApplicationFacade
from prievo_agent.knowledge.prior.retrieval import PriorRetrievalService
from prievo_agent.domain.errors import (
    ArtifactIntegrityError,
    LLMTimeoutError,
    PriorRepositoryUnavailableError,
)
from prievo_agent.domain.models import Candidate, OptimizationTask, Run, RunStatus
from prievo_agent.knowledge.prior.models import LANDSCAPE_METRICS, LandscapeProfile
from prievo_agent.infrastructure.composition import RuntimeComposition
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore
from prievo_agent.evaluation.queue import EvaluationQueueService


class UnavailablePriorRepository:
    def landscape_profiles(self):
        raise OSError("fixture repository offline")


class UnusedRefiner:
    def refine(self, target, candidates):
        raise AssertionError("repository 失败后不得调用 refiner")


class FailureModeTest(unittest.TestCase):
    def test_llm_timeout_marks_accepted_run_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            composition = RuntimeComposition(Path(directory))

            def timeout_executor(run_id):
                raise LLMTimeoutError("LLM provider timeout")

            facade = RunApplicationFacade(
                composition.open_store, timeout_executor, auto_start=False
            )
            run_id = facade.create_run("LLM 超时", 3, 60)["run_id"]
            with self.assertLogs(
                "prievo_agent.application.orchestration.run_facade", level="ERROR"
            ) as logs:
                facade.resume(run_id)
                facade.shutdown()
            self.assertIn("运行后台执行失败", "\n".join(logs.output))
            self.assertEqual("FAILED", facade.get_run(run_id)["status"])
            events = facade.events(run_id)
            self.assertEqual("RUN_FAILED", events[-1]["event_type"])

    def test_prior_repository_unavailable_fails_without_fake_prior(self):
        profile = LandscapeProfile(
            "target", {name: 0.0 for name in LANDSCAPE_METRICS}, 10, "fixture"
        )
        service = PriorRetrievalService(UnavailablePriorRepository(), UnusedRefiner())
        with self.assertRaises(PriorRepositoryUnavailableError):
            service.retrieve(profile)

    def test_artifact_missing_and_corrupt_have_same_taxonomy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            corrupt = store.put_artifact("run-a", "FIXTURE", b"valid", "text/plain")
            Path(corrupt.uri).write_bytes(b"corrupt")
            with self.assertRaises(ArtifactIntegrityError):
                store.artifact_content(corrupt.id)
            missing = store.put_artifact("run-b", "FIXTURE", b"missing", "text/plain")
            Path(missing.uri).unlink()
            with self.assertRaises(ArtifactIntegrityError):
                store.artifact_content(missing.id)
            store.close()

    def test_database_failure_rolls_back_job_and_budget_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            task = OptimizationTask("task-tx", "事务失败", "minimize", 3, 9)
            run = Run("run-tx", task.id, status=RunStatus.RUNNING)
            candidate = Candidate(
                "candidate-tx", run.id,
                "def run_tuners(file, budget, seed, maxlives): return 1",
                "事务 fixture", ["Revise"], {},
            )
            store.add_task(task)
            store.add_run(run)
            store.add_candidate(candidate)
            store.connection.execute(
                """CREATE TRIGGER fail_budget_reservation
                   BEFORE UPDATE OF reserved_evaluations ON runs
                   BEGIN SELECT RAISE(ABORT, 'fixture transaction failure'); END"""
            )
            store.connection.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                EvaluationQueueService(store).submit(
                    run.id, task.id, candidate.id, seed=1, budget=3
                )
            self.assertEqual([], list(store.evaluation_jobs_for_run(run.id)))
            self.assertEqual(0, store.get_run(run.id).reserved_evaluations)
            store.close()


if __name__ == "__main__":
    unittest.main()
