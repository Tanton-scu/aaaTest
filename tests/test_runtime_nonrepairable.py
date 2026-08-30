import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prievo_agent.domain.errors import CandidateRuntimeError
from prievo_agent.domain.models import (
    Candidate,
    CandidateStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.evaluation_queue import utc_clock
from prievo_agent.runtime.persistent_runtime import PersistentEvolutionRuntime


class _FailingEvaluator:
    version = "nonrepairable-fixture-v1"

    def evaluate(self, _candidate, _task):
        raise CandidateRuntimeError("fixture candidate runtime failure")


class _NonRepairableWorkflow:
    def __init__(self):
        self.calls = 0

    def repair(self, _run, _task, _candidate, _failed_job):
        self.calls += 1
        return SimpleNamespace(
            repaired_candidate=None,
            failure_artifact_id="failure-ref",
            repaired_candidate_draft_artifact_id=None,
            repair_decision_artifact_id="repair-decision-ref",
            repairable=False,
            skipped_reason="diagnosis marked repairable=false",
        )


class RuntimeNonRepairableTest(unittest.TestCase):
    def test_nonrepairable_is_business_skip_not_runtime_failure(self):
        with tempfile.TemporaryDirectory(prefix="prievo-nonrepairable-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task", "fixture", "minimize", 1, 3)
                run = Run("run", task.id, status=RunStatus.RUNNING)
                candidate = Candidate(
                    "candidate",
                    run.id,
                    "def run_tuners(file, budget, seed, maxlives):\n"
                    "    raise RuntimeError('broken')\n",
                    "broken candidate",
                    ["Sampling"],
                    {"generation": 1, "operator": "m1", "parents": []},
                )
                store.add_task(task)
                store.add_run(run)
                workflow = _NonRepairableWorkflow()
                runtime = PersistentEvolutionRuntime(
                    store,
                    _FailingEvaluator(),
                    SimpleNamespace(population_size=1),
                    repair_workflow=workflow,
                    runtime_owner_id="runtime-owner",
                )
                self.assertTrue(
                    store.claim_run_lease(
                        run.id,
                        runtime.runtime_owner_id,
                        utc_clock(),
                        runtime.runtime_lease_seconds,
                    )
                )

                values = runtime._evaluate_new(run, task, [candidate])

                self.assertEqual([], values)
                self.assertEqual(1, workflow.calls)
                self.assertEqual(
                    CandidateStatus.INVALID,
                    store.candidate_by_id(candidate.id).status,
                )
                skipped = [
                    event
                    for event in store.events_for_run(run.id)
                    if event.event_type == "CANDIDATE_REPAIR_SKIPPED"
                ]
                self.assertEqual(1, len(skipped))
                self.assertEqual(
                    "repair-decision-ref",
                    skipped[0].payload["repair_decision_artifact_id"],
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
