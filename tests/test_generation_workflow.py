from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.generation_workflow import DurableGenerationWorkflow
from prievo_agent.domain.models import Candidate, CandidateStatus, OptimizationTask, Run
from prievo_agent.domain.prior import (
    InstanceSpecificPrior,
    LandscapeProfile,
    SemanticRefinement,
)
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class DurableGenerationWorkflowTest(unittest.TestCase):
    def test_completed_draft_is_recovered_without_second_llm_call(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            store = SQLiteRuntimeStore(work / "state.sqlite3", work / "artifacts")
            try:
                task = OptimizationTask(
                    "task", "fixture", "minimize", 3, 30,
                    dataset_id="fixture", generations=1, population_size=1,
                )
                run = Run("run", task.id, dataset_id="fixture")
                store.add_task(task)
                store.add_run(run)
                parent = Candidate(
                    "parent", run.id,
                    "def run_tuners(file, budget, seed, maxlives):\n    value = 1\n    return value\n",
                    "parent", ["Random Search"], {"generation": 0},
                    CandidateStatus.EVALUATED, 0.5,
                )
                store.add_candidate(parent)
                from prievo_agent.domain.models import EvaluationResult
                store.add_result(EvaluationResult(
                    "result-parent", run.id, parent.id, 0.5,
                    [0.9, 0.5], {"x": 1}, 2,
                ))
                prior = InstanceSpecificPrior(
                    LandscapeProfile(
                        "fixture", {name: 0.1 for name in (
                            "FDC", "FBD", "PLO", "Skewness", "Kurtosis",
                            "CL", "MIE", "NBC",
                        )}, 100, "fixture",
                    ),
                    [], SemanticRefinement(["source"], "fixture", "fake"),
                    [], "prior-v1",
                )
                model = FakeLLM()
                workflow = DurableGenerationWorkflow(
                    store, SkillRegistry(root / "skills"), model
                )

                first, request_ref, draft_ref = workflow.generate(
                    run, task, prior, ["prior-ref"], "m2", [parent], 1, 0
                )
                second, second_request, second_draft = workflow.generate(
                    run, task, prior, ["prior-ref"], "m2", [parent], 1, 0
                )

                self.assertEqual(1, len(model.generation_agent_calls))
                self.assertEqual(first.code, second.code)
                self.assertEqual(request_ref, second_request)
                self.assertEqual(draft_ref, second_draft)
                tasks = list(store.agent_tasks_for_run(run.id))
                self.assertEqual(1, len(tasks))
                events = list(store.events_for_run(run.id))
                self.assertEqual(
                    1,
                    sum(item.event_type == "AGENT_TASK_COMPLETED" for item in events),
                )
                payload = json.loads(store.artifact_content(draft_ref))
                self.assertEqual("CandidateDraft", payload["result_type"])
                self.assertIn("skill_digest", payload)
                self.assertIn(request_ref, payload["input_artifact_refs"])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
