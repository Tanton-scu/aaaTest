from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.final_selection_workflow import (
    DurableFinalSelectionWorkflow,
)
from prievo_agent.agents.final_selection import NoQualifiedFinalCandidateError
from prievo_agent.domain.models import (
    Candidate,
    EvaluationResult,
    OptimizationTask,
    Run,
)
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


def _long_code(value):
    lines = ["def run_tuners(file, budget, seed, maxlives):"]
    lines.extend("    value_{} = {}".format(index, index) for index in range(55))
    lines.append("    return {}".format(value))
    return "\n".join(lines) + "\n"


class DurableFinalSelectionWorkflowTest(unittest.TestCase):
    def test_unique_direct_and_exact_tie_agent_paths_are_distinct(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task", "fixture", "minimize", 3, 30,
                    dataset_id="fixture", population_size=2,
                )
                run = Run("run", task.id, dataset_id="fixture")
                store.add_task(task)
                store.add_run(run)
                first = Candidate(
                    "a", run.id, _long_code(1), "a", ["A"], {}, objective=0.1
                )
                second = Candidate(
                    "b", run.id, _long_code(2), "b", ["B"], {}, objective=0.2
                )
                for candidate in (first, second):
                    store.add_candidate(candidate)
                    store.add_result(EvaluationResult(
                        "result-" + candidate.id,
                        run.id,
                        candidate.id,
                        candidate.objective,
                        [candidate.objective] * 3,
                        {"x": candidate.id},
                        3,
                    ))
                model = FakeLLM()
                workflow = DurableFinalSelectionWorkflow(
                    store, SkillRegistry(project / "skills"), model
                )

                selected, direct_ref, direct = workflow.select(
                    run.id, [first, second], 3
                )
                self.assertEqual("a", selected)
                self.assertFalse(direct.model_called)
                self.assertEqual([], model.final_selection_calls)
                self.assertEqual([], [
                    item for item in store.agent_tasks_for_run(run.id)
                    if item.task_type == "FINAL_SELECTION"
                ])
                self.assertTrue(direct_ref)

                second.objective = 0.1
                store.add_candidate(second)
                # result is an immutable benchmark fact; use a separate Run so the
                # second scenario can have a genuine exact tie.
                tie_run = Run("tie-run", task.id, dataset_id="fixture")
                store.add_run(tie_run)
                tied = []
                for source in (first, second):
                    candidate = Candidate(
                        source.id + "-tie", tie_run.id, source.code,
                        source.description, source.operators, {}, objective=0.1,
                    )
                    tied.append(candidate)
                    store.add_candidate(candidate)
                    store.add_result(EvaluationResult(
                        "result-" + candidate.id, tie_run.id, candidate.id,
                        0.1, [0.1] * 3, {"x": candidate.id}, 3,
                    ))
                selected, tie_ref, payload = workflow.select(
                    tie_run.id, tied, 3
                )
                self.assertEqual("a-tie", selected)
                self.assertTrue(tie_ref)
                self.assertEqual(1, len(model.final_selection_calls))
                tasks = [
                    item for item in store.agent_tasks_for_run(tie_run.id)
                    if item.task_type == "FINAL_SELECTION"
                ]
                self.assertEqual(1, len(tasks))
                self.assertEqual("COMPLETED", tasks[0].status.value)
                self.assertNotIn("FOREIGN_MEMORY_SENTINEL",
                                 model.final_selection_calls[0]["prompt"])
                self.assertNotIn("UNRELATED_RAG_CHUNK",
                                 model.final_selection_calls[0]["prompt"])
            finally:
                store.close()

    def test_faithful_mode_rejects_engineering_budget_completion(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-faithful-final", "fixture", "minimize", 3, 30,
                    dataset_id="fixture", population_size=2,
                )
                run = Run(
                    "run-faithful-final", task.id, dataset_id="fixture"
                )
                store.add_task(task)
                store.add_run(run)
                candidate = Candidate(
                    "candidate-budget-3",
                    run.id,
                    _long_code(1),
                    "candidate",
                    ["Sampling"],
                    {},
                    objective=0.1,
                )
                store.add_candidate(candidate)
                store.add_result(EvaluationResult(
                    "result-budget-3",
                    run.id,
                    candidate.id,
                    0.1,
                    [0.1] * 3,
                    {"x": 1},
                    3,
                ))
                skills = SkillRegistry(project / "skills")

                engineering = DurableFinalSelectionWorkflow(
                    store, skills, FakeLLM(), faithful_mode=False
                )
                selected, _, decision = engineering.select(
                    run.id, [candidate], 3
                )
                self.assertEqual(candidate.id, selected)
                self.assertEqual("engineering", decision.qualification_mode)

                faithful = DurableFinalSelectionWorkflow(
                    store, skills, FakeLLM(), faithful_mode=True
                )
                with self.assertRaises(NoQualifiedFinalCandidateError):
                    faithful.select(run.id, [candidate], 3)
            finally:
                store.close()

    def test_faithful_mode_accepts_exact_reference_trajectory(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-reference-final", "fixture", "minimize", 3, 30,
                    dataset_id="fixture", population_size=2,
                )
                run = Run(
                    "run-reference-final", task.id, dataset_id="fixture"
                )
                store.add_task(task)
                store.add_run(run)
                candidate = Candidate(
                    "candidate-reference-20",
                    run.id,
                    _long_code(1),
                    "candidate",
                    ["Sampling"],
                    {},
                    objective=0.1,
                )
                store.add_candidate(candidate)
                store.add_result(EvaluationResult(
                    "result-reference-20",
                    run.id,
                    candidate.id,
                    0.1,
                    [0.1] * 20,
                    {"x": 1},
                    3,
                ))
                model = FakeLLM()
                workflow = DurableFinalSelectionWorkflow(
                    store,
                    SkillRegistry(project / "skills"),
                    model,
                    faithful_mode=True,
                )

                selected, decision_ref, decision = workflow.select(
                    run.id, [candidate], 3
                )

                self.assertEqual(candidate.id, selected)
                self.assertTrue(decision_ref)
                self.assertEqual("reference", decision.qualification_mode)
                self.assertFalse(decision.fallback_used)
                self.assertFalse(decision.model_called)
                self.assertEqual([], model.final_selection_calls)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
