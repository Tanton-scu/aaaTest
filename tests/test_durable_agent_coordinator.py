import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.application.orchestration.durable_agent_coordinator import (
    DurableAgentCoordinator,
)
from prievo_agent.domain.models import (
    AgentCapability,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


class DurableAgentCoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")

    def tearDown(self):
        self.store.close()
        self.temporary_directory.cleanup()

    def _add_run(self, run_id):
        task = OptimizationTask(
            "task-{}".format(run_id),
            "durable coordinator",
            "minimize",
            evaluation_budget=3,
            total_budget=99,
            dataset_id="fixture",
            generations=4,
            population_size=10,
        )
        run = Run(
            run_id,
            task.id,
            status=RunStatus.RUNNING,
            generation=2,
            consumed_evaluations=17,
            reserved_evaluations=3,
            best_candidate_id="candidate-best",
            dataset_id=task.dataset_id,
        )
        self.store.add_task(task)
        self.store.add_run(run)
        return run

    def _artifact(self, run_id, kind, payload=None):
        content = json.dumps(
            payload if payload is not None else {"kind": kind},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return self.store.put_artifact(run_id, kind, content, "application/json")

    def test_reconcile_derives_three_agent_tasks_and_only_emits_created_events(self):
        coordinator = DurableAgentCoordinator(self.store)
        cases = {
            "GENERATION_REQUEST": (
                "HEURISTIC_GENERATION",
                AgentCapability.HEURISTIC_GENERATION,
            ),
            "KNOWLEDGE_GAP": (
                "LITERATURE_EVIDENCE",
                AgentCapability.LITERATURE_EVIDENCE,
            ),
            "CANDIDATE_FAILURE": (
                "CANDIDATE_REPAIR",
                AgentCapability.CANDIDATE_REPAIR,
            ),
        }
        all_created = []
        for index, (input_kind, expected) in enumerate(cases.items()):
            run = self._add_run("run-rule-{}".format(index))
            input_artifact = self._artifact(run.id, input_kind)
            before = self.store.get_run(run.id)

            created = coordinator.reconcile(run.id)
            self.assertEqual(1, len(created))
            task = created[0]
            all_created.append(task)
            self.assertEqual(expected[0], task.task_type)
            self.assertEqual(expected[1], task.required_capability)
            self.assertEqual([input_artifact.id], task.input_artifact_refs)

            # 再次 reconcile 不新增任务，也不重复写 AGENT_TASK_CREATED。
            self.assertEqual([], coordinator.reconcile(run.id))
            self.assertEqual(1, len(self.store.agent_tasks_for_run(run.id)))
            created_events = [
                event
                for event in self.store.events_for_run(run.id)
                if event.event_type == "AGENT_TASK_CREATED"
            ]
            self.assertEqual(1, len(created_events))
            self.assertEqual(task.id, created_events[0].payload["agent_task_id"])

            # Coordinator 只增加 AgentTask/Event，不改变算法事实或预算账本。
            after = self.store.get_run(run.id)
            self.assertEqual(before.status, after.status)
            self.assertEqual(before.generation, after.generation)
            self.assertEqual(before.consumed_evaluations, after.consumed_evaluations)
            self.assertEqual(before.reserved_evaluations, after.reserved_evaluations)
            self.assertEqual(before.best_candidate_id, after.best_candidate_id)
            self.assertEqual([], list(self.store.candidates_for_run(run.id)))

        self.assertEqual(3, len({task.id for task in all_created}))
        self.assertEqual(3, len({task.idempotency_key for task in all_created}))

    def test_corresponding_outputs_prevent_missing_work(self):
        run = self._add_run("run-with-outputs")
        for kind in [
            "TOP5_CANDIDATES",
            "GENERATION_REQUEST",
            "KNOWLEDGE_GAP",
            "CANDIDATE_FAILURE",
            "FINAL_TIE",
        ]:
            self._artifact(run.id, kind)
        for kind in [
            "SIMILARITY_DECISION",
            "CANDIDATE_DRAFT",
            "PRIOR_EXPLANATION",
            "LITERATURE_EVIDENCE",
            "REPAIR_DECISION",
            "FINAL_SELECTION_DECISION",
        ]:
            self._artifact(run.id, kind)

        created = DurableAgentCoordinator(self.store).reconcile(run.id)

        self.assertEqual([], created)
        self.assertEqual([], self.store.agent_tasks_for_run(run.id))
        self.assertEqual(
            [],
            [
                event
                for event in self.store.events_for_run(run.id)
                if event.event_type == "AGENT_TASK_CREATED"
            ],
        )

    def test_research_requires_both_outputs_and_multi_input_outputs_are_correlated(self):
        research_run = self._add_run("run-partial-research")
        self._artifact(research_run.id, "KNOWLEDGE_GAP")
        self._artifact(research_run.id, "PRIOR_EXPLANATION")

        research_tasks = DurableAgentCoordinator(self.store).reconcile(
            research_run.id
        )
        self.assertEqual(["LITERATURE_EVIDENCE"], [task.task_type for task in research_tasks])

        generation_run = self._add_run("run-multiple-generation-requests")
        first = self._artifact(
            generation_run.id, "GENERATION_REQUEST", {"request": 1}
        )
        second = self._artifact(
            generation_run.id, "GENERATION_REQUEST", {"request": 2}
        )
        self._artifact(
            generation_run.id,
            "CANDIDATE_DRAFT",
            {"input_artifact_refs": [first.id]},
        )

        generation_tasks = DurableAgentCoordinator(self.store).reconcile(
            generation_run.id
        )
        self.assertEqual(1, len(generation_tasks))
        self.assertEqual("HEURISTIC_GENERATION", generation_tasks[0].task_type)
        self.assertEqual([second.id], generation_tasks[0].input_artifact_refs)

    def test_sweep_is_stable_deduplicates_run_ids_and_is_idempotent(self):
        run_b = self._add_run("run-b")
        run_a = self._add_run("run-a")
        artifact_a = self._artifact(run_a.id, "GENERATION_REQUEST")
        artifact_b = self._artifact(run_b.id, "CANDIDATE_FAILURE")
        coordinator = DurableAgentCoordinator(self.store)

        created = coordinator.sweep([run_b.id, run_a.id, run_b.id, ""])

        self.assertEqual([run_a.id, run_b.id], [task.run_id for task in created])
        self.assertEqual(
            [[artifact_a.id], [artifact_b.id]],
            [task.input_artifact_refs for task in created],
        )
        first_identity = [(task.id, task.idempotency_key) for task in created]
        self.assertEqual([], coordinator.sweep([run_a.id, run_b.id]))
        persisted_identity = sorted(
            (task.id, task.idempotency_key)
            for run_id in (run_a.id, run_b.id)
            for task in self.store.agent_tasks_for_run(run_id)
        )
        self.assertEqual(sorted(first_identity), persisted_identity)


if __name__ == "__main__":
    unittest.main()
