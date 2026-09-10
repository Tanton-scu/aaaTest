from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prievo_agent.agents.nodes.final_selection import NoQualifiedFinalCandidateError
from prievo_agent.evolution.engine import PriEvOEngine
from prievo_agent.evaluation.evaluator import DatasetEvaluator
from prievo_agent.evaluation.datasets import DatasetRegistry
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.local.fake_llm import FakeLLM
from prievo_agent.infrastructure.local.sqlite_store import SQLiteRuntimeStore
from prievo_agent.domain.errors import CandidateRuntimeError


class _FailFirstPriorEvaluator(DatasetEvaluator):
    """只注入一次动态 prior 失败，证明 faithful Repair 排除语义。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._failed_prior = False

    def evaluate(self, candidate, task):
        if (
            not self._failed_prior
            and candidate.lineage.get("operator") == "preknowledge"
        ):
            self._failed_prior = True
            raise CandidateRuntimeError("faithful fixture: prior runtime failure")
        return super().evaluate(candidate, task)


class FaithfulModeTest(unittest.TestCase):
    def test_repair_is_excluded_and_strict_final_does_not_fallback(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-faithful",
                    "faithful",
                    "minimize",
                    3,
                    42,
                    dataset_id="xgboost-Covtype",
                    generations=1,
                    population_size=2,
                    random_seed=11,
                )
                run = Run("run-faithful", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                with patch(
                    "prievo_agent.evolution.engine.DatasetEvaluator",
                    _FailFirstPriorEvaluator,
                ):
                    with self.assertRaises(NoQualifiedFinalCandidateError) as raised:
                        PriEvOEngine(
                            store,
                            DatasetRegistry(project / "assets" / "datasets"),
                            project / "assets" / "prior",
                            llm=FakeLLM(),
                            evaluation_execution_mode="inline",
                            research_faithful_mode=True,
                        ).run(run.id)
                candidates = list(store.candidates_for_run(run.id))
                repaired = [
                    item for item in candidates
                    if item.lineage.get("creation_type") == "REPAIR"
                ]
                checkpoint = store.latest_checkpoint(run.id)
                payload = json.loads(store.artifact_content(checkpoint.artifact_id))
                event_types = [item.event_type for item in store.events_for_run(run.id)]
                task_types = [
                    item.task_type for item in store.agent_tasks_for_run(run.id)
                ]
                artifact_kinds = [
                    item.kind for item in store.artifacts_for_run(run.id)
                ]
                persisted = store.get_run(run.id)
            finally:
                store.close()

        rejected = " ".join(
            reason
            for reasons in raised.exception.rejected_reasons.values()
            for reason in reasons
        )
        self.assertIn("qualification_mode=reference", rejected)
        self.assertEqual(1, len(repaired))
        self.assertEqual("INVALID", repaired[0].status.value)
        self.assertNotIn(repaired[0].id, payload["population_ids"])
        self.assertIsNone(persisted.best_candidate_id)
        self.assertIn("CANDIDATE_REPAIR", task_types)
        self.assertNotIn("LITERATURE_EVIDENCE", task_types)
        self.assertIn("FAITHFUL_REPAIR_EXCLUDED", event_types)
        self.assertNotIn("FINAL_HEURISTIC", artifact_kinds)
        self.assertNotIn("FINAL_SELECTION_DECISION", artifact_kinds)


if __name__ == "__main__":
    unittest.main()
