import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prievo_agent.application.similarity_workflow import DurableSimilarityWorkflow
from prievo_agent.core.prior_retrieval import PriorRetrievalService
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.prior_adapters import DeterministicPriorRefiner
from prievo_agent.infrastructure.prior_repository import CsvPriorRepository
from prievo_agent.infrastructure.skill_registry import SkillRegistry
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class DurableSimilarityWorkflowTest(unittest.TestCase):
    def test_top5_node_decision_then_repository_prior(self):
        with tempfile.TemporaryDirectory(prefix="prievo-similarity-flow-") as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task-sim", "similarity", "minimize", 2, 20)
                run = Run("run-sim", task.id)
                store.add_task(task)
                store.add_run(run)
                repository = CsvPriorRepository(PROJECT_ROOT / "resources" / "prior_knowledge")
                target = next(
                    item for item in repository.landscape_profiles()
                    if item.instance_name == "xgboost-Covtype"
                )
                retrieval = PriorRetrievalService(repository, DeterministicPriorRefiner())
                numeric = retrieval.retrieve_numeric(target, 5)
                semantics = _metric_semantics(
                    PROJECT_ROOT / "resources" / "prior_knowledge" / "fl_metric.csv"
                )

                refinement, top5_ref, decision_ref = DurableSimilarityWorkflow(
                    store, SkillRegistry(PROJECT_ROOT / "skills"), FakeLLM()
                ).select(run.id, target, numeric, semantics)
                prior = retrieval.extract(target, numeric, refinement)

                self.assertEqual(2, len(prior.refinement.selected_instances))
                self.assertTrue(prior.optimizers)
                agent_tasks = store.agent_tasks_for_run(run.id)
                self.assertEqual(0, len(agent_tasks))
                kinds = {item.kind for item in store.artifacts_for_run(run.id)}
                self.assertTrue(
                    {"TOP5_CANDIDATES", "SIMILARITY_PROMPT", "SIMILARITY_DECISION"}
                    .issubset(kinds)
                )
                payload = json.loads(store.artifact_content(decision_ref))
                self.assertEqual(top5_ref, payload["input_artifact_refs"][0])
                self.assertIn("skill_digest", payload)
                events = [item.event_type for item in store.events_for_run(run.id)]
                self.assertEqual(
                    [
                        "SIMILARITY_CANDIDATES_READY",
                        "SIMILARITY_NODE_COMPLETED",
                    ],
                    events,
                )
            finally:
                store.close()


def _metric_semantics(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["metric"]: {
                "full_name": row["full_name"],
                "description": row["description"],
                "value_significance": row["value_significance"],
            }
            for row in csv.DictReader(handle)
        }


if __name__ == "__main__":
    unittest.main()
