import tempfile
import unittest
from pathlib import Path

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.application.agent_trace import AgentTraceQuery
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class AgentTraceTest(unittest.TestCase):
    def test_trace_exposes_durable_five_agent_path_and_causal_refs(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-trace", "trace", "minimize", 2, 50,
                    dataset_id="xgboost-Covtype", generations=1,
                    population_size=2, random_seed=31,
                )
                run = Run("run-trace", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                PriEvOEngine(
                    store, DatasetRegistry(project / "resources" / "datasets"),
                    project / "resources" / "prior_knowledge", llm=FakeLLM(),
                ).run(run.id)
                trace = AgentTraceQuery(store).trace(run.id)
            finally:
                store.close()

        task_types = [item["task_type"] for item in trace["tasks"]]
        artifact_kinds = {item["kind"] for item in trace["artifacts"]}
        event_types = {item["event_type"] for item in trace["steps"]}

        self.assertEqual(1, task_types.count("SEMANTIC_SIMILARITY_SELECTION"))
        self.assertEqual(8, task_types.count("HEURISTIC_GENERATION"))
        self.assertEqual(0, task_types.count("CANDIDATE_REPAIR"))
        self.assertTrue(all(item["status"] == "COMPLETED" for item in trace["tasks"]))
        self.assertTrue({
            "TOP5_CANDIDATES", "SIMILARITY_PROMPT", "SIMILARITY_DECISION",
            "GENERATION_REQUEST", "GENERATION_PROMPT", "CANDIDATE_DRAFT",
            "FINAL_SELECTION_DECISION",
        }.issubset(artifact_kinds))
        self.assertIn("AGENT_TASK_CREATED", event_types)
        self.assertIn("CANDIDATE_DRAFT_MATERIALIZED", event_types)
        self.assertTrue(trace["causal_edges"])
        self.assertEqual(len(trace["tasks"]), trace["summary"]["completed_task_count"])

        # 新产品 Trace 不再伪造旧三 Agent 的 advice/explanation 流程。
        self.assertFalse({
            "EVOLUTION_ADVICE", "RESEARCH_RESULT", "FINAL_EXPLANATION",
        } & artifact_kinds)


if __name__ == "__main__":
    unittest.main()
