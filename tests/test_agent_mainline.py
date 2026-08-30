import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore


class AgentMainlineTest(unittest.TestCase):
    def test_three_agent_two_node_architecture_replaces_legacy_boundary_agents(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            llm = FakeLLM()
            try:
                task = OptimizationTask(
                    "task-agent", "agent-mainline", "minimize", 3, 90,
                    dataset_id="xgboost-Covtype", generations=2,
                    population_size=2, random_seed=19,
                )
                run = Run("run-agent", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                completed = PriEvOEngine(
                    store,
                    DatasetRegistry(project / "resources" / "datasets"),
                    project / "resources" / "prior_knowledge",
                    llm=llm,
                    evaluation_execution_mode="inline",
                ).run(run.id)
                artifacts = list(store.artifacts_for_run(run.id))
                events = list(store.events_for_run(run.id))
                tasks = list(store.agent_tasks_for_run(run.id))
                checkpoint = store.latest_checkpoint(run.id)
                checkpoint_payload = json.loads(store.artifact_content(checkpoint.artifact_id))
                tool_rows = list(store.tool_calls_for_run(run.id))
                generation_prompts = [
                    store.artifact_content(item.id).decode("utf-8")
                    for item in artifacts if item.kind == "GENERATION_PROMPT"
                ]
                plans = list(store.generation_plans_for_run(run.id))
                traces = list(store.trace_records_for_run(run.id))
            finally:
                store.close()

        self.assertEqual("COMPLETED", completed.status.value)
        task_types = [item.task_type for item in tasks]
        self.assertEqual(0, task_types.count("SEMANTIC_SIMILARITY_SELECTION"))
        self.assertEqual(16, task_types.count("HEURISTIC_GENERATION"))
        self.assertEqual(0, task_types.count("CANDIDATE_REPAIR"))
        self.assertTrue(all(item.status.value == "COMPLETED" for item in tasks))
        kinds = {item.kind for item in artifacts}
        self.assertIn("SIMILARITY_DECISION", kinds)
        self.assertIn("CANDIDATE_DRAFT", kinds)
        self.assertIn("FINAL_SELECTION_DECISION", kinds)
        self.assertNotIn("EVOLUTION_ADVICE", kinds)
        self.assertNotIn("FINAL_EXPLANATION", kinds)
        self.assertNotIn("RESEARCH_RESULT", kinds)
        # 静态不兼容 prior 不再物化为失败 Candidate；无真实失败也无
        # KnowledgeGap 时，Repair/Literature 工具都保持零调用。
        self.assertEqual(0, len(tool_rows))
        # Artifact Store 按内容寻址；相同上下文的 Prompt 合法复用同一 ref。
        self.assertGreaterEqual(len(generation_prompts), 5)
        self.assertTrue(all("Original PriEvO prior" in item
                            for item in generation_prompts))
        self.assertTrue(all("Current strategy skill" in item
                            for item in generation_prompts))
        self.assertTrue(all("WHOLE_POPULATION" not in item
                            for item in generation_prompts))
        self.assertEqual("", checkpoint_payload["core_state"]["agent_prompt_context"])
        self.assertEqual(
            16,
            sum(item.event_type == "CANDIDATE_DRAFT_MATERIALIZED" for item in events),
        )
        self.assertEqual(16, len(plans))
        self.assertEqual(
            16,
            sum(item.span_type == "LLM_AGENT_PLAN" for item in traces),
        )
        self.assertEqual(
            0,
            sum(item.event_type == "REPAIRED_CANDIDATE_MATERIALIZED" for item in events),
        )
        self.assertIn("PRIOR_EXECUTION_COMPATIBILITY", kinds)


if __name__ == "__main__":
    unittest.main()
