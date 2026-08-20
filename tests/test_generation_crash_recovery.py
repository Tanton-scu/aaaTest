from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.application.agent_dispatcher import AgentDispatchError
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore


class _FailSecondGenerationCall(FakeLLM):
    def __init__(self):
        super().__init__()
        self.generation_attempts = 0

    def generate_heuristic_draft(self, prompt):
        self.generation_attempts += 1
        if self.generation_attempts == 2:
            raise RuntimeError("scripted LLM process death on second draft")
        return super().generate_heuristic_draft(prompt)


class GenerationCrashRecoveryTest(unittest.TestCase):
    def test_first_draft_and_candidate_survive_second_llm_failure(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    # 20 evolution + 2 seeds * 4 final optimization。
                    "task", "xgboost-Covtype", "minimize", 2, 28,
                    dataset_id="xgboost-Covtype", generations=1,
                    population_size=2, random_seed=13,
                )
                run = Run("run", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                model = _FailSecondGenerationCall()
                engine = PriEvOEngine(
                    store,
                    DatasetRegistry(project / "resources" / "datasets"),
                    project / "resources" / "prior_knowledge",
                    llm=model,
                )

                with self.assertRaises(AgentDispatchError):
                    engine.run(run.id)

                first_drafts = [
                    item for item in store.artifacts_for_run(run.id)
                    if item.kind == "CANDIDATE_DRAFT"
                ]
                first_candidates = [
                    item for item in store.candidates_for_run(run.id)
                    if item.lineage.get("candidate_draft_artifact_id")
                ]
                self.assertEqual(1, len(first_drafts))
                self.assertEqual(1, len(first_candidates))
                first_candidate_id = first_candidates[0].id
                calls_before_resume = len(model.generation_agent_calls)

                completed = engine.resume(run.id)

                self.assertEqual("COMPLETED", completed.status.value)
                # checkpoint 恢复会把 FakeLLM 的非事实调用列表回滚到 g0；其后
                # 只发生 7 次新调用，首个已完成 Draft 不会再次请求模型。
                self.assertEqual(7, len(model.generation_agent_calls))
                self.assertEqual(9, model.generation_attempts)
                self.assertEqual(1, calls_before_resume)
                recovered = store.candidate_by_id(first_candidate_id)
                self.assertIsNotNone(recovered.objective)
                tasks = list(store.agent_tasks_for_run(run.id))
                self.assertEqual(
                    8,
                    sum(item.task_type == "HEURISTIC_GENERATION" for item in tasks),
                )
                self.assertTrue(all(
                    item.status.value == "COMPLETED"
                    for item in tasks if item.task_type == "HEURISTIC_GENERATION"
                ))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
