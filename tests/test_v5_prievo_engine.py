import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.algorithm.prievo_engine import PriEvOEngine
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.infrastructure.sqlite_store import SQLiteRuntimeStore
from prievo_agent.infrastructure.fake_llm import FakeLLM


class V5PriEvOEngineTest(unittest.TestCase):
    def test_dataset_is_bound_through_prior_evaluation_and_checkpoint(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask(
                    "task-v5", "xgboost-Covtype", "minimize", 4, 88,
                    dataset_id="xgboost-Covtype", generations=2,
                    population_size=2, random_seed=7,
                )
                run = Run("run-v5", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                model = FakeLLM()
                engine = PriEvOEngine(
                    store, DatasetRegistry(project / "resources" / "datasets"),
                    project / "resources" / "prior_knowledge",
                    llm=model,
                )
                completed = engine.run(run.id)
                checkpoint = store.latest_checkpoint(run.id)
                payload = json.loads(store.artifact_content(checkpoint.artifact_id))
                results = [store.result_for_candidate(item.id)
                           for item in store.candidates_for_run(run.id)
                           if item.objective is not None]
                artifacts = store.artifacts_for_run(run.id)
                statuses = {item.status.value for item in store.candidates_for_run(run.id)}
                candidates = list(store.candidates_for_run(run.id))
                events = list(store.events_for_run(run.id))
                agent_tasks = list(store.agent_tasks_for_run(run.id))
            finally:
                store.close()
        self.assertEqual("COMPLETED", completed.status.value)
        self.assertEqual("xgboost-Covtype", checkpoint.dataset_id)
        self.assertEqual("xgboost-Covtype", payload["dataset_id"])
        self.assertTrue(checkpoint.prior_refs)
        self.assertTrue(results)
        self.assertIn("LANDSCAPE_SAMPLE", {item.kind for item in artifacts})
        self.assertIn("FINAL_OPTIMIZATION_REPORT", {item.kind for item in artifacts})
        # 静态不兼容的 original prior 仍保留为 Prompt/Evidence，但不会被
        # 物化成注定失败的 Candidate，也不会浪费一次 Repair 调用。
        final_clones = [
            item for item in candidates
            if item.lineage.get("creation_type") == "FINAL_OPTIMIZATION"
        ]
        self.assertEqual(2, len(final_clones))
        self.assertTrue(all(item.lineage.get("engineering_extension")
                            for item in final_clones))
        generated_early = [item.lineage.get("operator") for item in candidates
                           if item.lineage.get("generation") == 1]
        generated_late = [item.lineage.get("operator") for item in candidates
                          if item.lineage.get("generation") == 2]
        by_id = {item.id: item for item in candidates}
        self.assertEqual(
            {"e1": 2, "e2": 2, "m1": 2, "i1": 2},
            {operator: generated_early.count(operator)
             for operator in set(generated_early)},
        )
        self.assertEqual(
            {"e1": 2, "e2": 2, "m1": 2, "m2": 2},
            {operator: generated_late.count(operator)
             for operator in set(generated_late)},
        )
        # 同代四个 strategy 都只能从代初 retained P 选 parent：parent 的生成代
        # 必须严格早于 child（retained P 可保留更早代），不能偷看本代 offspring。
        for item in candidates:
            generation = int(item.lineage.get("generation", 0) or 0)
            if generation not in {1, 2} or item.lineage.get("creation_type"):
                continue
            for parent_id in item.lineage.get("parents", []):
                self.assertLess(
                    int(by_id[parent_id].lineage.get("generation", 0) or 0),
                    generation,
                    "同代 strategy 不得以先生成的 offspring 为 parent: child={} "
                    "parent={} child_lineage={} parent_lineage={}".format(
                        item.id, parent_id, item.lineage, by_id[parent_id].lineage
                    ),
                )
        starts = [item.payload["operator"] for item in events
                  if item.event_type == "OPERATOR_BATCH_STARTED"]
        self.assertEqual(
            ["i1", "e1", "e2", "m1", "e1", "e2", "m1", "m2"],
            starts,
        )
        selections = [item.payload for item in events
                      if item.event_type == "GENERATION_SELECTION_COMPLETED"]
        self.assertEqual(2, len(selections))
        self.assertTrue(all(item["offspring_count"] == 8 for item in selections))
        self.assertTrue(all(item["operator_batch_count"] == 4 for item in selections))
        self.assertTrue(all(item["selection_count"] == 1 for item in selections))
        self.assertTrue(all(item["combined_count"] == 10 for item in selections))
        self.assertTrue(all(len(item["population_ids"]) == 2 for item in selections))
        operator_selections = [item.payload for item in events
                               if item.event_type == "OPERATOR_SELECTION_COMPLETED"]
        self.assertEqual(2, len(operator_selections))
        self.assertTrue(all(item["offspring_count"] == 8
                            for item in operator_selections))
        self.assertTrue(all(item["combined_count"] == 10
                            for item in operator_selections))
        self.assertTrue(all(item["operator_batch_count"] == 4
                            for item in operator_selections))
        self.assertEqual(16, len(model.generation_agent_calls))
        self.assertEqual(
            16,
            sum(item.task_type == "HEURISTIC_GENERATION" for item in agent_tasks),
        )
        self.assertEqual(
            0,
            sum(item.task_type == "CANDIDATE_REPAIR" for item in agent_tasks),
        )
        self.assertEqual(0, len(model.repair_agent_calls))
        self.assertEqual(
            1,
            sum(item.task_type == "SEMANTIC_SIMILARITY_SELECTION"
                for item in agent_tasks),
        )
        evolved = [item for item in candidates
                   if item.lineage.get("generation", 0) > 0]
        self.assertTrue(all(item.lineage.get("candidate_draft_artifact_id")
                            for item in evolved))
        self.assertTrue(all(item.lineage.get("skill_digest") for item in evolved))
        self.assertEqual({"EVALUATED"}, statuses)
        self.assertEqual(
            0,
            sum(item.event_type == "CANDIDATE_FAILURE_CLASSIFIED" for item in events),
        )
        self.assertEqual(
            0,
            sum(item.event_type == "REPAIRED_CANDIDATE_MATERIALIZED" for item in events),
        )
        self.assertIn(
            "PRIOR_EXECUTION_COMPATIBILITY",
            {item.kind for item in artifacts},
        )
        self.assertEqual(
            1,
            sum(
                item.event_type == "PRIOR_EXECUTION_COMPATIBILITY_AUDITED"
                for item in events
            ),
        )
        self.assertTrue(all(item.best_configuration["dataset_id"] == "xgboost-Covtype"
                            for item in results))

    def test_checkpoint_rejects_changed_dataset_file(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_root = root / "datasets"
            dataset_root.mkdir()
            source = project / "resources" / "datasets" / "xgboost-Covtype.csv"
            target = dataset_root / source.name
            target.write_bytes(source.read_bytes())
            store = SQLiteRuntimeStore(root / "state.sqlite3", root / "artifacts")
            try:
                task = OptimizationTask("task", "dataset", "minimize", 3, 60,
                                        dataset_id="xgboost-Covtype", generations=0,
                                        population_size=2)
                run = Run("run", task.id, dataset_id=task.dataset_id)
                store.add_task(task)
                store.add_run(run)
                engine = PriEvOEngine(store, DatasetRegistry(dataset_root),
                                      project / "resources" / "prior_knowledge")
                engine.run(run.id)
                # 保持 CSV 合法但改变 digest，恢复必须拒绝。
                target.write_bytes(target.read_bytes() + b"\n")
                # 仅构造“checkpoint 后进程崩溃”的恢复 fixture；产品代码不得用
                # generic save_run 把终态 Run 复活。
                store.connection.execute(
                    "UPDATE runs SET status=? WHERE id=?",
                    ("RUNNING", run.id),
                )
                store.connection.commit()
                with self.assertRaisesRegex(RuntimeError, "Dataset digest"):
                    engine.resume(run.id)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
