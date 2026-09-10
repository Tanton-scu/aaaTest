import tempfile
import unittest
from pathlib import Path

from prievo_agent.application.orchestration.run_facade import RunApplicationFacade
from prievo_agent.infrastructure.composition import RuntimeComposition


class ObservabilityContractTest(unittest.TestCase):
    def test_metrics_derive_from_events_jobs_and_run_state(self):
        with tempfile.TemporaryDirectory() as directory:
            composition = RuntimeComposition(Path(directory))
            facade = RunApplicationFacade(
                composition.open_store, composition.execute, auto_start=False,
                dataset_registry=composition.dataset_registry,
            )
            try:
                run_id = facade.create_run(
                    "xgboost-Covtype", generations=2, population_size=3,
                    candidate_budget=3, random_seed=7,
                )["run_id"]
                facade.resume(run_id)
            finally:
                facade.shutdown()
            snapshot = facade.metrics(run_id)
            self.assertGreater(snapshot["run_duration_seconds"], 0)
            # 静态不兼容 prior 以 Artifact/Event 审计，不制造必败 Candidate。
            self.assertEqual(0.0, snapshot["candidate_invalid_rate"])
            # 27 个成功演化评价 + 2 个 final seed clone。
            self.assertEqual(29, snapshot["evaluation_attempts"])
            self.assertEqual(0, snapshot["evaluation_retries"])
            self.assertEqual(93, snapshot["budget_consumed"])
            self.assertGreaterEqual(snapshot["fitness_improvement"], 0)


if __name__ == "__main__":
    unittest.main()
