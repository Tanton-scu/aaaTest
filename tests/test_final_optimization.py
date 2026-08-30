from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 本测试只执行本地 evaluator；精简测试环境可能尚未安装产品 LLM 的 httpx
# 依赖。algorithm package 的聚合 __init__ 会间接导入它，因此仅在缺包时放置
# 一个不会被调用的模块占位，绝不替代 evaluator 或网络行为。
try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    sys.modules["httpx"] = types.ModuleType("httpx")

from prievo_agent.algorithm.executable_dataset_evaluator import (
    ExecutableDatasetEvaluator,
)
from prievo_agent.datasets.registry import DatasetRegistry
from prievo_agent.domain.errors import TransientEvaluationError
from prievo_agent.domain.models import (
    Candidate,
    CandidateStatus,
    OptimizationTask,
    Run,
    RunStatus,
)
from prievo_agent.infrastructure.testing.sqlite_store import SQLiteRuntimeStore
from prievo_agent.runtime.final_optimization import (
    FINAL_OPTIMIZATION_DISCLOSURE_CN,
    FinalOptimizationFailed,
    FinalOptimizationService,
)


SEEDED_HEURISTIC = """from util.Evaluate import evaluate
import random

def run_tuners(file, budget, seed, maxlives):
    used_budget = 0
    consecutive_no_improve = 0
    history_configs = {}
    best_result = float('inf')
    proposals = list(file.dict_search.keys())
    random.Random(seed).shuffle(proposals)
    for config in proposals[:budget]:
        used_budget, consecutive_no_improve, history_configs, best_result, score, mapped = evaluate(
            used_budget, consecutive_no_improve, history_configs, best_result, config
        )
    return best_result
"""


BROKEN_HEURISTIC = """def run_tuners(file, budget, seed, maxlives):
    raise ValueError('candidate failure in final optimization')
"""


class FakeClock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)


class TrackingExecutableEvaluator:
    version = "executable-final-test-v1"
    evaluation_parameters_version = "sandbox-test-v1"

    def __init__(self, delegate, transient_first=False):
        self.delegate = delegate
        self.transient_first = transient_first
        self.calls = []
        self.attempts = {}

    def evaluate(self, candidate, task):
        call = {
            "candidate_id": candidate.id,
            "seed": task.random_seed,
            "budget": task.evaluation_budget,
        }
        self.calls.append(call)
        attempt = self.attempts.get(candidate.id, 0) + 1
        self.attempts[candidate.id] = attempt
        if (
            self.transient_first
            and candidate.id.startswith("final-opt-")
            and attempt == 1
        ):
            raise TransientEvaluationError("temporary benchmark infrastructure")
        return self.delegate.evaluate(candidate, task)


class FinalOptimizationServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prievo-final-opt-")
        self.root = Path(self.temp.name)
        rows = ["x,$<loss"]
        # 20 个唯一 configuration，足以证明 final budget 大于 evolution budget，
        # 同时让不同 seed 真正改变 executable heuristic 的访问轨迹。
        for index in range(20):
            rows.append("{},{}".format(index, (index * 7) % 23 + 1))
        (self.root / "fixture.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        self.registry = DatasetRegistry(self.root)
        self.dataset = self.registry.load("fixture")
        self.store = SQLiteRuntimeStore(
            self.root / "runtime.sqlite3", self.root / "artifacts"
        )

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_evaluator_sandbox_parameters_are_part_of_durable_identity(self):
        first = FinalOptimizationService(
            self.store,
            ExecutableDatasetEvaluator(
                self.registry, timeout_seconds=3, memory_limit_mb=256, max_lives=50
            ),
        )
        changed_timeout = FinalOptimizationService(
            self.store,
            ExecutableDatasetEvaluator(
                self.registry, timeout_seconds=4, memory_limit_mb=256, max_lives=50
            ),
        )
        changed_memory = FinalOptimizationService(
            self.store,
            ExecutableDatasetEvaluator(
                self.registry, timeout_seconds=3, memory_limit_mb=512, max_lives=50
            ),
        )
        self.assertNotEqual(
            first.evaluation_parameters_version,
            changed_timeout.evaluation_parameters_version,
        )
        self.assertNotEqual(
            first.evaluation_parameters_version,
            changed_memory.evaluation_parameters_version,
        )

    def _facts(self, code=SEEDED_HEURISTIC, total_budget=40):
        task = OptimizationTask(
            id="task-final",
            name="final fixture",
            objective="minimize",
            evaluation_budget=2,
            total_budget=total_budget,
            dataset_id="fixture",
            generations=1,
            population_size=1,
            random_seed=17,
        )
        run = Run(
            "run-final", task.id, status=RunStatus.RUNNING,
            dataset_id="fixture",
        )
        selected = Candidate(
            id="selected-evolution-candidate",
            run_id=run.id,
            code=code,
            description="进化阶段胜出 Candidate",
            operators=["Random Search"],
            lineage={"generation": 1, "creation_type": "EVOLUTION"},
            status=CandidateStatus.EVALUATED,
            objective=9.0,
            code_artifact_id="original-code-artifact",
            evaluation_artifact_id="original-evaluation-artifact",
        )
        self.store.add_task(task)
        self.store.add_run(run)
        self.store.add_candidate(selected)
        return task, run, selected

    def test_real_executable_multi_seed_is_independent_budgeted_and_idempotent(self):
        task, run, selected = self._facts()
        evaluator = TrackingExecutableEvaluator(
            ExecutableDatasetEvaluator(self.registry, timeout_seconds=2)
        )
        service = FinalOptimizationService(
            self.store,
            evaluator,
            seeds=(29, 11),
            final_budget_per_seed=4,
        )

        report = service.optimize(run, task, selected, self.dataset.info.digest)

        self.assertEqual("SUCCESS", report.status)
        self.assertEqual((11, 29), report.seeds)
        self.assertEqual(2, report.successful_seed_count)
        self.assertEqual(8, report.total_used_budget)
        self.assertEqual(4, len(report.aggregate_trajectory))
        self.assertTrue(report.best_configuration)
        self.assertEqual(FINAL_OPTIMIZATION_DISCLOSURE_CN, report.disclosure_cn)

        final_calls = [
            item for item in evaluator.calls
            if item["candidate_id"].startswith("final-opt-")
        ]
        self.assertEqual(2, len(final_calls))
        self.assertEqual({11, 29}, {item["seed"] for item in final_calls})
        self.assertEqual({4}, {item["budget"] for item in final_calls})
        self.assertGreater(4, task.evaluation_budget)

        jobs = list(self.store.evaluation_jobs_for_run(run.id))
        self.assertEqual(2, len(jobs))
        self.assertEqual({11, 29}, {job.seed for job in jobs})
        self.assertEqual({4}, {job.budget for job in jobs})
        self.assertEqual(2, len({job.idempotency_key for job in jobs}))
        self.assertTrue(all(job.status.value == "SUCCESS" for job in jobs))

        candidates = list(self.store.candidates_for_run(run.id))
        clones = [
            item for item in candidates
            if item.lineage.get("creation_type") == "FINAL_OPTIMIZATION"
        ]
        self.assertEqual(2, len(clones))
        self.assertEqual({selected.id}, {
            item.lineage["source_candidate_id"] for item in clones
        })
        self.assertTrue(all(item.id != selected.id for item in clones))
        self.assertTrue(all(item.code == selected.code for item in clones))
        self.assertEqual(2, len({
            self.store.result_for_candidate(item.id).id for item in clones
        }))

        persisted_original = self.store.candidate_by_id(selected.id)
        self.assertEqual(9.0, persisted_original.objective)
        self.assertEqual("original-code-artifact", persisted_original.code_artifact_id)
        self.assertEqual(
            "original-evaluation-artifact",
            persisted_original.evaluation_artifact_id,
        )
        refreshed = self.store.get_run(run.id)
        self.assertEqual(8, refreshed.consumed_evaluations)
        self.assertEqual(0, refreshed.reserved_evaluations)

        artifact_count = len([
            item for item in self.store.artifacts_for_run(run.id)
            if item.kind == "FINAL_OPTIMIZATION_REPORT"
        ])
        event_count = len([
            item for item in self.store.events_for_run(run.id)
            if item.event_type == "FINAL_OPTIMIZATION_COMPLETED"
        ])
        call_count = len(evaluator.calls)
        recovered = service.optimize(run, task, selected, self.dataset.info.digest)

        self.assertEqual(report.artifact_id, recovered.artifact_id)
        self.assertEqual(call_count, len(evaluator.calls))
        self.assertEqual(2, len(list(self.store.evaluation_jobs_for_run(run.id))))
        self.assertEqual(8, self.store.get_run(run.id).consumed_evaluations)
        self.assertEqual(artifact_count, len([
            item for item in self.store.artifacts_for_run(run.id)
            if item.kind == "FINAL_OPTIMIZATION_REPORT"
        ]))
        self.assertEqual(event_count, len([
            item for item in self.store.events_for_run(run.id)
            if item.event_type == "FINAL_OPTIMIZATION_COMPLETED"
        ]))
        payload = json.loads(
            self.store.artifact_content(report.artifact_id).decode("utf-8")
        )
        self.assertTrue(payload["engineering_extension"])
        self.assertTrue(payload["reference_executable_missing"])
        self.assertIn("reference PriEvO", payload["disclosure_cn"])
        self.assertEqual(2, payload["successful_seed_count"])

    def test_transient_infrastructure_uses_queue_retry_without_double_charge(self):
        task, run, selected = self._facts()
        clock = FakeClock()
        evaluator = TrackingExecutableEvaluator(
            ExecutableDatasetEvaluator(self.registry, timeout_seconds=2),
            transient_first=True,
        )
        service = FinalOptimizationService(
            self.store,
            evaluator,
            seeds=(101, 202),
            final_budget_per_seed=4,
            clock=clock,
            sleeper=clock.sleep,
            max_poll_seconds=0.2,
        )

        report = service.optimize(run, task, selected, self.dataset.info.digest)

        self.assertEqual("SUCCESS", report.status)
        jobs = list(self.store.evaluation_jobs_for_run(run.id))
        self.assertEqual({2}, {job.attempts for job in jobs})
        self.assertEqual(4, len(evaluator.calls))
        self.assertTrue(clock.sleeps)
        self.assertEqual(8, self.store.get_run(run.id).consumed_evaluations)
        self.assertEqual(0, self.store.get_run(run.id).reserved_evaluations)
        self.assertEqual(2, len({job.result_id for job in jobs}))

    def test_candidate_failure_is_persisted_and_recovered_as_failure(self):
        task, run, selected = self._facts(code=BROKEN_HEURISTIC)
        evaluator = TrackingExecutableEvaluator(
            ExecutableDatasetEvaluator(self.registry, timeout_seconds=2)
        )
        service = FinalOptimizationService(
            self.store,
            evaluator,
            seeds=(3, 7),
            final_budget_per_seed=4,
        )

        with self.assertRaises(FinalOptimizationFailed) as raised:
            service.optimize(run, task, selected, self.dataset.info.digest)
        report = raised.exception.report

        self.assertEqual("FAILED", report.status)
        self.assertEqual(0, report.successful_seed_count)
        self.assertIsNone(report.best_objective)
        self.assertTrue(report.artifact_id)
        self.assertEqual({"DEAD"}, {trial.status for trial in report.trials})
        self.assertTrue(all(trial.error_code for trial in report.trials))
        self.assertEqual(0, self.store.get_run(run.id).consumed_evaluations)
        self.assertEqual(0, self.store.get_run(run.id).reserved_evaluations)
        calls = len(evaluator.calls)

        with self.assertRaises(FinalOptimizationFailed) as recovered:
            service.optimize(run, task, selected, self.dataset.info.digest)
        self.assertEqual(report.artifact_id, recovered.exception.report.artifact_id)
        self.assertEqual(calls, len(evaluator.calls))
        self.assertEqual(1, len([
            item for item in self.store.events_for_run(run.id)
            if item.event_type == "FINAL_OPTIMIZATION_FAILED"
        ]))


if __name__ == "__main__":
    unittest.main()
