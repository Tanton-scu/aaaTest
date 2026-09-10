import tempfile
import unittest
from pathlib import Path

from prievo_agent.evaluation.evaluator import DatasetEvaluator
from prievo_agent.evaluation.executor import (
    ExecutableDatasetEvaluator,
)
from prievo_agent.evaluation.datasets import DatasetRegistry
from prievo_agent.domain.errors import (
    AlgorithmOOMError,
    AlgorithmTimeoutError,
    CandidateInterfaceError,
    CandidateRuntimeError,
    CandidateSyntaxError,
)
from prievo_agent.domain.models import Candidate, OptimizationTask


HEURISTIC_A = """from util.Evaluate import evaluate

def run_tuners(file, budget, seed, maxlives):
    used_budget = 0
    consecutive_no_improve = 0
    history_configs = {}
    best_result = float('inf')
    proposals = [[0, 'a'], [2.2, 'b'], [1, 'a']]
    for config in proposals[:budget]:
        used_budget, consecutive_no_improve, history_configs, best_result, score, mapped = evaluate(
            used_budget, consecutive_no_improve, history_configs, best_result, config
        )
    return best_result
"""


HEURISTIC_B = """def run_tuners(file, budget, seed, maxlives):
    used_budget = 0
    consecutive_no_improve = 0
    history_configs = {}
    best_result = float('inf')
    proposals = [[4, 'a'], [3, 'b'], [0, 'b']]
    for config in proposals[:budget]:
        used_budget, consecutive_no_improve, history_configs, best_result, score, mapped = evaluate(
            used_budget, consecutive_no_improve, history_configs, best_result, config
        )
    return best_result
"""


def make_candidate(code, candidate_id="candidate-1"):
    return Candidate(
        id=candidate_id,
        run_id="run-evaluator",
        code=code,
        description="test heuristic",
        operators=["Synthesize"],
        lineage={"generation": 0},
    )


class ExecutableDatasetEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root / "mixed.csv").write_text(
            "x,mode,$<loss\n"
            "0,a,9\n"
            "0,b,8\n"
            "1,a,6\n"
            "1,b,5\n"
            "2,a,4\n"
            "2,b,1\n"
            "3,a,7\n"
            "3,b,3\n"
            "4,a,10\n"
            "4,b,2\n",
            encoding="utf-8",
        )
        self.registry = DatasetRegistry(root)

    def tearDown(self):
        self.directory.cleanup()

    @staticmethod
    def task(budget=3):
        return OptimizationTask(
            id="task-evaluator",
            name="mixed",
            objective="minimize",
            evaluation_budget=budget,
            dataset_id="mixed",
            generations=0,
            population_size=1,
            random_seed=17,
        )

    def test_two_heuristics_really_execute_and_produce_different_traces(self):
        evaluator = ExecutableDatasetEvaluator(self.registry, timeout_seconds=2)
        first = evaluator.execute(make_candidate(HEURISTIC_A, "A"), self.task())
        second = evaluator.execute(make_candidate(HEURISTIC_B, "B"), self.task())

        self.assertEqual(3, first.used_budget)
        self.assertEqual(3, second.used_budget)
        self.assertEqual((9.0, 1.0, 1.0), first.trajectory)
        self.assertEqual((10.0, 3.0, 3.0), second.trajectory)
        self.assertEqual(1.0, first.objective)
        self.assertEqual(3.0, second.objective)
        self.assertNotEqual(
            [item["requested_config"] for item in first.calls],
            [item["requested_config"] for item in second.calls],
        )
        self.assertEqual((2.0, "b"), first.best_configuration)
        self.assertFalse(first.calls[1]["exact_match"])
        self.assertEqual([2.0, "b"], first.calls[1]["mapped_config"])

    def test_facade_returns_domain_result_with_dataset_identity(self):
        result = DatasetEvaluator(self.registry, timeout_seconds=2).evaluate(
            make_candidate(HEURISTIC_A), self.task()
        )

        self.assertEqual("result-candidate-1", result.id)
        self.assertEqual(1.0, result.objective)
        self.assertEqual([9.0, 1.0, 1.0], result.trajectory)
        self.assertEqual(3, result.used_budget)
        self.assertEqual(2.0, result.best_configuration["x"])
        self.assertEqual("b", result.best_configuration["mode"])
        self.assertEqual("mixed", result.best_configuration["dataset_id"])
        self.assertTrue(result.best_configuration["dataset_digest"])

    def test_file_contract_is_available_but_dataset_lookup_remains_authoritative(self):
        code = """def run_tuners(file, budget, seed, maxlives):
    if file.features != ['x', 'mode']:
        raise ValueError('features missing')
    config = [file.independent_set[0][2], file.independent_set[1][1]]
    if tuple(config) not in file.dict_search:
        raise ValueError('dict_search missing')
    file.dict_search = {tuple(config): -999}
    used, lives, history, best, score, mapped = evaluate(0, 0, {}, float('inf'), config)
    return best
"""
        execution = ExecutableDatasetEvaluator(self.registry).execute(
            make_candidate(code), self.task(budget=1)
        )

        self.assertEqual((2.0, "b"), execution.best_configuration)
        self.assertEqual(1.0, execution.objective)

    def test_duplicate_mapped_configuration_does_not_consume_budget(self):
        code = """def run_tuners(file, budget, seed, maxlives):
    used_budget = 0
    consecutive_no_improve = 0
    history_configs = {}
    best_result = float('inf')
    for config in [[0, 'a'], [0.1, 'a'], [2, 'b'], [4, 'b']]:
        used_budget, consecutive_no_improve, history_configs, best_result, score, mapped = evaluate(
            used_budget, consecutive_no_improve, history_configs, best_result, config
        )
    return best_result
"""
        execution = ExecutableDatasetEvaluator(
            self.registry, timeout_seconds=2
        ).execute(make_candidate(code), self.task())

        self.assertEqual(3, execution.used_budget)
        self.assertEqual(4, len(execution.calls))
        self.assertEqual([1, 1, 2, 3], [item["used_budget"] for item in execution.calls])
        self.assertEqual([False, True, False, False], [item["duplicate"] for item in execution.calls])
        self.assertEqual(3, len(execution.trajectory))
        self.assertEqual(3, len(execution.evaluated_configurations))

    def test_syntax_interface_runtime_timeout_and_oom_are_typed(self):
        evaluator = ExecutableDatasetEvaluator(self.registry, timeout_seconds=0.35)
        cases = [
            (
                "def run_tuners(file, budget, seed, maxlives):\n    broken = (\n",
                CandidateSyntaxError,
            ),
            (
                "def run_tuners(file, budget):\n    return 1\n",
                CandidateInterfaceError,
            ),
            (
                "def run_tuners(file, budget, seed, maxlives):\n    return 1\n",
                CandidateInterfaceError,
            ),
            (
                "def run_tuners(file, budget, seed, maxlives):\n    raise ValueError('boom')\n",
                CandidateRuntimeError,
            ),
            (
                "def run_tuners(file, budget, seed, maxlives):\n    while True:\n        pass\n",
                AlgorithmTimeoutError,
            ),
            (
                "def run_tuners(file, budget, seed, maxlives):\n    raise MemoryError('oom-like')\n",
                AlgorithmOOMError,
            ),
        ]
        for code, error in cases:
            with self.subTest(error=error.__name__), self.assertRaises(error):
                evaluator.execute(make_candidate(code), self.task())

    def test_unknown_import_has_explicit_interface_classification(self):
        code = """import definitely_not_approved

def run_tuners(file, budget, seed, maxlives):
    return 1
"""
        with self.assertRaisesRegex(CandidateInterfaceError, "import 未获批准"):
            ExecutableDatasetEvaluator(self.registry).execute(
                make_candidate(code), self.task()
            )

    def test_budget_overrun_and_fabricated_best_are_rejected(self):
        overrun = """def run_tuners(file, budget, seed, maxlives):
    used_budget = 0
    consecutive_no_improve = 0
    history_configs = {}
    best_result = float('inf')
    for config in [[0, 'a'], [1, 'a'], [2, 'a'], [3, 'a']]:
        used_budget, consecutive_no_improve, history_configs, best_result, score, mapped = evaluate(
            used_budget, consecutive_no_improve, history_configs, best_result, config
        )
    return best_result
"""
        fabricated = """def run_tuners(file, budget, seed, maxlives):
    used_budget, lives, history, best, score, mapped = evaluate(0, 0, {}, float('inf'), [2, 'b'])
    return best - 1
"""
        evaluator = ExecutableDatasetEvaluator(self.registry, timeout_seconds=2)
        for code in (overrun, fabricated):
            with self.subTest(code=code[:30]), self.assertRaises(CandidateInterfaceError):
                evaluator.execute(make_candidate(code), self.task())

    def test_budget_cannot_exceed_unique_dataset_search_space(self):
        with self.assertRaisesRegex(CandidateInterfaceError, "超过 Dataset"):
            ExecutableDatasetEvaluator(self.registry).execute(
                make_candidate(HEURISTIC_A), self.task(budget=11)
            )


if __name__ == "__main__":
    unittest.main()
