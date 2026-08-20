from __future__ import annotations

import importlib.util
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.core.research_runner import ResearchRunner
from prievo_agent.core.schedule import operators_for_generation
from prievo_agent.core.selection import (
    select_parents,
    select_population_early,
    select_population_late,
)
from prievo_agent.core.serialization import candidate_from_dict, candidate_to_dict
from prievo_agent.domain.models import Candidate, OptimizationTask
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM


def candidate(identifier: str, objective: float, operators: list) -> Candidate:
    return Candidate(
        id=identifier,
        run_id="run-characterization",
        code="def run_tuners(): return {!r}".format(identifier),
        description=identifier,
        operators=operators,
        lineage={"operator": "fixture", "generation": 0},
        objective=objective,
    )


def load_read_only_reference(name: str, path: Path):
    """加载 reference characterization，不在只读参考树生成 bytecode。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


class PriEvoCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.population = [
            candidate("a", 0.1, ["Name: A", "Name: B"]),
            candidate("b", 0.1, ["Name: A"]),
            candidate("c", 0.2, ["Name: C"]),
            candidate("d", 0.3, ["Name: D", "Name: E"]),
        ]

    def test_operator_schedule_preserves_reference_halves(self) -> None:
        self.assertEqual(["i1", "e1", "e2", "m1"], operators_for_generation(1, 4))
        self.assertEqual(["i1", "e1", "e2", "m1"], operators_for_generation(2, 4))
        self.assertEqual(["e1", "e2", "m1", "m2"], operators_for_generation(3, 4))
        self.assertEqual(["e1", "e2", "m1", "m2"], operators_for_generation(4, 4))
        self.assertEqual(
            ["i1", "e1", "e2", "m1"], operators_for_generation(1, 1)
        )

    def test_parent_selection_matches_reference_formula(self) -> None:
        reference_path = (
            ROOT.parent / "references" / "prievo" / "prievo" / "methods" / "selection" / "prob_rank.py"
        )
        module = load_read_only_reference("reference_prob_rank", reference_path)
        random.seed(2024)
        expected = module.parent_selection(self.population, 5)
        actual = select_parents(self.population, 5, random.Random(2024))
        self.assertEqual([item.id for item in expected], [item.id for item in actual])

    def test_population_selection_matches_reference_on_fixture(self) -> None:
        reference_path = (
            ROOT.parent / "references" / "prievo" / "prievo" / "methods" / "management" / "pop_greedy.py"
        )
        module = load_read_only_reference("reference_pop_greedy", reference_path)
        reference_population = [
            {
                "id": item.id,
                "objective": item.objective,
                "operators": "\n".join(item.operators),
            }
            for item in self.population
        ]
        early_expected = module.population_management(reference_population, 3)
        late_expected = module.population_management_later(reference_population, 3)
        self.assertEqual(
            [item["id"] for item in early_expected],
            [item.id for item in select_population_early(self.population, 3)],
        )
        self.assertEqual(
            [item["id"] for item in late_expected],
            [item.id for item in select_population_late(self.population, 3)],
        )

    def test_candidate_serialization_round_trip(self) -> None:
        original = self.population[0]
        self.assertEqual(original, candidate_from_dict(candidate_to_dict(original)))

    def test_mocked_llm_generation_records_operator_and_lineage(self) -> None:
        llm = FakeLLM()
        core = PriEvoEvolutionCore(llm, population_size=2)
        task = OptimizationTask("task", "task", "minimize", 3)
        initial = core.propose(task, __import__("prievo_agent.domain.models", fromlist=["Run"]).Run("run", task.id))
        for item in initial:
            item.objective = 0.5
        offspring = core.evolve(initial, "run", 1, 2)
        self.assertEqual(["i1", "i1"], [call[0] for call in llm.calls[:2]])
        self.assertEqual(
            ["i1", "i1", "e1", "e1", "e2", "e2", "m1", "m1"],
            [item.lineage["operator"] for item in offspring],
        )
        self.assertTrue(all(item.code.startswith("def run_tuners") for item in offspring))

    def test_research_runner_uses_extracted_core(self) -> None:
        runner = ResearchRunner(PriEvoEvolutionCore(FakeLLM(), 3), FakeEvaluator())
        result = runner.run(OptimizationTask("research", "research", "minimize", 3), 2)
        self.assertEqual(3, len(result.population))
        self.assertEqual(27, result.evaluated_count)
        self.assertEqual(
            min(item.objective for item in result.population),
            result.best_candidate.objective,
        )


if __name__ == "__main__":
    unittest.main()
