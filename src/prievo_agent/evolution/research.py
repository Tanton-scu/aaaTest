from __future__ import annotations

from dataclasses import dataclass
from typing import List

from prievo_agent.domain.models import Candidate, CandidateStatus, OptimizationTask, Run
from prievo_agent.domain.ports import CandidateEvaluator

from .population import PriEvoEvolutionCore
from .schedule import is_early_generation, operators_for_generation
from .selection import select_population_early, select_population_late


@dataclass(frozen=True)
class ResearchRunResult:
    best_candidate: Candidate
    population: List[Candidate]
    evaluated_count: int


class ResearchRunner:
    """无 Runtime/DB 依赖的同步研究回归入口，复用同一 Evolution Core。"""

    def __init__(
        self, core: PriEvoEvolutionCore, evaluator: CandidateEvaluator
    ) -> None:
        self.core = core
        self.evaluator = evaluator

    def run(
        self, task: OptimizationTask, generations: int = 2
    ) -> ResearchRunResult:
        run = Run(id="research-{}".format(task.id), task_id=task.id)
        population = self.core.propose(task, run)
        evaluated_count = self._evaluate(population, task)
        population = select_population_early(population, self.core.population_size)

        for generation in range(1, generations + 1):
            generation_offspring = []
            for operator in operators_for_generation(generation, generations):
                offspring = self.core.evolve_operator(
                    population, run.id, generation, operator
                )
                evaluated_count += self._evaluate(offspring, task)
                generation_offspring.extend(offspring)
            # 论文/用户契约：四个策略都基于代初 P 生成各 P 个候选，统一将
            # 原 P + 4P 按当前阶段的管理规则筛回 P。
            combined = [*population, *generation_offspring]
            if is_early_generation(generation, generations):
                population = select_population_early(
                    combined, self.core.population_size
                )
            else:
                population = select_population_late(
                    combined, self.core.population_size
                )

        best = min(population, key=lambda item: float(item.objective))
        for candidate in population:
            candidate.status = CandidateStatus.EVALUATED
        return ResearchRunResult(best, population, evaluated_count)

    def _evaluate(
        self, candidates: List[Candidate], task: OptimizationTask
    ) -> int:
        for candidate in candidates:
            result = self.evaluator.evaluate(candidate, task)
            candidate.objective = result.objective
            candidate.status = CandidateStatus.EVALUATED
        return len(candidates)
