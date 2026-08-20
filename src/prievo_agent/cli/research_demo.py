from __future__ import annotations

import json

from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.core.research_runner import ResearchRunner
from prievo_agent.domain.models import OptimizationTask
from prievo_agent.infrastructure.fake_evaluator import FakeEvaluator
from prievo_agent.infrastructure.fake_llm import FakeLLM


def main() -> int:
    task = OptimizationTask(
        id="research-demo",
        name="PriEvO Core 研究路径演示",
        objective="minimize",
        evaluation_budget=3,
    )
    result = ResearchRunner(
        PriEvoEvolutionCore(FakeLLM(), population_size=3), FakeEvaluator()
    ).run(task, generations=2)
    print(
        json.dumps(
            {
                "message": "PriEvO-derived Core 研究路径运行完成",
                "best_candidate_id": result.best_candidate.id,
                "objective": result.best_candidate.objective,
                "evaluated_count": result.evaluated_count,
                "operators": result.best_candidate.operators,
                "lineage": result.best_candidate.lineage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
