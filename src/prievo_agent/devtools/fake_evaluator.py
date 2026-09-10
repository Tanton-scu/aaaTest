from __future__ import annotations

import hashlib

from prievo_agent.domain.models import Candidate, EvaluationResult, OptimizationTask


class FakeEvaluator:
    """无需 benchmark/API Key 的确定性评价适配器。"""

    # 故障注入子类仍代表同一评价语义；显式版本可让 pause/restart 在更换
    # wrapper 实例时安全复用已有 Result，同时不会放宽真实 evaluator 校验。
    version = "fake-evaluator-v1"

    def evaluate(self, candidate: Candidate, task: OptimizationTask) -> EvaluationResult:
        digest = hashlib.sha256(candidate.code.encode("utf-8")).hexdigest()
        objective = round(0.1 + (int(digest[:8], 16) % 8000) / 10000, 4)
        trajectory = [1.0, 0.6, objective]
        return EvaluationResult(
            id="result-{}".format(candidate.id),
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            objective=objective,
            trajectory=trajectory,
            best_configuration={
                "strategy": str(candidate.lineage.get("operator", "unknown"))
            },
            used_budget=min(task.evaluation_budget, len(trajectory)),
        )
