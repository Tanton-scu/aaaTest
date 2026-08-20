from __future__ import annotations

from typing import Dict, Sequence

from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    LandscapeProfile,
    SemanticRefinement,
    SimilarInstance,
)


class RecordedLandscapeAnalyzer:
    """消费由 PriEvO reference-compatible FLA 离线计算的记录；不改写指标定义。"""

    def __init__(self, recorded_metrics: Dict[str, Dict[str, float]]) -> None:
        self.recorded_metrics = recorded_metrics

    def analyze(
        self, instance_name: str, samples: Sequence[Dict[str, object]]
    ) -> LandscapeProfile:
        metrics = self.recorded_metrics[instance_name]
        return LandscapeProfile(
            instance_name,
            {name: float(metrics[name]) for name in LANDSCAPE_METRICS},
            len(samples),
            "recorded PriEvO-compatible FLA metrics",
        )


class DeterministicPriorRefiner:
    def __init__(self, selected_count: int = 2) -> None:
        self.selected_count = selected_count

    def refine(
        self, target: LandscapeProfile, candidates: Sequence[SimilarInstance]
    ) -> SemanticRefinement:
        selected = [item.instance_name for item in candidates[: self.selected_count]]
        return SemanticRefinement(
            selected,
            "Deterministic fake：保留数值距离最小的 {} 个历史实例".format(len(selected)),
            "deterministic-fake-refiner",
            raw_response="[[{}]]".format("]] [[".join(selected)),
        )
