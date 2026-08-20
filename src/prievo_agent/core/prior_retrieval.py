from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    InstanceSpecificPrior,
    LandscapeProfile,
    SemanticRefinement,
    SimilarInstance,
)
from prievo_agent.domain.prior_ports import PriorRepository, PriorSemanticRefiner
from prievo_agent.domain.errors import PriorRepositoryUnavailableError


SMALLER_BETTER = {"FBD", "PLO", "MIE", "NBC"}
CLOSE_TO_ZERO = {"FDC", "Skewness", "Kurtosis"}


class NumericalLandscapeRetriever:
    """保持 source `dataset_distance.py` 的定向、min-max 与等权欧氏距离。"""

    def top_k(
        self,
        target: LandscapeProfile,
        histories: Sequence[LandscapeProfile],
        k: int = 5,
    ) -> List[SimilarInstance]:
        candidates = [
            item
            for item in histories
            if item.instance_name.strip().lower()
            != target.instance_name.strip().lower()
        ]
        if not candidates:
            return []
        normalized_by_metric: Dict[str, List[float]] = {}
        target_normalized: Dict[str, float] = {}
        for metric in LANDSCAPE_METRICS:
            raw = [float(item.metrics[metric]) for item in candidates]
            transformed, target_value = self._forward_and_normalize(
                metric, raw, float(target.metrics[metric])
            )
            normalized_by_metric[metric] = transformed
            target_normalized[metric] = target_value

        results = []
        for index, item in enumerate(candidates):
            normalized = {
                metric: normalized_by_metric[metric][index]
                for metric in LANDSCAPE_METRICS
            }
            distance = math.sqrt(
                sum(
                    (normalized[metric] - target_normalized[metric]) ** 2
                    for metric in LANDSCAPE_METRICS
                )
            )
            results.append(
                SimilarInstance(
                    item.instance_name,
                    distance,
                    dict(item.metrics),
                    normalized,
                )
            )
        ranked = sorted(
            enumerate(results), key=lambda pair: (pair[1].distance, pair[0])
        )
        return [pair[1] for pair in ranked[:k]]

    def _forward_and_normalize(
        self, metric: str, raw: List[float], target: float
    ) -> Tuple[List[float], float]:
        raw_min, raw_max = min(raw), max(raw)
        clamped = min(max(target, raw_min), raw_max)
        if metric in SMALLER_BETTER:
            transformed = [raw_max - value for value in raw]
            target_forward = raw_max - clamped
        elif metric in CLOSE_TO_ZERO:
            max_abs = max(abs(value) for value in raw)
            if max_abs == 0:
                transformed = [0.5 for _ in raw]
                return transformed, 0.5
            transformed = [1 - abs(value) / max_abs for value in raw]
            target_forward = 1 - abs(clamped) / max_abs
        else:  # CL larger-better
            transformed = list(raw)
            target_forward = clamped
        low, high = min(transformed), max(transformed)
        if high - low < 1e-6:
            return [0.5 for _ in transformed], 0.5
        return (
            [(value - low) / (high - low) for value in transformed],
            (target_forward - low) / (high - low),
        )


class PriorRetrievalService:
    def __init__(
        self,
        repository: PriorRepository,
        refiner: PriorSemanticRefiner = None,
        numerical: NumericalLandscapeRetriever = None,
    ) -> None:
        self.repository = repository
        self.refiner = refiner
        self.numerical = numerical or NumericalLandscapeRetriever()

    def retrieve_numeric(
        self, target: LandscapeProfile, top_k: int = 5
    ) -> List[SimilarInstance]:
        """只执行 deterministic numeric Top-K，便于在语义阶段前持久化事实。"""
        try:
            histories = self.repository.landscape_profiles()
        except Exception as exc:
            raise PriorRepositoryUnavailableError(
                "structured prior repository 不可用"
            ) from exc
        numeric = self.numerical.top_k(target, histories, top_k)
        if not numeric:
            raise ValueError("结构化 prior knowledge 中没有可比较的历史实例")
        return numeric

    def extract(
        self,
        target: LandscapeProfile,
        numeric: Sequence[SimilarInstance],
        refinement: SemanticRefinement,
    ) -> InstanceSpecificPrior:
        """校验 semantic decision 后从 repository 提取真实 Prior evidence。"""
        allowed = {item.instance_name for item in numeric}
        selected = list(dict.fromkeys(refinement.selected_instances))
        if not 1 <= len(selected) <= 3 or any(name not in allowed for name in selected):
            raise ValueError("语义精排必须只从 numeric Top-K 选择 1-3 个实例")
        validated = SemanticRefinement(
            selected,
            refinement.reason,
            refinement.refiner,
            refinement.raw_response,
            refinement.fallback_used,
        )
        optimizers = []
        try:
            for instance_name in validated.selected_instances:
                optimizers.extend(self.repository.strong_optimizers(instance_name))
            evidence_version = self.repository.evidence_version()
        except Exception as exc:
            raise PriorRepositoryUnavailableError(
                "structured prior evidence 读取失败"
            ) from exc
        return InstanceSpecificPrior(
            target, list(numeric), validated, optimizers, evidence_version
        )

    def retrieve(self, target: LandscapeProfile, top_k: int = 5) -> InstanceSpecificPrior:
        """兼容的一步式入口；语义阶段失败必须显式失败，禁止 numeric 冒充 LLM。"""

        numeric = self.retrieve_numeric(target, top_k)
        if self.refiner is None:
            raise RuntimeError(
                "一步式 retrieve 未配置 semantic refiner；产品路径应使用 SimilarityAgent"
            )
        refinement = self.refiner.refine(target, numeric)
        return self.extract(target, numeric, refinement)
