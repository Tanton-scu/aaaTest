from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


LANDSCAPE_METRICS = [
    "FDC",
    "FBD",
    "PLO",
    "Skewness",
    "Kurtosis",
    "CL",
    "MIE",
    "NBC",
]


@dataclass(frozen=True)
class LandscapeProfile:
    instance_name: str
    metrics: Dict[str, float]
    sample_count: int
    analyzer: str

    def __post_init__(self) -> None:
        missing = [name for name in LANDSCAPE_METRICS if name not in self.metrics]
        if missing:
            raise ValueError("LandscapeProfile 缺少指标：{}".format(", ".join(missing)))


@dataclass(frozen=True)
class SimilarInstance:
    instance_name: str
    distance: float
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]


@dataclass(frozen=True)
class OperatorEvidence:
    operator_id: str
    name: str
    module: str
    description: str
    code: str
    source_instance: str
    optimizer: str
    rank: str


@dataclass(frozen=True)
class OptimizerEvidence:
    name: str
    rank: str
    source_instance: str
    description: str
    code: str
    operators: List[OperatorEvidence]


@dataclass(frozen=True)
class SemanticRefinement:
    selected_instances: List[str]
    reason: str
    refiner: str
    raw_response: Optional[str] = None
    fallback_used: bool = False


@dataclass(frozen=True)
class InstanceSpecificPrior:
    target: LandscapeProfile
    numeric_candidates: List[SimilarInstance]
    refinement: SemanticRefinement
    optimizers: List[OptimizerEvidence]
    evidence_version: str
