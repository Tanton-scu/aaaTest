from __future__ import annotations

from typing import Dict, List, Protocol, Sequence

from .models import (
    LandscapeProfile,
    OptimizerEvidence,
    SemanticRefinement,
    SimilarInstance,
)


class PriorRepository(Protocol):
    def landscape_profiles(self) -> Sequence[LandscapeProfile]: ...
    def strong_optimizers(self, instance_name: str) -> List[OptimizerEvidence]: ...
    def evidence_version(self) -> str: ...


class PriorSemanticRefiner(Protocol):
    def refine(
        self, target: LandscapeProfile, candidates: Sequence[SimilarInstance]
    ) -> SemanticRefinement: ...


class LandscapeAnalyzer(Protocol):
    def analyze(self, instance_name: str, samples: Sequence[Dict[str, object]]) -> LandscapeProfile: ...
