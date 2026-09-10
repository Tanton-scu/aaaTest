from __future__ import annotations

import json
from dataclasses import asdict

from prievo_agent.knowledge.prior.retrieval import PriorRetrievalService
from prievo_agent.domain.events import EventType
from prievo_agent.knowledge.prior.models import InstanceSpecificPrior, LandscapeProfile
from prievo_agent.domain.ports import RuntimeStore


class PriorService:
    def __init__(self, store: RuntimeStore, retrieval: PriorRetrievalService) -> None:
        self.store = store
        self.retrieval = retrieval

    def retrieve_and_persist(
        self, run_id: str, profile: LandscapeProfile
    ) -> InstanceSpecificPrior:
        profile_artifact = self.store.put_artifact(
            run_id,
            "LANDSCAPE_PROFILE",
            json.dumps(asdict(profile), ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run_id,
            EventType.LANDSCAPE_ANALYZED.value,
            "目标实例 landscape 已分析",
            artifact_id=profile_artifact.id,
            metrics=profile.metrics,
            sample_count=profile.sample_count,
        )
        prior = self.retrieval.retrieve(profile)
        prior_artifact = self.store.put_artifact(
            run_id,
            "INSTANCE_SPECIFIC_PRIOR",
            json.dumps(asdict(prior), ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run_id,
            EventType.PRIOR_RETRIEVED.value,
            "instance-specific prior 已检索",
            artifact_id=prior_artifact.id,
            numeric_candidates=[
                {"instance": item.instance_name, "distance": item.distance}
                for item in prior.numeric_candidates
            ],
            semantic_selected=prior.refinement.selected_instances,
            fallback_used=prior.refinement.fallback_used,
            evidence_version=prior.evidence_version,
        )
        return prior
