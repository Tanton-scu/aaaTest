from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from prievo_agent.application.prior_service import PriorApplicationService
from prievo_agent.core.evolution import PriEvoEvolutionCore
from prievo_agent.core.prior_retrieval import PriorRetrievalService
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import OptimizationTask, Run
from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    LandscapeProfile,
    OperatorEvidence,
    OptimizerEvidence,
)
from prievo_agent.infrastructure.fake_llm import FakeLLM
from prievo_agent.infrastructure.memory import InMemoryRuntimeStore
from prievo_agent.infrastructure.prior_adapters import (
    DeterministicPriorRefiner,
    RecordedLandscapeAnalyzer,
)
from prievo_agent.infrastructure.prior_repository import StructuredPriorRepository


def _metrics(offset: float) -> Dict[str, float]:
    return {
        "FDC": 0.1 + offset,
        "FBD": 1.0 + offset,
        "PLO": 0.2 + offset,
        "Skewness": 0.3 + offset,
        "Kurtosis": 1.2 + offset,
        "CL": 0.8 + offset,
        "MIE": 0.6 + offset,
        "NBC": 0.7 + offset,
    }


def _optimizer(instance: str, name: str, rank: str, operator_id: str) -> OptimizerEvidence:
    operator = OperatorEvidence(
        operator_id,
        "Operator {}".format(operator_id),
        "Candidate Generation",
        "fixture operator evidence",
        "def operator(): return {!r}".format(operator_id),
        instance,
        name,
        rank,
    )
    return OptimizerEvidence(
        name,
        rank,
        instance,
        "{} 的历史强优化器".format(instance),
        "def run_tuners(file, budget, seed, maxlives): return {!r}".format(name),
        [operator],
    )


@dataclass(frozen=True)
class PriorHarnessReport:
    nearest_instance: str
    nearest_distance: float
    numeric_count: int
    semantic_selected: List[str]
    optimizer_count: int
    operator_provenance_complete: bool
    trace_events: List[str]
    artifact_kinds: List[str]
    seeded_candidate_count: int

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class PriorRetrievalHarness:
    def run(self) -> PriorHarnessReport:
        histories = [
            LandscapeProfile("history-nearest", _metrics(0.0), 100, "fixture"),
            LandscapeProfile("history-second", _metrics(0.15), 100, "fixture"),
            LandscapeProfile("history-far", _metrics(0.8), 100, "fixture"),
        ]
        repository = StructuredPriorRepository(
            histories,
            {
                "history-nearest": [_optimizer("history-nearest", "FLASH", "rank1", "b3a")],
                "history-second": [_optimizer("history-second", "HEBO", "rank1", "b1a")],
            },
            "prior-fixture-2026-08",
        )
        analyzer = RecordedLandscapeAnalyzer({"target": _metrics(0.0)})
        samples = [{"sample_id": index} for index in range(100)]
        target = analyzer.analyze("target", samples)

        store = InMemoryRuntimeStore()
        task = OptimizationTask("task-prior", "Prior Harness", "minimize", 3)
        run = Run("run-prior", task.id)
        store.add_task(task)
        store.add_run(run)
        store.append_event(run.id, EventType.RUN_CREATED.value, "运行已创建")
        prior = PriorApplicationService(
            store,
            PriorRetrievalService(repository, DeterministicPriorRefiner(2)),
        ).retrieve_and_persist(run.id, target)

        complete = all(
            operator.source_instance
            and operator.optimizer
            and operator.rank
            and operator.operator_id
            for optimizer in prior.optimizers
            for operator in optimizer.operators
        )
        seeded = PriEvoEvolutionCore(
            FakeLLM(), population_size=2, prior=prior
        ).propose(task, run)
        if not all(item.lineage["operator"] == "preknowledge" for item in seeded):
            raise AssertionError("Prior optimizer 未进入 initial population")
        events = list(store.events_for_run(run.id))
        artifacts = sorted(item.kind for item in store.artifacts_for_run(run.id))
        return PriorHarnessReport(
            prior.numeric_candidates[0].instance_name,
            prior.numeric_candidates[0].distance,
            len(prior.numeric_candidates),
            prior.refinement.selected_instances,
            len(prior.optimizers),
            complete,
            [event.event_type for event in events],
            artifacts,
            len(seeded),
        )
