from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from prievo_agent.domain.models import CandidateStatus, RunStatus
from prievo_agent.domain.ports import RuntimeStore


class LifecycleInvariantError(AssertionError):
    pass


@dataclass(frozen=True)
class LifecycleInspection:
    run_id: str
    status: str
    event_types: List[str]
    candidate_count: int
    evaluation_result_count: int
    unfinished_required_evaluations: int
    best_candidate_id: str
    artifact_kinds: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "event_types": self.event_types,
            "candidate_count": self.candidate_count,
            "evaluation_result_count": self.evaluation_result_count,
            "unfinished_required_evaluations": self.unfinished_required_evaluations,
            "best_candidate_id": self.best_candidate_id,
            "artifact_kinds": self.artifact_kinds,
        }


class LifecycleHarness:
    """完成后从 store 检查运行，不依赖 console 文本。"""

    def __init__(self, store: RuntimeStore) -> None:
        self.store = store

    def inspect_completed(self, run_id: str) -> LifecycleInspection:
        run = self.store.get_run(run_id)
        if run.status != RunStatus.COMPLETED:
            raise LifecycleInvariantError("Run 尚未完成")
        candidates = list(self.store.candidates_for_run(run_id))
        if not candidates:
            raise LifecycleInvariantError("Run 没有候选")
        result_count = 0
        unfinished = 0
        for candidate in candidates:
            if not candidate.code_artifact_id:
                raise LifecycleInvariantError("候选缺少代码 artifact：{}".format(candidate.id))
            self.store.artifact_content(candidate.code_artifact_id)
            if candidate.status in {CandidateStatus.CREATED, CandidateStatus.VALIDATED,
                                     CandidateStatus.EVALUATION_PENDING,
                                     CandidateStatus.EVALUATING}:
                unfinished += 1
                continue
            result = self.store.result_for_candidate(candidate.id)
            if result.candidate_id != candidate.id:
                raise LifecycleInvariantError("评价结果与候选关联错误")
            if not candidate.evaluation_artifact_id:
                raise LifecycleInvariantError("已评价候选缺少结果 artifact")
            self.store.artifact_content(candidate.evaluation_artifact_id)
            result_count += 1
        if unfinished:
            raise LifecycleInvariantError("COMPLETED Run 仍有未完成的必需评价")
        if not run.best_candidate_id:
            raise LifecycleInvariantError("COMPLETED Run 缺少最终候选")
        best = next(
            (item for item in candidates if item.id == run.best_candidate_id), None
        )
        if best is None or best.status != CandidateStatus.EVALUATED:
            raise LifecycleInvariantError("最终候选不存在或未被选中")

        artifact_kinds = sorted(
            metadata.kind for metadata in self.store.artifacts_for_run(run_id)
        )
        for required in {"POPULATION_SNAPSHOT", "FINAL_HEURISTIC"}:
            if required not in artifact_kinds:
                raise LifecycleInvariantError("缺少必需 artifact：{}".format(required))
        events = list(self.store.events_for_run(run_id))
        if [event.sequence for event in events] != list(range(1, len(events) + 1)):
            raise LifecycleInvariantError("事件 sequence 不连续")
        return LifecycleInspection(
            run_id=run.id,
            status=run.status.value,
            event_types=[event.event_type for event in events],
            candidate_count=len(candidates),
            evaluation_result_count=result_count,
            unfinished_required_evaluations=unfinished,
            best_candidate_id=run.best_candidate_id,
            artifact_kinds=artifact_kinds,
        )
