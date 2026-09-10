from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, Iterable, List

from prievo_agent.domain.models import (
    ArtifactMetadata,
    Candidate,
    CheckpointMetadata,
    EvaluationResult,
    Event,
    OptimizationTask,
    Run,
)


class InMemoryRuntimeStore:
    def __init__(self) -> None:
        self.tasks: Dict[str, OptimizationTask] = {}
        self.runs: Dict[str, Run] = {}
        self.candidates: Dict[str, Candidate] = {}
        self.results: Dict[str, EvaluationResult] = {}
        self.events: Dict[str, List[Event]] = defaultdict(list)
        self.artifacts: Dict[str, bytes] = {}
        self.artifact_metadata: Dict[str, ArtifactMetadata] = {}
        self.checkpoints: Dict[str, List[CheckpointMetadata]] = defaultdict(list)

    def add_task(self, task: OptimizationTask) -> None:
        if task.id in self.tasks:
            raise ValueError("OptimizationTask 已存在：{}".format(task.id))
        self.tasks[task.id] = task

    def add_run(self, run: Run) -> None:
        if run.id in self.runs:
            raise ValueError("Run 已存在：{}".format(run.id))
        self.runs[run.id] = run

    def get_task(self, task_id: str) -> OptimizationTask:
        return self.tasks[task_id]

    def get_run(self, run_id: str) -> Run:
        return self.runs[run_id]

    def save_run(self, run: Run) -> None:
        self.runs[run.id] = run

    def add_candidate(self, candidate: Candidate) -> None:
        self.candidates[candidate.id] = candidate

    def candidates_for_run(self, run_id: str) -> Iterable[Candidate]:
        return [item for item in self.candidates.values() if item.run_id == run_id]

    def candidate_by_id(self, candidate_id: str) -> Candidate:
        return self.candidates[candidate_id]

    def add_result(self, result: EvaluationResult) -> None:
        self.results[result.candidate_id] = result

    def result_for_candidate(self, candidate_id: str) -> EvaluationResult:
        return self.results[candidate_id]

    def append_event(
        self, run_id: str, event_type: str, message: str, **payload: object
    ) -> Event:
        event = Event(
            sequence=len(self.events[run_id]) + 1,
            run_id=run_id,
            event_type=event_type,
            message=message,
            payload=dict(payload),
        )
        self.events[run_id].append(event)
        return event

    def events_for_run(self, run_id: str) -> Iterable[Event]:
        return list(self.events[run_id])

    def put_artifact(
        self, run_id: str, kind: str, content: bytes, media_type: str
    ) -> ArtifactMetadata:
        digest = hashlib.sha256(content).hexdigest()
        artifact_id = "artifact-{}-{}".format(kind.lower(), digest[:16])
        metadata = ArtifactMetadata(
            id=artifact_id,
            run_id=run_id,
            kind=kind,
            media_type=media_type,
            size=len(content),
            digest=digest,
            uri="memory://{}".format(artifact_id),
        )
        self.artifacts[artifact_id] = content
        self.artifact_metadata[artifact_id] = metadata
        return metadata

    def artifact_content(self, artifact_id: str) -> bytes:
        return self.artifacts[artifact_id]

    def artifacts_for_run(self, run_id: str) -> Iterable[ArtifactMetadata]:
        return [
            metadata
            for metadata in self.artifact_metadata.values()
            if metadata.run_id == run_id
        ]

    def add_checkpoint(self, checkpoint: CheckpointMetadata) -> None:
        self.checkpoints[checkpoint.run_id].append(checkpoint)

    def latest_checkpoint(self, run_id: str) -> CheckpointMetadata:
        values = self.checkpoints[run_id]
        if not values:
            raise KeyError(run_id)
        return values[-1]
