"""由 durable Store 记录重建的 Run Blackboard 只读投影。

Blackboard 不拥有 Population、PriEvO Prior、Candidate code 或 Artifact 内容。
它只投影 AgentTask、Event 的相关引用和 ArtifactMetadata 引用；所有变更必须
先通过 Store 提交，再重新构建投影。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Protocol

from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    AgentTaskStatus,
    ArtifactMetadata,
    Event,
)


class BlackboardStore(Protocol):
    """构建投影所需的最小 Store 读接口。"""

    def agent_tasks_for_run(self, run_id: str) -> Iterable[AgentTask]: ...

    def events_for_run(self, run_id: str) -> Iterable[Event]: ...

    def artifacts_for_run(self, run_id: str) -> Iterable[ArtifactMetadata]: ...


@dataclass(frozen=True)
class AgentTaskView:
    id: str
    run_id: str
    task_type: str
    required_capability: AgentCapability
    idempotency_key: str
    input_artifact_refs: tuple[str, ...]
    output_artifact_refs: tuple[str, ...]
    status: AgentTaskStatus
    claimed_by: str
    attempts: int
    max_attempts: int
    error_message: str
    created_at: datetime
    updated_at: datetime
    claim_token: str
    lease_expires_at: datetime | None

    @classmethod
    def from_task(cls, task: AgentTask) -> "AgentTaskView":
        return cls(
            id=task.id,
            run_id=task.run_id,
            task_type=task.task_type,
            required_capability=task.required_capability,
            idempotency_key=task.idempotency_key,
            input_artifact_refs=tuple(task.input_artifact_refs),
            output_artifact_refs=tuple(task.output_artifact_refs),
            status=task.status,
            claimed_by=task.claimed_by,
            attempts=task.attempts,
            max_attempts=task.max_attempts,
            error_message=task.error_message,
            created_at=task.created_at,
            updated_at=task.updated_at,
            claim_token=task.claim_token,
            lease_expires_at=task.lease_expires_at,
        )


@dataclass(frozen=True)
class BlackboardEventView:
    """Event 的安全投影，只保留顺序和关联引用，不复制任意 payload。"""

    sequence: int
    run_id: str
    event_type: str
    message: str
    task_refs: tuple[str, ...]
    artifact_refs: tuple[str, ...]
    occurred_at: datetime

    @classmethod
    def from_event(cls, event: Event) -> "BlackboardEventView":
        return cls(
            sequence=event.sequence,
            run_id=event.run_id,
            event_type=event.event_type,
            message=event.message,
            task_refs=_references(event.payload, "task"),
            artifact_refs=_references(event.payload, "artifact"),
            occurred_at=event.occurred_at,
        )


@dataclass(frozen=True)
class ArtifactRef:
    """Artifact metadata 引用；故意不提供内容读取接口。"""

    id: str
    run_id: str
    kind: str
    media_type: str
    size: int
    digest: str
    uri: str

    @classmethod
    def from_metadata(cls, artifact: ArtifactMetadata) -> "ArtifactRef":
        return cls(
            artifact.id,
            artifact.run_id,
            artifact.kind,
            artifact.media_type,
            artifact.size,
            artifact.digest,
            artifact.uri,
        )


@dataclass(frozen=True)
class Blackboard:
    """某个 Run 的不可变协作现场投影。"""

    run_id: str
    tasks: tuple[AgentTaskView, ...] = field(default_factory=tuple)
    events: tuple[BlackboardEventView, ...] = field(default_factory=tuple)
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    _tasks_by_ref: Mapping[str, AgentTaskView] = field(
        init=False, repr=False, compare=False
    )
    _artifacts_by_ref: Mapping[str, ArtifactRef] = field(
        init=False, repr=False, compare=False
    )
    _events_by_sequence: Mapping[int, BlackboardEventView] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_tasks_by_ref", MappingProxyType(_unique_index(self.tasks, "id"))
        )
        object.__setattr__(
            self,
            "_artifacts_by_ref",
            MappingProxyType(_unique_index(self.artifacts, "id")),
        )
        object.__setattr__(
            self,
            "_events_by_sequence",
            MappingProxyType(_unique_index(self.events, "sequence")),
        )

    @classmethod
    def from_store(cls, store: BlackboardStore, run_id: str) -> "Blackboard":
        """只调用三个 durable list 接口重建投影。"""

        tasks = tuple(
            sorted(
                (AgentTaskView.from_task(item) for item in store.agent_tasks_for_run(run_id)),
                key=lambda item: (item.created_at, item.id),
            )
        )
        events = tuple(
            sorted(
                (BlackboardEventView.from_event(item) for item in store.events_for_run(run_id)),
                key=lambda item: item.sequence,
            )
        )
        artifacts = tuple(
            sorted(
                (ArtifactRef.from_metadata(item) for item in store.artifacts_for_run(run_id)),
                key=lambda item: item.id,
            )
        )
        _require_same_run(run_id, tasks, events, artifacts)
        return cls(run_id=run_id, tasks=tasks, events=events, artifacts=artifacts)

    def open_tasks(self) -> tuple[AgentTaskView, ...]:
        """返回尚未领取的 durable PENDING tasks。"""

        return tuple(item for item in self.tasks if item.status == AgentTaskStatus.PENDING)

    def tasks_by_type(self, task_type: str) -> tuple[AgentTaskView, ...]:
        return tuple(item for item in self.tasks if item.task_type == task_type)

    def events_by_type(self, event_type: str) -> tuple[BlackboardEventView, ...]:
        return tuple(item for item in self.events if item.event_type == event_type)

    def artifacts_by_kind(self, kind: str) -> tuple[ArtifactRef, ...]:
        return tuple(item for item in self.artifacts if item.kind == kind)

    def has_artifact_kind(self, kind: str) -> bool:
        return any(item.kind == kind for item in self.artifacts)

    def task_by_ref(self, ref: str) -> AgentTaskView:
        try:
            return self._tasks_by_ref[ref]
        except KeyError as exc:
            raise KeyError("Blackboard 中不存在 AgentTask ref：{}".format(ref)) from exc

    def artifact_by_ref(self, ref: str) -> ArtifactRef:
        try:
            return self._artifacts_by_ref[ref]
        except KeyError as exc:
            raise KeyError("Blackboard 中不存在 Artifact ref：{}".format(ref)) from exc

    def event_by_sequence(self, sequence: int) -> BlackboardEventView:
        try:
            return self._events_by_sequence[sequence]
        except KeyError as exc:
            raise KeyError("Blackboard 中不存在 Event sequence：{}".format(sequence)) from exc


# 名称明确的别名，便于文档和调用端强调它不是可写 Domain aggregate。
BlackboardProjection = Blackboard


def _references(payload: Mapping[str, object], category: str) -> tuple[str, ...]:
    refs: list[str] = []
    for key, value in payload.items():
        normalized = str(key).lower()
        if not _is_reference_key(normalized, category):
            continue
        refs.extend(_reference_values(value))
    return tuple(dict.fromkeys(refs))


def _is_reference_key(key: str, category: str) -> bool:
    singular = "{}_ref".format(category)
    identifier = "{}_id".format(category)
    return (
        key in {singular, singular + "s", identifier, identifier + "s"}
        or key.endswith("_" + singular)
        or key.endswith("_" + singular + "s")
        or key.endswith("_" + identifier)
        or key.endswith("_" + identifier + "s")
    )


def _reference_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (tuple, list, set, frozenset)):
        return [str(item) for item in value if isinstance(item, str) and item]
    return []


def _unique_index(items: Iterable[object], attribute: str) -> dict:
    result = {}
    for item in items:
        key = getattr(item, attribute)
        if key in result:
            raise ValueError("Blackboard durable projection 出现重复引用：{}".format(key))
        result[key] = item
    return result


def _require_same_run(run_id: str, *groups: Iterable[object]) -> None:
    for group in groups:
        for item in group:
            if getattr(item, "run_id") != run_id:
                raise ValueError(
                    "Blackboard 投影混入其他 Run：expected={} actual={}".format(
                        run_id, getattr(item, "run_id")
                    )
                )
