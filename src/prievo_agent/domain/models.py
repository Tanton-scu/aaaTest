from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CandidateStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    EVALUATION_PENDING = "EVALUATION_PENDING"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    INVALID = "INVALID"


class EvaluationJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RETRY_WAIT = "RETRY_WAIT"
    SUCCESS = "SUCCESS"
    DEAD = "DEAD"
    CANCELLED = "CANCELLED"


class AgentCapability(str, Enum):
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    HEURISTIC_GENERATION = "HEURISTIC_GENERATION"
    PRIOR_RESEARCH = "PRIOR_RESEARCH"
    CANDIDATE_REPAIR = "CANDIDATE_REPAIR"
    FINAL_SELECTION = "FINAL_SELECTION"


class AgentTaskStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class OptimizationTask:
    id: str
    name: str
    objective: str
    evaluation_budget: int
    total_budget: int = 100
    created_at: datetime = field(default_factory=utc_now)
    dataset_id: str = ""
    generations: int = 2
    population_size: int = 3
    random_seed: int = 2024

    def __post_init__(self) -> None:
        if self.evaluation_budget <= 0:
            raise ValueError("评价预算必须大于 0")
        if self.total_budget <= 0:
            raise ValueError("总预算必须大于 0")


@dataclass
class Run:
    id: str
    task_id: str
    status: RunStatus = RunStatus.PENDING
    generation: int = 0
    consumed_evaluations: int = 0
    reserved_evaluations: int = 0
    best_candidate_id: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    dataset_id: str = ""
    # 控制请求与执行所有权必须落 durable store。API 只写 request；Runtime 在
    # 一致性安全点确认 pause/cancel，不能依赖进程内 Event/flag。
    pause_requested: bool = False
    cancel_requested: bool = False
    control_reason: str = ""
    runtime_cursor_artifact_id: str = ""
    runtime_owner_id: str = ""
    runtime_lease_expires_at: Optional[datetime] = None

@dataclass
class Candidate:
    id: str
    run_id: str
    code: str
    description: str
    operators: List[str]
    lineage: Dict[str, Any]
    status: CandidateStatus = CandidateStatus.CREATED
    objective: Optional[float] = None
    code_artifact_id: Optional[str] = None
    evaluation_artifact_id: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    run_id: str
    candidate_id: str
    objective: float
    trajectory: List[float]
    best_configuration: Dict[str, Any]
    used_budget: int


@dataclass
class EvaluationJob:
    id: str
    run_id: str
    task_id: str
    candidate_id: str
    seed: int
    budget: int
    idempotency_key: str
    status: EvaluationJobStatus = EvaluationJobStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    available_at: datetime = field(default_factory=utc_now)
    lease_expires_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    result_id: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class Event:
    sequence: int
    run_id: str
    event_type: str
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ArtifactMetadata:
    id: str
    run_id: str
    kind: str
    media_type: str
    size: int
    digest: str
    uri: str


@dataclass(frozen=True)
class CheckpointMetadata:
    id: str
    run_id: str
    generation: int
    population_ids: List[str]
    consumed_budget: int
    remaining_budget: int
    artifact_id: str
    schema_version: int
    code_version: str
    created_at: datetime = field(default_factory=utc_now)
    dataset_id: str = ""
    prior_refs: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentMemory:
    id: str
    run_id: str
    dataset_id: str
    memory_type: str
    subject: str
    content: str
    evidence_artifact_id: str = ""
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class AgentTask:
    """可从 durable state 重建的 Agent 工作项；不承载 Population 等算法事实。"""

    id: str
    run_id: str
    task_type: str
    required_capability: AgentCapability
    idempotency_key: str
    input_artifact_refs: List[str] = field(default_factory=list)
    output_artifact_refs: List[str] = field(default_factory=list)
    status: AgentTaskStatus = AgentTaskStatus.PENDING
    claimed_by: str = ""
    attempts: int = 0
    max_attempts: int = 3
    error_message: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    # claimed_by 只是人类可读的 Agent 名称；真正的 fencing identity 是每次
    # claim 都重新生成、不可复用的 token。过期 token 不能结算或覆盖接管者。
    claim_token: str = ""
    lease_expires_at: Optional[datetime] = None


@dataclass(frozen=True)
class ToolCallRecord:
    id: str
    run_id: str
    tool_name: str
    status: str
    request: Dict[str, Any]
    response: Dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)
