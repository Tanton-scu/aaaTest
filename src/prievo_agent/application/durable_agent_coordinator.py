"""由持久事实派生 AgentTask 的确定性 missing-work coordinator。

该组件只负责发现缺失工作并创建 durable task。它不调用 Agent，也不读取或
修改 Run 的状态、预算、Population 与 Candidate。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    AgentCapability,
    AgentTask,
    ArtifactMetadata,
    utc_now,
)
from prievo_agent.domain.ports import RuntimeStore


@dataclass(frozen=True)
class MissingWorkRule:
    """一个输入事实、可接受输出与所需 Agent capability 的静态映射。"""

    input_kind: str
    task_type: str
    required_capability: AgentCapability
    # 任一 group 完整存在即表示工作已有结果；不同 group 表示替代输出。
    output_alternatives: Tuple[frozenset[str], ...]


MISSING_WORK_RULES: Tuple[MissingWorkRule, ...] = (
    MissingWorkRule(
        "TOP5_CANDIDATES",
        "SEMANTIC_SIMILARITY_SELECTION",
        AgentCapability.SEMANTIC_SIMILARITY,
        (frozenset({"SIMILARITY_DECISION"}),),
    ),
    MissingWorkRule(
        "GENERATION_REQUEST",
        "HEURISTIC_GENERATION",
        AgentCapability.HEURISTIC_GENERATION,
        (
            frozenset({"CANDIDATE_DRAFT"}),
            frozenset({"KNOWLEDGE_GAP"}),
        ),
    ),
    MissingWorkRule(
        "KNOWLEDGE_GAP",
        "PRIOR_RESEARCH",
        AgentCapability.PRIOR_RESEARCH,
        (frozenset({"PRIOR_EXPLANATION", "LITERATURE_EVIDENCE"}),),
    ),
    MissingWorkRule(
        "CANDIDATE_FAILURE",
        "CANDIDATE_REPAIR",
        AgentCapability.CANDIDATE_REPAIR,
        (
            frozenset({"REPAIR_DECISION"}),
            frozenset({"REPAIRED_CANDIDATE_DRAFT"}),
        ),
    ),
    MissingWorkRule(
        "FINAL_TIE",
        "FINAL_SELECTION",
        AgentCapability.FINAL_SELECTION,
        (frozenset({"FINAL_SELECTION_DECISION"}),),
    ),
)


class DurableAgentCoordinator:
    """从 Artifact 投影中补齐唯一 AgentTask，不执行任何 Agent。"""

    def __init__(
        self,
        store: RuntimeStore,
        rules: Sequence[MissingWorkRule] = MISSING_WORK_RULES,
    ) -> None:
        self.store = store
        self.rules = tuple(rules)

    def reconcile(self, run_id: str) -> List[AgentTask]:
        """为一个 Run 创建当前缺失的任务，并只返回本次新建的任务。"""

        if not run_id:
            raise ValueError("run_id 不能为空")

        artifacts = sorted(
            self.store.artifacts_for_run(run_id),
            key=lambda artifact: (artifact.kind, artifact.id),
        )
        artifacts_by_kind: Dict[str, List[ArtifactMetadata]] = {}
        for artifact in artifacts:
            artifacts_by_kind.setdefault(artifact.kind, []).append(artifact)

        known_tasks = {
            task.idempotency_key: task
            for task in self.store.agent_tasks_for_run(run_id)
        }
        created_tasks: List[AgentTask] = []

        for rule in self.rules:
            input_artifacts = artifacts_by_kind.get(rule.input_kind, [])
            for input_artifact in input_artifacts:
                task_id, idempotency_key = self._task_identity(
                    run_id, rule, input_artifact.id
                )
                if idempotency_key in known_tasks:
                    continue
                if self._has_corresponding_output(
                    rule,
                    input_artifact,
                    len(input_artifacts),
                    artifacts_by_kind,
                ):
                    continue

                now = utc_now()
                proposed = AgentTask(
                    id=task_id,
                    run_id=run_id,
                    task_type=rule.task_type,
                    required_capability=rule.required_capability,
                    idempotency_key=idempotency_key,
                    input_artifact_refs=[input_artifact.id],
                    created_at=now,
                    updated_at=now,
                )
                persisted, created = self.store.add_agent_task(proposed)
                # add_agent_task 的唯一约束是并发 reconcile 的最终防线。
                known_tasks[idempotency_key] = persisted
                if not created:
                    continue

                self.store.append_event(
                    run_id,
                    EventType.AGENT_TASK_CREATED.value,
                    "已从持久化事实派生 AgentTask：{}".format(rule.task_type),
                    agent_task_id=persisted.id,
                    task_type=persisted.task_type,
                    required_capability=persisted.required_capability.value,
                    idempotency_key=persisted.idempotency_key,
                    input_artifact_refs=list(persisted.input_artifact_refs),
                )
                created_tasks.append(persisted)

        return created_tasks

    def sweep(self, active_run_ids: Iterable[str]) -> List[AgentTask]:
        """按稳定顺序 reconcile 调用方给出的 active Run，作为周期兜底。"""

        created_tasks: List[AgentTask] = []
        for run_id in sorted(set(active_run_ids)):
            if not run_id:
                continue
            created_tasks.extend(self.reconcile(run_id))
        return created_tasks

    @staticmethod
    def _task_identity(
        run_id: str, rule: MissingWorkRule, input_artifact_id: str
    ) -> Tuple[str, str]:
        source_digest = hashlib.sha256(
            "{}|{}".format(rule.input_kind, input_artifact_id).encode("utf-8")
        ).hexdigest()[:32]
        idempotency_key = "missing-work:v1:{}:{}".format(
            rule.task_type, source_digest
        )
        task_digest = hashlib.sha256(
            "{}|{}".format(run_id, idempotency_key).encode("utf-8")
        ).hexdigest()[:32]
        return "agent-task-{}".format(task_digest), idempotency_key

    def _has_corresponding_output(
        self,
        rule: MissingWorkRule,
        input_artifact: ArtifactMetadata,
        same_kind_input_count: int,
        artifacts_by_kind: Dict[str, List[ArtifactMetadata]],
    ) -> bool:
        relevant_kinds = set().union(*rule.output_alternatives)
        matching_outputs: List[ArtifactMetadata] = []

        for output_kind in sorted(relevant_kinds):
            for output in artifacts_by_kind.get(output_kind, []):
                # 单例输入兼容尚未携带 source ref 的早期 artifact schema。
                # 多输入时必须通过 JSON 内容关联，避免一个输出错误满足全部请求。
                if same_kind_input_count == 1 or self._references_source(
                    output, input_artifact.id
                ):
                    matching_outputs.append(output)

        present_kinds = {artifact.kind for artifact in matching_outputs}
        return any(
            required_kinds.issubset(present_kinds)
            for required_kinds in rule.output_alternatives
        )

    def _references_source(
        self, output: ArtifactMetadata, input_artifact_id: str
    ) -> bool:
        try:
            payload = json.loads(
                self.store.artifact_content(output.id).decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            return False
        return _contains_value(payload, input_artifact_id)


def _contains_value(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_value(item, expected) for item in value)
    return value == expected
