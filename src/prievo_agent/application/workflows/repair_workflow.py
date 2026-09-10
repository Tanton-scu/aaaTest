"""Candidate failure 到新版本 Candidate 的 durable Repair 工作流。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional

from prievo_agent.agents.registry import AgentRegistry
from prievo_agent.agents.nodes.repair import (
    FailureEvidence,
    RepairAgent,
    RepairAgentResult,
)
from prievo_agent.application.orchestration.agent_dispatcher import AgentTaskDispatcher
from prievo_agent.application.orchestration.durable_agent_coordinator import DurableAgentCoordinator
from prievo_agent.application.memory.run_local_memory import (
    RunLocalMemoryService,
    memory_context,
)
from prievo_agent.application.memory.history_compactor import HistoryCompactor
from prievo_agent.agents.tools import (
    CandidateInspectionTool,
    ToolGovernanceGateway,
)
from prievo_agent.domain.models import Candidate


class RepairTaskHandler:
    name = RepairAgent.name
    capability = RepairAgent.capability

    def __init__(self, store, agent, skill_registry, inspection_tool):
        self.store = store
        self.agent = agent
        self.skill_registry = skill_registry
        self.inspection_tool = inspection_tool

    def handle(self, task, board):
        if len(task.input_artifact_refs) != 1:
            raise ValueError("RepairTask 必须且只能引用一个 CANDIDATE_FAILURE")
        artifact = board.artifact_by_ref(task.input_artifact_refs[0])
        if artifact.kind != "CANDIDATE_FAILURE":
            raise ValueError("RepairTask 输入必须是 CANDIDATE_FAILURE")
        payload = json.loads(self.store.artifact_content(artifact.id))
        candidate = self.store.candidate_by_id(payload["candidate_id"])
        inspection = self.inspection_tool.inspect(
            task.run_id,
            candidate.id,
            "RepairAgent 诊断已分类的 Candidate failure",
        )
        tool_call_ref = str(inspection["tool_call_ref"])
        evidence = FailureEvidence(
            error_code=payload["error_code"],
            message=payload.get("error_message", ""),
            return_code=payload.get("return_code"),
            stderr=payload.get("stderr", ""),
            run_id=task.run_id,
            candidate_id=candidate.id,
            artifact_refs=(artifact.id, tool_call_ref),
        )
        relevant_failures = list(payload.get("relevant_failures", []))
        relevant_failures.append({"candidate_inspection": inspection})
        return self.agent.execute(
            candidate,
            evidence,
            self.skill_registry.require("candidate_code_repair"),
            diagnosis_skill=self.skill_registry.require(
                "candidate_failure_diagnosis"
            ),
            repair_history=payload.get("repair_history", []),
            relevant_failures=relevant_failures,
            repair_attempt=int(payload.get("repair_attempt", 0)),
            max_attempts=int(payload.get("max_repair_attempts", 3)),
            repair_budget=int(payload.get("repair_budget", 1)),
            context_refs=(artifact.id,),
        )


class RepairArtifactWriter:
    def __init__(self, store):
        self.store = store

    def __call__(self, task, result: RepairAgentResult):
        diagnosis_prompt = self.store.put_artifact(
            task.run_id,
            "REPAIR_DIAGNOSIS_PROMPT",
            result.diagnosis.diagnosis_prompt.encode("utf-8"),
            "text/plain",
        )
        diagnosis_payload = result.diagnosis.to_dict()
        diagnosis_payload.pop("diagnosis_prompt", None)
        decision_payload = {
            "input_artifact_refs": list(task.input_artifact_refs),
            "agent_task_id": task.id,
            "diagnosis_prompt_artifact_id": diagnosis_prompt.id,
            "decision": {
                "failure_type": result.decision.failure_type.value,
                "action": result.decision.action.value,
                "reason": result.decision.reason,
                "error_code": result.decision.error_code,
            },
            "diagnosis": diagnosis_payload,
            "skipped_reason": result.skipped_reason,
        }
        decision = self.store.put_artifact(
            task.run_id,
            "REPAIR_DECISION",
            json.dumps(
                decision_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        refs = [diagnosis_prompt, decision]
        if result.repaired_candidate is not None:
            repair_prompt = self.store.put_artifact(
                task.run_id,
                "REPAIR_PROMPT",
                result.repaired_candidate.repair_prompt.encode("utf-8"),
                "text/plain",
            )
            payload = result.repaired_candidate.to_dict()
            payload.pop("repair_prompt", None)
            payload.update({
                "input_artifact_refs": list(task.input_artifact_refs),
                "agent_task_id": task.id,
                "repair_decision_artifact_id": decision.id,
                "repair_prompt_artifact_id": repair_prompt.id,
            })
            draft = self.store.put_artifact(
                task.run_id,
                "REPAIRED_CANDIDATE_DRAFT",
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                "application/json",
            )
            refs.extend([repair_prompt, draft])
        return refs


@dataclass(frozen=True)
class RepairWorkflowResult:
    """Repair 的结构化业务结果；成功路径仍支持旧式三元解包。"""

    repaired_candidate: Optional[Candidate]
    failure_artifact_id: str
    repaired_candidate_draft_artifact_id: Optional[str]
    repair_decision_artifact_id: str
    repairable: bool
    diagnosis: Mapping
    skipped_reason: str = ""

    def __iter__(self):
        # 保持 ``candidate, failure_ref, draft_ref = repair(...)`` 兼容。
        yield self.repaired_candidate
        yield self.failure_artifact_id
        yield self.repaired_candidate_draft_artifact_id


class DurableRepairWorkflow:
    def __init__(
        self,
        store,
        skill_registry,
        model,
        max_repair_attempts=3,
        working_memory=None,
    ):
        self.store = store
        self.max_repair_attempts = int(max_repair_attempts)
        self.memory = RunLocalMemoryService(store, working_memory)
        self.history_compactor = HistoryCompactor(
            store, skill_registry, model, working_memory
        )
        handler = RepairTaskHandler(
            store,
            RepairAgent(model),
            skill_registry,
            CandidateInspectionTool(ToolGovernanceGateway(store), store),
        )
        self.coordinator = DurableAgentCoordinator(store)
        self.dispatcher = AgentTaskDispatcher(
            store, AgentRegistry([handler]), RepairArtifactWriter(store)
        )

    def repair(self, run, task, candidate, failed_job):
        attempt = int(candidate.lineage.get("repair_attempt", 0))
        remaining = (
            task.total_budget
            - run.consumed_evaluations
            - run.reserved_evaluations
        )
        repair_records = memory_context(
            self.history_compactor.context_records(
                run.id,
                run.dataset_id or task.dataset_id,
                "repair",
                recent_window=2,
            )
        )
        candidate_family = {
            candidate.id,
            str(candidate.lineage.get("repair_parent_id", "")),
        }
        related = [
            item
            for item in repair_records
            if _repair_memory_related(item, candidate_family)
        ]
        payload = {
            "candidate_id": candidate.id,
            "evaluation_job_id": failed_job.id,
            "error_code": failed_job.error_code or "UNKNOWN",
            "error_message": failed_job.error_message or "",
            "repair_attempt": attempt,
            "max_repair_attempts": self.max_repair_attempts,
            "repair_budget": 1 if remaining >= task.evaluation_budget else 0,
            "repair_history": related[-3:],
            "relevant_failures": repair_records[-3:],
        }
        failure = self.store.put_artifact(
            run.id,
            "CANDIDATE_FAILURE",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run.id,
            "CANDIDATE_FAILURE_CLASSIFIED",
            "Candidate failure 已持久化并路由 RepairAgent",
            artifact_id=failure.id,
            candidate_id=candidate.id,
            evaluation_job_id=failed_job.id,
            error_code=failed_job.error_code,
        )
        self.coordinator.reconcile(run.id)
        tasks = [
            item for item in self.store.agent_tasks_for_run(run.id)
            if item.task_type == "CANDIDATE_REPAIR"
            and failure.id in item.input_artifact_refs
        ]
        if len(tasks) != 1:
            raise RuntimeError("无法为 CANDIDATE_FAILURE 唯一定位 RepairTask")
        dispatched = self.dispatcher.dispatch_with_retries(tasks[0].id)
        artifacts_by_id = {
            item.id: item for item in self.store.artifacts_for_run(run.id)
        }
        decisions = [
            ref for ref in dispatched.artifact_refs
            if ref in artifacts_by_id
            and artifacts_by_id[ref].kind == "REPAIR_DECISION"
        ]
        if len(decisions) != 1:
            raise RuntimeError("RepairAgent 必须产生唯一 REPAIR_DECISION")
        decision_data = json.loads(self.store.artifact_content(decisions[0]))
        diagnosis_data = decision_data.get("diagnosis", {})
        if not isinstance(diagnosis_data, dict):
            raise RuntimeError("REPAIR_DECISION diagnosis 必须是结构化对象")
        drafts = [
            ref for ref in dispatched.artifact_refs
            if ref in artifacts_by_id
            and artifacts_by_id[ref].kind == "REPAIRED_CANDIDATE_DRAFT"
        ]
        if not drafts:
            if diagnosis_data.get("repairable") is not False:
                raise RuntimeError(
                    "RepairAgent 判定可修复但未产生 repaired Candidate draft"
                )
            skipped_reason = str(
                decision_data.get("skipped_reason", "")
                or "Diagnose 判定 Candidate 不可安全修复"
            )
            repair_memory = self.memory.persist(
                run_id=run.id,
                dataset_id=run.dataset_id or task.dataset_id,
                scope="repair",
                memory_type="REPAIR_DECISION",
                subject="{} / not repairable".format(candidate.id),
                content={
                    "candidate_id": candidate.id,
                    "failure_type": payload["error_code"],
                    "failure_message": payload["error_message"],
                    "diagnosis_ref": diagnosis_data.get("id", ""),
                    "candidate_failure_artifact_id": failure.id,
                    "repair_decision_artifact_id": decisions[0],
                    "repairable": False,
                    "skipped_reason": skipped_reason,
                    "outcome": "NOT_REPAIRABLE",
                },
                evidence_artifact_id=decisions[0],
                identity_material="{}|{}|{}|not-repairable".format(
                    failed_job.id, candidate.id, decisions[0]
                ),
            )
            self.store.append_event(
                run.id,
                "REPAIR_SKIPPED_NOT_REPAIRABLE",
                "RepairAgent 已持久化不可修复诊断，未创建 Candidate draft",
                candidate_id=candidate.id,
                candidate_failure_artifact_id=failure.id,
                repair_decision_artifact_id=decisions[0],
                diagnosis_ref=diagnosis_data.get("id", ""),
                skipped_reason=skipped_reason,
            )
            self.store.append_event(
                run.id,
                "REPAIR_MEMORY_UPDATED",
                "RepairAgent 已写入同 Run repair scope 的不可修复历史",
                agent_memory_id=repair_memory.id,
                candidate_id=candidate.id,
                outcome="NOT_REPAIRABLE",
            )
            return RepairWorkflowResult(
                repaired_candidate=None,
                failure_artifact_id=failure.id,
                repaired_candidate_draft_artifact_id=None,
                repair_decision_artifact_id=decisions[0],
                repairable=False,
                diagnosis=diagnosis_data,
                skipped_reason=skipped_reason,
            )
        if len(drafts) != 1:
            raise RuntimeError("RepairAgent 产生了多个 repaired Candidate draft")
        data = json.loads(self.store.artifact_content(drafts[0]))
        lineage = dict(data["lineage"])
        lineage.update({
            "repaired_candidate_draft_artifact_id": drafts[0],
            "candidate_failure_artifact_id": failure.id,
            "repair_task_id": tasks[0].id,
        })
        repaired = Candidate(
            id=data["id"],
            run_id=run.id,
            code=data["code"],
            description=data["description"],
            operators=list(data["operators"]),
            lineage=lineage,
        )
        self.store.add_candidate(repaired)
        self.store.append_event(
            run.id,
            "REPAIRED_CANDIDATE_MATERIALIZED",
            "RepairAgent 已创建新 Candidate version，原 Candidate 保持不可变",
            candidate_id=repaired.id,
            repair_parent_id=candidate.id,
            repair_attempt=data["repair_attempt"],
            repaired_candidate_draft_artifact_id=drafts[0],
        )
        repair_memory = self.memory.persist(
            run_id=run.id,
            dataset_id=run.dataset_id or task.dataset_id,
            scope="repair",
            memory_type="REPAIR_DECISION",
            subject="{} -> {} / attempt {}".format(
                candidate.id, repaired.id, data["repair_attempt"]
            ),
            content={
                "candidate_id": candidate.id,
                "repaired_candidate_id": repaired.id,
                "repair_attempt": data["repair_attempt"],
                "failure_type": payload["error_code"],
                "failure_message": payload["error_message"],
                "diagnosis_ref": data.get("diagnosis_ref", ""),
                "candidate_failure_artifact_id": failure.id,
                "repaired_candidate_draft_artifact_id": drafts[0],
                "outcome": "REPAIRED_DRAFT_CREATED",
            },
            evidence_artifact_id=drafts[0],
            identity_material="{}|{}|{}".format(
                failed_job.id, candidate.id, drafts[0]
            ),
        )
        self.store.append_event(
            run.id,
            "REPAIR_MEMORY_UPDATED",
            "RepairAgent 已写入同 Run repair scope 历史",
            agent_memory_id=repair_memory.id,
            candidate_id=candidate.id,
            repaired_candidate_id=repaired.id,
            repair_attempt=data["repair_attempt"],
        )
        return RepairWorkflowResult(
            repaired_candidate=repaired,
            failure_artifact_id=failure.id,
            repaired_candidate_draft_artifact_id=drafts[0],
            repair_decision_artifact_id=decisions[0],
            repairable=True,
            diagnosis=diagnosis_data,
        )


def _repair_memory_related(item, candidate_family):
    content = item.get("content", {})
    if not isinstance(content, dict):
        return False
    identifiers = {
        str(content.get("candidate_id", "")),
        str(content.get("repaired_candidate_id", "")),
    }
    return bool({value for value in identifiers if value} & candidate_family)
