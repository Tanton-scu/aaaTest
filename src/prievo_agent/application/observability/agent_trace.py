import json


# Trace 只展开 Agent 协作面 Artifact；Candidate code、整群 snapshot 与 checkpoint
# 仍通过各自 API 按需读取，避免一个 trace 响应复制大对象。
AGENT_ARTIFACT_KINDS = {
    "TOP5_CANDIDATES",
    "PLANNER_CONTEXT",
    "GENERATION_PLAN",
    "SIMILARITY_PROMPT",
    "SIMILARITY_DECISION",
    "GENERATION_REQUEST",
    "GENERATION_RESUME_REQUEST",
    "GENERATION_PROMPT",
    "GENERATION_RESUME_PROMPT",
    "CANDIDATE_DRAFT",
    "RESUMED_CANDIDATE_DRAFT",
    "RESUMED_KNOWLEDGE_GAP",
    "KNOWLEDGE_GAP",
    "LITERATURE_EVIDENCE_QUERY_PROMPT",
    "LITERATURE_EVIDENCE_EXPLANATION_PROMPT",
    "PRIOR_EXPLANATION",
    "LITERATURE_EVIDENCE",
    "CANDIDATE_FAILURE",
    "REPAIR_DIAGNOSIS_PROMPT",
    "REPAIR_DECISION",
    "REPAIR_PROMPT",
    "REPAIRED_CANDIDATE_DRAFT",
    "FINAL_TIE",
    "FINAL_SELECTION_PROMPT",
    "FINAL_SELECTION_DECISION",
}


class AgentTraceQuery:
    """由 durable Event/Task/Artifact/ToolCall 投影 Agent 因果链。

    Trace 不是第二份可变状态，也不声称完整 Event Sourcing。Task 状态与 refs
    来自事实表，进程重启后仍可重建；Artifact payload 读取失败时显式标注。
    """

    def __init__(self, store):
        self.store = store

    def trace(self, run_id):
        run = self.store.get_run(run_id)
        events = list(self.store.events_for_run(run_id))
        tasks = list(self.store.agent_tasks_for_run(run_id))
        artifacts = [
            item
            for item in self.store.artifacts_for_run(run_id)
            if item.kind in AGENT_ARTIFACT_KINDS
        ]
        tool_calls = list(self.store.tool_calls_for_run(run_id))
        trace_records = (
            list(self.store.trace_records_for_run(run_id))
            if hasattr(self.store, "trace_records_for_run") else []
        )
        generation_plans = (
            list(self.store.generation_plans_for_run(run_id))
            if hasattr(self.store, "generation_plans_for_run") else []
        )
        task_rows = [self._task(item) for item in tasks]
        return {
            "run_id": run.id,
            "dataset_id": run.dataset_id,
            "status": run.status.value,
            "steps": [
                self._event(event)
                for event in events
                if self._is_agent_event(event.event_type)
            ],
            "tasks": task_rows,
            "artifacts": [self._artifact(item) for item in artifacts],
            "generation_plans": [self._generation_plan(item) for item in generation_plans],
            "trace_records": [self._trace_record(item) for item in trace_records],
            "tool_calls": [self._tool_call(item) for item in tool_calls],
            "causal_edges": self._causal_edges(tasks, events),
            "summary": {
                "task_count": len(tasks),
                "completed_task_count": sum(
                    item.status.value == "COMPLETED" for item in tasks
                ),
                "failed_task_count": sum(
                    item.status.value == "FAILED" for item in tasks
                ),
                "task_counts_by_type": self._counts(
                    item.task_type for item in tasks
                ),
                "artifact_counts_by_kind": self._counts(
                    item.kind for item in artifacts
                ),
                "tool_call_count": len(tool_calls),
                "trace_record_count": len(trace_records),
                "generation_plan_count": len(generation_plans),
            },
        }

    @staticmethod
    def _task(task):
        return {
            "task_id": task.id,
            "task_type": task.task_type,
            "required_capability": task.required_capability.value,
            "status": task.status.value,
            "claimed_by": task.claimed_by,
            "attempts": task.attempts,
            "max_attempts": task.max_attempts,
            "input_artifact_refs": list(task.input_artifact_refs),
            "output_artifact_refs": list(task.output_artifact_refs),
            "error_message": task.error_message,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
            "lease_expires_at": (
                task.lease_expires_at.isoformat()
                if task.lease_expires_at else None
            ),
        }

    def _artifact(self, artifact):
        try:
            payload = json.loads(
                self.store.artifact_content(artifact.id).decode("utf-8")
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            payload = {"unavailable": True}
        return {
            "artifact_id": artifact.id,
            "kind": artifact.kind,
            "digest": artifact.digest,
            "size": artifact.size,
            "payload": payload,
        }

    @staticmethod
    def _tool_call(item):
        return {
            "call_id": item.id,
            "tool_name": item.tool_name,
            "status": item.status,
            "caller": item.request.get("caller", ""),
            "reason": item.request.get("reason", ""),
            "request": item.request,
            "response": item.response,
            "created_at": item.created_at.isoformat(),
        }

    @staticmethod
    def _generation_plan(item):
        return {
            "plan_id": item.plan_id,
            "generation": item.generation,
            "sequence": item.sequence,
            "generation_strategy": item.generation_strategy,
            "parent_selection_policy": item.parent_selection_policy,
            "decision_reason": item.decision_reason,
            "required_parent_count": item.required_parent_count,
            "context_artifact_id": item.context_artifact_id,
            "prompt_artifact_id": item.prompt_artifact_id,
            "created_at": item.created_at.isoformat(),
        }

    @staticmethod
    def _trace_record(item):
        return {
            "trace_id": item.id,
            "span_type": item.span_type,
            "actor": item.actor,
            "status": item.status,
            "plan_id": item.plan_id,
            "candidate_id": item.candidate_id,
            "evaluation_job_id": item.evaluation_job_id,
            "skill_name": item.skill_name,
            "context_refs": list(item.context_refs),
            "tool_call_id": item.tool_call_id,
            "latency_ms": item.latency_ms,
            "token_usage": dict(item.token_usage),
            "payload": dict(item.payload),
            "created_at": item.created_at.isoformat(),
        }

    @staticmethod
    def _is_agent_event(event_type):
        return (
            event_type.startswith("AGENT_TASK_")
            or event_type.startswith("TOOL_CALL_")
            or event_type
            in {
                "SIMILARITY_CANDIDATES_READY",
                "SIMILARITY_NODE_COMPLETED",
                "GENERATION_PLANNED",
                "GENERATION_REQUESTED",
                "GENERATION_SUSPENDED_FOR_RESEARCH",
                "GENERATION_RESUMED_AFTER_RESEARCH",
                "LITERATURE_EVIDENCE_REQUESTED",
                "LITERATURE_EVIDENCE_COMPLETED",
                "CANDIDATE_DRAFT_MATERIALIZED",
                "KNOWLEDGE_GAP_IDENTIFIED",
                "LITERATURE_SEARCHED",
                "LITERATURE_EVIDENCE_USED",
                "CANDIDATE_FAILURE_CLASSIFIED",
                "REPAIRED_CANDIDATE_MATERIALIZED",
                "FINAL_TIE_DETECTED",
            }
        )

    @staticmethod
    def _event(event):
        return {
            "sequence": event.sequence,
            "event_type": event.event_type,
            "actor": event.payload.get("agent")
            or event.payload.get("caller")
            or "DurableAgentCoordinator",
            "message": event.message,
            "payload": event.payload,
            "occurred_at": event.occurred_at.isoformat(),
        }

    @staticmethod
    def _causal_edges(tasks, events):
        edges = []
        seen = set()
        for task in tasks:
            for ref in task.input_artifact_refs:
                edge = ("artifact", ref, "task", task.id, "INPUT")
                if edge not in seen:
                    seen.add(edge)
                    edges.append({
                        "from_type": edge[0], "from_id": edge[1],
                        "to_type": edge[2], "to_id": edge[3],
                        "relation": edge[4],
                    })
            for ref in task.output_artifact_refs:
                edge = ("task", task.id, "artifact", ref, "OUTPUT")
                if edge not in seen:
                    seen.add(edge)
                    edges.append({
                        "from_type": edge[0], "from_id": edge[1],
                        "to_type": edge[2], "to_id": edge[3],
                        "relation": edge[4],
                    })
        for event in events:
            task_id = event.payload.get("agent_task_id") or event.payload.get("task_id")
            if task_id:
                edge = ("event", str(event.sequence), "task", task_id, "OBSERVES")
                if edge not in seen:
                    seen.add(edge)
                    edges.append({
                        "from_type": edge[0], "from_id": edge[1],
                        "to_type": edge[2], "to_id": edge[3],
                        "relation": edge[4],
                    })
        return edges

    @staticmethod
    def _counts(values):
        result = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))
