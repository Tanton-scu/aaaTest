from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
import uuid

from prievo_agent.domain.events import EventType
from prievo_agent.domain.ports import RuntimeStore
from prievo_agent.domain.models import ToolCallRecord, utc_now


class ToolCallDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    tool_name: str
    allowed_callers: frozenset
    max_calls_per_run: int
    side_effect: str
    cost_class: str


class ToolGovernanceGateway:
    def __init__(self, store: RuntimeStore) -> None:
        self.store = store

    def execute(
        self,
        run_id: str,
        caller: str,
        policy: ToolPolicy,
        reason: str,
        operation: Callable[[], Any],
        input_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        result, _ = self.execute_with_audit(
            run_id,
            caller,
            policy,
            reason,
            operation,
            input_metadata=input_metadata,
        )
        return result

    def execute_with_audit(
        self,
        run_id: str,
        caller: str,
        policy: ToolPolicy,
        reason: str,
        operation: Callable[[], Any],
        input_metadata: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Any, str]:
        call_id = "tool-call-{}".format(uuid.uuid4().hex)
        started_at = utc_now()
        request = {
            "caller": caller,
            "reason": reason,
            "policy": policy.tool_name,
            "input_metadata": dict(input_metadata or {}),
            "started_at": started_at.isoformat(),
        }
        budget_scope = str(
            request["input_metadata"].get("tool_budget_scope") or run_id
        )
        denied_reason = ""
        if caller not in policy.allowed_callers:
            denied_reason = "caller 不在 allowlist"
        elif not reason.strip():
            denied_reason = "缺少 tool call reason"
        else:
            prior_calls = sum(
                1
                for event in self.store.events_for_run(run_id)
                if event.event_type == EventType.TOOL_CALL_COMPLETED.value
                and event.payload.get("tool_name") == policy.tool_name
                and str(event.payload.get("tool_budget_scope") or run_id) == budget_scope
            )
            if prior_calls >= policy.max_calls_per_run:
                denied_reason = "tool scoped 调用上限已用尽"
        if denied_reason:
            finished_at = utc_now()
            self._record(ToolCallRecord(
                call_id, run_id, policy.tool_name, "DENIED", request,
                {
                    "reason": denied_reason,
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                },
            ))
            self.store.append_event(
                run_id,
                EventType.TOOL_CALL_DENIED.value,
                "Tool 调用被拒绝",
                tool_name=policy.tool_name,
                caller=caller,
                reason=reason,
                tool_budget_scope=budget_scope,
                denied_reason=denied_reason,
            )
            raise ToolCallDenied(denied_reason)
        self.store.append_event(
            run_id,
            EventType.TOOL_CALL_STARTED.value,
            "Tool 调用已授权",
            tool_name=policy.tool_name,
            caller=caller,
            tool_budget_scope=budget_scope,
            reason=reason,
            side_effect=policy.side_effect,
            cost_class=policy.cost_class,
        )
        self._record(ToolCallRecord(
            call_id, run_id, policy.tool_name, "STARTED", request, {},
        ))
        try:
            result = operation()
        except ToolCallDenied as exc:
            finished_at = utc_now()
            denied_reason = str(exc)[:500] or "tool operation 被策略拒绝"
            self._record(ToolCallRecord(
                call_id, run_id, policy.tool_name, "DENIED", request,
                {
                    "reason": denied_reason,
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                },
            ))
            self.store.append_event(
                run_id,
                EventType.TOOL_CALL_DENIED.value,
                "Tool 调用被拒绝",
                tool_name=policy.tool_name,
                caller=caller,
                reason=reason,
                tool_budget_scope=budget_scope,
                denied_reason=denied_reason,
            )
            raise
        except Exception as exc:
            finished_at = utc_now()
            self._record(ToolCallRecord(
                call_id, run_id, policy.tool_name, "FAILED", request,
                {
                    "error_type": type(exc).__name__,
                    "failure": str(exc)[:500],
                    "finished_at": finished_at.isoformat(),
                    "duration_ms": _duration_ms(started_at, finished_at),
                },
            ))
            self.store.append_event(
                run_id,
                EventType.TOOL_CALL_FAILED.value,
                "Tool 调用失败",
                tool_name=policy.tool_name,
                caller=caller,
                tool_budget_scope=budget_scope,
                error_type=type(exc).__name__,
            )
            raise
        finished_at = utc_now()
        result_count = _result_count(result)
        self.store.append_event(
            run_id,
            EventType.TOOL_CALL_COMPLETED.value,
            "Tool 调用完成",
            tool_name=policy.tool_name,
            caller=caller,
            tool_budget_scope=budget_scope,
            result_count=result_count,
        )
        self._record(ToolCallRecord(
            call_id, run_id, policy.tool_name, "COMPLETED", request,
            {
                "result_count": result_count,
                "finished_at": finished_at.isoformat(),
                "duration_ms": _duration_ms(started_at, finished_at),
            },
        ))
        return result, call_id

    def _record(self, record):
        # 旧的纯内存 harness 没有 SQL audit 表；主线 SQLite/MySQL 必须持久化。
        if hasattr(self.store, "record_tool_call"):
            self.store.record_tool_call(record)


class CandidateInspectionTool:
    policy = ToolPolicy(
        "candidate_inspection", frozenset({"RepairAgent", "HeuristicGenerationAgent"}), 10,
        "READ_ONLY", "LOCAL_LOW",
    )

    def __init__(self, gateway, store):
        self.gateway = gateway
        self.store = store

    def inspect(
        self,
        run_id,
        candidate_id,
        reason,
        caller="RepairAgent",
        tool_budget_scope="",
    ):
        inspection, call_id = self.gateway.execute_with_audit(
            run_id, caller, self.policy, reason,
            lambda: self._inspect(run_id, candidate_id),
            input_metadata={
                "candidate_id": candidate_id,
                "tool_budget_scope": tool_budget_scope or candidate_id,
            },
        )
        value = dict(inspection)
        value["tool_call_ref"] = call_id
        return value

    def _inspect(self, run_id, candidate_id):
        candidate = self.store.candidate_by_id(candidate_id)
        if candidate.run_id != run_id:
            raise ToolCallDenied(
                "Candidate 不属于当前 Run，拒绝跨 Run 检查"
            )
        try:
            result = self.store.result_for_candidate(candidate_id)
            evaluation = {
                "objective": result.objective,
                "trajectory": result.trajectory,
                "used_budget": result.used_budget,
            }
        except KeyError:
            evaluation = None
        jobs = [job for job in self.store.evaluation_jobs_for_run(candidate.run_id)
                if job.candidate_id == candidate_id]
        job = jobs[-1] if jobs else None
        parents = []
        for parent_id in candidate.lineage.get("parents", []):
            try:
                parent = self.store.candidate_by_id(parent_id)
                if parent.run_id != run_id:
                    parents.append({"id": parent_id, "forbidden": True})
                    continue
                parents.append({"id": parent.id, "objective": parent.objective,
                                "description": parent.description})
            except KeyError:
                parents.append({"id": parent_id, "missing": True})
        return {
            "candidate_id": candidate.id,
            "operator": candidate.lineage.get("operator", "unknown"),
            "generation": candidate.lineage.get("generation"),
            "status": candidate.status.value,
            "description": candidate.description,
            "code_excerpt": candidate.code[:2400],
            "parents": parents,
            "evaluation": evaluation,
            "job": ({"id": job.id, "status": job.status.value,
                     "attempts": job.attempts, "error_code": job.error_code,
                     "stderr": job.error_message or "", "stdout": ""}
                    if job else None),
        }


def _result_count(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 1


def _duration_ms(started_at, finished_at) -> int:
    return max(0, int((finished_at - started_at).total_seconds() * 1000))
