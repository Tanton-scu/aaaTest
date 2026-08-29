"""Durable EvolutionPlannerAgent workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from prievo_agent.agents.evolution_planner import EvolutionPlannerAgent
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import (
    AgentMemory,
    PreviousPlanFeedback,
    TraceRecord,
    utc_now,
)


class DurablePlanningWorkflow:
    """为每次 Candidate generation 持久化 GenerationPlan。"""

    def __init__(self, store, model, working_memory=None):
        self.store = store
        self.agent = EvolutionPlannerAgent(model)
        self.working_memory = working_memory

    def plan(
        self,
        run,
        task,
        population,
        generation,
        sequence,
        scheduled_strategy,
        previous_feedback=None,
    ):
        feedback = _feedback(previous_feedback)
        payload = {
            "run_id": run.id,
            "generation": int(generation),
            "sequence": int(sequence),
            "scheduled_strategy": scheduled_strategy,
            "task": {
                "dataset_id": task.dataset_id,
                "objective": task.objective,
                "generation": int(generation),
                "population_size": task.population_size,
                "candidate_budget": task.evaluation_budget,
                "total_budget": task.total_budget,
                "consumed_budget": run.consumed_evaluations,
                "reserved_budget": run.reserved_evaluations,
                "available_budget": (
                    task.total_budget
                    - run.consumed_evaluations
                    - run.reserved_evaluations
                ),
            },
            "phase": {
                "name": "generation_operator_batch",
                "scheduled_strategy": scheduled_strategy,
                "sequence": int(sequence),
            },
            "population_summary": _population_summary(population),
            "population_ranking": _population_ranking(population),
            "strategy_statistics": _strategy_statistics(
                self.store.candidates_for_run(run.id)
            ),
            "previous_plan_feedback": (
                asdict(feedback) if feedback is not None else {}
            ),
            "planner_memory": _planner_memory(self.store, run.id),
        }
        request = self.store.put_artifact(
            run.id,
            "PLANNER_CONTEXT",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        payload["context_artifact_id"] = request.id
        plan = self.agent.plan(payload)
        plan_payload = {
            "plan_id": plan.plan_id,
            "run_id": plan.run_id,
            "generation": plan.generation,
            "sequence": plan.sequence,
            "generation_strategy": plan.generation_strategy,
            "parent_selection_policy": plan.parent_selection_policy,
            "decision_reason": plan.decision_reason,
            "required_parent_count": plan.required_parent_count,
            "previous_feedback": (
                asdict(plan.previous_feedback)
                if plan.previous_feedback is not None else {}
            ),
            "context_artifact_id": request.id,
        }
        prompt_artifact = self.store.put_artifact(
            run.id,
            "GENERATION_PLAN",
            json.dumps(plan_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        plan = type(plan)(
            plan.plan_id,
            plan.run_id,
            plan.generation,
            plan.sequence,
            plan.generation_strategy,
            plan.parent_selection_policy,
            plan.decision_reason,
            plan.required_parent_count,
            plan.previous_feedback,
            request.id,
            prompt_artifact.id,
            plan.created_at,
        )
        persisted, created = self.store.add_generation_plan(plan)
        if created:
            self.store.append_event(
                run.id,
                "GENERATION_PLANNED",
                "EvolutionPlannerAgent 已产生结构化 GenerationPlan",
                plan_id=plan.plan_id,
                generation=generation,
                sequence=sequence,
                generation_strategy=plan.generation_strategy,
                parent_selection_policy=plan.parent_selection_policy,
                required_parent_count=plan.required_parent_count,
                context_artifact_id=request.id,
                plan_artifact_id=prompt_artifact.id,
            )
            self._record_trace(plan, request.id, prompt_artifact.id)
            self._record_memory(run, task, plan, request.id, prompt_artifact.id)
        return persisted

    def _record_trace(self, plan, context_ref, plan_ref):
        if not hasattr(self.store, "record_trace"):
            return
        trace_id = "trace-{}".format(
            hashlib.sha256(
                "{}|{}".format(plan.run_id, plan.plan_id).encode("utf-8")
            ).hexdigest()[:24]
        )
        self.store.record_trace(
            TraceRecord(
                id=trace_id,
                run_id=plan.run_id,
                span_type="LLM_AGENT_PLAN",
                actor=EvolutionPlannerAgent.name,
                status="COMPLETED",
                plan_id=plan.plan_id,
                context_refs=[context_ref, plan_ref],
                payload={
                    "generation_strategy": plan.generation_strategy,
                    "parent_selection_policy": plan.parent_selection_policy,
                    "decision_reason": plan.decision_reason,
                },
                created_at=utc_now(),
            )
        )

    def _record_memory(self, run, task, plan, context_ref, plan_ref):
        memory_id = "planner-memory-{}".format(
            hashlib.sha256(plan.plan_id.encode("utf-8")).hexdigest()[:24]
        )
        content = json.dumps(
            {
                "plan_id": plan.plan_id,
                "generation": plan.generation,
                "sequence": plan.sequence,
                "generation_strategy": plan.generation_strategy,
                "parent_selection_policy": plan.parent_selection_policy,
                "required_parent_count": plan.required_parent_count,
                "decision_reason": plan.decision_reason,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        self.store.add_agent_memory(
            AgentMemory(
                id=memory_id,
                run_id=run.id,
                dataset_id=task.dataset_id,
                memory_type="PLANNING_EXPERIENCE",
                subject="{}:{}:{}".format(
                    plan.generation, plan.sequence, plan.generation_strategy
                ),
                content=content,
                evidence_artifact_id=plan_ref,
                agent_name=EvolutionPlannerAgent.name,
                detail_level="DETAILED",
                sequence_no=int(plan.sequence),
                source_refs=[context_ref, plan_ref],
            )
        )
        if self.working_memory is not None:
            self.working_memory.append(
                run.id,
                "PLANNING_EXPERIENCE",
                content,
                scope="planner",
                plan_id=plan.plan_id,
                evidence_artifact_id=plan_ref,
            )


def _population_summary(population):
    values = [item for item in population if item.objective is not None]
    objectives = [float(item.objective) for item in values]
    return {
        "candidate_count": len(population),
        "evaluated_count": len(values),
        "best_objective": min(objectives) if objectives else None,
        "worst_objective": max(objectives) if objectives else None,
    }


def _population_ranking(population):
    values = sorted(
        (item for item in population if item.objective is not None),
        key=lambda item: (float(item.objective), item.id),
    )
    return [
        {
            "rank": rank,
            "candidate_id": item.id,
            "fitness": float(item.objective),
            "strategy": item.generation_strategy or item.lineage.get("operator", ""),
            "parent_ids": list(item.selected_parent_ids or item.lineage.get("parents", [])),
            "operators": list(item.operators),
        }
        for rank, item in enumerate(values, 1)
    ]


def _strategy_statistics(candidates):
    result = {}
    for candidate in candidates:
        strategy = candidate.generation_strategy or candidate.lineage.get("operator", "")
        if not strategy:
            continue
        bucket = result.setdefault(strategy, {"count": 0, "fitness_values": []})
        bucket["count"] += 1
        if candidate.objective is not None:
            bucket["fitness_values"].append(float(candidate.objective))
    for value in result.values():
        fitness_values = value["fitness_values"]
        value["best_fitness"] = min(fitness_values) if fitness_values else None
        value["fitness_values"] = fitness_values[-8:]
    return result


def _planner_memory(store, run_id):
    try:
        memories = store.agent_memories_for_run(run_id, "planner", limit=6)
    except Exception:
        return []
    return [
        {
            "memory_type": item.memory_type,
            "subject": item.subject,
            "content": item.content,
            "evidence_artifact_id": item.evidence_artifact_id,
        }
        for item in memories
    ]


def _feedback(value):
    if value is None or value == {}:
        return None
    if isinstance(value, PreviousPlanFeedback):
        return value
    return PreviousPlanFeedback(**dict(value))
