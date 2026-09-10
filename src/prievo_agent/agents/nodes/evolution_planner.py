"""EvolutionPlannerNode：只规划下一次生成策略，不生成 Candidate code。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from prievo_agent.agents.context_policies import ContextPolicyFramework
from prievo_agent.domain.models import AgentCapability, GenerationPlan, PreviousPlanFeedback, utc_now


ALLOWED_STRATEGIES = frozenset({"i1", "e1", "e2", "m1", "m2"})
PARENT_POLICIES = frozenset({"roulette", "greedy", "random", "null"})
PARENT_COUNTS = {"i1": 0, "e1": 2, "e2": 2, "m1": 1, "m2": 1}


class PlannerModel(Protocol):
    def plan_generation(self, prompt: str) -> Mapping[str, Any]: ...


class EvolutionPlannerError(RuntimeError):
    pass


class MalformedGenerationPlanError(EvolutionPlannerError):
    pass


class EvolutionPlannerNode:
    """根据 Runtime 状态输出结构化 GenerationPlan。"""

    name = "EvolutionPlannerNode"
    capability = AgentCapability.EVOLUTION_PLANNING

    def __init__(self, model: PlannerModel, context_framework=None):
        self.model = model
        self.context_framework = context_framework or ContextPolicyFramework()

    def plan(self, request: Mapping[str, Any]) -> GenerationPlan:
        if not isinstance(request, Mapping):
            raise TypeError("Planner request 必须是 Mapping")
        strategy = str(request.get("scheduled_strategy", "")).strip()
        if strategy not in ALLOWED_STRATEGIES:
            raise ValueError("scheduled_strategy 必须来自 PriEvO operator 集合")
        parent_policy = _default_parent_policy(strategy)
        payload = {
            "task": dict(request.get("task", {})),
            "phase": dict(request.get("phase", {})),
            "population_summary": dict(request.get("population_summary", {})),
            "population_ranking": list(request.get("population_ranking", [])),
            "strategy_statistics": dict(request.get("strategy_statistics", {})),
            "previous_plan_feedback": dict(request.get("previous_plan_feedback", {})),
            "planner_memory": list(request.get("planner_memory", [])),
            "allowed_strategies": sorted(ALLOWED_STRATEGIES),
            "allowed_parent_policies": sorted(PARENT_POLICIES),
            "output_schema": {
                "generation_strategy": strategy,
                "parent_selection_policy": sorted(PARENT_POLICIES),
                "decision_reason": "short rationale",
                "required_parent_count": PARENT_COUNTS[strategy],
            },
        }
        context = self.context_framework.build("planner", payload)
        prompt = _planner_prompt(strategy, parent_policy, context.text)
        raw = self._call_model(prompt, strategy, parent_policy)
        plan = self._parse_plan(
            raw,
            run_id=str(request["run_id"]),
            generation=int(request["generation"]),
            sequence=int(request["sequence"]),
            scheduled_strategy=strategy,
            context_artifact_id=str(request.get("context_artifact_id", "")),
            prompt_artifact_id=str(request.get("prompt_artifact_id", "")),
            previous_feedback=_feedback(request.get("previous_plan_feedback")),
        )
        return plan

    def _call_model(self, prompt, strategy, parent_policy):
        if self.model is not None and hasattr(self.model, "plan_generation"):
            return self.model.plan_generation(prompt)
        return {
            "generation_strategy": strategy,
            "parent_selection_policy": parent_policy,
            "decision_reason": (
                "Deterministic fallback keeps the PriEvO scheduled operator; "
                "no production LLM planner is configured."
            ),
            "required_parent_count": PARENT_COUNTS[strategy],
        }

    @staticmethod
    def _parse_plan(
        data,
        *,
        run_id,
        generation,
        sequence,
        scheduled_strategy,
        context_artifact_id,
        prompt_artifact_id,
        previous_feedback,
    ):
        if not isinstance(data, Mapping):
            raise MalformedGenerationPlanError("Planner 输出必须是 JSON object")
        strategy = str(data.get("generation_strategy", "")).strip()
        if strategy != scheduled_strategy:
            raise MalformedGenerationPlanError(
                "Planner 不能覆盖 PriEvO 当前调度 strategy：{} -> {}".format(
                    scheduled_strategy, strategy
                )
            )
        policy = str(data.get("parent_selection_policy", "")).strip()
        if policy not in PARENT_POLICIES:
            raise MalformedGenerationPlanError("parent_selection_policy 非法")
        expected_parent_count = PARENT_COUNTS[strategy]
        raw_parent_count = data.get("required_parent_count", expected_parent_count)
        try:
            required_parent_count = int(raw_parent_count)
        except (TypeError, ValueError) as exc:
            raise MalformedGenerationPlanError("required_parent_count 必须是整数") from exc
        reason = str(data.get("decision_reason", "")).strip()
        if not reason:
            raise MalformedGenerationPlanError("decision_reason 不能为空")
        corrections = []
        if required_parent_count != expected_parent_count:
            corrections.append(
                "required_parent_count 已由模型输出 {} 规范化为 PriEvO contract {}".format(
                    required_parent_count, expected_parent_count
                )
            )
            required_parent_count = expected_parent_count
        expected_policy = _default_parent_policy(strategy)
        if strategy == "i1" and policy != "null":
            corrections.append(
                "i1 parent_selection_policy 已由 {} 规范化为 null".format(policy)
            )
            policy = "null"
        elif strategy != "i1" and policy == "null":
            corrections.append(
                "{} parent_selection_policy 已由 null 规范化为 {}".format(
                    strategy, expected_policy
                )
            )
            policy = expected_policy
        if corrections:
            reason = "{} | Backend contract normalization: {}".format(
                reason, "; ".join(corrections)
            )
        identity = {
            "run_id": run_id,
            "generation": generation,
            "sequence": sequence,
            "strategy": strategy,
            "policy": policy,
        }
        plan_id = "plan-{}".format(
            hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24]
        )
        return GenerationPlan(
            plan_id=plan_id,
            run_id=run_id,
            generation=generation,
            sequence=sequence,
            generation_strategy=strategy,
            parent_selection_policy=policy,
            decision_reason=reason,
            required_parent_count=required_parent_count,
            previous_feedback=previous_feedback,
            context_artifact_id=context_artifact_id,
            prompt_artifact_id=prompt_artifact_id,
            created_at=utc_now(),
        )


def _default_parent_policy(strategy):
    return "null" if strategy == "i1" else "roulette"


def _planner_prompt(strategy, policy, context_text):
    return (
        "You are EvolutionPlannerNode in PriEvO. Decide only the parent "
        "selection policy for the already scheduled PriEvO generation strategy. "
        "Do not generate code, do not select concrete parents, and do not change "
        "the scheduled strategy {strategy}. Default policy is {policy}. Return "
        "JSON only with generation_strategy, parent_selection_policy, "
        "decision_reason, required_parent_count.\n\n{context}"
    ).format(strategy=strategy, policy=policy, context=context_text)


def _feedback(value):
    if isinstance(value, PreviousPlanFeedback):
        return value
    if not value:
        return None
    if isinstance(value, Mapping):
        return PreviousPlanFeedback(**dict(value))
    raise TypeError("previous_plan_feedback 必须是 Mapping")


def feedback_to_dict(feedback):
    if feedback is None:
        return {}
    if isinstance(feedback, PreviousPlanFeedback):
        return asdict(feedback)
    return dict(feedback)
