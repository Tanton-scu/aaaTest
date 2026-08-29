"""最终 heuristic 的确定性资格过滤与 durable exact-tie Agent 工作流。"""

from __future__ import annotations

import json

from prievo_agent.agents.final_selection import (
    DeterministicFinalSelector,
    ENGINEERING_QUALIFICATION_MODE,
    FinalCandidate,
    FinalSelectionNode,
    FinalSelectionSkill,
    REFERENCE_QUALIFICATION_MODE,
)
from prievo_agent.domain.events import EventType


class FinalSelectionTaskHandler:
    name = FinalSelectionNode.name
    capability = FinalSelectionNode.capability

    def __init__(self, store, agent, skill):
        self.store = store
        self.agent = agent
        self.skill = skill

    def handle(self, task, board):
        if len(task.input_artifact_refs) != 1:
            raise ValueError("FinalSelectionTask 必须且只能引用一个 FINAL_TIE")
        artifact = board.artifact_by_ref(task.input_artifact_refs[0])
        if artifact.kind != "FINAL_TIE":
            raise ValueError("FinalSelectionTask 输入必须是 FINAL_TIE")
        payload = json.loads(self.store.artifact_content(artifact.id))
        candidates = [_final_candidate(item) for item in payload["candidates"]]
        return self.agent.select(candidates, skill=self.skill)


class FinalSelectionArtifactWriter:
    def __init__(self, store):
        self.store = store

    def __call__(self, task, decision):
        refs = []
        if decision.prompt:
            refs.append(self.store.put_artifact(
                task.run_id,
                "FINAL_SELECTION_PROMPT",
                decision.prompt.encode("utf-8"),
                "text/plain",
            ))
        payload = decision.to_dict()
        payload.pop("prompt", None)
        payload.update({
            "agent_task_id": task.id,
            "input_artifact_refs": list(task.input_artifact_refs),
            "prompt_artifact_id": refs[0].id if refs else "",
        })
        refs.append(self.store.put_artifact(
            task.run_id,
            "FINAL_SELECTION_DECISION",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        ))
        return refs


class DurableFinalSelectionWorkflow:
    """唯一 best 不建 AgentTask；只有 exact tie 才进入 durable Agent path。"""

    def __init__(self, store, skill_registry, model, *, faithful_mode=False):
        self.store = store
        self.faithful_mode = bool(faithful_mode)
        skill_definition = skill_registry.require("final_heuristic_audit")
        skill = FinalSelectionSkill(
            skill_definition.name,
            skill_definition.content_digest,
            skill_definition.instructions,
            "skill:{}:{}".format(
                skill_definition.name, skill_definition.content_digest[:16]
            ),
        )
        qualification_mode = (
            REFERENCE_QUALIFICATION_MODE
            if self.faithful_mode
            else ENGINEERING_QUALIFICATION_MODE
        )
        selector = DeterministicFinalSelector(qualification_mode)
        agent = FinalSelectionNode(model, selector=selector)
        self.agent = agent
        self.skill = skill
        self.selector = selector

    def select(self, run_id, candidates, candidate_budget):
        final_candidates = [
            self._project(candidate, candidate_budget) for candidate in candidates
        ]
        empty_policy = "raise" if self.faithful_mode else "stable_fallback"
        resolution = self.selector.resolve(final_candidates, empty_policy=empty_policy)
        if resolution.controlled_fallback_candidate is not None:
            decision = self.agent.select(
                final_candidates, empty_policy="stable_fallback"
            )
            decision_ref = self._persist_direct(run_id, decision)
            return decision.selected_candidate_id, decision_ref, decision
        if len(resolution.tied_best) == 1:
            decision = self.agent.select(final_candidates)
            decision_ref = self._persist_direct(run_id, decision)
            return decision.selected_candidate_id, decision_ref, decision

        payload = {
            "candidates": [_candidate_dict(item) for item in final_candidates],
            "tied_candidate_ids": [item.id for item in resolution.tied_best],
            "best_objective": resolution.best_objective,
            "candidate_budget": candidate_budget,
        }
        tie = self.store.put_artifact(
            run_id,
            "FINAL_TIE",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run_id,
            EventType.FINAL_TIE_DETECTED.value,
            "最终合格候选出现 exact objective tie，调用 FinalSelectionNode",
            artifact_id=tie.id,
            tied_candidate_ids=payload["tied_candidate_ids"],
            best_objective=resolution.best_objective,
        )
        decision = self.agent.select(final_candidates, skill=self.skill)
        decision_ref = self._persist_direct(run_id, decision)
        decision_payload = decision.to_dict()
        decision_payload.pop("prompt", None)
        return decision.selected_candidate_id, decision_ref, decision_payload

    def _project(self, candidate, candidate_budget):
        result = self.store.result_for_candidate(candidate.id)
        return FinalCandidate(
            id=candidate.id,
            code=candidate.code,
            description=candidate.description,
            operators=tuple(candidate.operators),
            objective=float(candidate.objective),
            trajectory=tuple(result.trajectory),
            candidate_budget=int(candidate_budget),
            used_budget=int(result.used_budget),
        )

    def _persist_direct(self, run_id, decision):
        prompt_artifact_id = ""
        if decision.prompt:
            prompt_artifact = self.store.put_artifact(
                run_id,
                "FINAL_SELECTION_PROMPT",
                decision.prompt.encode("utf-8"),
                "text/plain",
            )
            prompt_artifact_id = prompt_artifact.id
        payload = decision.to_dict()
        payload.pop("prompt", None)
        payload["prompt_artifact_id"] = prompt_artifact_id
        artifact = self.store.put_artifact(
            run_id,
            "FINAL_SELECTION_DECISION",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        return artifact.id


def _candidate_dict(candidate):
    return {
        "id": candidate.id,
        "code": candidate.code,
        "description": candidate.description,
        "operators": list(candidate.operators),
        "objective": candidate.objective,
        "trajectory": list(candidate.trajectory),
        "candidate_budget": candidate.candidate_budget,
        "used_budget": candidate.used_budget,
    }


def _final_candidate(payload):
    return FinalCandidate(
        id=payload["id"],
        code=payload["code"],
        description=payload["description"],
        operators=tuple(payload["operators"]),
        objective=float(payload["objective"]),
        trajectory=tuple(payload["trajectory"]),
        candidate_budget=int(payload["candidate_budget"]),
        used_budget=int(payload["used_budget"]),
    )
