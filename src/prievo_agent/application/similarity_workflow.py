"""Numeric Top-5 到 durable SimilarityAgent decision 再到 PriorExtractor 的用例。"""

from __future__ import annotations

import json
from dataclasses import asdict

from prievo_agent.agents.registry import AgentRegistry
from prievo_agent.agents.similarity import SimilarityAgent
from prievo_agent.application.agent_dispatcher import AgentTaskDispatcher
from prievo_agent.application.durable_agent_coordinator import DurableAgentCoordinator
from prievo_agent.domain.events import EventType
from prievo_agent.domain.prior import (
    LandscapeProfile,
    SemanticRefinement,
    SimilarInstance,
)


class SimilarityTaskHandler:
    """从 input Artifact 读取受限上下文，再调用纯 SimilarityAgent。"""

    name = SimilarityAgent.name
    capability = SimilarityAgent.capability

    def __init__(self, store, agent):
        self.store = store
        self.agent = agent

    def handle(self, task, board):
        if len(task.input_artifact_refs) != 1:
            raise ValueError("SimilarityTask 必须且只能引用一个 TOP5_CANDIDATES Artifact")
        artifact = board.artifact_by_ref(task.input_artifact_refs[0])
        if artifact.kind != "TOP5_CANDIDATES":
            raise ValueError("SimilarityTask 输入必须是 TOP5_CANDIDATES")
        payload = json.loads(
            self.store.artifact_content(artifact.id).decode("utf-8")
        )
        target_data = payload["target_profile"]
        target = LandscapeProfile(
            target_data["instance_name"],
            dict(target_data["metrics"]),
            int(target_data["sample_count"]),
            target_data["analyzer"],
        )
        candidates = [
            SimilarInstance(
                item["instance_name"],
                float(item["distance"]),
                dict(item["raw_metrics"]),
                dict(item["normalized_metrics"]),
            )
            for item in payload["numeric_candidates"]
        ]
        return self.agent.select(
            target, candidates, dict(payload["metric_semantics"])
        )


class SimilarityArtifactWriter:
    """将完整 Prompt 与结构化 Decision 分开持久化并互相引用。"""

    def __init__(self, store):
        self.store = store

    def __call__(self, task, decision):
        prompt = self.store.put_artifact(
            task.run_id,
            "SIMILARITY_PROMPT",
            decision.selection_prompt.encode("utf-8"),
            "text/plain",
        )
        payload = decision.to_dict()
        payload.pop("selection_prompt", None)
        payload.update(
            {
                "input_artifact_refs": list(task.input_artifact_refs),
                "prompt_artifact_id": prompt.id,
                "agent_task_id": task.id,
            }
        )
        output = self.store.put_artifact(
            task.run_id,
            "SIMILARITY_DECISION",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        return [prompt, output]


class DurableSimilarityWorkflow:
    """把 SimilarityAgent 接进真实 AgentTask/Blackboard/Registry/Artifact 路径。"""

    def __init__(self, store, skill_registry, model):
        self.store = store
        handler = SimilarityTaskHandler(
            store, SimilarityAgent(skill_registry, model)
        )
        self.coordinator = DurableAgentCoordinator(store)
        self.dispatcher = AgentTaskDispatcher(
            store, AgentRegistry([handler]), SimilarityArtifactWriter(store)
        )

    def select(self, run_id, target, numeric_candidates, metric_semantics):
        input_payload = {
            "target_profile": asdict(target),
            "numeric_candidates": [asdict(item) for item in numeric_candidates],
            "metric_semantics": dict(metric_semantics),
        }
        top5 = self.store.put_artifact(
            run_id,
            "TOP5_CANDIDATES",
            json.dumps(input_payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run_id,
            EventType.SIMILARITY_CANDIDATES_READY.value,
            "[先验] deterministic FLA numeric Top-5 已就绪",
            artifact_id=top5.id,
            candidate_count=len(numeric_candidates),
            ranked_instance_ids=[item.instance_name for item in numeric_candidates],
        )
        created = self.coordinator.reconcile(run_id)
        tasks = [
            item for item in created
            if item.task_type == "SEMANTIC_SIMILARITY_SELECTION"
            and top5.id in item.input_artifact_refs
        ]
        if not tasks:
            tasks = [
                item for item in self.store.agent_tasks_for_run(run_id)
                if item.task_type == "SEMANTIC_SIMILARITY_SELECTION"
                and top5.id in item.input_artifact_refs
            ]
        if len(tasks) != 1:
            raise RuntimeError("无法为 TOP5_CANDIDATES 唯一定位 SimilarityTask")
        result = self.dispatcher.dispatch_with_retries(tasks[0].id)
        decision_refs = [
            ref for ref in result.artifact_refs
            if any(
                artifact.id == ref and artifact.kind == "SIMILARITY_DECISION"
                for artifact in self.store.artifacts_for_run(run_id)
            )
        ]
        if len(decision_refs) != 1:
            raise RuntimeError("SimilarityTask 未产生唯一 SIMILARITY_DECISION")
        payload = json.loads(
            self.store.artifact_content(decision_refs[0]).decode("utf-8")
        )
        refinement = SemanticRefinement(
            list(payload["selected_instance_ids"]),
            payload["reason_summary"],
            "SimilarityAgent",
            raw_response=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            fallback_used=False,
        )
        return refinement, top5.id, decision_refs[0]
