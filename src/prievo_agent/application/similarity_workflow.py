"""Numeric Top-5 到 durable SimilaritySelectionNode decision 再到 PriorExtractor。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from prievo_agent.agents.similarity import SimilarityAgent, SimilaritySelectionNode
from prievo_agent.domain.events import EventType
from prievo_agent.domain.models import TraceRecord
from prievo_agent.domain.prior import (
    LandscapeProfile,
    SemanticRefinement,
    SimilarInstance,
)


class SimilarityTaskHandler:
    """从 input Artifact 读取受限上下文，再调用兼容 Agent facade。"""

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
    """SimilaritySelectionNode 是一次结构化 LLM Node，不进入 AgentTask。"""

    def __init__(self, store, skill_registry, model):
        self.store = store
        self.node = SimilaritySelectionNode(skill_registry, model)

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
        decision = self.node.select(target, numeric_candidates, metric_semantics)
        prompt = self.store.put_artifact(
            run_id,
            "SIMILARITY_PROMPT",
            decision.selection_prompt.encode("utf-8"),
            "text/plain",
        )
        payload = decision.to_dict()
        payload.pop("selection_prompt", None)
        payload.update(
            {
                "input_artifact_refs": [top5.id],
                "prompt_artifact_id": prompt.id,
                "node_name": "SimilaritySelectionNode",
            }
        )
        output = self.store.put_artifact(
            run_id,
            "SIMILARITY_DECISION",
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            run_id,
            "SIMILARITY_NODE_COMPLETED",
            "SimilaritySelectionNode 已从 numeric Top-5 中完成结构化语义筛选",
            artifact_id=output.id,
            prompt_artifact_id=prompt.id,
            selected_instance_ids=list(payload["selected_instance_ids"]),
        )
        if hasattr(self.store, "record_trace"):
            self.store.record_trace(
                TraceRecord(
                    id="trace-{}".format(
                        hashlib.sha256(output.id.encode("utf-8")).hexdigest()[:24]
                    ),
                    run_id=run_id,
                    span_type="LLM_NODE",
                    actor="SimilaritySelectionNode",
                    status="COMPLETED",
                    payload={
                        "selected_instance_ids": list(
                            payload["selected_instance_ids"]
                        )
                    },
                    context_refs=[top5.id, prompt.id, output.id],
                    skill_name=payload.get("skill_name", ""),
                )
            )
        refinement = SemanticRefinement(
            list(payload["selected_instance_ids"]),
            payload["reason_summary"],
            "SimilaritySelectionNode",
            raw_response=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            fallback_used=False,
        )
        return refinement, top5.id, output.id
