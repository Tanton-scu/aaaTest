"""KnowledgeGap -> durable PriorResearchTask -> evidence/explanation 的产品工作流。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from prievo_agent.agents.prior_research import (
    KnowledgeGap as ResearchKnowledgeGap,
    PriorResearchAgent,
    ResearchSkill,
    RetrievedLiteratureChunk,
    ToolExecution,
)
from prievo_agent.agents.registry import AgentRegistry
from prievo_agent.application.agent_dispatcher import AgentTaskDispatcher
from prievo_agent.application.durable_agent_coordinator import DurableAgentCoordinator
from prievo_agent.application.run_local_memory import (
    RunLocalMemoryService,
    memory_context,
)
from prievo_agent.application.history_compactor import HistoryCompactor
from prievo_agent.application.tool_governance import (
    ToolGovernanceGateway,
    ToolPolicy,
)
from prievo_agent.domain.literature import LiteratureQuery
from prievo_agent.domain.models import AgentCapability


class PriorResearchModelAdapter:
    """把产品/Fake model 收敛为 PriorResearchAgent 的两个局部能力。"""

    def __init__(self, model) -> None:
        self.model = model

    def formulate_query(self, prompt):
        if hasattr(self.model, "formulate_query"):
            return self.model.formulate_query(prompt)
        if hasattr(self.model, "_agent_json"):
            value = self.model._agent_json(
                prompt,
                "Return JSON only with one non-empty string key: query.",
            )
            return str(value.get("query", "")).strip()
        # Fake/offline mode 的查询仍有界，不向 Prompt 注入整个 Prior。
        return "PriEvO fitness landscape heuristic prior mechanism"

    def explain_prior(self, prompt):
        if hasattr(self.model, "explain_prior"):
            return self.model.explain_prior(prompt)
        if hasattr(self.model, "_agent_json"):
            return self.model._agent_json(
                prompt,
                "Return JSON keys summary, relationship_to_landscape, "
                "strategy_implications. JSON only.",
            )
        return {
            "summary": "Fake/offline annotation grounded only in retrieved evidence.",
            "relationship_to_landscape": (
                "Retrieved evidence may clarify the named Prior mechanism; benchmark "
                "validation remains authoritative."
            ),
            "strategy_implications": [
                "Keep Original Prior immutable and use this annotation as bounded context."
            ],
        }


class LiteratureRetrieverAdapter:
    """把现有 Local/Hybrid literature backend 转为 Agent chunk schema。"""

    def __init__(self, backend) -> None:
        self.backend = backend

    def search(self, query):
        backend_query = LiteratureQuery(
            algorithm_names=["PriEvO"],
            topic=query.text,
            purpose="Resolve bounded Generation KnowledgeGap",
            reason="PriorResearchAgent KnowledgeGap {}".format(
                query.knowledge_gap_ref
            ),
            max_results=query.max_results,
        )
        if hasattr(self.backend, "retrieve"):
            values = [
                item.to_dict()
                for item in self.backend.retrieve(
                    backend_query,
                    mode="hybrid",
                    rerank=True,
                    top_k=min(query.max_results, 5),
                    expand_neighbors=True,
                )
            ]
        else:
            values = self.backend.search(backend_query)
        return [self._chunk(value) for value in values]

    @staticmethod
    def _chunk(value):
        if isinstance(value, RetrievedLiteratureChunk):
            return value
        if isinstance(value, Mapping):
            metadata = dict(value.get("source_metadata", {}))
            for key in (
                "authors", "year", "identifier", "source_path", "primary_source",
                "adjacent_chunk_ids", "bm25_score", "vector_score", "fusion_score",
                "rerank_score", "retrieval_mode", "vector_backend",
                "vector_is_production_semantic", "page_start", "page_end",
                "chunk_index", "expanded_chunk_ids", "bm25_raw_score",
                "bm25_normalized_score", "vector_raw_score",
                "vector_normalized_score", "rerank_bonus", "final_score",
            ):
                if key in value:
                    metadata[key] = value[key]
            return RetrievedLiteratureChunk(
                paper_ref=str(value.get("paper_ref", value.get("paper_id", ""))),
                title=str(value.get("title", "")),
                section=str(value.get("section", "")),
                chunk_ref=str(value.get("chunk_ref", value.get("chunk_id", ""))),
                content=str(value.get("content", "")),
                score=float(
                    value.get(
                        "score",
                        value.get("final_score", value.get("rerank_score", 0.0)),
                    )
                ),
                source_metadata=metadata,
            )
        metadata = {
            "authors": list(getattr(value, "authors", [])),
            "year": getattr(value, "year", ""),
            "identifier": getattr(value, "identifier", ""),
            "source_path": getattr(value, "source_path", ""),
            "primary_source": getattr(value, "primary_source", False),
            "adjacent_chunk_ids": list(
                getattr(value, "adjacent_chunk_ids", [])
            ),
        }
        for key in ("bm25_score", "vector_score", "fusion_score", "rerank_score"):
            if hasattr(value, key):
                metadata[key] = getattr(value, key)
        return RetrievedLiteratureChunk(
            paper_ref="paper:{}".format(getattr(value, "paper_id", "unknown")),
            title=str(getattr(value, "title", "")),
            section=str(getattr(value, "section", "")),
            chunk_ref=str(getattr(value, "chunk_id", "")),
            content=str(getattr(value, "content", "")),
            score=float(getattr(value, "score", 0.0)),
            source_metadata=metadata,
        )


class LiteratureSearchTool:
    """PriorResearchAgent 的受治理、可追溯 Literature Search Tool。"""

    policy = ToolPolicy(
        "literature_search",
        frozenset({"PriorResearchAgent"}),
        16,
        "READ_ONLY",
        "LOCAL_LOW",
    )

    def __init__(self, store) -> None:
        self.gateway = ToolGovernanceGateway(store)

    def execute(self, run_id, caller, tool_name, reason, operation):
        if tool_name != self.policy.tool_name:
            raise ValueError("PriorResearch workflow 只允许 literature_search")
        result, call_id = self.gateway.execute_with_audit(
            run_id,
            caller,
            self.policy,
            reason,
            lambda: list(operation()),
            input_metadata={"tool_name": tool_name},
        )
        return ToolExecution(call_id, tuple(result), "COMPLETED")


class PriorResearchTaskHandler:
    name = PriorResearchAgent.name
    capability = AgentCapability.PRIOR_RESEARCH

    def __init__(self, store, skill_registry, agent, memory, history_compactor) -> None:
        self.store = store
        self.skill_registry = skill_registry
        self.agent = agent
        self.memory = memory
        self.history_compactor = history_compactor

    def handle(self, task, board):
        if len(task.input_artifact_refs) != 1:
            raise ValueError("PriorResearchTask 必须且只能引用一个 KNOWLEDGE_GAP")
        gap_artifact = board.artifact_by_ref(task.input_artifact_refs[0])
        if gap_artifact.kind != "KNOWLEDGE_GAP":
            raise ValueError("PriorResearchTask 输入必须是 KNOWLEDGE_GAP")
        gap_payload = _artifact_json(self.store, gap_artifact.id)
        request_refs = list(gap_payload.get("input_artifact_refs", []))
        if len(request_refs) != 1:
            raise ValueError("KNOWLEDGE_GAP 必须关联唯一 Generation request")
        request_artifact = board.artifact_by_ref(request_refs[0])
        if request_artifact.kind not in {
            "GENERATION_REQUEST", "GENERATION_RESUME_REQUEST"
        }:
            raise ValueError("KnowledgeGap source 必须是 Generation request")
        request_payload = _artifact_json(self.store, request_artifact.id)
        original_prior = request_payload["original_prior_slice"]
        original_digest = request_payload.get(
            "original_prior_digest", _json_digest(original_prior)
        )
        if original_digest != _json_digest(original_prior):
            raise ValueError("Generation request 的 Original Prior digest 不一致")

        definition = self.skill_registry.require("prior_explanation")
        review_definition = self.skill_registry.require(
            "literature_evidence_review"
        )
        skill = _research_skill(definition)
        evidence_review_skill = _research_skill(review_definition)
        research_gap = ResearchKnowledgeGap(
            ref=gap_artifact.id,
            summary=str(gap_payload["knowledge_gap"]),
            reason=str(gap_payload["reason"]),
            # Generation request 是 immutable prior snapshot 的 durable root ref。
            original_prior_ref=request_payload.get(
                "stable_generation_request_id", request_artifact.id
            ),
        )
        parents = [
            {
                "candidate_ref": item.get("candidate_id", ""),
                "description": item.get("description", ""),
                "fitness": item.get("fitness"),
                "operators": list(item.get("operators", [])),
                "lineage": item.get("lineage", {}),
            }
            for item in request_payload.get("parents", [])
        ]
        landscape = {
            "target_instance": original_prior.get("target_instance", ""),
            "metrics": original_prior.get("target_landscape_metrics", {}),
        }
        return self.agent.research(
            run_id=task.run_id,
            knowledge_gap=research_gap,
            original_prior_slice=original_prior,
            landscape_summary=landscape,
            strategy=request_payload["strategy"],
            skill=skill,
            evidence_review_skill=evidence_review_skill,
            parent_summary=parents,
            research_history=memory_context(
                self.history_compactor.context_records(
                    task.run_id,
                    str(request_payload.get("task_contract", {}).get("dataset_id", "")),
                    "research",
                    recent_window=4,
                )
            ),
        )


class PriorResearchArtifactWriter:
    def __init__(self, store) -> None:
        self.store = store

    def __call__(self, task, result):
        query_prompt = self.store.put_artifact(
            task.run_id,
            "PRIOR_RESEARCH_QUERY_PROMPT",
            result.query_prompt.encode("utf-8"),
            "text/plain",
        )
        explanation_prompt = self.store.put_artifact(
            task.run_id,
            "PRIOR_RESEARCH_EXPLANATION_PROMPT",
            result.explanation_prompt.encode("utf-8"),
            "text/plain",
        )
        original_prior_digest = hashlib.sha256(
            result.original_prior_snapshot.encode("utf-8")
        ).hexdigest()
        common = {
            "input_artifact_refs": list(task.input_artifact_refs),
            "knowledge_gap_artifact_id": result.knowledge_gap.ref,
            "agent_task_id": task.id,
            "original_prior_ref": result.knowledge_gap.original_prior_ref,
            "original_prior_digest": original_prior_digest,
            "prior_explanation_skill_ref": result.skill_ref,
            "skill_provenance": [
                dict(item) for item in result.skill_provenance
            ],
            "query_prompt_artifact_id": query_prompt.id,
            "explanation_prompt_artifact_id": explanation_prompt.id,
            "context_metadata": dict(result.context_metadata),
        }
        evidence_payload = result.literature_evidence.to_dict()
        evidence_payload.update(common)
        evidence = self.store.put_artifact(
            task.run_id,
            "LITERATURE_EVIDENCE",
            json.dumps(
                evidence_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        explanation_payload = result.explanation.to_dict()
        explanation_payload.update(common)
        explanation_payload["literature_evidence_artifact_id"] = evidence.id
        explanation = self.store.put_artifact(
            task.run_id,
            "PRIOR_EXPLANATION",
            json.dumps(
                explanation_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        self.store.append_event(
            task.run_id,
            "PRIOR_RESEARCH_COMPLETED",
            "PriorResearchAgent 已完成有界研究并保留 Original Prior",
            agent_task_id=task.id,
            knowledge_gap_artifact_id=result.knowledge_gap.ref,
            prior_explanation_artifact_id=explanation.id,
            literature_evidence_artifact_id=evidence.id,
            evidence_status=result.literature_evidence.status,
            evidence_count=len(result.literature_evidence.items),
            original_prior_digest=original_prior_digest,
        )
        return [query_prompt, explanation_prompt, evidence, explanation]


def _research_skill(definition):
    return ResearchSkill(
        name=definition.name,
        version=str(definition.version),
        digest=definition.content_digest,
        instructions=definition.instructions,
        ref="skill:{}@{}#{}".format(
            definition.name, definition.version, definition.content_digest
        ),
    )


@dataclass(frozen=True)
class PriorResearchResolution:
    run_id: str
    knowledge_gap_artifact_id: str
    research_task_id: str
    prior_explanation_artifact_id: str
    literature_evidence_artifact_id: str
    query_prompt_artifact_id: str
    explanation_prompt_artifact_id: str
    original_prior_ref: str
    original_prior_digest: str
    prior_explanation: Mapping[str, Any]
    literature_evidence: Mapping[str, Any]

    @property
    def evidence_status(self):
        return str(self.literature_evidence["status"])

    @property
    def context_refs(self):
        return [
            self.knowledge_gap_artifact_id,
            self.prior_explanation_artifact_id,
            self.literature_evidence_artifact_id,
            self.query_prompt_artifact_id,
            self.explanation_prompt_artifact_id,
        ]

    def generation_evidence(self):
        return {
            "run_id": self.run_id,
            "annotation_type": "PRIOR_RESEARCH",
            "knowledge_gap_ref": self.knowledge_gap_artifact_id,
            "original_prior_ref": self.original_prior_ref,
            "original_prior_digest": self.original_prior_digest,
            "prior_explanation_ref": self.prior_explanation_artifact_id,
            "literature_evidence_ref": self.literature_evidence_artifact_id,
            "evidence_status": self.evidence_status,
            "summary": self.prior_explanation.get("summary", ""),
            "relationship_to_landscape": self.prior_explanation.get(
                "relationship_to_landscape", ""
            ),
            "strategy_implications": list(
                self.prior_explanation.get("strategy_implications", [])
            ),
            "skill_provenance": list(
                self.literature_evidence.get("skill_provenance", [])
            ),
            "items": list(self.literature_evidence.get("items", [])),
        }


class DurablePriorResearchWorkflow:
    """用 Coordinator/Dispatcher 执行且可恢复的唯一 PriorResearchTask。"""

    def __init__(
        self,
        store,
        skill_registry,
        model,
        literature_backend,
        working_memory=None,
    ) -> None:
        self.store = store
        self.memory = RunLocalMemoryService(store, working_memory)
        self.history_compactor = HistoryCompactor(
            store, skill_registry, model, working_memory
        )
        agent = PriorResearchAgent(
            PriorResearchModelAdapter(model),
            LiteratureRetrieverAdapter(literature_backend),
            LiteratureSearchTool(store),
            max_results=5,
        )
        handler = PriorResearchTaskHandler(
            store,
            skill_registry,
            agent,
            self.memory,
            self.history_compactor,
        )
        self.coordinator = DurableAgentCoordinator(store)
        self.dispatcher = AgentTaskDispatcher(
            store,
            AgentRegistry([handler]),
            PriorResearchArtifactWriter(store),
        )

    def resolve(self, run_id, knowledge_gap_artifact_id):
        artifacts = {item.id: item for item in self.store.artifacts_for_run(run_id)}
        gap = artifacts.get(knowledge_gap_artifact_id)
        if gap is None or gap.kind != "KNOWLEDGE_GAP":
            raise ValueError("KnowledgeGap artifact 不属于当前 Run")
        created = self.coordinator.reconcile(run_id)
        tasks = [
            item
            for item in self.store.agent_tasks_for_run(run_id)
            if item.task_type == "PRIOR_RESEARCH"
            and item.input_artifact_refs == [knowledge_gap_artifact_id]
        ]
        if len(tasks) != 1:
            raise RuntimeError("KnowledgeGap 未映射到唯一 PriorResearchTask")
        if any(item.id == tasks[0].id for item in created):
            self.store.append_event(
                run_id,
                "PRIOR_RESEARCH_REQUESTED",
                "KnowledgeGap 已派生唯一 PriorResearchTask",
                agent_task_id=tasks[0].id,
                knowledge_gap_artifact_id=knowledge_gap_artifact_id,
            )
        dispatched = self.dispatcher.dispatch_with_retries(tasks[0].id)
        resolution = self._resolution(
            run_id,
            knowledge_gap_artifact_id,
            tasks[0].id,
            dispatched.artifact_refs,
        )
        self._remember(resolution)
        return resolution

    def _remember(self, resolution):
        gap = _artifact_json(self.store, resolution.knowledge_gap_artifact_id)
        before = {
            item.id
            for item in self.store.agent_memories_for_run(
                resolution.run_id, "research", limit=1_000_000
            )
        }
        items = list(resolution.literature_evidence.get("items", []))
        memory = self.memory.persist(
            run_id=resolution.run_id,
            dataset_id=str(gap.get("dataset_id", "")),
            scope="research",
            memory_type="RESEARCH_RESULT",
            subject="KnowledgeGap {} / {}".format(
                resolution.knowledge_gap_artifact_id,
                resolution.evidence_status,
            ),
            content={
                "knowledge_gap_ref": resolution.knowledge_gap_artifact_id,
                "knowledge_gap": gap.get("knowledge_gap", ""),
                "query": resolution.literature_evidence.get("query", ""),
                "query_ref": resolution.literature_evidence.get("query_ref", ""),
                "retrieved_papers": [
                    {
                        "paper_ref": item.get("paper_ref", ""),
                        "chunk_ref": item.get("chunk_ref", ""),
                        "section": item.get("section", ""),
                    }
                    for item in items
                ],
                "evidence_status": resolution.evidence_status,
                "evidence_summary": resolution.prior_explanation.get(
                    "summary", ""
                ),
                "skill_provenance": list(
                    resolution.literature_evidence.get(
                        "skill_provenance", []
                    )
                ),
                "prior_explanation_artifact_id": (
                    resolution.prior_explanation_artifact_id
                ),
                "literature_evidence_artifact_id": (
                    resolution.literature_evidence_artifact_id
                ),
            },
            evidence_artifact_id=resolution.literature_evidence_artifact_id,
            identity_material="{}|{}".format(
                resolution.knowledge_gap_artifact_id,
                resolution.literature_evidence_artifact_id,
            ),
        )
        if memory.id not in before:
            self.store.append_event(
                resolution.run_id,
                "RESEARCH_MEMORY_UPDATED",
                "PriorResearchAgent 已写入同 Run research scope 历史",
                agent_memory_id=memory.id,
                knowledge_gap_artifact_id=resolution.knowledge_gap_artifact_id,
                query_ref=resolution.literature_evidence.get("query_ref", ""),
                retrieved_paper_count=len(items),
            )

    def _resolution(self, run_id, gap_ref, task_id, refs):
        artifacts = {item.id: item for item in self.store.artifacts_for_run(run_id)}
        by_kind = {}
        for ref in refs:
            artifact = artifacts.get(ref)
            if artifact is not None:
                by_kind[artifact.kind] = artifact
        required = {
            "PRIOR_EXPLANATION",
            "LITERATURE_EVIDENCE",
            "PRIOR_RESEARCH_QUERY_PROMPT",
            "PRIOR_RESEARCH_EXPLANATION_PROMPT",
        }
        if not required.issubset(by_kind):
            raise RuntimeError("PriorResearchTask durable outputs 不完整")
        explanation = _artifact_json(self.store, by_kind["PRIOR_EXPLANATION"].id)
        evidence = _artifact_json(self.store, by_kind["LITERATURE_EVIDENCE"].id)
        if explanation.get("knowledge_gap_artifact_id") != gap_ref:
            raise RuntimeError("PriorExplanation 与 KnowledgeGap 关联不一致")
        if evidence.get("knowledge_gap_artifact_id") != gap_ref:
            raise RuntimeError("LiteratureEvidence 与 KnowledgeGap 关联不一致")
        if explanation["original_prior_digest"] != evidence["original_prior_digest"]:
            raise RuntimeError("Research outputs 的 Original Prior digest 不一致")
        return PriorResearchResolution(
            run_id=run_id,
            knowledge_gap_artifact_id=gap_ref,
            research_task_id=task_id,
            prior_explanation_artifact_id=by_kind["PRIOR_EXPLANATION"].id,
            literature_evidence_artifact_id=by_kind["LITERATURE_EVIDENCE"].id,
            query_prompt_artifact_id=by_kind["PRIOR_RESEARCH_QUERY_PROMPT"].id,
            explanation_prompt_artifact_id=by_kind[
                "PRIOR_RESEARCH_EXPLANATION_PROMPT"
            ].id,
            original_prior_ref=str(explanation["original_prior_ref"]),
            original_prior_digest=str(explanation["original_prior_digest"]),
            prior_explanation=explanation,
            literature_evidence=evidence,
        )


def _artifact_json(store, artifact_id):
    return json.loads(store.artifact_content(artifact_id).decode("utf-8"))


def _json_digest(value):
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
