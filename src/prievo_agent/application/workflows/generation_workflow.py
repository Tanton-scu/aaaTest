"""Durable HeuristicGenerationAgent 工作流。

Core 先决定 strategy 与 parents，再把最小 GenerationRequest 持久化。LLM 输出先成为
CandidateDraft Artifact；即使 Runtime 在创建 Candidate 前退出，恢复时也只消费该
Artifact，不会重新调用模型。
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from dataclasses import asdict

from prievo_agent.agents.nodes.heuristic_generation import (
    CandidateDraft,
    GenerationParent,
    GenerationRequest,
    HeuristicGenerationAgent,
    KnowledgeGap,
)
from prievo_agent.agents.registry import AgentRegistry
from prievo_agent.application.orchestration.agent_dispatcher import AgentTaskDispatcher
from prievo_agent.application.orchestration.durable_agent_coordinator import (
    DurableAgentCoordinator,
    MissingWorkRule,
)
from prievo_agent.application.memory.history_compactor import HistoryCompactor
from prievo_agent.domain.models import AgentCapability, AgentMemory


class _DurableOnlyGenerationMemory:
    """无 Redis adapter 时直接读取同 Run durable history 的应用层 Null Object。"""

    def load_with_fallback(self, run_id, scope, store, limit=None):
        if scope != "generation":
            raise ValueError("Generation workflow 只能读取 generation scope")
        memories = store.agent_memories_for_run(
            run_id, "generation", limit=max(1, min(6, int(limit or 6)))
        )
        return [
            {
                "memory_type": memory.memory_type,
                "scope": "generation",
                "content": memory.content,
                "metadata": {
                    "agent_memory_id": memory.id,
                    "subject": memory.subject,
                    "evidence_artifact_id": memory.evidence_artifact_id,
                },
                "created_at": memory.created_at.isoformat(),
            }
            for memory in memories
        ]

    def load_recent(self, run_id, limit=None, scope="generation"):
        return []

    def append(self, run_id, memory_type, content, scope="generation", **metadata):
        return False


class GenerationTaskHandler:
    name = HeuristicGenerationAgent.name
    capability = HeuristicGenerationAgent.capability

    def __init__(self, store, agent):
        self.store = store
        self.agent = agent

    def handle(self, task, board):
        if len(task.input_artifact_refs) != 1:
            raise ValueError("GenerationTask 必须且只能引用一个 GENERATION_REQUEST")
        artifact = board.artifact_by_ref(task.input_artifact_refs[0])
        if artifact.kind not in {"GENERATION_REQUEST", "GENERATION_RESUME_REQUEST"}:
            raise ValueError(
                "GenerationTask 输入必须是 GENERATION_REQUEST/GENERATION_RESUME_REQUEST"
            )
        payload = json.loads(
            self.store.artifact_content(artifact.id).decode("utf-8")
        )
        parents = [GenerationParent(**item) for item in payload["parents"]]
        request = GenerationRequest(
            run_id=task.run_id,
            task_contract=payload["task_contract"],
            original_prior_slice=payload["original_prior_slice"],
            original_prior_refs=payload["original_prior_refs"],
            strategy=payload["strategy"],
            parents=parents,
            relevant_run_memory=payload.get("relevant_run_memory", []),
            evidence=payload.get("evidence", []),
            context_refs=[artifact.id, *payload.get("context_refs", [])],
        )
        return self.agent.generate(request)


class GenerationArtifactWriter:
    def __init__(self, store):
        self.store = store

    def __call__(self, task, output):
        resumed = task.task_type == "HEURISTIC_GENERATION_RESUME"
        prompt = self.store.put_artifact(
            task.run_id,
            "GENERATION_RESUME_PROMPT" if resumed else "GENERATION_PROMPT",
            output.prompt.encode("utf-8"),
            "text/plain",
        )
        payload = output.to_dict()
        payload.pop("prompt", None)
        payload.update(
            {
                "input_artifact_refs": list(task.input_artifact_refs),
                "prompt_artifact_id": prompt.id,
                "agent_task_id": task.id,
                "result_type": type(output).__name__,
            }
        )
        if isinstance(output, CandidateDraft):
            kind = "RESUMED_CANDIDATE_DRAFT" if resumed else "CANDIDATE_DRAFT"
        else:
            kind = "RESUMED_KNOWLEDGE_GAP" if resumed else "KNOWLEDGE_GAP"
        artifact = self.store.put_artifact(
            task.run_id,
            kind,
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            "application/json",
        )
        return [prompt, artifact]


class DurableGenerationWorkflow:
    """构造、路由并恢复一个 CandidateDraft/KnowledgeGap。"""

    def __init__(
        self,
        store,
        skill_registry,
        model,
        literature_evidence_workflow=None,
        working_memory=None,
    ):
        self.store = store
        self.literature_evidence_workflow = literature_evidence_workflow
        # Redis/InMemory 只是近期缓存；即使没有缓存适配器，Null adapter 也会
        # 通过 load_with_fallback 从 SQLite/MySQL 的同 Run 事实记录回源。
        self.working_memory = (
            working_memory
            if working_memory is not None
            else _DurableOnlyGenerationMemory()
        )
        self.history_compactor = HistoryCompactor(
            store, skill_registry, model, self.working_memory
        )
        handler = GenerationTaskHandler(
            store, HeuristicGenerationAgent(skill_registry, model)
        )
        self.coordinator = DurableAgentCoordinator(store)
        self.dispatcher = AgentTaskDispatcher(
            store, AgentRegistry([handler]), GenerationArtifactWriter(store)
        )
        self.resume_coordinator = DurableAgentCoordinator(
            store,
            rules=(
                MissingWorkRule(
                    "GENERATION_RESUME_REQUEST",
                    "HEURISTIC_GENERATION_RESUME",
                    AgentCapability.HEURISTIC_GENERATION,
                    (
                        frozenset({"RESUMED_CANDIDATE_DRAFT"}),
                        frozenset({"RESUMED_KNOWLEDGE_GAP"}),
                    ),
                ),
            ),
        )

    def generate(
        self,
        run,
        task,
        prior,
        prior_refs,
        strategy,
        parents,
        generation,
        sequence,
        relevant_run_memory=(),
        evidence=(),
    ):
        parent_views = [self._parent_view(parent) for parent in parents]
        prior_slice = _prior_slice(prior)
        original_prior_digest = _json_digest(prior_slice)
        automatic_memory = self.history_compactor.context_records(
            run.id,
            run.dataset_id or task.dataset_id,
            "generation",
            recent_window=3,
        )
        automatic_memory = [
            item
            for item in automatic_memory
            if not _is_same_generation_request_memory(
                item, generation, sequence, strategy
            )
        ]
        merged_memory = _merge_run_local_memory(
            run.id, automatic_memory, relevant_run_memory, limit=6
        )
        request_payload = {
            "run_id": run.id,
            "generation": int(generation),
            "sequence": int(sequence),
            "strategy": strategy,
            "task_contract": {
                "dataset_id": task.dataset_id,
                "objective": task.objective,
                "generation": int(generation),
                "population_size": task.population_size,
                "candidate_budget": task.evaluation_budget,
                "random_seed": task.random_seed,
                "interface": "run_tuners(file, budget, seed, maxlives)",
                "evaluate_contract": "injected evaluate tracks unique mapped configurations",
            },
            "original_prior_slice": prior_slice,
            "original_prior_digest": original_prior_digest,
            "original_prior_refs": list(prior_refs),
            "parents": parent_views,
            "relevant_run_memory": merged_memory,
            "evidence": list(evidence),
            "context_refs": [],
        }
        existing_artifacts = {
            item.id for item in self.store.artifacts_for_run(run.id)
        }
        request = self.store.put_artifact(
            run.id,
            "GENERATION_REQUEST",
            json.dumps(
                request_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        if request.id not in existing_artifacts:
            self.store.append_event(
                run.id,
                "GENERATION_REQUESTED",
                "PriEvO Core 已确定 strategy 与 parents，等待生成 Agent",
                artifact_id=request.id,
                generation=generation,
                sequence=sequence,
                strategy=strategy,
                parent_ids=[parent.id for parent in parents],
                original_prior_digest=original_prior_digest,
            )
        output, output_ref = self._dispatch_request(
            run.id, request.id, self.coordinator, "HEURISTIC_GENERATION"
        )
        if not isinstance(output, KnowledgeGap):
            self._remember_candidate_draft(
                run, task, output, output_ref, generation, sequence, strategy
            )
            return output, request.id, output_ref
        if self.literature_evidence_workflow is None:
            # 兼容纯领域/旧 composition；产品 Engine 会注入 durable Research workflow。
            return output, request.id, output_ref

        resolution = self.literature_evidence_workflow.resolve(run.id, output_ref)
        if resolution.original_prior_digest != original_prior_digest:
            raise RuntimeError("Research annotation 改变了 Original Prior digest")
        resume_payload = json.loads(
            json.dumps(request_payload, ensure_ascii=False, sort_keys=True)
        )
        resume_payload.update(
            {
                "stable_generation_request_id": request.id,
                "original_generation_request_id": request.id,
                "knowledge_gap_artifact_id": output_ref,
                "research_task_id": resolution.research_task_id,
                "prior_explanation_artifact_id": (
                    resolution.prior_explanation_artifact_id
                ),
                "literature_evidence_artifact_id": (
                    resolution.literature_evidence_artifact_id
                ),
                "resume_attempt": 1,
                "resume_policy": (
                    "Research is bounded and complete. Generate a best-effort "
                    "CandidateDraft using immutable Original Prior plus supplied "
                    "annotation. Do not emit another KnowledgeGap merely because "
                    "literature status is EMPTY."
                ),
                "evidence": [
                    *resume_payload.get("evidence", []),
                    resolution.generation_evidence(),
                ],
                "context_refs": list(
                    dict.fromkeys(
                        [
                            request.id,
                            output_ref,
                            *resume_payload.get("context_refs", []),
                            *resolution.context_refs,
                        ]
                    )
                ),
            }
        )
        resume_payload["task_contract"]["research_resolution"] = {
            "status": resolution.evidence_status,
            "bounded_retry": 1,
            "policy": resume_payload["resume_policy"],
        }
        if _json_digest(resume_payload["original_prior_slice"]) != original_prior_digest:
            raise RuntimeError("Generation resume request 改变了 Original Prior")
        existing_artifacts = {
            item.id for item in self.store.artifacts_for_run(run.id)
        }
        resume = self.store.put_artifact(
            run.id,
            "GENERATION_RESUME_REQUEST",
            json.dumps(
                resume_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            "application/json",
        )
        if resume.id not in existing_artifacts:
            self.store.append_event(
                run.id,
                "GENERATION_RESUMED_AFTER_RESEARCH",
                "研究完成，使用同一 Generation identity 恢复生成",
                stable_generation_request_id=request.id,
                generation_resume_request_artifact_id=resume.id,
                knowledge_gap_artifact_id=output_ref,
                research_task_id=resolution.research_task_id,
                prior_explanation_artifact_id=resolution.prior_explanation_artifact_id,
                literature_evidence_artifact_id=resolution.literature_evidence_artifact_id,
                evidence_status=resolution.evidence_status,
                original_prior_digest=original_prior_digest,
            )
        resumed, resumed_ref = self._dispatch_request(
            run.id,
            resume.id,
            self.resume_coordinator,
            "HEURISTIC_GENERATION_RESUME",
        )
        if isinstance(resumed, KnowledgeGap):
            raise RuntimeError(
                "bounded LiteratureEvidence 解析后 GenerationAgent 仍返回 KnowledgeGap；"
                "不会创建无界 Research loop"
            )
        self._remember_candidate_draft(
            run, task, resumed, resumed_ref, generation, sequence, strategy
        )
        return resumed, request.id, resumed_ref

    def _remember_candidate_draft(
        self, run, task, draft, draft_artifact_id, generation, sequence, strategy
    ):
        """把成功 Draft 作为同 Run generation 记忆持久化并回温缓存。

        该步骤位于 durable task 消费之后，因此任务已完成但进程在 Candidate
        materialize 前退出时，重放同一请求仍会补齐记忆，且不会再次调用 LLM。
        """

        material = "{}|generation|{}".format(run.id, draft_artifact_id)
        memory_id = "agent-memory-{}".format(
            hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        )
        content = json.dumps(
            {
                "generation": int(generation),
                "sequence": int(sequence),
                "strategy": str(strategy),
                "description": draft.description,
                "operators": list(draft.operators),
                "generation_note": draft.generation_note,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        # 避免 SQLite INSERT OR REPLACE 在重放时改变 created_at；稳定 ID 加
        # 预查询使 durable 记录本身也保持幂等，而不只是“没有重复行”。
        existing = next(
            (
                item
                for item in self.store.agent_memories_for_run(
                    run.id, "generation", limit=1_000_000
                )
                if item.id == memory_id
            ),
            None,
        )
        memory = existing or AgentMemory(
            memory_id,
            run.id,
            run.dataset_id or task.dataset_id,
            "GENERATION_SUMMARY",
            "第 {} 代 {} / 序号 {} 候选草案".format(
                generation, strategy, sequence
            ),
            content,
            draft_artifact_id,
        )
        if existing is None:
            self.store.add_agent_memory(memory)

        cached = self.working_memory.load_recent(
            run.id, limit=6, scope="generation"
        )
        if not any(
            _memory_record_id(item) == memory.id for item in cached
        ):
            self.working_memory.append(
                run.id,
                memory.memory_type,
                memory.content,
                scope="generation",
                agent_memory_id=memory.id,
                subject=memory.subject,
                evidence_artifact_id=memory.evidence_artifact_id,
                generation=int(generation),
                sequence=int(sequence),
                strategy=str(strategy),
            )

    def record_evaluation(self, run, task, candidate, result):
        """评价成功后补写 fitness/trajectory 事实，供后代与 HistorySummary 使用。"""

        draft_ref = str(candidate.lineage.get("candidate_draft_artifact_id", ""))
        if not draft_ref:
            return None
        material = "{}|generation-evaluation|{}|{}".format(
            run.id, candidate.id, result.id
        )
        memory_id = "agent-memory-{}".format(
            hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        )
        existing = next(
            (
                item
                for item in self.store.agent_memories_for_run(
                    run.id, "generation", limit=1_000_000
                )
                if item.id == memory_id
            ),
            None,
        )
        content = json.dumps(
            {
                "candidate_id": candidate.id,
                "generation": int(candidate.lineage.get("generation", 0) or 0),
                "strategy": str(candidate.lineage.get("operator", "")),
                "parent_ids": list(
                    candidate.lineage.get(
                        "parent_ids", candidate.lineage.get("parents", [])
                    )
                ),
                "operators": list(candidate.operators),
                "fitness": float(result.objective),
                "used_budget": int(result.used_budget),
                "trajectory": list(result.trajectory),
                "candidate_draft_artifact_id": draft_ref,
                "evaluation_artifact_id": str(candidate.evaluation_artifact_id or ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        memory = existing or AgentMemory(
            memory_id,
            run.id,
            run.dataset_id or task.dataset_id,
            "GENERATION_EVALUATION",
            "Candidate {} 评价结果".format(candidate.id),
            content,
            str(candidate.evaluation_artifact_id or draft_ref),
        )
        if existing is None:
            self.store.add_agent_memory(memory)
            self.store.append_event(
                run.id,
                "GENERATION_MEMORY_UPDATED",
                "已将 Candidate fitness/trajectory 写入同 Run generation history",
                agent_memory_id=memory.id,
                candidate_id=candidate.id,
                evaluation_result_id=result.id,
                fitness=result.objective,
            )
        cached = self.working_memory.load_recent(
            run.id, limit=6, scope="generation"
        )
        if not any(_memory_record_id(item) == memory.id for item in cached):
            self.working_memory.append(
                run.id,
                memory.memory_type,
                memory.content,
                scope="generation",
                agent_memory_id=memory.id,
                subject=memory.subject,
                evidence_artifact_id=memory.evidence_artifact_id,
                candidate_id=candidate.id,
                fitness=float(result.objective),
            )
        return memory

    def _dispatch_request(self, run_id, request_id, coordinator, task_type):
        coordinator.reconcile(run_id)
        tasks = [
            item for item in self.store.agent_tasks_for_run(run_id)
            if item.task_type == task_type
            and request_id in item.input_artifact_refs
        ]
        if len(tasks) != 1:
            raise RuntimeError(
                "无法为 {} 唯一定位 {}".format(request_id, task_type)
            )
        dispatched = self.dispatcher.dispatch_with_retries(tasks[0].id)
        artifacts = {
            item.id: item for item in self.store.artifacts_for_run(run_id)
        }
        output_refs = [
            ref for ref in dispatched.artifact_refs
            if ref in artifacts
            and artifacts[ref].kind in {
                "CANDIDATE_DRAFT",
                "KNOWLEDGE_GAP",
                "RESUMED_CANDIDATE_DRAFT",
                "RESUMED_KNOWLEDGE_GAP",
            }
        ]
        if len(output_refs) != 1:
            raise RuntimeError("GenerationTask 未产生唯一 Draft/KnowledgeGap")
        payload = json.loads(
            self.store.artifact_content(output_refs[0]).decode("utf-8")
        )
        if payload["result_type"] == "KnowledgeGap":
            return KnowledgeGap(
                knowledge_gap=payload["knowledge_gap"],
                reason=payload["reason"],
                required_evidence=list(payload["required_evidence"]),
                **_audit_from_payload(payload, self.store, run_id),
            ), output_refs[0]
        if payload["result_type"] != "CandidateDraft":
            raise RuntimeError("未知 GenerationTask result_type")
        return CandidateDraft(
            code=payload["code"],
            description=payload["description"],
            operators=list(payload["operators"]),
            generation_note=payload.get("generation_note", ""),
            **_audit_from_payload(payload, self.store, run_id),
        ), output_refs[0]

    def _parent_view(self, parent):
        result = self.store.result_for_candidate(parent.id)
        return asdict(
            GenerationParent(
                candidate_id=parent.id,
                run_id=parent.run_id,
                code=parent.code,
                description=parent.description,
                fitness=float(parent.objective),
                trajectory=list(result.trajectory),
                operators=list(parent.operators),
                lineage=self._recent_parent_lineage(parent, max_steps=3),
                used_budget=result.used_budget,
            )
        )

    def _recent_parent_lineage(self, parent, max_steps=3):
        """从 durable Candidate/Result 还原最近 2~3 步，而不是只传一层 lineage。"""

        values = []
        current = parent
        visited = set()
        while current is not None and len(values) < max(1, int(max_steps)):
            if current.id in visited or current.run_id != parent.run_id:
                break
            visited.add(current.id)
            try:
                current_result = self.store.result_for_candidate(current.id)
                fitness = float(current_result.objective)
            except KeyError:
                fitness = current.objective
            values.append(
                {
                    "candidate_id": current.id,
                    "generation": current.lineage.get("generation", 0),
                    "creation_strategy": current.lineage.get(
                        "creation_strategy",
                        current.lineage.get("operator", current.lineage.get("creation_type", "")),
                    ),
                    "parent_ids": list(
                        current.lineage.get(
                            "parent_ids", current.lineage.get("parents", [])
                        )
                    ),
                    "fitness": fitness,
                    "operators": list(current.operators),
                    "candidate_draft_artifact_id": current.lineage.get(
                        "candidate_draft_artifact_id", ""
                    ),
                }
            )
            parent_ids = list(
                current.lineage.get("parent_ids", current.lineage.get("parents", []))
            )
            if not parent_ids:
                break
            try:
                ancestor = self.store.candidate_by_id(parent_ids[0])
            except KeyError:
                break
            current = ancestor if ancestor.run_id == parent.run_id else None
        # Context 按时间从旧到新展示 C3 -> C18 -> C29。
        return list(reversed(values))


def _prior_slice(prior):
    if prior is None:
        return {"source": "no-prior", "optimizers": []}
    return {
        "target_instance": prior.target.instance_name,
        "target_landscape_metrics": dict(prior.target.metrics),
        "evidence_version": prior.evidence_version,
        "selected_instances": list(prior.refinement.selected_instances),
        "optimizers": [
            {
                "name": item.name,
                "source_instance": item.source_instance,
                "rank": item.rank,
                "description": item.description,
                "code": item.code,
                "operators": [asdict(operator) for operator in item.operators],
            }
            for item in prior.optimizers[:10]
        ],
    }


def _audit_from_payload(payload, store, run_id):
    prompt = store.artifact_content(payload["prompt_artifact_id"]).decode("utf-8")
    return {
        "strategy": payload["strategy"],
        "skill_name": payload["skill_name"],
        "skill_version": payload["skill_version"],
        "skill_digest": payload["skill_digest"],
        "original_prior_refs": list(payload["original_prior_refs"]),
        "context_refs": list(payload["context_refs"]),
        "context_metadata": dict(payload["context_metadata"]),
        "prompt": prompt,
        "prompt_digest": payload["prompt_digest"],
    }


def _json_digest(value):
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _merge_run_local_memory(run_id, automatic, explicit, limit=6):
    """合并自动回源与调用方显式记忆，去重后保留最近的有界窗口。"""

    merged = []
    positions = {}
    for raw in [*automatic, *explicit]:
        if not isinstance(raw, Mapping):
            raise ValueError("Generation Memory 必须是结构化记录")
        item = dict(raw)
        record_run_id = item.get("run_id")
        if record_run_id is not None and str(record_run_id) != str(run_id):
            raise ValueError("Generation Memory 禁止跨 Run 注入")
        record_scope = str(item.get("scope", "generation")).strip().lower()
        if record_scope != "generation":
            raise ValueError("Generation workflow 只能读取 generation scope")
        item["run_id"] = str(run_id)
        item["scope"] = "generation"
        identity = _memory_record_identity(item)
        if identity in positions:
            merged[positions[identity]] = None
        positions[identity] = len(merged)
        merged.append(item)
    return [item for item in merged if item is not None][-max(1, int(limit)):]


def _memory_record_identity(item):
    record_id = _memory_record_id(item)
    if record_id:
        return "id:{}".format(record_id)
    evidence_id = item.get("evidence_artifact_id")
    metadata = item.get("metadata")
    if not evidence_id and isinstance(metadata, Mapping):
        evidence_id = metadata.get("evidence_artifact_id")
    if evidence_id:
        return "evidence:{}".format(evidence_id)
    # created_at 只描述缓存时间，不应让同一语义记录绕过去重。
    stable = {key: value for key, value in item.items() if key != "created_at"}
    return "payload:{}".format(_json_digest(stable))


def _memory_record_id(item):
    if not isinstance(item, Mapping):
        return ""
    direct = item.get("agent_memory_id")
    metadata = item.get("metadata")
    nested = metadata.get("agent_memory_id") if isinstance(metadata, Mapping) else ""
    return str(direct or nested or "")


def _is_same_generation_request_memory(item, generation, sequence, strategy):
    """排除当前请求自己写出的 Draft，保持重放请求内容寻址稳定。"""

    if not isinstance(item, Mapping):
        return False
    metadata = item.get("metadata")
    values = dict(metadata) if isinstance(metadata, Mapping) else {}
    content = item.get("content")
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, Mapping):
            values = {**decoded, **values}
    try:
        return (
            int(values.get("generation")) == int(generation)
            and int(values.get("sequence")) == int(sequence)
            and str(values.get("strategy")) == str(strategy)
        )
    except (TypeError, ValueError):
        return False
