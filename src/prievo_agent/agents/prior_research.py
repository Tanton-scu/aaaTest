"""由 KnowledgeGap 触发的纯领域 PriorResearchAgent。

该模块不依赖 Store、具体 LLM SDK、具体 Retriever 或应用层 Tool Gateway。
这些不可控边界均通过局部 Protocol 注入。Agent 只创建可持久化的结构化输出，
且始终保留 Original Prior，不用研究结果覆盖它。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from prievo_agent.agents.context_policies import ContextPolicyFramework


@dataclass(frozen=True)
class KnowledgeGap:
    ref: str
    summary: str
    reason: str
    original_prior_ref: str

    def __post_init__(self):
        if not self.ref.strip():
            raise ValueError("KnowledgeGap ref 不能为空")
        if not self.summary.strip():
            raise ValueError("KnowledgeGap summary 不能为空")
        if not self.original_prior_ref.strip():
            raise ValueError("KnowledgeGap 必须引用 Original Prior")


@dataclass(frozen=True)
class ResearchSkill:
    name: str
    digest: str
    instructions: str
    ref: str = ""
    version: str = "1"

    def __post_init__(self):
        if self.name not in {"prior_explanation", "literature_evidence_review"}:
            raise ValueError(
                "PriorResearchAgent 只允许 prior_explanation 或 "
                "literature_evidence_review Skill"
            )
        if not self.digest.strip() or not self.instructions.strip():
            raise ValueError("Skill digest 和 instructions 不能为空")
        if not str(self.version).strip():
            raise ValueError("Skill version 不能为空")

    @property
    def effective_ref(self):
        return self.ref or "skill:{}:{}".format(self.name, self.digest[:16])


@dataclass(frozen=True)
class LiteratureSearchQuery:
    text: str
    query_ref: str
    knowledge_gap_ref: str
    original_prior_ref: str
    context_ref: str
    skill_ref: str
    max_results: int = 5

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Literature query 不能为空")
        if not 1 <= self.max_results <= 10:
            raise ValueError("Literature query max_results 必须在 1~10")


@dataclass(frozen=True)
class RetrievedLiteratureChunk:
    paper_ref: str
    title: str
    section: str
    chunk_ref: str
    content: str
    score: float
    source_metadata: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecution:
    tool_call_ref: str
    result: Sequence
    status: str = "COMPLETED"


class PriorResearchModel(Protocol):
    def formulate_query(self, prompt: str) -> str: ...

    def explain_prior(self, prompt: str) -> Mapping[str, Any]: ...


class LiteratureRetriever(Protocol):
    def search(self, query: LiteratureSearchQuery) -> Sequence[RetrievedLiteratureChunk]: ...


class ToolGateway(Protocol):
    def execute(
        self,
        run_id: str,
        caller: str,
        tool_name: str,
        reason: str,
        operation: Callable[[], Sequence],
        tool_budget_scope: str = "",
    ) -> ToolExecution: ...


@dataclass(frozen=True)
class LiteratureEvidenceItem:
    evidence_ref: str
    paper_ref: str
    title: str
    section: str
    chunk_ref: str
    content: str
    score: float
    query_ref: str
    tool_call_ref: str
    provenance: tuple

    def to_dict(self):
        values = asdict(self)
        values["provenance"] = dict(self.provenance)
        return values


@dataclass(frozen=True)
class LiteratureEvidence:
    """一次受治理检索的完整证据包；允许显式 EMPTY。"""

    ref: str
    status: str
    query: str
    query_ref: str
    tool_call_ref: str
    original_prior_ref: str
    context_ref: str
    skill_ref: str
    items: tuple
    message: str
    skill_version: str = ""
    skill_digest: str = ""

    def to_dict(self):
        return {
            "ref": self.ref,
            "status": self.status,
            "query": self.query,
            "query_ref": self.query_ref,
            "tool_call_ref": self.tool_call_ref,
            "original_prior_ref": self.original_prior_ref,
            "context_ref": self.context_ref,
            "skill_ref": self.skill_ref,
            "skill_version": self.skill_version,
            "skill_digest": self.skill_digest,
            "items": [item.to_dict() for item in self.items],
            "message": self.message,
        }


@dataclass(frozen=True)
class PriorExplanation:
    ref: str
    knowledge_gap_ref: str
    original_prior_ref: str
    summary: str
    relationship_to_landscape: str
    strategy_implications: tuple
    evidence_refs: tuple
    evidence_bundle_ref: str
    skill_ref: str
    skill_digest: str
    context_ref: str
    prompt_ref: str
    grounded: bool
    skill_version: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PriorResearchResult:
    run_id: str
    knowledge_gap: KnowledgeGap
    original_prior_snapshot: str
    explanation: PriorExplanation
    literature_evidence: LiteratureEvidence
    skill_ref: str
    initial_context_ref: str
    evidence_context_ref: str
    query_prompt_ref: str
    explanation_prompt_ref: str
    context_metadata: Mapping
    query_prompt: str
    explanation_prompt: str
    skill_provenance: tuple = ()

    @property
    def original_prior_slice(self):
        return json.loads(self.original_prior_snapshot)

    def resume_context(self):
        """供 Generation resume 持久化/恢复的最小结构化上下文。"""

        return {
            "knowledge_gap_ref": self.knowledge_gap.ref,
            "original_prior_ref": self.knowledge_gap.original_prior_ref,
            "original_prior_slice": self.original_prior_slice,
            "prior_explanation": self.explanation.to_dict(),
            "literature_evidence": self.literature_evidence.to_dict(),
            "skill_ref": self.skill_ref,
            "skill_provenance": [dict(item) for item in self.skill_provenance],
            "context_refs": [self.initial_context_ref, self.evidence_context_ref],
            "prompt_refs": [self.query_prompt_ref, self.explanation_prompt_ref],
        }


class PriorResearchAgent:
    name = "PriorResearchAgent"

    def __init__(
        self,
        model: PriorResearchModel,
        retriever: LiteratureRetriever,
        tool_gateway: ToolGateway,
        context_framework=None,
        max_results=5,
    ):
        self.model = model
        self.retriever = retriever
        self.tool_gateway = tool_gateway
        self.context_framework = context_framework or ContextPolicyFramework()
        self.max_results = min(10, max(1, int(max_results)))

    def research(
        self,
        run_id,
        knowledge_gap,
        original_prior_slice,
        landscape_summary,
        strategy,
        skill,
        parent_summary=None,
        research_history=(),
        extra_context=None,
        evidence_review_skill=None,
    ):
        """执行一次有界研究，并返回不覆盖 Original Prior 的结构化结果。"""

        if not str(run_id).strip():
            raise ValueError("run_id 不能为空")
        if not isinstance(knowledge_gap, KnowledgeGap):
            raise TypeError("knowledge_gap 必须是 KnowledgeGap")
        if not isinstance(skill, ResearchSkill):
            raise TypeError("skill 必须是 ResearchSkill")
        _validate_research_skill(skill, "prior_explanation")
        if evidence_review_skill is None:
            # 兼容旧的单 Skill 调用；产品工作流始终显式注入审阅 Skill。
            evidence_review_skill = skill
        elif not isinstance(evidence_review_skill, ResearchSkill):
            raise TypeError("evidence_review_skill 必须是 ResearchSkill")
        else:
            _validate_research_skill(
                evidence_review_skill, "literature_evidence_review"
            )
        skill_provenance = (
            _skill_provenance("query_and_prior_explanation", skill),
            _skill_provenance("literature_evidence_review", evidence_review_skill),
        )

        # JSON snapshot 同时验证输入是可持久化的结构化数据。后续只使用副本，
        # 不会向调用方传入的 dict/list 写入 Agent annotation。
        prior_snapshot = _canonical_json(original_prior_slice)
        prior_copy = json.loads(prior_snapshot)
        base_payload = {
            "knowledge_gap": asdict(knowledge_gap),
            "prior_slice": prior_copy,
            "landscape_summary": _json_copy(landscape_summary),
            "strategy": _json_copy(strategy),
            "parent_summary": _json_copy(parent_summary),
            "skill": _skill_context(skill),
            "research_history": _json_copy(list(research_history)),
            "output_schema": {
                "outputs": ["PriorExplanation", "LiteratureEvidence"],
                "original_prior_policy": "immutable",
            },
        }
        if extra_context:
            base_payload.update(dict(extra_context))

        initial_context = self.context_framework.build("research", base_payload)
        initial_context_ref = _content_ref("context", initial_context.text)
        query_prompt = _query_prompt(
            skill, initial_context.text, initial_context_ref,
            knowledge_gap.original_prior_ref,
        )
        query_prompt_ref = _content_ref("prompt", query_prompt)
        query_text = str(self.model.formulate_query(query_prompt)).strip()
        if not query_text:
            # 不在空 query 时偷偷改用整个 Prior；使用 KnowledgeGap 的有界文本。
            query_text = knowledge_gap.summary.strip()
        query_ref = _content_ref("literature-query", query_text)
        query = LiteratureSearchQuery(
            text=query_text,
            query_ref=query_ref,
            knowledge_gap_ref=knowledge_gap.ref,
            original_prior_ref=knowledge_gap.original_prior_ref,
            context_ref=initial_context_ref,
            skill_ref=skill.effective_ref,
            max_results=self.max_results,
        )
        execute_kwargs = {
            "run_id": str(run_id),
            "caller": self.name,
            "tool_name": "literature_search",
            "reason": "Resolve KnowledgeGap {} without replacing Original Prior".format(
                knowledge_gap.ref
            ),
            "operation": lambda: self.retriever.search(query),
        }
        if _supports_tool_budget_scope(self.tool_gateway):
            execute_kwargs["tool_budget_scope"] = knowledge_gap.ref
        execution = self.tool_gateway.execute(**execute_kwargs)
        if not isinstance(execution, ToolExecution):
            raise TypeError("ToolGateway 必须返回 ToolExecution")
        chunks = tuple(_chunk(item) for item in execution.result)[: self.max_results]

        evidence_context_items = [
            {
                "paper_ref": item.paper_ref,
                "section": item.section,
                "chunk_ref": item.chunk_ref,
                "score": float(item.score),
                "content": item.content,
                "query_ref": query_ref,
                "tool_call_ref": execution.tool_call_ref,
            }
            for item in chunks
        ]
        if not evidence_context_items:
            evidence_context_items = [{
                "status": "EMPTY",
                "query_ref": query_ref,
                "tool_call_ref": execution.tool_call_ref,
                "message": "检索未返回可追溯的文献证据；禁止编造来源。",
            }]
        evidence_payload = dict(base_payload)
        evidence_payload["skill"] = {
            "prior_explanation": _skill_context(skill),
            "literature_evidence_review": _skill_context(
                evidence_review_skill
            ),
        }
        evidence_payload["evidence"] = evidence_context_items
        evidence_context = self.context_framework.build("research", evidence_payload)
        evidence_context_ref = _content_ref("context", evidence_context.text)

        evidence_items = tuple(
            LiteratureEvidenceItem(
                evidence_ref=_content_ref(
                    "literature-evidence",
                    "{}|{}|{}|{}".format(
                        item.paper_ref, item.section, item.chunk_ref, query_ref
                    ),
                ),
                paper_ref=item.paper_ref,
                title=item.title,
                section=item.section,
                chunk_ref=item.chunk_ref,
                content=item.content,
                score=float(item.score),
                query_ref=query_ref,
                tool_call_ref=execution.tool_call_ref,
                provenance=_provenance(item.source_metadata),
            )
            for item in chunks
        )
        evidence_status = "FOUND" if evidence_items else "EMPTY"
        evidence_message = (
            "检索到 {} 条可追溯文献证据。".format(len(evidence_items))
            if evidence_items
            else "检索结果为空；保留 Original Prior，不添加无来源的算法结论。"
        )
        evidence_ref = _content_ref(
            "literature-evidence-bundle",
            "{}|{}|{}".format(query_ref, execution.tool_call_ref, evidence_status),
        )
        literature_evidence = LiteratureEvidence(
            ref=evidence_ref,
            status=evidence_status,
            query=query_text,
            query_ref=query_ref,
            tool_call_ref=execution.tool_call_ref,
            original_prior_ref=knowledge_gap.original_prior_ref,
            context_ref=evidence_context_ref,
            skill_ref=evidence_review_skill.effective_ref,
            items=evidence_items,
            message=evidence_message,
            skill_version=str(evidence_review_skill.version),
            skill_digest=evidence_review_skill.digest,
        )

        explanation_prompt = _explanation_prompt(
            evidence_review_skill, evidence_context.text, evidence_context_ref,
            knowledge_gap.original_prior_ref, literature_evidence,
        )
        explanation_prompt_ref = _content_ref("prompt", explanation_prompt)
        if evidence_items:
            model_output = self.model.explain_prior(explanation_prompt)
            explanation_values = _explanation_values(model_output)
            grounded = True
        else:
            # 没有来源时不请求模型“解释”算法知识，避免生成看似合理的伪证据。
            explanation_values = {
                "summary": "当前检索没有找到可追溯证据，Original Prior 保持不变。",
                "relationship_to_landscape": "证据不足，无法新增 landscape 关系判断。",
                "strategy_implications": (),
            }
            grounded = False
        explanation_ref = _content_ref(
            "prior-explanation",
            "{}|{}|{}".format(
                knowledge_gap.ref, evidence_ref, explanation_prompt_ref
            ),
        )
        explanation = PriorExplanation(
            ref=explanation_ref,
            knowledge_gap_ref=knowledge_gap.ref,
            original_prior_ref=knowledge_gap.original_prior_ref,
            summary=explanation_values["summary"],
            relationship_to_landscape=explanation_values[
                "relationship_to_landscape"
            ],
            strategy_implications=explanation_values["strategy_implications"],
            evidence_refs=tuple(item.evidence_ref for item in evidence_items),
            evidence_bundle_ref=evidence_ref,
            skill_ref=evidence_review_skill.effective_ref,
            skill_digest=evidence_review_skill.digest,
            context_ref=evidence_context_ref,
            prompt_ref=explanation_prompt_ref,
            grounded=grounded,
            skill_version=str(evidence_review_skill.version),
        )

        # 两阶段 context metadata 均可持久化并用于泄漏/恢复断言；这里只记录
        # included/excluded key 等 metadata，不复制未授权字段的值。
        context_metadata = MappingProxyType({
            "initial": initial_context.metadata(),
            "with_evidence": evidence_context.metadata(),
        })
        return PriorResearchResult(
            run_id=str(run_id),
            knowledge_gap=knowledge_gap,
            original_prior_snapshot=prior_snapshot,
            explanation=explanation,
            literature_evidence=literature_evidence,
            skill_ref=skill.effective_ref,
            initial_context_ref=initial_context_ref,
            evidence_context_ref=evidence_context_ref,
            query_prompt_ref=query_prompt_ref,
            explanation_prompt_ref=explanation_prompt_ref,
            context_metadata=context_metadata,
            query_prompt=query_prompt,
            explanation_prompt=explanation_prompt,
            skill_provenance=skill_provenance,
        )


def _validate_research_skill(skill, expected_name):
    if skill.name != expected_name:
        raise ValueError(
            "PriorResearchAgent {} 阶段必须使用 {} Skill，收到 {}".format(
                expected_name, expected_name, skill.name
            )
        )


def _skill_context(skill):
    return {
        "name": skill.name,
        "version": str(skill.version),
        "digest": skill.digest,
        "ref": skill.effective_ref,
        "instructions": skill.instructions,
    }


def _skill_provenance(role, skill):
    return {
        "role": role,
        "name": skill.name,
        "version": str(skill.version),
        "digest": skill.digest,
        "ref": skill.effective_ref,
    }


def _query_prompt(skill, context, context_ref, original_prior_ref):
    return (
        "You are PriorResearchAgent. Form one bounded literature query for the "
        "declared KnowledgeGap. Never replace or rewrite Original Prior. Return "
        "only the query text.\n\n"
        "Skill ref: {skill_ref}\nSkill version: {version}\n"
        "Skill digest: {digest}\n"
        "Context ref: {context_ref}\nOriginal prior ref: {prior_ref}\n\n"
        "Skill instructions:\n{instructions}\n\nResearch context:\n{context}"
    ).format(
        skill_ref=skill.effective_ref,
        version=skill.version,
        digest=skill.digest,
        context_ref=context_ref,
        prior_ref=original_prior_ref,
        instructions=skill.instructions,
        context=context,
    )


def _explanation_prompt(
    skill, context, context_ref, original_prior_ref, literature_evidence
):
    return (
        "You are PriorResearchAgent. Explain the immutable Original Prior only "
        "to the extent supported by the supplied LiteratureEvidence. Return a "
        "mapping with summary, relationship_to_landscape, strategy_implications. "
        "Do not invent fitness or overwrite the Prior.\n\n"
        "Skill ref: {skill_ref}\nSkill version: {version}\n"
        "Skill digest: {digest}\n"
        "Context ref: {context_ref}\nOriginal prior ref: {prior_ref}\n"
        "Evidence bundle ref: {evidence_ref}\n\n"
        "Skill instructions:\n{instructions}\n\nGrounded context:\n{context}"
    ).format(
        skill_ref=skill.effective_ref,
        version=skill.version,
        digest=skill.digest,
        context_ref=context_ref,
        prior_ref=original_prior_ref,
        evidence_ref=literature_evidence.ref,
        instructions=skill.instructions,
        context=context,
    )


def _explanation_values(value):
    if not isinstance(value, Mapping):
        raise TypeError("Prior explanation 模型输出必须是 Mapping")
    summary = str(value.get("summary", "")).strip()
    relationship = str(value.get("relationship_to_landscape", "")).strip()
    implications = value.get("strategy_implications", ())
    if not summary or not relationship:
        raise ValueError("Prior explanation 缺少 summary/relationship_to_landscape")
    if isinstance(implications, str):
        implications = (implications,)
    elif isinstance(implications, Sequence):
        implications = tuple(str(item) for item in implications)
    else:
        raise TypeError("strategy_implications 必须是字符串序列")
    return {
        "summary": summary,
        "relationship_to_landscape": relationship,
        "strategy_implications": implications,
    }


def _chunk(value):
    if isinstance(value, RetrievedLiteratureChunk):
        return value
    if isinstance(value, Mapping):
        return RetrievedLiteratureChunk(
            paper_ref=str(value["paper_ref"]),
            title=str(value.get("title", "")),
            section=str(value["section"]),
            chunk_ref=str(value["chunk_ref"]),
            content=str(value["content"]),
            score=float(value["score"]),
            source_metadata=dict(value.get("source_metadata", {})),
        )
    raise TypeError("Retriever result 必须是 RetrievedLiteratureChunk 或 Mapping")


def _provenance(metadata):
    if not isinstance(metadata, Mapping):
        raise TypeError("Literature provenance 必须是 Mapping")
    return tuple(
        (str(key), _stable_metadata_value(value))
        for key, value in sorted(metadata.items(), key=lambda item: str(item[0]))
    )


def _stable_metadata_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)


def _canonical_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("Original Prior slice 必须是可持久化的结构化 JSON") from exc


def _json_copy(value):
    if value is None:
        return None
    return json.loads(_canonical_json(value))


def _content_ref(kind, content):
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
    return "{}:{}".format(kind, digest[:24])


def _supports_tool_budget_scope(gateway) -> bool:
    try:
        return "tool_budget_scope" in inspect.signature(gateway.execute).parameters
    except (TypeError, ValueError):
        return False
