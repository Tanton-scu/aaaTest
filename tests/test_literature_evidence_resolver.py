import copy
import unittest

from prievo_agent.agents.common.context import AgentContextBuilder
from prievo_agent.agents.common.context_policies import ContextPolicyFramework
from prievo_agent.application.literature.evidence_resolver import (
    KnowledgeGap,
    LiteratureEvidenceResolver,
    ResearchSkill,
    RetrievedLiteratureChunk,
    ToolExecution,
)


class RecordingModel:
    def __init__(self):
        self.query_prompts = []
        self.explanation_prompts = []

    def formulate_query(self, prompt):
        self.query_prompts.append(prompt)
        return "decision tree surrogate rugged fitness landscape evolutionary HPO"

    def explain_prior(self, prompt):
        self.explanation_prompts.append(prompt)
        return {
            "summary": "Tree surrogate can model non-linear response surfaces.",
            "relationship_to_landscape": "The evidence is relevant to a rugged landscape.",
            "strategy_implications": [
                "Preserve the original surrogate prior.",
                "Use the evidence only as an annotation for imitation.",
            ],
        }


class FixedRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return list(self.results)


class RecordingGateway:
    def __init__(self):
        self.calls = []

    def execute(self, run_id, caller, tool_name, reason, operation):
        self.calls.append({
            "run_id": run_id,
            "caller": caller,
            "tool_name": tool_name,
            "reason": reason,
        })
        return ToolExecution("tool-call:literature:001", operation())


def skill():
    return ResearchSkill(
        name="prior_explanation",
        digest="a" * 64,
        ref="skill:prior_explanation:v1",
        instructions=(
            "Resolve the bounded gap with traceable primary evidence. "
            "Never overwrite Original Prior and never invent a citation."
        ),
    )


def evidence_review_skill():
    return ResearchSkill(
        name="literature_evidence_review",
        version="7",
        digest="b" * 64,
        ref="skill:literature_evidence_review@7#{}".format("b" * 64),
        instructions=(
            "Review only traceable evidence IDs and distinguish supported from "
            "unsupported claims. REVIEW_SKILL_MARKER_7719"
        ),
    )


def gap():
    return KnowledgeGap(
        ref="knowledge-gap:run-86:1",
        summary="Decision Tree Surrogate 与当前 rugged landscape 的关系不清楚",
        reason="GenerationAgent 无法理解该 prior slice",
        original_prior_ref="prior:run-86:instance-12",
    )


def agent(model, retriever, gateway):
    return LiteratureEvidenceResolver(
        model,
        retriever,
        gateway,
        context_framework=ContextPolicyFramework(
            AgentContextBuilder(max_chars=16000)
        ),
        max_results=3,
    )


class LiteratureEvidenceResolverTests(unittest.TestCase):
    def test_knowledge_gap_retrieval_produces_traceable_recoverable_outputs(self):
        model = RecordingModel()
        retriever = FixedRetriever([
            RetrievedLiteratureChunk(
                paper_ref="paper:smith-2024",
                title="Surrogate Models for HPO",
                section="3.2 Decision-tree surrogates",
                chunk_ref="paper:smith-2024:section-3.2:chunk-4",
                content="Tree ensembles capture non-linear interactions.",
                score=0.91,
                source_metadata={
                    "identifier": "doi:10.1000/example",
                    "source_path": "data/literature/smith-2024.pdf",
                    "page": 7,
                },
            ),
        ])
        gateway = RecordingGateway()
        original_prior = {
            "optimizer_ref": "optimizer:instance-12:rank-1",
            "description": "Decision Tree Surrogate",
            "operators": ["fit_tree", "suggest_by_ei"],
        }

        result = agent(model, retriever, gateway).research(
            run_id="run-86",
            knowledge_gap=gap(),
            original_prior_slice=original_prior,
            landscape_summary={"NBC": 0.84, "summary": "rugged"},
            strategy="imitate",
            parent_summary={"candidate_ref": "candidate:c9", "operators": ["TPE"]},
            skill=skill(),
            evidence_review_skill=evidence_review_skill(),
            research_history=[{"query_ref": "literature-query:old"}],
        )

        self.assertEqual(1, len(gateway.calls))
        self.assertEqual("LiteratureEvidenceResolver", gateway.calls[0]["caller"])
        self.assertEqual("literature_search", gateway.calls[0]["tool_name"])
        self.assertEqual(1, len(retriever.queries))
        query = retriever.queries[0]
        self.assertEqual(gap().ref, query.knowledge_gap_ref)
        self.assertEqual(gap().original_prior_ref, query.original_prior_ref)
        self.assertTrue(query.context_ref.startswith("context:"))
        self.assertEqual("skill:prior_explanation:v1", query.skill_ref)

        evidence = result.literature_evidence
        self.assertEqual("FOUND", evidence.status)
        self.assertEqual("tool-call:literature:001", evidence.tool_call_ref)
        self.assertEqual(query.query_ref, evidence.query_ref)
        self.assertEqual(1, len(evidence.items))
        item = evidence.items[0]
        self.assertEqual("paper:smith-2024", item.paper_ref)
        self.assertEqual("3.2 Decision-tree surrogates", item.section)
        self.assertEqual("paper:smith-2024:section-3.2:chunk-4", item.chunk_ref)
        self.assertEqual(0.91, item.score)
        self.assertEqual(query.query_ref, item.query_ref)
        self.assertEqual("tool-call:literature:001", item.tool_call_ref)
        self.assertEqual(
            "doi:10.1000/example", dict(item.provenance)["identifier"]
        )

        explanation = result.explanation
        self.assertTrue(explanation.grounded)
        self.assertEqual(gap().original_prior_ref, explanation.original_prior_ref)
        self.assertEqual((item.evidence_ref,), explanation.evidence_refs)
        self.assertEqual(
            evidence_review_skill().effective_ref, evidence.skill_ref
        )
        self.assertEqual("7", evidence.skill_version)
        self.assertEqual("b" * 64, evidence.skill_digest)
        self.assertEqual(evidence_review_skill().effective_ref, explanation.skill_ref)
        self.assertEqual("7", explanation.skill_version)
        self.assertEqual("b" * 64, explanation.skill_digest)
        self.assertEqual(result.evidence_context_ref, explanation.context_ref)
        self.assertEqual(result.explanation_prompt_ref, explanation.prompt_ref)
        self.assertIn(skill().effective_ref, model.query_prompts[0])
        self.assertNotIn("REVIEW_SKILL_MARKER_7719", model.query_prompts[0])
        self.assertIn(result.initial_context_ref, model.query_prompts[0])
        self.assertIn(item.chunk_ref, model.explanation_prompts[0])
        self.assertIn(
            evidence_review_skill().effective_ref, model.explanation_prompts[0]
        )
        self.assertIn("Skill version: 7", model.explanation_prompts[0])
        self.assertIn("REVIEW_SKILL_MARKER_7719", model.explanation_prompts[0])
        self.assertIn(result.evidence_context_ref, model.explanation_prompts[0])
        provenance = {
            item["role"]: item for item in result.skill_provenance
        }
        self.assertEqual(
            "prior_explanation",
            provenance["query_and_prior_explanation"]["name"],
        )
        self.assertEqual(
            {
                "name": "literature_evidence_review",
                "version": "7",
                "digest": "b" * 64,
                "ref": evidence_review_skill().effective_ref,
            },
            {
                key: provenance["literature_evidence_review"][key]
                for key in ("name", "version", "digest", "ref")
            },
        )

        recovered = result.resume_context()
        self.assertEqual(original_prior, recovered["original_prior_slice"])
        self.assertEqual(gap().original_prior_ref, recovered["original_prior_ref"])
        self.assertEqual("FOUND", recovered["literature_evidence"]["status"])
        self.assertEqual(
            [result.initial_context_ref, result.evidence_context_ref],
            recovered["context_refs"],
        )
        self.assertEqual(
            [result.query_prompt_ref, result.explanation_prompt_ref],
            recovered["prompt_refs"],
        )
        self.assertEqual(2, len(recovered["skill_provenance"]))

    def test_empty_retrieval_is_explicit_and_does_not_ask_model_to_invent(self):
        model = RecordingModel()
        retriever = FixedRetriever([])
        gateway = RecordingGateway()

        result = agent(model, retriever, gateway).research(
            "run-86",
            gap(),
            {"description": "Decision Tree Surrogate"},
            {"summary": "rugged"},
            "imitate",
            skill(),
            research_history=(),
        )

        self.assertEqual("EMPTY", result.literature_evidence.status)
        self.assertEqual((), result.literature_evidence.items)
        self.assertIn("检索结果为空", result.literature_evidence.message)
        self.assertFalse(result.explanation.grounded)
        self.assertEqual((), result.explanation.evidence_refs)
        self.assertIn("Original Prior 保持不变", result.explanation.summary)
        self.assertEqual([], model.explanation_prompts)
        self.assertIn("禁止编造来源", result.explanation_prompt)
        self.assertEqual(1, len(gateway.calls))

    def test_original_prior_is_immutable_and_annotation_is_separate(self):
        model = RecordingModel()
        retriever = FixedRetriever([
            RetrievedLiteratureChunk(
                "paper:p1", "Paper", "methods", "paper:p1:methods:1",
                "Evidence", 0.8, {"identifier": "p1"},
            )
        ])
        gateway = RecordingGateway()
        original_prior = {
            "description": "Decision Tree Surrogate",
            "operators": [
                {"id": "op-tree", "parameters": {"depth": 6}},
            ],
        }
        before = copy.deepcopy(original_prior)

        result = agent(model, retriever, gateway).research(
            "run-86", gap(), original_prior, {"summary": "rugged"},
            "imitate", skill(),
        )

        self.assertEqual(before, original_prior)
        self.assertEqual(before, result.original_prior_slice)
        self.assertEqual(
            before, result.resume_context()["original_prior_slice"]
        )
        self.assertNotIn("prior_explanation", original_prior)
        self.assertEqual(
            gap().original_prior_ref,
            result.resume_context()["prior_explanation"]["original_prior_ref"],
        )

    def test_generation_and_repair_memory_cannot_leak_into_research_prompt(self):
        model = RecordingModel()
        retriever = FixedRetriever([])
        gateway = RecordingGateway()
        generation_secret = "GENERATION_MEMORY_SECRET_9281"
        repair_secret = "REPAIR_MEMORY_SECRET_7720"

        result = agent(model, retriever, gateway).research(
            "run-86",
            gap(),
            {"description": "Decision Tree Surrogate"},
            {"summary": "rugged"},
            "imitate",
            skill(),
            research_history=[{"query": "bounded same-run research history"}],
            extra_context={
                "generation_memory": generation_secret,
                "repair_history": repair_secret,
                "population": "POPULATION_SECRET_1180",
            },
        )

        all_prompts = "\n".join([
            *model.query_prompts,
            *model.explanation_prompts,
            result.query_prompt,
            result.explanation_prompt,
        ])
        self.assertIn("bounded same-run research history", all_prompts)
        self.assertNotIn(generation_secret, all_prompts)
        self.assertNotIn(repair_secret, all_prompts)
        self.assertNotIn("POPULATION_SECRET_1180", all_prompts)
        excluded = set(
            result.context_metadata["initial"]["excluded_keys"]
        )
        self.assertEqual(
            {"generation_memory", "population", "repair_history"}, excluded
        )
        # metadata 只记录被排除的 key，绝不能保存被排除的 secret value。
        self.assertNotIn(
            generation_secret, str(dict(result.context_metadata))
        )
        self.assertNotIn(repair_secret, str(dict(result.context_metadata)))


if __name__ == "__main__":
    unittest.main()
