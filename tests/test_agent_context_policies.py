import unittest

from prievo_agent.agents.context import AgentContextBuilder
from prievo_agent.agents.context_policies import (
    ContextPolicyFramework,
    build,
)


class ContextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.framework = ContextPolicyFramework(AgentContextBuilder(max_chars=12000))

    def test_similarity_uses_only_fla_top5_and_skill(self):
        result = self.framework.build("SimilarityAgent", {
            "target_landscape": {"profile_ref": "landscape-1"},
            "fla_metrics": {"ela_meta.lin_simple.adj_r2": 0.81},
            "metric_semantics": {"ela_meta.lin_simple.adj_r2": "linearity"},
            "top5_candidates": [{"id": "h{}".format(i)} for i in range(1, 7)],
            "numeric_distance_ranking": ["h1", "h2", "h3", "h4", "h5"],
            "skill": {"name": "semantic_similarity_selection", "digest": "s1"},
            "memory": "SIMILARITY_MEMORY_MUST_NOT_LEAK",
            "rag": "SIMILARITY_RAG_MUST_NOT_LEAK",
            "population": "SIMILARITY_POPULATION_MUST_NOT_LEAK",
        })

        self.assertIn("semantic_similarity_selection", result.text)
        self.assertIn("h5", result.text)
        self.assertNotIn("h6", result.text)
        self.assertNotIn("MUST_NOT_LEAK", result.text)
        self.assertEqual(
            {"memory", "population", "rag"}, set(result.excluded_keys)
        )
        self.assertEqual(1, result.omitted_items["top5_candidates"])
        self.assertEqual((), result.recent)
        self.assertEqual((), result.evidence)

    def test_generation_keeps_three_lineage_steps_and_relevant_run_context(self):
        result = build("generation", {
            "task": {"dataset": "covtype", "objective": "minimize"},
            "prior": {"ref": "prior-1", "operators": ["surrogate"]},
            "strategy_skill": {"name": "recombine", "digest": "skill-1"},
            "parents": [{"id": "c18"}, {"id": "c27"}],
            "lineage": [
                {"id": "old-1"}, {"id": "old-2"},
                {"id": "recent-1"}, {"id": "recent-2"}, {"id": "recent-3"},
            ],
            "relevant_run_memory": [{"ref": "memory-run-86"}],
            "evidence": [{"chunk_ref": "paper-1:chunk-4"}],
            "output_schema": {"oneOf": ["CandidateDraft", "KnowledgeGap"]},
            "population": "WHOLE_POPULATION_MUST_NOT_LEAK",
            "repair_history": "REPAIR_HISTORY_MUST_NOT_LEAK",
            "other_run_memory": "RUN_87_MEMORY_MUST_NOT_LEAK",
        }, builder=AgentContextBuilder(max_chars=12000))

        for expected in [
            "prior-1", "recombine", "c18", "recent-1", "recent-2",
            "recent-3", "memory-run-86", "paper-1:chunk-4", "KnowledgeGap",
        ]:
            self.assertIn(expected, result.text)
        self.assertNotIn("old-1", result.text)
        self.assertNotIn("old-2", result.text)
        self.assertNotIn("MUST_NOT_LEAK", result.text)
        self.assertEqual(2, result.omitted_items["lineage"])
        self.assertIn("relevant_run_memory", result.included_keys)
        self.assertIn("evidence", result.included_keys)
        self.assertEqual(
            {"other_run_memory", "population", "repair_history"},
            set(result.excluded_keys),
        )
        self.assertTrue(result.recent)
        self.assertTrue(result.evidence)

    def test_research_minimizes_context_and_caps_its_own_history(self):
        result = self.framework.build("PriorResearchAgent", {
            "knowledge_gap": {"term": "Decision Tree Surrogate"},
            "prior_slice": {"ref": "prior-slice-2"},
            "landscape_summary": {"ruggedness": "high"},
            "strategy": "imitate",
            "parent_summary": {"id": "c9", "operators": ["TPE"]},
            "skill": "literature_evidence_review",
            "research_history": [
                "query-old", "query-1", "query-2", "query-3", "query-4", "query-5"
            ],
            "evidence": [{"chunk_ref": "paper-2:methods:1"}],
            "generation_memory": "GENERATION_MEMORY_MUST_NOT_LEAK",
            "repair_history": "REPAIR_MEMORY_MUST_NOT_LEAK",
            "population": "POPULATION_MUST_NOT_LEAK",
        })

        self.assertIn("Decision Tree Surrogate", result.text)
        self.assertIn("query-1", result.text)
        self.assertIn("query-5", result.text)
        self.assertNotIn("query-old", result.text)
        self.assertIn("paper-2:methods:1", result.text)
        self.assertNotIn("MUST_NOT_LEAK", result.text)
        self.assertEqual(1, result.omitted_items["research_history"])
        self.assertEqual(
            {"generation_memory", "population", "repair_history"},
            set(result.excluded_keys),
        )

    def test_repair_contains_only_candidate_failure_and_bounded_repair_context(self):
        result = self.framework.build("repair", {
            "candidate": {"id": "c17", "code_ref": "artifact-code-c17"},
            "failure": {"type": "SYNTAX_ERROR", "ref": "failure-c17"},
            "skill": "candidate_code_repair",
            "repair_history": ["attempt-0", "attempt-1", "attempt-2", "attempt-3"],
            "relevant_failures": ["failure-a", "failure-b", "failure-c", "failure-d"],
            "output_schema": {"type": "RepairedCandidateDraft"},
            "prior": "PRIOR_MUST_NOT_LEAK",
            "research_history": "RESEARCH_MUST_NOT_LEAK",
            "rag": "RAG_MUST_NOT_LEAK",
        })

        for expected in [
            "c17", "SYNTAX_ERROR", "candidate_code_repair", "attempt-1",
            "attempt-3", "failure-b", "failure-d", "RepairedCandidateDraft",
        ]:
            self.assertIn(expected, result.text)
        self.assertNotIn("attempt-0", result.text)
        self.assertNotIn("failure-a", result.text)
        self.assertNotIn("MUST_NOT_LEAK", result.text)
        self.assertEqual(1, result.omitted_items["repair_history"])
        self.assertEqual(1, result.omitted_items["relevant_failures"])

    def test_final_selection_has_no_memory_or_rag(self):
        result = self.framework.build("FinalSelectionAgent", {
            "tied_candidates": [
                {"id": "c29", "fitness": 0.1, "code_ref": "code-c29"},
                {"id": "c31", "fitness": 0.1, "code_ref": "code-c31"},
            ],
            "skill": "final_heuristic_audit",
            "output_schema": {"type": "FinalSelectionDecision"},
            "memory": "FINAL_MEMORY_MUST_NOT_LEAK",
            "rag": "FINAL_RAG_MUST_NOT_LEAK",
            "population": "NON_TIED_POPULATION_MUST_NOT_LEAK",
        })

        self.assertIn("c29", result.text)
        self.assertIn("c31", result.text)
        self.assertIn("final_heuristic_audit", result.text)
        self.assertNotIn("MUST_NOT_LEAK", result.text)
        self.assertEqual({"memory", "population", "rag"}, set(result.excluded_keys))
        self.assertEqual((), result.recent)
        self.assertEqual((), result.evidence)
        metadata = result.metadata()
        self.assertEqual("final_selection", metadata["policy_name"])
        self.assertEqual(
            ["tied_candidates", "skill", "output_schema"],
            metadata["included_keys"],
        )


if __name__ == "__main__":
    unittest.main()
