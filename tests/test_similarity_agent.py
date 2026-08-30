from __future__ import annotations

import unittest
from pathlib import Path

from prievo_agent.agents.nodes.similarity import (
    MalformedSimilarityDecisionError,
    SimilaritySelectionNode,
)
from prievo_agent.domain.models import AgentCapability
from prievo_agent.domain.prior import (
    LANDSCAPE_METRICS,
    LandscapeProfile,
    SimilarInstance,
)
from prievo_agent.infrastructure.testing.fake_llm import FakeLLM
from prievo_agent.infrastructure.skill_registry import SkillRegistry


def _metrics(offset: float) -> dict[str, float]:
    return {
        metric: float(index) / 10.0 + offset
        for index, metric in enumerate(LANDSCAPE_METRICS)
    }


def _candidate(name: str, rank: int) -> SimilarInstance:
    return SimilarInstance(name, rank / 10.0, _metrics(rank / 100.0), _metrics(0.0))


def _semantics() -> dict[str, str]:
    return {
        metric: "{} definition, range, and value significance".format(metric)
        for metric in LANDSCAPE_METRICS
    }


class _StaticModel:
    def __init__(self, result):
        self.result = result

    def generate_similarity_decision(self, prompt, allowed_instance_ids):
        return self.result


class SimilaritySelectionNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.registry = SkillRegistry(cls.project / "skills")
        cls.target = LandscapeProfile("target", _metrics(0.0), 100, "fixture")
        cls.candidates = [_candidate("A", 1), _candidate("b", 2), _candidate("C", 3)]

    def test_fake_selects_top_two_and_preserves_prompt_provenance(self):
        llm = FakeLLM()
        agent = SimilaritySelectionNode(self.registry, llm)

        decision = agent.select(self.target, self.candidates, _semantics())

        self.assertEqual("SimilaritySelectionNode", agent.name)
        self.assertEqual(AgentCapability.SEMANTIC_SIMILARITY, agent.capability)
        self.assertEqual(["A", "b"], decision.selected_instance_ids)
        self.assertEqual("semantic_similarity_selection", decision.skill_name)
        self.assertTrue(decision.skill_digest)
        self.assertEqual({"A", "b"}, set(decision.metric_evidence))
        self.assertEqual(set(LANDSCAPE_METRICS), set(decision.metric_evidence["A"]))
        self.assertEqual(1, len(llm.similarity_calls))
        self.assertEqual([], llm.calls, "semantic call 不得污染既有 candidate call 记录")
        for metric in LANDSCAPE_METRICS:
            self.assertIn(metric, decision.selection_prompt)
            self.assertIn(_semantics()[metric], decision.selection_prompt)
        self.assertIn('"rank": 1', decision.selection_prompt)
        self.assertIn('"numeric_distance": 0.1', decision.selection_prompt)
        self.assertIn("Return exactly one JSON object", decision.selection_prompt)
        self.assertEqual("similarity", decision.context_metadata["policy_name"])
        self.assertNotIn("Recent Agent Working Memory", decision.selection_prompt)
        self.assertEqual(
            decision.selected_instance_ids,
            decision.to_dict()["selected_instance_ids"],
        )

    def test_allowlist_is_exact_case_sensitive_and_deduplicated(self):
        valid_evidence = {
            metric: "close under supplied definition" for metric in LANDSCAPE_METRICS
        }
        malformed_results = [
            {
                "selected_instance_ids": ["a"],
                "reason_summary": "wrong case",
                "metric_evidence": {"a": valid_evidence},
            },
            {
                "selected_instance_ids": ["A", "A"],
                "reason_summary": "duplicate",
                "metric_evidence": {"A": valid_evidence},
            },
            {
                "selected_instance_ids": ["A", "b", "C", "outside"],
                "reason_summary": "too many",
                "metric_evidence": {},
            },
        ]
        for result in malformed_results:
            with self.subTest(result=result), self.assertRaises(
                MalformedSimilarityDecisionError
            ):
                SimilaritySelectionNode(self.registry, _StaticModel(result)).select(
                    self.target, self.candidates, _semantics()
                )

    def test_malformed_response_fails_explicitly_without_numeric_fallback(self):
        for result in ["not-an-object", {}, {"selected_instance_ids": []}]:
            with self.subTest(result=result), self.assertRaises(
                MalformedSimilarityDecisionError
            ):
                SimilaritySelectionNode(self.registry, _StaticModel(result)).select(
                    self.target, self.candidates, _semantics()
                )

    def test_requires_all_metric_semantics(self):
        semantics = _semantics()
        semantics.pop("NBC")
        with self.assertRaisesRegex(ValueError, "NBC"):
            SimilaritySelectionNode(self.registry, FakeLLM()).select(
                self.target, self.candidates, semantics
            )


if __name__ == "__main__":
    unittest.main()
