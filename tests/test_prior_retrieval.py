from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prievo_agent.core.prior_retrieval import PriorRetrievalService
from prievo_agent.domain.prior import LandscapeProfile, SemanticRefinement
from prievo_agent.infrastructure.prior_adapters import DeterministicPriorRefiner
from prievo_agent.infrastructure.prior_repository import StructuredPriorRepository
from prievo_agent.runtime.prior_harness import PriorRetrievalHarness, _metrics


class BrokenRefiner:
    def refine(self, target, candidates):
        raise RuntimeError("fake refiner unavailable")


class PriorRetrievalTest(unittest.TestCase):
    def test_numeric_semantic_provenance_trace_and_core_seed(self) -> None:
        report = PriorRetrievalHarness().run()
        self.assertEqual("history-nearest", report.nearest_instance)
        self.assertAlmostEqual(0.0, report.nearest_distance)
        self.assertEqual(3, report.numeric_count)
        self.assertEqual(["history-nearest", "history-second"], report.semantic_selected)
        self.assertTrue(report.operator_provenance_complete)
        self.assertEqual(2, report.seeded_candidate_count)
        self.assertIn("LANDSCAPE_ANALYZED", report.trace_events)
        self.assertIn("PRIOR_RETRIEVED", report.trace_events)
        self.assertIn("LANDSCAPE_PROFILE", report.artifact_kinds)
        self.assertIn("INSTANCE_SPECIFIC_PRIOR", report.artifact_kinds)

    def test_semantic_refinement_failure_is_explicit_without_numeric_fallback(self) -> None:
        histories = [
            LandscapeProfile("a", _metrics(0.0), 100, "fixture"),
            LandscapeProfile("b", _metrics(0.2), 100, "fixture"),
            LandscapeProfile("c", _metrics(0.4), 100, "fixture"),
        ]
        with self.assertRaisesRegex(RuntimeError, "fake refiner unavailable"):
            PriorRetrievalService(
                StructuredPriorRepository(histories, {}), BrokenRefiner()
            ).retrieve(LandscapeProfile("target", _metrics(0.0), 100, "fixture"))

    def test_numeric_and_prior_extraction_are_separate_and_allowlisted(self) -> None:
        histories = [
            LandscapeProfile("a", _metrics(0.0), 100, "fixture"),
            LandscapeProfile("b", _metrics(0.2), 100, "fixture"),
            LandscapeProfile("c", _metrics(0.4), 100, "fixture"),
        ]
        target = LandscapeProfile("target", _metrics(0.0), 100, "fixture")
        service = PriorRetrievalService(
            StructuredPriorRepository(histories, {}), DeterministicPriorRefiner()
        )
        numeric = service.retrieve_numeric(target, top_k=3)
        decision = SemanticRefinement([numeric[0].instance_name], "fixture", "SimilaritySelectionNode")

        prior = service.extract(target, numeric, decision)

        self.assertEqual(["a"], prior.refinement.selected_instances)
        with self.assertRaisesRegex(ValueError, "numeric Top-K"):
            service.extract(
                target,
                numeric,
                SemanticRefinement(["not-in-top5"], "bad", "SimilaritySelectionNode"),
            )


if __name__ == "__main__":
    unittest.main()
