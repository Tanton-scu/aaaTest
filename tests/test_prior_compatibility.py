from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.evolution.population import PriEvoEvolutionCore
from prievo_agent.knowledge.prior.compatibility import (
    assess_prior_code,
    load_prior_compatibility_report,
)
from prievo_agent.domain.models import Run
from prievo_agent.knowledge.prior.models import (
    InstanceSpecificPrior,
    LandscapeProfile,
    OptimizerEvidence,
    SemanticRefinement,
)
from prievo_agent.infrastructure.local.fake_llm import FakeLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PriorCompatibilityTest(unittest.TestCase):
    def test_all_31_records_have_stable_matrix(self):
        source = PROJECT_ROOT / "assets" / "prior" / "prior_population.json"
        first = load_prior_compatibility_report(source)
        second = load_prior_compatibility_report(source)

        self.assertEqual(first, second)
        self.assertEqual(31, first["total"])
        self.assertEqual(21, first["supported_count"])
        self.assertEqual(10, first["unsupported_count"])
        self.assertEqual(
            {
                "HEBO",
                "PromiseTune",
                "SWAY",
                "CMAES",
                "ACO",
                "ResTune",
                "ROBOTune",
                "Hyperband",
                "BOHB",
                "DEHB",
            },
            {
                row["name"]
                for row in first["records"]
                if row["status"] == "UNSUPPORTED"
            },
        )

    def test_unsupported_prior_remains_evidence_but_is_not_materialized(self):
        unsupported = "import torch\ndef run_tuners(file, budget, seed, maxlives):\n    return 1\n"
        supported = (
            "def run_tuners(file, budget, seed, maxlives):\n"
            "    generated_config = [values[0] for values in file.independent_set]\n"
            "    return evaluate(0, 0, {}, float('inf'), generated_config)[3]\n"
        )
        profile = LandscapeProfile(
            "target",
            {name: 0.0 for name in ("FDC", "FBD", "PLO", "Skewness", "Kurtosis", "CL", "MIE", "NBC")},
            100,
            "fixture",
        )
        prior = InstanceSpecificPrior(
            profile,
            [],
            SemanticRefinement(["source"], "fixture", "fixture"),
            [
                OptimizerEvidence("blocked", "rank1", "source", "blocked", unsupported, []),
                OptimizerEvidence("safe", "rank2", "source", "safe", supported, []),
            ],
            "fixture-v1",
        )

        self.assertFalse(assess_prior_code(unsupported).supported)
        self.assertEqual(2, len(prior.optimizers), "immutable prior evidence 不得被覆盖")
        seeds = PriEvoEvolutionCore(
            FakeLLM(), population_size=2, prior=prior
        ).propose_prior_seeds(Run("run-prior", "task-prior"))
        self.assertEqual(["safe"], [item.lineage["optimizer"] for item in seeds])
        self.assertEqual("SUPPORTED", seeds[0].lineage["prior_compatibility_status"])


if __name__ == "__main__":
    unittest.main()
