from __future__ import annotations

import importlib.util
import itertools
import math
import unittest
from pathlib import Path

from prievo_agent.knowledge.prior.landscape import (
    DatasetLandscapeSampler,
    LandscapeAnalysisService,
    UnsupportedLandscapeAnalysisError,
)
from prievo_agent.evaluation.datasets import DatasetInfo, LoadedDataset
from prievo_agent.knowledge.prior.models import LANDSCAPE_METRICS, LandscapeProfile


def _dataset(instance_name="known", full_factorial=True):
    columns = ["x", "kind", "$<objective"]
    independent = ["x", "kind"]
    configurations = (
        [[x, kind] for x in [0.0, 5.0, 10.0] for kind in ["a", "b", "c"]]
        if full_factorial
        else [[0.0, "a"], [5.0, "b"], [10.0, "c"]]
    )
    objectives = [float(index + 1) for index in range(len(configurations))]
    rows = [
        {"x": config[0], "kind": config[1], "$<objective": objectives[index]}
        for index, config in enumerate(configurations)
    ]
    info = DatasetInfo(
        instance_name,
        instance_name + ".csv",
        instance_name,
        Path(instance_name + ".csv"),
        columns,
        independent,
        ["$<objective"],
        [],
        len(rows),
        "fixture-digest",
    )
    return LoadedDataset(info, rows, configurations, objectives)


def _metrics(offset=0.0):
    return {
        metric: float(index + 1) / 10.0 + offset
        for index, metric in enumerate(LANDSCAPE_METRICS)
    }


class _UnavailableAnalyzer:
    def analyze(self, instance_name, samples):
        raise UnsupportedLandscapeAnalysisError("fixture 缺少 pflacco")


class _RealAnalyzer:
    def analyze(self, instance_name, samples):
        return LandscapeProfile(instance_name, _metrics(), len(samples), "fixture real")


class _BrokenComputationAnalyzer:
    def analyze(self, instance_name, samples):
        raise ValueError("fixture metric computation failed")


class _RecordedRepository:
    def __init__(self, profiles):
        self.profiles = profiles

    def landscape_profiles(self):
        return list(self.profiles)


class LandscapeAnalysisTest(unittest.TestCase):
    def test_lhs_like_sampling_is_deterministic_and_maps_only_independent_set(self):
        dataset = _dataset(full_factorial=False)
        sampler = DatasetLandscapeSampler()

        first = sampler.sample(dataset, seed=17, sample_size=6)
        second = sampler.sample(dataset, seed=17, sample_size=6)
        different = sampler.sample(dataset, seed=18, sample_size=6)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(
            first.sampled_configurations, different.sampled_configurations
        )
        self.assertEqual(6, len(first.rows))
        self.assertEqual(6, first.exact_match_count + first.nearest_match_count)
        self.assertGreater(first.nearest_match_count, 0)
        for row in first.rows:
            self.assertIn(row["param_1"], [0.0, 5.0, 10.0])
            self.assertIn(row["param_2"], ["a", "b", "c"])
            self.assertIsInstance(row["objective"], float)
        for sampled, matched in zip(
            first.sampled_configurations, first.matched_configurations
        ):
            self.assertIn(matched, dataset.configurations)
            if sampled not in dataset.configurations:
                self.assertNotEqual(sampled, matched)

    def test_recorded_fallback_has_explicit_provenance(self):
        profile = LandscapeProfile("known", _metrics(), 100, "PriEvO dataset_fl.csv")
        service = LandscapeAnalysisService(
            analyzer=_UnavailableAnalyzer(),
            recorded_repository=_RecordedRepository([profile]),
        )

        result = service.analyze(_dataset("known"), seed=7, sample_size=5)

        self.assertEqual("recorded", result.source)
        self.assertIn("缺少 pflacco", result.fallback_reason)
        self.assertIn("recorded fallback", result.profile.analyzer)
        self.assertEqual(_metrics(), result.profile.metrics)
        self.assertEqual(5, len(result.samples.rows))
        self.assertEqual("recorded", result.to_dict()["source"])

    def test_unseen_target_is_rejected_when_real_analysis_is_unavailable(self):
        known = LandscapeProfile("known", _metrics(), 100, "recorded")
        service = LandscapeAnalysisService(
            analyzer=_UnavailableAnalyzer(),
            recorded_repository=_RecordedRepository([known]),
        )
        with self.assertRaisesRegex(
            UnsupportedLandscapeAnalysisError, "unseen"
        ):
            service.analyze(_dataset("unseen"), seed=7, sample_size=5)

    def test_real_analyzer_result_is_not_marked_as_fallback(self):
        service = LandscapeAnalysisService(
            analyzer=_RealAnalyzer(),
            recorded_repository=_RecordedRepository([]),
        )
        result = service.analyze(_dataset("new-target"), seed=9, sample_size=5)
        self.assertEqual("real", result.source)
        self.assertEqual("", result.fallback_reason)
        self.assertEqual("fixture real", result.profile.analyzer)
        self.assertEqual(set(LANDSCAPE_METRICS), set(result.profile.metrics))

    def test_metric_computation_error_is_not_hidden_by_recorded_profile(self):
        recorded = LandscapeProfile("known", _metrics(), 100, "recorded")
        service = LandscapeAnalysisService(
            analyzer=_BrokenComputationAnalyzer(),
            recorded_repository=_RecordedRepository([recorded]),
        )
        with self.assertRaisesRegex(ValueError, "computation failed"):
            service.analyze(_dataset("known"), seed=3, sample_size=5)

    def test_impossible_unique_sample_size_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "搜索空间"):
            DatasetLandscapeSampler().sample(_dataset(), seed=1, sample_size=10)

    @unittest.skipUnless(
        all(
            importlib.util.find_spec(name) is not None
            for name in ("numpy", "pandas", "sklearn", "pflacco")
        ),
        "未安装 research FLA extras",
    )
    def test_optional_reference_pflacco_path_returns_all_finite_metrics(self):
        values = range(5)
        configurations = [list(item) for item in itertools.product(values, repeat=3)]
        objectives = [
            float((x - 2) ** 2 + (y - 1) ** 2 + (z - 3) ** 2 + 0.1 * x * y)
            for x, y, z in configurations
        ]
        info = DatasetInfo(
            "synthetic-real",
            "synthetic-real.csv",
            "synthetic-real",
            Path("synthetic-real.csv"),
            ["x", "y", "z", "$<objective"],
            ["x", "y", "z"],
            ["$<objective"],
            [],
            len(configurations),
            "fixture-real-digest",
        )
        rows = [
            {"x": config[0], "y": config[1], "z": config[2], "$<objective": objective}
            for config, objective in zip(configurations, objectives)
        ]
        dataset = LoadedDataset(info, rows, configurations, objectives)

        result = LandscapeAnalysisService().analyze(
            dataset, seed=42, sample_size=100
        )

        self.assertEqual("real", result.source)
        self.assertEqual(set(LANDSCAPE_METRICS), set(result.profile.metrics))
        self.assertTrue(all(math.isfinite(value) for value in result.profile.metrics.values()))


if __name__ == "__main__":
    unittest.main()
