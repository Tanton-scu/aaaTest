"""目标 CTP Dataset 的可追溯 Fitness Landscape Analysis。

该模块实现 PriEvO reference 的三段式入口：space-filling sampling、Dataset
exact/nearest query、8-metric FLA。可选 research 依赖缺失时只允许使用同名目标
实例的 recorded profile，并显式记录 fallback；绝不生成占位指标。
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from prievo_agent.datasets.registry import LoadedDataset
from prievo_agent.domain.prior import LANDSCAPE_METRICS, LandscapeProfile


class UnsupportedLandscapeAnalysisError(RuntimeError):
    """当前环境无法真实计算，且没有目标实例的可信 recorded profile。"""


class LandscapeComputationError(RuntimeError):
    """research extras 已进入计算，但真实 FLA 结果失败或无效。"""


@dataclass(frozen=True)
class LandscapeSampleBatch:
    """采样与 Dataset query 的完整 provenance。"""

    rows: List[Dict[str, Any]]
    sampled_configurations: List[List[Any]]
    matched_configurations: List[List[Any]]
    exact_match_count: int
    nearest_match_count: int
    seed: int
    requested_sample_size: int

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LandscapeAnalysisResult:
    profile: LandscapeProfile
    samples: LandscapeSampleBatch
    source: str
    fallback_reason: str = ""

    def to_dict(self):
        return asdict(self)


class DatasetLandscapeSampler:
    """纯 stdlib、确定性的 LHS-like 离散空间采样器。"""

    performance_key = "objective"

    def sample(
        self, dataset: LoadedDataset, seed: int, sample_size: int
    ) -> LandscapeSampleBatch:
        _validate_loaded_dataset(dataset)
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size 必须是正整数")
        spaces = _configuration_spaces(dataset)
        total_space = math.prod(len(options) for options in spaces)
        if sample_size > total_space:
            raise ValueError(
                "sample_size {} 超过离散搜索空间 {}，无法生成唯一采样".format(
                    sample_size, total_space
                )
            )

        rng = random.Random(int(seed))
        configurations = _lhs_like_configurations(spaces, sample_size, rng)
        rows = []
        matched = []
        exact_count = 0
        nearest_count = 0
        for configuration in configurations:
            objective, matched_configuration, exact = _query_dataset(
                dataset, configuration, spaces
            )
            row = {
                "param_{}".format(index + 1): value
                for index, value in enumerate(configuration)
            }
            row[self.performance_key] = float(objective)
            rows.append(row)
            matched.append(list(matched_configuration))
            if exact:
                exact_count += 1
            else:
                nearest_count += 1
        return LandscapeSampleBatch(
            rows=rows,
            sampled_configurations=[list(item) for item in configurations],
            matched_configurations=matched,
            exact_match_count=exact_count,
            nearest_match_count=nearest_count,
            seed=int(seed),
            requested_sample_size=sample_size,
        )


class RecordedLandscapeAdapter:
    """只按目标 instance ID 精确加载 recorded FLA；不做相似实例替代。"""

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def profile_for(self, instance_name: str) -> LandscapeProfile:
        wanted = instance_name.strip().casefold()
        profiles = self._profiles()
        for profile in profiles:
            if profile.instance_name.strip().casefold() != wanted:
                continue
            _validate_profile(profile)
            return LandscapeProfile(
                instance_name=instance_name,
                metrics={name: float(profile.metrics[name]) for name in LANDSCAPE_METRICS},
                sample_count=int(profile.sample_count),
                analyzer="recorded fallback from {}".format(profile.analyzer),
            )
        raise UnsupportedLandscapeAnalysisError(
            "记录库中不存在目标实例 {} 的 exact FLA profile".format(instance_name)
        )

    def _profiles(self) -> List[LandscapeProfile]:
        if self.repository is None:
            return []
        if hasattr(self.repository, "landscape_profiles"):
            return list(self.repository.landscape_profiles())
        if isinstance(self.repository, Mapping):
            profiles = []
            for name, value in self.repository.items():
                if isinstance(value, LandscapeProfile):
                    profiles.append(value)
                    continue
                if not isinstance(value, Mapping):
                    raise TypeError("recorded landscape Mapping value 必须是 profile/metrics")
                metrics_value = value.get("metrics", value)
                profiles.append(
                    LandscapeProfile(
                        str(name),
                        {metric: float(metrics_value[metric]) for metric in LANDSCAPE_METRICS},
                        int(value.get("sample_count", 0)),
                        str(value.get("analyzer", "recorded repository")),
                    )
                )
            return profiles
        raise TypeError("recorded_repository 必须提供 landscape_profiles() 或 Mapping")


class LandscapeAnalysisService:
    """真实 FLA 优先、仅 dependency-unavailable 时可记录回退。"""

    def __init__(
        self,
        analyzer: Any = None,
        recorded_repository: Any = None,
        sampler: DatasetLandscapeSampler = None,
    ) -> None:
        self.analyzer = analyzer
        self.recorded = RecordedLandscapeAdapter(recorded_repository)
        self.sampler = sampler or DatasetLandscapeSampler()

    def analyze(
        self, dataset: LoadedDataset, seed: int = 42, sample_size: int = 100
    ) -> LandscapeAnalysisResult:
        samples = self.sampler.sample(dataset, seed=seed, sample_size=sample_size)
        instance_name = str(dataset.info.id)
        analyzer = self.analyzer or ReferencePflaccoAnalyzer(seed=seed)
        try:
            profile = analyzer.analyze(instance_name, samples.rows)
        except Exception as exc:
            if not _is_dependency_unavailable(exc):
                raise
            fallback_reason = str(exc) or exc.__class__.__name__
            try:
                recorded = self.recorded.profile_for(instance_name)
            except UnsupportedLandscapeAnalysisError as missing:
                raise UnsupportedLandscapeAnalysisError(
                    "真实 FLA 不可用（{}），且 {}".format(fallback_reason, missing)
                ) from exc
            return LandscapeAnalysisResult(
                profile=recorded,
                samples=samples,
                source="recorded",
                fallback_reason=fallback_reason,
            )
        _validate_profile(profile)
        if profile.instance_name != instance_name:
            raise LandscapeComputationError(
                "真实 analyzer 返回了其他 instance：{}".format(profile.instance_name)
            )
        return LandscapeAnalysisResult(
            profile=profile,
            samples=samples,
            source="real",
            fallback_reason="",
        )


class ReferencePflaccoAnalyzer:
    """尽可能忠实移植 PriEvO `prior_util/fl_compute.py` 的 8-metric 计算。"""

    def __init__(
        self, seed: int = 42, performance_key: str = "objective", minimize: bool = True
    ) -> None:
        self.seed = int(seed)
        self.performance_key = performance_key
        self.minimize = bool(minimize)

    def analyze(
        self, instance_name: str, samples: Sequence[Dict[str, Any]]
    ) -> LandscapeProfile:
        try:
            import numpy as np
            import pandas as pd
            import sklearn  # noqa: F401 - 明确校验 research extra 完整性
            from pflacco.classical_ela_features import (
                calculate_ela_distribution,
                calculate_information_content,
                calculate_nbc,
            )
            from pflacco.misc_features import calculate_fitness_distance_correlation
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise UnsupportedLandscapeAnalysisError(
                "缺少 numpy/pandas/scikit-learn/pflacco research 可选依赖"
            ) from exc

        if len(samples) < 3:
            raise LandscapeComputationError("真实 FLA 至少需要 3 条 sample")
        data = pd.DataFrame(samples)
        if self.performance_key not in data.columns:
            raise LandscapeComputationError(
                "sample 缺少 performance key：{}".format(self.performance_key)
            )

        encoded = data.copy()
        for column in encoded.columns:
            if pd.api.types.is_bool_dtype(encoded[column]):
                encoded[column] = encoded[column].astype(int)
            elif not pd.api.types.is_numeric_dtype(encoded[column]):
                encoded[column] = LabelEncoder().fit_transform(
                    encoded[column].fillna("NaN")
                )

        parameter_columns = [
            column for column in encoded.columns if column != self.performance_key
        ]
        x = encoded[parameter_columns]
        y = encoded[self.performance_key]
        params = x.values
        performances = y.values
        param_types = [
            _infer_reference_param_type(encoded[column].tolist(), np)
            for column in parameter_columns
        ]
        optimum_value = min(performances) if self.minimize else max(performances)
        optimum_configs = [
            tuple(params[index])
            for index, value in enumerate(performances)
            if value == optimum_value
        ]
        if not optimum_configs:
            raise LandscapeComputationError("sample 无法确定最优配置")

        try:
            fdc_result = calculate_fitness_distance_correlation(
                x, y, minimize=self.minimize
            )
            fdc = fdc_result["fitness_distance.fd_correlation"]
            if pd.isna(fdc):
                # 与 reference 一致的 documented abnormal value。
                fdc = 0.5
            fbd = _mean(
                [
                    min(
                        _mixed_distance(tuple(config), best, param_types)
                        for best in optimum_configs
                    )
                    for config in params
                ]
            )
            plo = self._proportion_local_optima(
                encoded, parameter_columns, params, performances, param_types
            )
            distribution = calculate_ela_distribution(x, y)
            skewness = distribution["ela_distr.skewness"]
            kurtosis = distribution["ela_distr.kurtosis"]
            correlation_length = self._correlation_length(
                params, performances, optimum_configs, param_types, np
            )
            mie = calculate_information_content(x, y, seed=100)["ic.h_max"]
            nbc = calculate_nbc(x, y)["nbc.nn_nb.mean_ratio"]
        except LandscapeComputationError:
            raise
        except Exception as exc:
            raise LandscapeComputationError(
                "pflacco/reference FLA 计算失败：{}".format(exc)
            ) from exc

        values = [fdc, fbd, plo, skewness, kurtosis, correlation_length, mie, nbc]
        metrics = {
            name: float(value) for name, value in zip(LANDSCAPE_METRICS, values)
        }
        if any(not math.isfinite(value) for value in metrics.values()):
            raise LandscapeComputationError("真实 FLA 返回 NaN/Infinity，拒绝伪造替代值")
        return LandscapeProfile(
            instance_name,
            metrics,
            len(samples),
            "pflacco PriEvO reference-compatible",
        )

    def _proportion_local_optima(
        self, data, columns, params, performances, param_types, max_neighbors=5
    ):
        rng = random.Random(self.seed)
        mapping = {
            tuple(configuration): performance
            for configuration, performance in zip(params, performances)
        }
        local_count = 0
        for index, configuration in enumerate(params):
            neighbors = []
            for axis, column in enumerate(columns):
                current = configuration[axis]
                values = list(data[column].unique())
                if param_types[axis] == "continuous":
                    ordered = sorted(values)
                    position = _insertion_index(ordered, current)
                    if position >= len(ordered) or ordered[position] != current:
                        position = min(position, len(ordered) - 1)
                    candidates = [
                        ordered[item]
                        for item in range(
                            max(0, position - max_neighbors),
                            min(len(ordered), position + max_neighbors + 1),
                        )
                        if item != position
                    ]
                else:
                    candidates = [value for value in values if value != current]
                    if len(candidates) > max_neighbors:
                        candidates = rng.sample(candidates, max_neighbors)
                for value in candidates:
                    neighbor = configuration.copy()
                    neighbor[axis] = value
                    neighbors.append(tuple(neighbor))
            current_performance = performances[index]
            better_or_equal = (
                (lambda left, right: left <= right)
                if self.minimize
                else (lambda left, right: left >= right)
            )
            if all(
                neighbor not in mapping
                or better_or_equal(current_performance, mapping[neighbor])
                for neighbor in neighbors
            ):
                local_count += 1
        return local_count / float(len(params))

    def _correlation_length(
        self, params, performances, optimum_configs, param_types, np
    ):
        rng = random.Random(self.seed)
        populations = [tuple(configuration) for configuration in params]
        mapping = {
            tuple(configuration): performance
            for configuration, performance in zip(params, performances)
        }

        def min_distance(configuration):
            values = [
                _mixed_distance(configuration, base, param_types)
                for base in optimum_configs
            ]
            non_zero = [value for value in values if value != 0]
            return min(non_zero) if non_zero else 1e8

        ordered = sorted(populations, key=min_distance)
        max_distance = max(
            _mixed_distance(left, right, param_types)
            for left in ordered
            for right in optimum_configs
        )
        total_autocorrelation = 0.0
        valid_samples = 0
        for _ in range(50):
            walk = list(optimum_configs)
            current = rng.choice(optimum_configs)
            for _ in range(50):
                radius = 0.1
                neighbors = []
                while radius <= max_distance + 0.2:
                    neighbors = [
                        candidate
                        for candidate in ordered
                        if 0 < _mixed_distance(candidate, current, param_types) <= radius
                    ]
                    if neighbors:
                        break
                    radius += 0.1
                if not neighbors:
                    break
                current = rng.choice(neighbors)
                walk.append(current)
            fitness = [mapping[item] for item in walk]
            if len(fitness) < 2:
                continue
            mean = float(np.mean(fitness))
            autocorrelation = sum(
                (fitness[index] - mean) * (fitness[index + 1] - mean)
                for index in range(len(fitness) - 1)
            ) / float(len(fitness) - 1)
            deviation = float(np.std(fitness))
            if deviation != 0:
                total_autocorrelation += autocorrelation / (deviation ** 2)
                valid_samples += 1
        if not valid_samples:
            return 0.5
        average = total_autocorrelation / valid_samples
        if average == 0 or abs(average) >= 1:
            return 0.5
        return -1.0 / math.log(abs(average))


def build_landscape_samples(
    dataset: LoadedDataset, seed: int = 42, sample_size: int = 100
) -> LandscapeSampleBatch:
    """函数式入口，便于 Harness 单独验证 sampling/query。"""

    return DatasetLandscapeSampler().sample(dataset, seed, sample_size)


def _validate_loaded_dataset(dataset):
    if not isinstance(dataset, LoadedDataset):
        raise TypeError("dataset 必须是 LoadedDataset")
    if not dataset.configurations or not dataset.objectives:
        raise ValueError("LoadedDataset 没有 configuration/objective")
    if len(dataset.configurations) != len(dataset.objectives):
        raise ValueError("LoadedDataset configuration/objective 数量不一致")
    if not dataset.info.independent_columns:
        raise ValueError("LoadedDataset independent columns 为空")


def _configuration_spaces(dataset):
    dimensions = len(dataset.info.independent_columns)
    spaces = []
    for axis in range(dimensions):
        values = _stable_unique(configuration[axis] for configuration in dataset.configurations)
        try:
            values = sorted(values)
        except TypeError:
            values = sorted(values, key=lambda item: (type(item).__name__, repr(item)))
        if not values:
            raise ValueError("independent_set 第 {} 维为空".format(axis))
        spaces.append(values)
    for configuration in dataset.configurations:
        if len(configuration) != dimensions:
            raise ValueError("configuration 维数与 independent columns 不一致")
    return spaces


def _lhs_like_configurations(spaces, sample_size, rng):
    coordinates_by_axis = []
    for _ in spaces:
        strata = list(range(sample_size))
        rng.shuffle(strata)
        coordinates_by_axis.append(
            [(stratum + rng.random()) / sample_size for stratum in strata]
        )
    result = []
    seen = set()
    for row_index in range(sample_size):
        configuration = tuple(
            options[min(int(coordinates_by_axis[axis][row_index] * len(options)), len(options) - 1)]
            for axis, options in enumerate(spaces)
        )
        if configuration not in seen:
            seen.add(configuration)
            result.append(configuration)

    # 离散映射可能使多个 LHS strata 落到同一组合；先随机补齐，再以稳定枚举兜底。
    attempts = 0
    max_attempts = max(100, sample_size * 20)
    while len(result) < sample_size and attempts < max_attempts:
        configuration = tuple(rng.choice(options) for options in spaces)
        attempts += 1
        if configuration in seen:
            continue
        seen.add(configuration)
        result.append(configuration)
    if len(result) < sample_size:
        for configuration in itertools.product(*spaces):
            if configuration in seen:
                continue
            seen.add(configuration)
            result.append(configuration)
            if len(result) == sample_size:
                break
    return result


def _query_dataset(dataset, configuration, spaces):
    key = tuple(configuration)
    if key in dataset.lookup:
        return float(dataset.lookup[key]), key, True

    parameter_types = [
        _infer_query_param_type(options) for options in spaces
    ]
    numeric_ranges = []
    for options, kind in zip(spaces, parameter_types):
        if kind == "numeric":
            numeric = [float(value) for value in options]
            numeric_ranges.append((min(numeric), max(numeric)))
        else:
            numeric_ranges.append((0.0, 0.0))

    best = None
    for index, candidate in enumerate(dataset.configurations):
        continuous_squared = 0.0
        categorical_distance = 0.0
        for axis, (left, right) in enumerate(zip(configuration, candidate)):
            if parameter_types[axis] == "numeric":
                minimum, maximum = numeric_ranges[axis]
                span = maximum - minimum
                delta = 0.0 if span == 0 else (float(left) - float(right)) / span
                continuous_squared += delta * delta
            elif left != right:
                categorical_distance += 1.0
        distance = math.sqrt(continuous_squared) + categorical_distance
        ranked = (distance, index)
        if best is None or ranked < best[0]:
            best = (ranked, float(dataset.objectives[index]), tuple(candidate))
    if best is None:
        raise ValueError("LoadedDataset 无法执行 nearest query")
    return best[1], best[2], False


def _infer_query_param_type(values):
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "numeric"
    return "categorical"


def _infer_reference_param_type(values, np):
    if any(isinstance(value, str) for value in values):
        return "categorical"
    if not all(isinstance(value, (int, float, np.number)) for value in values):
        return "categorical"
    unique = list(set(values))
    ratio = len(unique) / len(values) if values else 0
    deviation = np.std(values) if len(values) > 1 else 0
    if ratio > 0.1 and deviation > 1:
        return "continuous"
    if len(unique) <= 10:
        return "discrete"
    return "categorical"


def _mixed_distance(left, right, parameter_types):
    hamming = 0.0
    euclidean = 0.0
    categorical_count = 0
    continuous_count = 0
    for first, second, kind in zip(left, right, parameter_types):
        if kind == "categorical":
            hamming += first != second
            categorical_count += 1
        else:
            euclidean += (first - second) ** 2
            continuous_count += 1
    return (
        (hamming / categorical_count if categorical_count else 0.0)
        + (math.sqrt(euclidean) / math.sqrt(continuous_count) if continuous_count else 0.0)
    )


def _validate_profile(profile):
    if not isinstance(profile, LandscapeProfile):
        raise LandscapeComputationError("analyzer 必须返回 LandscapeProfile")
    if set(profile.metrics) != set(LANDSCAPE_METRICS):
        raise LandscapeComputationError("FLA profile 必须且只能包含 reference 8 metrics")
    for name in LANDSCAPE_METRICS:
        try:
            value = float(profile.metrics[name])
        except (TypeError, ValueError) as exc:
            raise LandscapeComputationError("FLA metric {} 不是数值".format(name)) from exc
        if not math.isfinite(value):
            raise LandscapeComputationError("FLA metric {} 是 NaN/Infinity".format(name))


def _is_dependency_unavailable(exc):
    current = exc
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(
            current,
            (UnsupportedLandscapeAnalysisError, ImportError, ModuleNotFoundError),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _stable_unique(values):
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _mean(values):
    return sum(float(value) for value in values) / float(len(values))


def _insertion_index(values, target):
    low = 0
    high = len(values)
    while low < high:
        middle = (low + high) // 2
        if values[middle] < target:
            low = middle + 1
        else:
            high = middle
    return low
