from __future__ import annotations

import math
import random
from typing import Dict, Sequence

from prievo_agent.domain.prior import LANDSCAPE_METRICS, LandscapeProfile


class PflaccoLandscapeAnalyzer:
    """按 PriEvO reference 公式计算 8 个 FLA 指标的可选 research adapter。"""

    def __init__(self, performance_key: str = "objective", minimize: bool = True) -> None:
        self.performance_key = performance_key
        self.minimize = minimize

    def analyze(
        self, instance_name: str, samples: Sequence[Dict[str, object]]
    ) -> LandscapeProfile:
        try:
            import numpy as np
            import pandas as pd
            from pflacco.classical_ela_features import (
                calculate_ela_distribution,
                calculate_information_content,
                calculate_nbc,
            )
            from pflacco.misc_features import calculate_fitness_distance_correlation
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise RuntimeError("请安装 research 可选依赖后计算真实 FLA 指标") from exc
        if not samples:
            raise ValueError("landscape samples 不能为空")
        data = pd.DataFrame(samples)
        if self.performance_key not in data.columns:
            raise ValueError("sample 缺少 performance key：{}".format(self.performance_key))
        encoder = LabelEncoder()
        for column in data.columns:
            if pd.api.types.is_bool_dtype(data[column]):
                data[column] = data[column].astype(int)
            elif not pd.api.types.is_numeric_dtype(data[column]):
                data[column] = encoder.fit_transform(data[column].fillna("NaN"))
        y = data[self.performance_key]
        x = data[[column for column in data.columns if column != self.performance_key]]
        params = x.values
        performances = y.values
        param_types = [self._infer_type(data[column].tolist(), np) for column in x.columns]
        optimum = min(performances) if self.minimize else max(performances)
        optimum_configs = [tuple(params[index]) for index, value in enumerate(performances) if value == optimum]

        fdc = calculate_fitness_distance_correlation(x, y, minimize=self.minimize)["fitness_distance.fd_correlation"]
        if pd.isna(fdc):
            fdc = 0.5
        fbd = float(np.mean([
            min(self._mixed_distance(tuple(config), best, param_types, math) for best in optimum_configs)
            for config in params
        ])) if optimum_configs else 0.5
        plo = self._plo(data, x.columns.tolist(), params, performances, param_types, np)
        distribution = calculate_ela_distribution(x, y)
        skewness = distribution["ela_distr.skewness"]
        kurtosis = distribution["ela_distr.kurtosis"]
        cl = self._correlation_length(params, performances, optimum_configs, param_types, np, math)
        mie = calculate_information_content(x, y, seed=100)["ic.h_max"]
        nbc = calculate_nbc(x, y)["nbc.nn_nb.mean_ratio"]
        metrics = dict(zip(LANDSCAPE_METRICS, map(float, [fdc, fbd, plo, skewness, kurtosis, cl, mie, nbc])))
        return LandscapeProfile(instance_name, metrics, len(samples), "pflacco PriEvO reference-compatible")

    @staticmethod
    def _infer_type(values, np):
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

    @staticmethod
    def _mixed_distance(left, right, types, math):
        hamming = euclidean = 0.0
        categorical = continuous = 0
        for first, second, kind in zip(left, right, types):
            if kind == "categorical":
                hamming += first != second
                categorical += 1
            else:
                euclidean += (first - second) ** 2
                continuous += 1
        return ((hamming / categorical) if categorical else 0) + (
            (math.sqrt(euclidean) / math.sqrt(continuous)) if continuous else 0
        )

    def _plo(self, data, columns, params, performances, types, np, max_neighbors=5):
        mapping = {tuple(config): perf for config, perf in zip(params, performances)}
        local = 0
        for index, config in enumerate(params):
            neighbors = []
            for axis, column in enumerate(columns):
                values = data[column].unique()
                current = config[axis]
                if types[axis] == "continuous":
                    ordered = sorted(values)
                    position = np.searchsorted(ordered, current)
                    position = min(position, len(ordered) - 1)
                    candidates = [
                        ordered[item]
                        for item in range(max(0, position - max_neighbors), min(len(ordered), position + max_neighbors + 1))
                        if item != position
                    ]
                else:
                    candidates = [value for value in values if value != current]
                    if len(candidates) > max_neighbors:
                        candidates = np.random.choice(candidates, max_neighbors, replace=False)
                for value in candidates:
                    neighbor = config.copy()
                    neighbor[axis] = value
                    neighbors.append(tuple(neighbor))
            better_or_equal = (
                lambda a, b: a <= b
            ) if self.minimize else (lambda a, b: a >= b)
            if all(neighbor not in mapping or better_or_equal(performances[index], mapping[neighbor]) for neighbor in neighbors):
                local += 1
        return local / len(params)

    def _correlation_length(self, params, performances, optimum_configs, types, np, math):
        if not optimum_configs:
            return 0.5
        random.seed(42)
        np.random.seed(42)
        populations = [tuple(config) for config in params]
        def min_distance(value):
            distances = [self._mixed_distance(value, base, types, math) for base in optimum_configs]
            non_zero = [item for item in distances if item != 0]
            return min(non_zero) if non_zero else 1e8
        ordered = sorted(populations, key=min_distance)
        mapping = {tuple(config): perf for config, perf in zip(params, performances)}
        total = valid = 0
        for _ in range(50):
            walk = list(optimum_configs)
            current = random.choice(optimum_configs)
            for _ in range(50):
                radius = 0.1
                neighbors = []
                while not neighbors or all(self._mixed_distance(item, current, types, math) == 0 for item in neighbors):
                    neighbors = [item for item in ordered if self._mixed_distance(item, current, types, math) <= radius]
                    radius += 0.1
                current = random.choice(neighbors)
                walk.append(current)
            fitness = [mapping[item] for item in walk]
            mean = np.mean(fitness)
            autocorrelation = sum((fitness[i] - mean) * (fitness[i + 1] - mean) for i in range(len(fitness) - 1)) / (len(fitness) - 1)
            deviation = np.std(fitness)
            if deviation != 0:
                total += autocorrelation / (deviation ** 2)
                valid += 1
        if not valid:
            return 0.5
        average = total / valid
        return 0.5 if average == 0 or abs(average) >= 1 else -1.0 / math.log(abs(average))
