from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

from prievo_agent.domain.models import Candidate


def operator_names(candidate: Candidate) -> List[str]:
    return sorted({name.strip() for name in candidate.operators if name.strip()})


def diversity_gain(current: Set[str], candidate_operators: Iterable[str]) -> int:
    return len(set(candidate_operators) - current)


def select_population_early(population: Sequence[Candidate], size: int) -> List[Candidate]:
    """忠实表达参考 `population_management` 的 fitness-tier + diversity 行为。"""
    valid = [candidate for candidate in population if candidate.objective is not None]
    if size <= 0 or not valid:
        return []
    if size >= len(valid):
        return list(valid)

    groups: Dict[float, List[Candidate]] = defaultdict(list)
    for candidate in valid:
        groups[float(candidate.objective)].append(candidate)
    tiers = [groups[value] for value in sorted(groups)]
    processed = []
    global_operators: Set[str] = set()
    selected: List[Candidate] = []

    for tier_index, tier in enumerate(tiers):
        if len(selected) >= size:
            break
        pairs = [(candidate, operator_names(candidate)) for candidate in tier]
        if tier_index == 0:
            pairs.sort(
                key=lambda item: (
                    len(item[1]),
                    diversity_gain(global_operators, item[1]),
                ),
                reverse=True,
            )
        else:
            pairs.sort(
                key=lambda item: diversity_gain(global_operators, item[1]),
                reverse=True,
            )
        chosen, operators = pairs[0]
        selected.append(chosen)
        global_operators.update(operators)
        processed.append((pairs, {0}))

    while len(selected) < size:
        found = False
        for pairs, used_indexes in processed:
            for index, (candidate, operators) in enumerate(pairs):
                if index in used_indexes:
                    continue
                selected.append(candidate)
                global_operators.update(operators)
                used_indexes.add(index)
                found = True
                break
            if len(selected) >= size:
                break
        if not found:
            break
    return selected


def select_population_late(population: Sequence[Candidate], size: int) -> List[Candidate]:
    """保持参考实现的 objective 优先、operator 数量作为 tie-break。"""
    valid = [candidate for candidate in population if candidate.objective is not None]
    if size <= 0 or not valid:
        return []
    if len(valid) <= size:
        return list(valid)
    return sorted(
        valid,
        key=lambda candidate: (
            float(candidate.objective),
            -len(operator_names(candidate)),
        ),
    )[:size]


def select_parents(
    population: Sequence[Candidate], count: int, rng: random.Random
) -> List[Candidate]:
    """保持参考 `prob_rank.parent_selection` 的权重公式。"""
    if not population:
        raise ValueError("population 不能为空")
    weights = [1 / (rank + 1 + len(population)) for rank in range(len(population))]
    return rng.choices(list(population), weights=weights, k=count)
