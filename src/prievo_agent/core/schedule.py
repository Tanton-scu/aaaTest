from __future__ import annotations

from typing import List


EARLY_OPERATORS = ["i1", "e1", "e2", "m1"]
LATE_OPERATORS = ["e1", "e2", "m1", "m2"]


def operators_for_generation(generation: int, total_generations: int) -> List[str]:
    """保持参考实现的前半探索、后半利用调度，generation 从 1 开始。"""
    if generation < 1 or generation > total_generations:
        raise ValueError("generation 必须位于 1..total_generations")
    # G=1 的 smoke/小预算运行也必须先经历论文 early exploration；否则唯一一代
    # 会直接跳到 m2，既看不到 i1，也违背“前期 i1/e1/e2/m1”的用户契约。
    if is_early_generation(generation, total_generations):
        return list(EARLY_OPERATORS)
    return list(LATE_OPERATORS)


def is_early_generation(generation: int, total_generations: int) -> bool:
    if generation < 1 or generation > total_generations:
        raise ValueError("generation 必须位于 1..total_generations")
    return generation <= max(1, total_generations // 2)
