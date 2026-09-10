from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GeneratedCandidate:
    code: str
    description: str
    operators: List[str]
