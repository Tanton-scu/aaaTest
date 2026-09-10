from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LiteratureQuery:
    algorithm_names: List[str]
    topic: str
    purpose: str
    reason: str
    max_results: int = 3


@dataclass(frozen=True)
class LiteratureEvidence:
    evidence_id: str
    paper_id: str
    title: str
    authors: List[str]
    year: int
    identifier: str
    source_path: str
    section: str
    chunk_id: str
    adjacent_chunk_ids: List[str]
    content: str
    score: float
    primary_source: bool
