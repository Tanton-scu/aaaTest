"""BM25 + optional Vector + RRF fusion + deterministic rerank 的本地 Literature RAG。"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from prievo_agent.domain.literature import LiteratureEvidence, LiteratureQuery
from prievo_agent.infrastructure.rag.literature_bm25 import (
    LocalLiteratureBM25,
    tokenize,
)


class VectorPort(Protocol):
    """可替换的 embedding 端口；生产 adapter 应声明真实 provider/model。"""

    name: str
    production_semantic_embedding: bool

    def encode(self, text: str) -> Sequence[float]:
        ...


class RerankerPort(Protocol):
    """Cross-encoder reranker port used after coarse retrieval."""

    name: str
    production_cross_encoder: bool

    def score(self, query_text: str, documents: Sequence[str]) -> Sequence[float]:
        ...


class DeterministicHashingVectorizer:
    """离线可复验的 lexical hashing vector，明确不是语义生产 embedding。"""

    name = "deterministic-token-hashing-v1"
    production_semantic_embedding = False

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32 or dimensions > 4096:
            raise ValueError("hashing vector dimensions 必须在 [32, 4096]")
        self.dimensions = int(dimensions)

    def encode(self, text: str) -> tuple[float, ...]:
        terms = tokenize(text)
        features = list(terms)
        features.extend(
            "{}::{}".format(left, right)
            for left, right in zip(terms, terms[1:])
        )
        counts = Counter(features)
        vector = [0.0] * self.dimensions
        for feature, frequency in counts.items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0 + math.log(float(frequency))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


@dataclass(frozen=True)
class LiteratureRetrievalResult:
    rank: int
    retrieval_mode: str
    vector_backend: str
    vector_is_production_semantic: bool
    paper_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    identifier: str
    source_path: str
    primary: bool
    section: str
    page_start: int | None
    page_end: int | None
    chunk_id: str
    chunk_index: int
    adjacent_chunk_ids: tuple[str, ...]
    expanded_chunk_ids: tuple[str, ...]
    content: str
    bm25_raw_score: float
    bm25_normalized_score: float
    vector_raw_score: float
    vector_normalized_score: float
    fusion_score: float
    cross_encoder_score: float | None
    cross_encoder_backend: str
    rerank_bonus: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_evidence(self) -> LiteratureEvidence:
        return LiteratureEvidence(
            evidence_id="evidence:{}".format(self.chunk_id),
            paper_id=self.paper_id,
            title=self.title,
            authors=list(self.authors),
            year=self.year,
            identifier=self.identifier,
            source_path=self.source_path,
            section=self.section,
            chunk_id=self.chunk_id,
            adjacent_chunk_ids=list(self.adjacent_chunk_ids),
            content=self.content,
            score=round(self.final_score, 6),
            primary_source=self.primary,
        )


class LocalHybridLiteratureRAG:
    """统一四种可评测路径：BM25、Vector、Hybrid(RRF)、Hybrid(RRF)+Rerank。"""

    MODES = {"bm25", "vector", "hybrid"}

    def __init__(
        self,
        corpus_path,
        *,
        vector_port: VectorPort | None = None,
        reranker_port: RerankerPort | None = None,
        bm25_weight: float = 0.58,
        vector_weight: float = 0.42,
        rerank_candidate_pool: int = 40,
        chunk_words: int = 55,
        overlap_words: int = 8,
    ) -> None:
        if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight <= 0:
            raise ValueError("fusion weights 必须非负且至少一个大于 0")
        total = bm25_weight + vector_weight
        self.bm25_weight = float(bm25_weight) / total
        self.vector_weight = float(vector_weight) / total
        self.bm25 = LocalLiteratureBM25(
            corpus_path, chunk_words=chunk_words, overlap_words=overlap_words
        )
        self.vector_port = vector_port or DeterministicHashingVectorizer()
        self.reranker_port = reranker_port
        if isinstance(rerank_candidate_pool, bool) or int(rerank_candidate_pool) <= 0:
            raise ValueError("rerank_candidate_pool must be a positive integer")
        self.rerank_candidate_pool = int(rerank_candidate_pool)
        self._chunk_vectors = {
            chunk["chunk_id"]: self._checked_vector(
                _encode_document(self.vector_port, chunk["search_text"]),
                chunk["chunk_id"],
            )
            for chunk in self.bm25.chunks
        }

    @property
    def vector_metadata(self) -> dict[str, Any]:
        return {
            "backend": str(getattr(self.vector_port, "name", type(self.vector_port).__name__)),
            "production_semantic_embedding": bool(
                getattr(self.vector_port, "production_semantic_embedding", False)
            ),
            "warning": (
                "当前 deterministic hashing vector 只提供离线、可解释、可复验的 lexical vector；"
                "它不是生产语义 embedding。"
                if not getattr(self.vector_port, "production_semantic_embedding", False)
                else ""
            ),
        }

    @property
    def reranker_metadata(self) -> dict[str, Any]:
        return {
            "backend": (
                str(getattr(self.reranker_port, "name", type(self.reranker_port).__name__))
                if self.reranker_port is not None
                else "deterministic-rerank-bonus-v1"
            ),
            "production_cross_encoder": bool(
                getattr(self.reranker_port, "production_cross_encoder", False)
            ),
            "candidate_pool": self.rerank_candidate_pool,
        }

    def search(self, query: LiteratureQuery) -> list[LiteratureEvidence]:
        return [
            item.to_evidence()
            for item in self.retrieve(
                query,
                mode="hybrid",
                rerank=True,
                top_k=min(max(query.max_results, 1), 5),
                expand_neighbors=True,
            )
        ]

    def retrieve(
        self,
        query: LiteratureQuery,
        *,
        mode: str = "hybrid",
        rerank: bool = True,
        top_k: int = 5,
        expand_neighbors: bool = True,
    ) -> list[LiteratureRetrievalResult]:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in self.MODES:
            raise ValueError("未知 retrieval mode：{}".format(mode))
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        query_text = _query_text(query)
        query_vector = self._checked_vector(
            _encode_query(self.vector_port, query_text), "<query>"
        )
        bm25_raw = self.bm25.raw_scores(query)
        vector_raw = {
            chunk_id: _cosine(query_vector, vector)
            for chunk_id, vector in self._chunk_vectors.items()
        }
        bm25_normalized = _minmax_scores(
            {chunk["chunk_id"]: bm25_raw.get(chunk["chunk_id"], 0.0)
             for chunk in self.bm25.chunks}
        )
        vector_normalized = _minmax_scores(vector_raw)
        bm25_rrf_ranked = [
            chunk["chunk_id"]
            for chunk in sorted(
                self.bm25.chunks,
                key=lambda item: (
                    -bm25_raw.get(item["chunk_id"], 0.0),
                    item["chunk_id"],
                ),
            )
        ]
        vector_rrf_ranked = [
            chunk["chunk_id"]
            for chunk in sorted(
                self.bm25.chunks,
                key=lambda item: (
                    -vector_raw.get(item["chunk_id"], 0.0),
                    item["chunk_id"],
                ),
            )
        ]
        rrf_scores = _weighted_rrf_scores(
            (bm25_rrf_ranked, vector_rrf_ranked),
            (self.bm25_weight, self.vector_weight),
        )

        rough_ranked = []
        for chunk in self.bm25.chunks:
            chunk_id = chunk["chunk_id"]
            bm25_score = bm25_raw.get(chunk_id, 0.0)
            vector_score = vector_raw.get(chunk_id, 0.0)
            if normalized_mode == "bm25":
                fusion = bm25_normalized[chunk_id]
                if bm25_score <= 0:
                    continue
            elif normalized_mode == "vector":
                fusion = vector_normalized[chunk_id]
            else:
                fusion = rrf_scores[chunk_id]
            rough_ranked.append(
                {
                    "chunk": chunk,
                    "bm25_raw": bm25_score,
                    "bm25_normalized": bm25_normalized[chunk_id],
                    "vector_raw": vector_score,
                    "vector_normalized": vector_normalized[chunk_id],
                    "fusion": fusion,
                    "bonus": 0.0,
                    "cross_encoder": None,
                    "final": fusion,
                }
            )
        rough_ranked.sort(
            key=lambda item: (-item["final"], item["chunk"]["chunk_id"])
        )

        if rerank and self.reranker_port is not None:
            pool = rough_ranked[: self.rerank_candidate_pool]
            scores = tuple(
                float(value)
                for value in self.reranker_port.score(
                    query_text, [item["chunk"]["search_text"] for item in pool]
                )
            )
            if len(scores) != len(pool):
                raise ValueError("RerankerPort returned score count mismatch")
            for item, score in zip(pool, scores):
                item["cross_encoder"] = score
                item["final"] = score
            ranked = pool + rough_ranked[self.rerank_candidate_pool :]
            ranked.sort(
                key=lambda item: (
                    -(item["cross_encoder"] if item["cross_encoder"] is not None else float("-inf")),
                    -item["fusion"],
                    item["chunk"]["chunk_id"],
                )
            )
        else:
            for item in rough_ranked:
                bonus = _rerank_bonus(query, item["chunk"]) if rerank else 0.0
                item["bonus"] = bonus
                item["final"] = item["fusion"] + bonus
            ranked = rough_ranked
            ranked.sort(
                key=lambda item: (-item["final"], item["chunk"]["chunk_id"])
            )

        selected = []
        seen_sections = set()
        for item in ranked:
            chunk = item["chunk"]
            section_key = (chunk["paper_id"], chunk["section"])
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            selected.append(item)
            if len(selected) >= top_k:
                break
        mode_label = normalized_mode
        if normalized_mode == "hybrid":
            mode_label += "+rrf"
        if rerank and self.reranker_port is not None:
            mode_label += "+cross-rerank"
        elif rerank:
            mode_label += "+rerank"
        return [
            self._result(
                rank,
                mode_label,
                item,
                expand_neighbors=expand_neighbors,
            )
            for rank, item in enumerate(selected, 1)
        ]

    def _result(self, rank, mode, item, expand_neighbors):
        chunk = item["chunk"]
        adjacent = tuple(chunk["adjacent_chunk_ids"]) if expand_neighbors else ()
        evidence = self.bm25.evidence_from_chunk(
            chunk["chunk_id"], item["final"], adjacent_chunk_ids=adjacent
        )
        expanded_ids = []
        previous = chunk.get("previous_chunk_id")
        following = chunk.get("next_chunk_id")
        if previous and previous in adjacent:
            expanded_ids.append(previous)
        expanded_ids.append(chunk["chunk_id"])
        if following and following in adjacent:
            expanded_ids.append(following)
        return LiteratureRetrievalResult(
            rank=rank,
            retrieval_mode=mode,
            vector_backend=str(self.vector_metadata["backend"]),
            vector_is_production_semantic=bool(
                self.vector_metadata["production_semantic_embedding"]
            ),
            paper_id=chunk["paper_id"],
            title=chunk["title"],
            authors=tuple(chunk["authors"]),
            year=int(chunk["year"]),
            identifier=chunk["identifier"],
            source_path=chunk["source_path"],
            primary=bool(chunk["primary"]),
            section=chunk["section"],
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            chunk_id=chunk["chunk_id"],
            chunk_index=int(chunk["chunk_index"]),
            adjacent_chunk_ids=adjacent,
            expanded_chunk_ids=tuple(expanded_ids),
            content=evidence.content,
            bm25_raw_score=round(float(item["bm25_raw"]), 8),
            bm25_normalized_score=round(float(item["bm25_normalized"]), 8),
            vector_raw_score=round(float(item["vector_raw"]), 8),
            vector_normalized_score=round(float(item["vector_normalized"]), 8),
            fusion_score=round(float(item["fusion"]), 8),
            cross_encoder_score=(
                None if item["cross_encoder"] is None
                else round(float(item["cross_encoder"]), 8)
            ),
            cross_encoder_backend=str(self.reranker_metadata["backend"]),
            rerank_bonus=round(float(item["bonus"]), 8),
            final_score=round(float(item["final"]), 8),
        )

    @staticmethod
    def _checked_vector(vector, subject):
        if isinstance(vector, (str, bytes)) or not isinstance(vector, Sequence):
            raise ValueError("VectorPort {} 输出必须是数值 sequence".format(subject))
        try:
            normalized = tuple(float(item) for item in vector)
        except (TypeError, ValueError) as exc:
            raise ValueError("VectorPort {} 输出包含非数值".format(subject)) from exc
        if not normalized or any(not math.isfinite(item) for item in normalized):
            raise ValueError("VectorPort {} 输出为空或包含非有限值".format(subject))
        return normalized


def _query_text(query: LiteratureQuery) -> str:
    return "{} {} {}".format(
        " ".join(query.algorithm_names), query.topic, query.purpose
    )


def _encode_query(vector_port: VectorPort, text: str) -> Sequence[float]:
    """Encode query text.

    BGE-style retrievers normally need a query instruction prefix, while
    document chunks should stay as plain text.  The generic VectorPort keeps
    ``encode`` for backwards compatibility; production adapters can expose
    ``encode_query`` and ``encode_document`` to make that distinction explicit.
    """

    method = getattr(vector_port, "encode_query", None)
    if callable(method):
        return method(text)
    return vector_port.encode(text)


def _encode_document(vector_port: VectorPort, text: str) -> Sequence[float]:
    method = getattr(vector_port, "encode_document", None)
    if callable(method):
        return method(text)
    return vector_port.encode(text)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("VectorPort query/document 维度不一致")
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        value = 1.0 if high > 0 else 0.0
        return {key: value for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _weighted_rrf_scores(
    ranked_lists: Sequence[Sequence[str]],
    weights: Sequence[float],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion with deterministic tie behavior.

    BM25 与 embedding 先各自产生完整排序，再用 RRF 合并。分数只用于排序和
    审计展示，不伪装成概率或相似度。
    """

    if len(ranked_lists) != len(weights):
        raise ValueError("RRF ranked_lists 与 weights 数量不一致")
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        normalized_weight = float(weight)
        for rank, chunk_id in enumerate(ranked, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                normalized_weight / float(k + rank)
            )
    return scores


def _rerank_bonus(query: LiteratureQuery, chunk: dict[str, Any]) -> float:
    search = chunk["search_text"].lower()
    title_terms = set(tokenize(chunk["title"]))
    section_terms = set(tokenize(chunk["section"]))
    query_terms = set(tokenize(_query_text(query)))
    exact_algorithms = sum(
        1
        for name in query.algorithm_names
        if name.strip() and name.strip().lower() in search
    )
    title_overlap = len(query_terms.intersection(title_terms)) / max(1, len(query_terms))
    section_overlap = len(query_terms.intersection(section_terms)) / max(
        1, len(query_terms)
    )
    primary_bonus = 0.015 if chunk["primary"] else 0.0
    return min(0.12, exact_algorithms * 0.055) + 0.04 * title_overlap + 0.025 * section_overlap + primary_bonus


def build_literature_rag_from_env(corpus_path) -> LocalHybridLiteratureRAG:
    """Build the production literature RAG from local configuration.

    默认不强行加载大模型，便于单元测试和没有 GPU 的机器启动；如果在
    ``.env`` 中设置 ``RAG_EMBEDDING_BACKEND=bge`` 或
    ``RAG_RERANKER_BACKEND=bge``，才会通过 FlagEmbedding 加载 BGE-M3 /
    BGE-Reranker。
    """

    from prievo_agent.infrastructure.rag.literature_bge import (
        build_bge_reranker_from_env,
        build_bge_vectorizer_from_env,
    )

    return LocalHybridLiteratureRAG(
        corpus_path,
        vector_port=build_bge_vectorizer_from_env(),
        reranker_port=build_bge_reranker_from_env(),
        rerank_candidate_pool=_env_positive_int("RAG_RERANK_CANDIDATE_POOL", 40),
    )


def _env_positive_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return int(default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return parsed


__all__ = [
    "build_literature_rag_from_env",
    "DeterministicHashingVectorizer",
    "LiteratureRetrievalResult",
    "LocalHybridLiteratureRAG",
    "RerankerPort",
    "VectorPort",
]
