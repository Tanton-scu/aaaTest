"""统一 Literature corpus schema 与可审计 BM25 检索。"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from prievo_agent.domain.literature import LiteratureEvidence, LiteratureQuery


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*|[\u4e00-\u9fff]")
REQUIRED_PAPER_FIELDS = (
    "paper_id",
    "title",
    "authors",
    "year",
    "identifier",
    "source_path",
    "primary",
    "sections",
)


class LiteratureCorpusError(ValueError):
    pass


def tokenize(text: str) -> List[str]:
    return [item.lower() for item in TOKEN_PATTERN.findall(str(text or ""))]


@dataclass(frozen=True)
class BM25ChunkHit:
    chunk_id: str
    paper_id: str
    raw_score: float
    chunk: Mapping[str, Any]


def load_literature_corpus(path) -> list[dict[str, Any]]:
    """加载唯一产品 schema；支持 curated + PDF corpus 显式合并。"""

    if isinstance(path, (list, tuple)):
        papers = []
        seen = set()
        for item in path:
            item_path = Path(item)
            if not item_path.exists():
                continue
            for paper in load_literature_corpus(item_path):
                if paper["paper_id"] in seen:
                    raise LiteratureCorpusError(
                        "跨 corpus paper_id 重复：{}".format(paper["paper_id"])
                    )
                seen.add(paper["paper_id"])
                papers.append(paper)
        return papers

    corpus_path = Path(path)
    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiteratureCorpusError("Literature corpus 无法读取：{}".format(path)) from exc
    if not isinstance(payload, list):
        raise LiteratureCorpusError("Literature corpus 顶层必须是 paper 数组")
    papers = []
    seen_ids = set()
    for index, paper in enumerate(payload):
        if not isinstance(paper, Mapping):
            raise LiteratureCorpusError("paper[{}] 必须是 object".format(index))
        missing = [name for name in REQUIRED_PAPER_FIELDS if name not in paper]
        if missing:
            raise LiteratureCorpusError(
                "paper[{}] 缺少统一字段：{}".format(index, ", ".join(missing))
            )
        paper_id = str(paper["paper_id"] or "").strip()
        if not paper_id or paper_id in seen_ids:
            raise LiteratureCorpusError("paper_id 为空或重复：{}".format(paper_id))
        seen_ids.add(paper_id)
        authors = paper["authors"]
        if (
            not isinstance(authors, list)
            or any(not isinstance(item, str) or not item.strip() for item in authors)
        ):
            raise LiteratureCorpusError("{} authors 必须是字符串数组".format(paper_id))
        try:
            year = int(paper["year"])
        except (TypeError, ValueError) as exc:
            raise LiteratureCorpusError("{} year 必须是整数".format(paper_id)) from exc
        if isinstance(paper["primary"], bool) is False:
            raise LiteratureCorpusError("{} primary 必须是 boolean".format(paper_id))
        if not isinstance(paper["sections"], (Mapping, list)):
            raise LiteratureCorpusError("{} sections 必须是 object 或数组".format(paper_id))
        normalized = dict(paper)
        normalized.update(
            {
                "paper_id": paper_id,
                "title": str(paper["title"] or "").strip(),
                "authors": [item.strip() for item in authors],
                "year": year,
                "identifier": str(paper["identifier"] or "").strip(),
                "source_path": str(paper["source_path"] or "").strip(),
                "primary": bool(paper["primary"]),
            }
        )
        if not normalized["title"] or not normalized["identifier"] or not normalized["source_path"]:
            raise LiteratureCorpusError(
                "{} title/identifier/source_path 不能为空".format(paper_id)
            )
        papers.append(normalized)
    return papers


def build_literature_chunks(
    papers: Sequence[Mapping[str, Any]],
    chunk_words: int = 55,
    overlap_words: int = 8,
) -> list[dict[str, Any]]:
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("chunk_words/overlap_words 参数无效")
    chunks = []
    seen_chunk_ids = set()
    for paper in papers:
        paper_chunks = []
        for section in _section_records(paper):
            precomputed = section.get("chunks") or []
            if precomputed:
                for part, raw_chunk in enumerate(precomputed):
                    if not isinstance(raw_chunk, Mapping):
                        raise LiteratureCorpusError("PDF section chunks 必须是 object 数组")
                    content = str(raw_chunk.get("content", "") or "").strip()
                    if not content:
                        continue
                    chunk_id = str(raw_chunk.get("chunk_id", "") or "").strip()
                    if not chunk_id:
                        chunk_id = _chunk_id(paper["paper_id"], section["section"], part)
                    paper_chunks.append(
                        _chunk_record(
                            paper,
                            section,
                            chunk_id,
                            int(raw_chunk.get("chunk_index", part)),
                            content,
                            raw_chunk.get("page_start", section.get("page_start")),
                            raw_chunk.get("page_end", section.get("page_end")),
                        )
                    )
            else:
                words = str(section.get("content", "") or "").split()
                start = 0
                part = 0
                while start < len(words):
                    end = min(len(words), start + chunk_words)
                    paper_chunks.append(
                        _chunk_record(
                            paper,
                            section,
                            _chunk_id(paper["paper_id"], section["section"], part),
                            part,
                            " ".join(words[start:end]),
                            section.get("page_start"),
                            section.get("page_end"),
                        )
                    )
                    if end == len(words):
                        break
                    start = max(start + 1, end - overlap_words)
                    part += 1

        for index, chunk in enumerate(paper_chunks):
            if chunk["chunk_id"] in seen_chunk_ids:
                raise LiteratureCorpusError(
                    "chunk_id 重复：{}".format(chunk["chunk_id"])
                )
            seen_chunk_ids.add(chunk["chunk_id"])
            adjacent = []
            if index > 0 and paper_chunks[index - 1]["section"] == chunk["section"]:
                adjacent.append(paper_chunks[index - 1]["chunk_id"])
            if (
                index + 1 < len(paper_chunks)
                and paper_chunks[index + 1]["section"] == chunk["section"]
            ):
                adjacent.append(paper_chunks[index + 1]["chunk_id"])
            chunk["adjacent_chunk_ids"] = adjacent
            chunk["previous_chunk_id"] = (
                paper_chunks[index - 1]["chunk_id"]
                if index > 0
                and paper_chunks[index - 1]["section"] == chunk["section"]
                else ""
            )
            chunk["next_chunk_id"] = (
                paper_chunks[index + 1]["chunk_id"]
                if index + 1 < len(paper_chunks)
                and paper_chunks[index + 1]["section"] == chunk["section"]
                else ""
            )
            chunk["search_text"] = "{} {} {} {} {}".format(
                chunk["title"],
                " ".join(chunk["authors"]),
                chunk["identifier"],
                chunk["section"],
                chunk["content"],
            )
            chunks.append(chunk)
    return chunks


class LocalLiteratureBM25:
    """小语料 section-aware BM25；保留来源、页码、chunk 与 ±1 邻居。"""

    def __init__(
        self, corpus_path, chunk_words: int = 55, overlap_words: int = 8
    ) -> None:
        self.corpus_path = corpus_path
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.papers = load_literature_corpus(self.corpus_path)
        self.chunks = build_literature_chunks(
            self.papers, chunk_words=chunk_words, overlap_words=overlap_words
        )
        self.chunks_by_id = {item["chunk_id"]: item for item in self.chunks}
        self.term_frequencies = [
            Counter(tokenize(item["search_text"])) for item in self.chunks
        ]
        self.lengths = [sum(values.values()) for values in self.term_frequencies]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        )
        self.document_frequency = Counter()
        for values in self.term_frequencies:
            self.document_frequency.update(values.keys())

    def search(self, query: LiteratureQuery) -> List[LiteratureEvidence]:
        hits = self.rank_chunks(query)
        evidence = []
        seen = set()
        for hit in hits:
            chunk = hit.chunk
            dedupe_key = (chunk["paper_id"], chunk["section"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            evidence.append(self.evidence_from_chunk(hit.chunk_id, hit.raw_score))
            if len(evidence) >= min(max(query.max_results, 1), 5):
                break
        return evidence

    def rank_chunks(
        self, query: LiteratureQuery, limit: int | None = None
    ) -> list[BM25ChunkHit]:
        terms = _query_terms(query)
        hits = []
        for index, chunk in enumerate(self.chunks):
            lexical_score = self._bm25(index, terms)
            algorithm_match = any(
                name.strip()
                and name.lower() in chunk["search_text"].lower()
                for name in query.algorithm_names
            )
            # primary 只在已有词项/算法名相关性的候选间加权，不能凭空造命中。
            if lexical_score <= 0 and not algorithm_match:
                continue
            score = lexical_score
            if chunk["primary"]:
                score += 0.08
            if algorithm_match:
                score += 0.2
            hits.append(
                BM25ChunkHit(chunk["chunk_id"], chunk["paper_id"], score, chunk)
            )
        hits.sort(key=lambda item: (-item.raw_score, item.chunk_id))
        return hits if limit is None else hits[: max(0, int(limit))]

    def raw_scores(self, query: LiteratureQuery) -> dict[str, float]:
        return {item.chunk_id: item.raw_score for item in self.rank_chunks(query)}

    def evidence_from_chunk(
        self,
        chunk_id: str,
        score: float,
        adjacent_chunk_ids: Sequence[str] | None = None,
    ) -> LiteratureEvidence:
        try:
            chunk = self.chunks_by_id[chunk_id]
        except KeyError as exc:
            raise LiteratureCorpusError("未知 chunk_id：{}".format(chunk_id)) from exc
        adjacent = list(
            chunk["adjacent_chunk_ids"]
            if adjacent_chunk_ids is None
            else adjacent_chunk_ids
        )
        ordered_ids = []
        previous = chunk.get("previous_chunk_id")
        following = chunk.get("next_chunk_id")
        if previous and previous in adjacent:
            ordered_ids.append(previous)
        ordered_ids.append(chunk_id)
        if following and following in adjacent:
            ordered_ids.append(following)
        ordered_ids.extend(
            item for item in adjacent if item not in ordered_ids and item != chunk_id
        )
        expanded = []
        for expanded_id in ordered_ids:
            item = self.chunks_by_id.get(expanded_id)
            if item is not None and item["paper_id"] == chunk["paper_id"]:
                page = _page_label(item)
                expanded.append(
                    "[{}{} | {}] {}".format(
                        item["section"], page, item["chunk_id"], item["content"]
                    )
                )
        return LiteratureEvidence(
            evidence_id="evidence:{}".format(chunk["chunk_id"]),
            paper_id=chunk["paper_id"],
            title=chunk["title"],
            authors=list(chunk["authors"]),
            year=int(chunk["year"]),
            identifier=chunk["identifier"],
            source_path=chunk["source_path"],
            section=chunk["section"],
            chunk_id=chunk["chunk_id"],
            adjacent_chunk_ids=adjacent,
            content="\n".join(expanded) or chunk["content"],
            score=round(float(score), 6),
            primary_source=bool(chunk["primary"]),
        )

    def _bm25(self, index: int, terms: Sequence[str]) -> float:
        values = self.term_frequencies[index]
        length = self.lengths[index]
        score = 0.0
        k1, b = 1.5, 0.75
        total = len(self.chunks)
        if not total:
            return 0.0
        for term in terms:
            frequency = values.get(term, 0)
            if not frequency:
                continue
            df = self.document_frequency[term]
            inverse = math.log(1 + (total - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * length / max(self.average_length, 1e-12)
            )
            score += inverse * frequency * (k1 + 1) / denominator
        return score


def _query_terms(query: LiteratureQuery) -> list[str]:
    return tokenize(
        " ".join(query.algorithm_names)
        + " "
        + query.topic
        + " "
        + query.purpose
    )


def _section_records(paper: Mapping[str, Any]) -> list[dict[str, Any]]:
    sections = paper["sections"]
    result = []
    if isinstance(sections, Mapping):
        iterable = sections.items()
        for name, value in iterable:
            if isinstance(value, str):
                record = {"section": str(name), "content": value}
            elif isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("section", str(name))
            else:
                raise LiteratureCorpusError(
                    "{} section {} 必须是 string/object".format(
                        paper["paper_id"], name
                    )
                )
            result.append(record)
    else:
        for index, value in enumerate(sections):
            if not isinstance(value, Mapping):
                raise LiteratureCorpusError("sections 数组元素必须是 object")
            record = dict(value)
            record.setdefault("section", "section-{}".format(index))
            result.append(record)
    return result


def _chunk_record(paper, section, chunk_id, chunk_index, content, page_start, page_end):
    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "authors": list(paper["authors"]),
        "year": int(paper["year"]),
        "identifier": paper["identifier"],
        "source_path": paper["source_path"],
        "primary": bool(paper["primary"]),
        "section": str(section.get("section", "unknown")),
        "page_start": _optional_positive_int(page_start),
        "page_end": _optional_positive_int(page_end),
        "chunk_id": str(chunk_id),
        "chunk_index": int(chunk_index),
        "content": str(content).strip(),
    }


def _chunk_id(paper_id: str, section: str, part: int) -> str:
    safe_section = re.sub(r"[^a-z0-9]+", "-", str(section).lower()).strip("-")
    return "{}:{}:{}".format(paper_id, safe_section or "section", part)


def _optional_positive_int(value):
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LiteratureCorpusError("page metadata 必须是整数") from exc
    if parsed <= 0:
        raise LiteratureCorpusError("page metadata 必须从 1 开始")
    return parsed


def _page_label(chunk):
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None:
        return ""
    if end is None or end == start:
        return " p.{}".format(start)
    return " pp.{}-{}".format(start, end)


__all__ = [
    "BM25ChunkHit",
    "LiteratureCorpusError",
    "LocalLiteratureBM25",
    "REQUIRED_PAPER_FIELDS",
    "build_literature_chunks",
    "load_literature_corpus",
    "tokenize",
]
