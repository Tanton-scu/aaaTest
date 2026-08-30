import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.domain.literature import LiteratureQuery
from prievo_agent.infrastructure.rag.literature_bm25 import (
    LiteratureCorpusError,
    LocalLiteratureBM25,
    REQUIRED_PAPER_FIELDS,
    load_literature_corpus,
)
from prievo_agent.infrastructure.rag.literature_hybrid import (
    DeterministicHashingVectorizer,
    LocalHybridLiteratureRAG,
)
from scripts.ingest_papers import ingest_pdfs


ROOT = Path(__file__).resolve().parents[1]


class _Page:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class _Reader:
    metadata = {
        "/Title": "Needle Retrieval Paper",
        "/Author": "Alice Example; Bob Example",
        "/CreationDate": "D:20220101000000",
        "/Subject": "doi:10.1234/NEEDLE.2022",
    }
    pages = [
        _Page(
            "1 Introduction\nUniqueNeedle allocation mechanism searches configurations "
            "with bounded resources and deterministic evidence for retrieval testing."
        ),
        _Page(
            "2 Limitations\nUniqueNeedle depends on representative measurements and "
            "does not replace empirical validation in production systems."
        ),
    ]


class _QueryDocumentVectorizer:
    name = "fake-bge-m3"
    production_semantic_embedding = True

    def __init__(self):
        self.queries = []
        self.documents = []

    def encode(self, text):
        raise AssertionError("hybrid RAG should call encode_query/encode_document when available")

    def encode_query(self, text):
        self.queries.append(text)
        return (1.0, 0.0, 0.0)

    def encode_document(self, text):
        self.documents.append(text)
        lowered = text.lower()
        if "hyperband" in lowered:
            return (1.0, 0.0, 0.0)
        if "dehb" in lowered:
            return (0.8, 0.2, 0.0)
        return (0.0, 1.0, 0.0)


class _RecordingCrossEncoderReranker:
    name = "fake-bge-reranker-v2-m3"
    production_cross_encoder = True

    def __init__(self):
        self.calls = []

    def score(self, query_text, documents):
        self.calls.append((query_text, tuple(documents)))
        return tuple(float(index) for index in range(len(documents), 0, -1))


class HybridLiteratureRAGTest(unittest.TestCase):
    def test_no_pdf_writes_empty_product_schema_without_pypdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_root = root / "papers"
            output = root / "corpus.json"
            result = ingest_pdfs(paper_root, output)
            self.assertEqual((), result.papers)
            self.assertEqual([], json.loads(output.read_text(encoding="utf-8")))

    def test_legacy_curated_schema_search_and_true_empty_result(self):
        corpus = ROOT / "data" / "literature" / "corpus.json"
        bm25 = LocalLiteratureBM25(corpus)
        evidence = bm25.search(
            LiteratureQuery(
                ["DEHB"], "differential evolution hyperband", "mechanism", "test", 3
            )
        )
        self.assertTrue(evidence)
        self.assertEqual("awad-2021-dehb", evidence[0].paper_id)
        self.assertTrue(evidence[0].title.startswith("DEHB:"))
        self.assertEqual(
            [],
            bm25.search(
                LiteratureQuery(
                    [], "zzzz_no_matching_term_987654", "none", "test", 3
                )
            ),
        )

    def test_hybrid_has_stage_scores_provenance_and_is_deterministic(self):
        rag = LocalHybridLiteratureRAG(
            ROOT / "data" / "literature" / "corpus.json"
        )
        query = LiteratureQuery(
            ["Hyperband"],
            "successive halving brackets resource allocation",
            "explain mechanism",
            "test",
            3,
        )
        first = rag.retrieve(query, mode="hybrid", rerank=True, top_k=3)
        second = rag.retrieve(query, mode="hybrid", rerank=True, top_k=3)

        self.assertEqual(
            [item.to_dict() for item in first], [item.to_dict() for item in second]
        )
        top = first[0]
        # 小语料里 DEHB 的 mechanism 同时包含 Hyperband/resource allocation；
        # 这里只验证真实融合/provenance，不把当前微型集上的 top-1 偶然顺序写成契约。
        self.assertIn("li-2018-hyperband", [item.paper_id for item in first])
        self.assertIn(
            "li-2018-hyperband:mechanism:0", [item.chunk_id for item in first]
        )
        self.assertTrue(top.identifier)
        self.assertTrue(top.source_path)
        self.assertTrue(top.authors)
        self.assertGreaterEqual(top.bm25_normalized_score, 0)
        self.assertLessEqual(top.bm25_normalized_score, 1)
        self.assertGreaterEqual(top.vector_normalized_score, 0)
        self.assertLessEqual(top.vector_normalized_score, 1)
        self.assertEqual("hybrid+rrf+rerank", top.retrieval_mode)
        self.assertFalse(top.vector_is_production_semantic)
        self.assertIn("不是生产语义 embedding", rag.vector_metadata["warning"])
        self.assertEqual(top.chunk_id, top.to_evidence().chunk_id)

    def test_pdf_ingest_roundtrip_uses_product_schema_pages_chunks_and_neighbors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_root = root / "papers"
            paper_root.mkdir()
            (paper_root / "needle.pdf").write_bytes(b"fake-pdf-for-injected-reader")
            output = root / "pdf_corpus.json"
            result = ingest_pdfs(
                paper_root,
                output,
                reader_factory=lambda path: _Reader(),
                chunk_words=6,
                overlap_words=2,
            )
            second = ingest_pdfs(
                paper_root,
                output,
                reader_factory=lambda path: _Reader(),
                chunk_words=6,
                overlap_words=2,
            )
            papers = load_literature_corpus(output)
            rag = LocalHybridLiteratureRAG(
                output, chunk_words=6, overlap_words=2
            )
            hits = rag.retrieve(
                LiteratureQuery(
                    ["UniqueNeedle"], "bounded resources", "mechanism", "test", 3
                ),
                mode="hybrid",
                rerank=True,
                top_k=3,
            )

        self.assertEqual(1, len(result.papers))
        self.assertEqual((), result.skipped)
        self.assertEqual(
            result.papers[0]["paper_id"], second.papers[0]["paper_id"]
        )
        self.assertEqual(set(REQUIRED_PAPER_FIELDS), set(REQUIRED_PAPER_FIELDS).intersection(papers[0]))
        self.assertEqual(2022, papers[0]["year"])
        self.assertTrue(papers[0]["identifier"].startswith("DOI:"))
        sections = papers[0]["sections"]
        self.assertGreaterEqual(len(sections), 2)
        all_chunks = [chunk for section in sections for chunk in section["chunks"]]
        self.assertTrue(all(chunk["page_start"] >= 1 for chunk in all_chunks))
        self.assertTrue(any(chunk["next_chunk_id"] for chunk in all_chunks))
        self.assertTrue(hits)
        self.assertEqual(papers[0]["paper_id"], hits[0].paper_id)
        self.assertTrue(hits[0].expanded_chunk_ids)
        self.assertIn("UniqueNeedle", hits[0].content)

    def test_schema_errors_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([{"paper_id": "bad"}]), encoding="utf-8")
            with self.assertRaisesRegex(LiteratureCorpusError, "缺少统一字段"):
                load_literature_corpus(path)

    def test_hash_vector_is_repeatable_and_explicitly_nonsemantic(self):
        vectorizer = DeterministicHashingVectorizer(64)
        self.assertEqual(vectorizer.encode("DEHB mechanism"), vectorizer.encode("DEHB mechanism"))
        self.assertFalse(vectorizer.production_semantic_embedding)
        self.assertAlmostEqual(
            1.0,
            sum(value * value for value in vectorizer.encode("DEHB mechanism")),
            places=10,
        )

    def test_cross_encoder_reranker_only_scores_coarse_top_n(self):
        vectorizer = _QueryDocumentVectorizer()
        reranker = _RecordingCrossEncoderReranker()
        rag = LocalHybridLiteratureRAG(
            ROOT / "data" / "literature" / "corpus.json",
            vector_port=vectorizer,
            reranker_port=reranker,
            rerank_candidate_pool=2,
        )
        results = rag.retrieve(
            LiteratureQuery(
                ["Hyperband"],
                "successive halving resource allocation",
                "explain mechanism",
                "test",
                3,
            ),
            mode="hybrid",
            rerank=True,
            top_k=2,
        )

        self.assertEqual(1, len(reranker.calls))
        self.assertEqual(2, len(reranker.calls[0][1]))
        self.assertEqual(1, len(vectorizer.queries))
        self.assertGreater(len(vectorizer.documents), len(reranker.calls[0][1]))
        self.assertEqual("hybrid+rrf+cross-rerank", results[0].retrieval_mode)
        self.assertEqual("fake-bge-m3", results[0].vector_backend)
        self.assertTrue(results[0].vector_is_production_semantic)
        self.assertEqual("fake-bge-reranker-v2-m3", results[0].cross_encoder_backend)
        self.assertIsNotNone(results[0].cross_encoder_score)
        self.assertEqual(0.0, results[0].rerank_bonus)

    def test_product_loader_merges_curated_and_pdf_corpora(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "pdf_corpus.json"
            pdf.write_text("[]\n", encoding="utf-8")
            paths = [ROOT / "data" / "literature" / "corpus.json", pdf]
            papers = load_literature_corpus(paths)
            rag = LocalHybridLiteratureRAG(paths)
            results = rag.retrieve(
                LiteratureQuery(
                    ["DEHB"], "differential evolution hyperband", "mechanism", "test", 2
                ),
                mode="hybrid", rerank=True, top_k=2,
            )
        self.assertEqual(3, len(papers))
        self.assertTrue(results)
        self.assertEqual("hybrid+rrf+rerank", results[0].retrieval_mode)


if __name__ == "__main__":
    unittest.main()
