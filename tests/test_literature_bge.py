import sys
import types
import unittest
from unittest.mock import patch

from prievo_agent.infrastructure.rag import literature_bge


class _FakeFlagModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.queries = []
        self.corpus = []

    def encode_queries(self, texts):
        self.queries.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def encode_corpus(self, texts):
        self.corpus.extend(texts)
        return [[0.0, 1.0, 0.0] for _ in texts]


class _FakeFlagReranker:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.pairs = []

    def compute_score(self, pairs, normalize=True):
        self.pairs.extend(pairs)
        return [0.9 - index * 0.1 for index, _ in enumerate(pairs)]


class BGEAdapterTest(unittest.TestCase):
    def test_factories_are_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(literature_bge.build_bge_vectorizer_from_env())
            self.assertIsNone(literature_bge.build_bge_reranker_from_env())

    def test_embedding_query_adds_instruction_but_document_does_not(self):
        fake_module = types.SimpleNamespace(FlagModel=_FakeFlagModel)
        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            vectorizer = literature_bge.BGEEmbeddingVectorizer(
                "BAAI/bge-m3",
                query_instruction="query-prefix:",
                use_fp16=False,
                device="cpu",
            )
            self.assertEqual((1.0, 0.0, 0.0), vectorizer.encode_query("hello"))
            self.assertEqual((0.0, 1.0, 0.0), vectorizer.encode_document("hello"))
            self.assertEqual(["query-prefix:hello"], vectorizer.model.queries)
            self.assertEqual(["hello"], vectorizer.model.corpus)
            self.assertEqual("cpu", vectorizer.model.kwargs["device"])
            self.assertFalse(vectorizer.model.kwargs["use_fp16"])

    def test_reranker_scores_query_document_pairs(self):
        fake_module = types.SimpleNamespace(FlagReranker=_FakeFlagReranker)
        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            reranker = literature_bge.BGEReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=False,
                device="cpu",
                normalize=True,
            )
            scores = reranker.score("query", ["doc-a", "doc-b"])
            self.assertEqual((0.9, 0.8), scores)
            self.assertEqual([["query", "doc-a"], ["query", "doc-b"]], reranker.model.pairs)


if __name__ == "__main__":
    unittest.main()
