import json
import tempfile
import unittest
from pathlib import Path

from prievo_agent.devtools.rag_eval.evaluator import (
    compute_ranking_metrics,
    run_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


class RAGEvaluationTest(unittest.TestCase):
    def test_metric_formula_uses_first_gold_rank_and_macro_average(self):
        metrics = compute_ranking_metrics(
            rankings=[["x", "gold-a", "y"], ["z"]],
            gold_sets=[["gold-a"], ["gold-b"]],
            k=3,
        )
        self.assertEqual(0.5, metrics["hit_rate_at_k"])
        self.assertEqual(0.25, metrics["mrr_at_k"])
        self.assertEqual(0.5, metrics["recall_at_k"])

    def test_real_four_configuration_evaluation_is_reproducible(self):
        corpus = ROOT / "assets" / "literature" / "corpus.json"
        cases = ROOT / "assets" / "literature" / "eval_cases.json"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            markdown = Path(directory) / "report.md"
            first = run_evaluation(
                corpus, cases, k=3, output_path=output, markdown_path=markdown
            )
            second = run_evaluation(corpus, cases, k=3)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            rendered = markdown.read_text(encoding="utf-8")

        self.assertEqual(
            {"BM25", "Vector", "Hybrid", "Hybrid+Rerank"},
            set(first["configurations"]),
        )
        self.assertEqual(5, first["case_count"])
        self.assertEqual(6, first["corpus"]["chunk_count"])
        self.assertFalse(first["vector"]["production_semantic_embedding"])
        self.assertEqual(
            {name: value["metrics"] for name, value in first["configurations"].items()},
            {name: value["metrics"] for name, value in second["configurations"].items()},
        )
        self.assertEqual(
            first["configurations"]["BM25"]["metrics"],
            persisted["configurations"]["BM25"]["metrics"],
        )
        for configuration in first["configurations"].values():
            for case in configuration["cases"]:
                self.assertTrue(case["gold_paper_ids"])
                self.assertTrue(case["gold_chunk_ids"])
                self.assertTrue(case["ranked_chunk_ids"])
                self.assertIn("source_path", case["results"][0])
                self.assertIn("bm25_raw_score", case["results"][0])
                self.assertIn("vector_raw_score", case["results"][0])
                self.assertIn("expanded_chunk_ids", case["results"][0])
        self.assertIn("没有复用 MindBridge", rendered)
        self.assertIn("少量", "".join(first["limitations"]))

    def test_missing_fixed_gold_id_fails_instead_of_title_matching(self):
        corpus = ROOT / "assets" / "literature" / "corpus.json"
        with tempfile.TemporaryDirectory() as directory:
            cases = Path(directory) / "cases.json"
            cases.write_text(
                json.dumps(
                    [
                        {
                            "case_id": "bad-gold",
                            "query": {
                                "algorithm_names": ["DEHB"],
                                "topic": "mechanism",
                                "purpose": "test",
                                "reason": "test",
                                "max_results": 3,
                            },
                            "gold_paper_ids": ["missing-paper"],
                            "gold_chunk_ids": ["missing-chunk"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "gold IDs 不存在"):
                run_evaluation(corpus, cases)


if __name__ == "__main__":
    unittest.main()
