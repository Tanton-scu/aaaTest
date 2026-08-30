"""使用项目当前 corpus 和固定 gold IDs 实际计算 Literature RAG 指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from prievo_agent.domain.literature import LiteratureQuery
from prievo_agent.infrastructure.rag.literature_hybrid import LocalHybridLiteratureRAG


VARIANTS = (
    ("BM25", "bm25", False),
    ("Vector", "vector", False),
    ("Hybrid", "hybrid", False),
    ("Hybrid+Rerank", "hybrid", True),
)


def compute_ranking_metrics(
    rankings: Sequence[Sequence[str]],
    gold_sets: Sequence[Sequence[str]],
    k: int,
) -> dict[str, float]:
    """计算 macro HitRate/MRR/Recall；MRR 取首个 gold 的 reciprocal rank。"""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k 必须是正整数")
    if len(rankings) != len(gold_sets) or not rankings:
        raise ValueError("rankings/gold_sets 必须是等长非空数组")
    hits = []
    reciprocals = []
    recalls = []
    for ranking, gold in zip(rankings, gold_sets):
        gold_ids = set(gold)
        if not gold_ids:
            raise ValueError("每个 eval case 至少有一个 gold ID")
        top = list(ranking)[:k]
        matched = gold_ids.intersection(top)
        hits.append(1.0 if matched else 0.0)
        first_rank = next(
            (index for index, item in enumerate(top, 1) if item in gold_ids), None
        )
        reciprocals.append(0.0 if first_rank is None else 1.0 / first_rank)
        recalls.append(len(matched) / len(gold_ids))
    count = float(len(rankings))
    return {
        "hit_rate_at_k": round(sum(hits) / count, 6),
        "mrr_at_k": round(sum(reciprocals) / count, 6),
        "recall_at_k": round(sum(recalls) / count, 6),
    }


def run_evaluation(
    corpus_path: Path,
    cases_path: Path,
    *,
    k: int = 3,
    output_path: Path | None = None,
    markdown_path: Path | None = None,
) -> dict[str, Any]:
    corpus = Path(corpus_path)
    cases_file = Path(cases_path)
    cases = _load_cases(cases_file)
    retriever = LocalHybridLiteratureRAG(corpus)
    _validate_gold(cases, retriever)
    configurations = {}
    for label, mode, rerank in VARIANTS:
        case_results = []
        chunk_rankings = []
        document_rankings = []
        chunk_gold = []
        document_gold = []
        for case in cases:
            query = LiteratureQuery(**case["query"])
            results = retriever.retrieve(
                query,
                mode=mode,
                rerank=rerank,
                top_k=k,
                expand_neighbors=True,
            )
            ranked_chunks = [item.chunk_id for item in results]
            ranked_documents = [item.paper_id for item in results]
            chunk_rankings.append(ranked_chunks)
            document_rankings.append(ranked_documents)
            chunk_gold.append(case["gold_chunk_ids"])
            document_gold.append(case["gold_paper_ids"])
            case_results.append(
                {
                    "case_id": case["case_id"],
                    "gold_paper_ids": list(case["gold_paper_ids"]),
                    "gold_chunk_ids": list(case["gold_chunk_ids"]),
                    "ranked_paper_ids": ranked_documents,
                    "ranked_chunk_ids": ranked_chunks,
                    "results": [item.to_dict() for item in results],
                }
            )
        chunk_metrics = compute_ranking_metrics(chunk_rankings, chunk_gold, k)
        document_metrics = compute_ranking_metrics(
            document_rankings, document_gold, k
        )
        configurations[label] = {
            "mode": mode,
            "rerank": rerank,
            "metrics": {
                # 主指标严格使用固定 gold chunk IDs。
                **chunk_metrics,
                "document_hit_rate_at_k": document_metrics["hit_rate_at_k"],
                "document_mrr_at_k": document_metrics["mrr_at_k"],
                "document_recall_at_k": document_metrics["recall_at_k"],
            },
            "cases": case_results,
        }

    report = {
        "schema_version": "literature-rag-eval-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "k": k,
        "case_count": len(cases),
        "corpus": {
            "path": corpus.as_posix(),
            "sha256": _file_digest(corpus),
            "paper_count": len(retriever.bm25.papers),
            "chunk_count": len(retriever.bm25.chunks),
        },
        "eval_cases": {
            "path": cases_file.as_posix(),
            "sha256": _file_digest(cases_file),
            "gold_identity": "固定 paper_id + chunk_id；不使用标题模糊匹配",
        },
        "vector": retriever.vector_metadata,
        "fusion": {
            "bm25_weight": retriever.bm25_weight,
            "vector_weight": retriever.vector_weight,
            "normalization": "per-query min-max over canonical chunks",
            "rerank": "deterministic algorithm/title/section overlap + primary bonus",
            "neighbor_expansion": "anchor chunk ±1 within the same paper section",
        },
        "configurations": configurations,
        "limitations": [
            "当前 curated corpus 只有少量论文和 section，指标方差很大，不能外推到生产规模。",
            "gold cases 由当前 corpus 人工固定，尚未经过多人标注一致性检验。",
            "默认 Vector 是 deterministic lexical hashing，不是语义 embedding；生产部署应接入真实 embedding provider 后重跑同一评测。",
            "当前评测只衡量 retrieval ranking，不衡量下游生成答案的事实正确性。",
        ],
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    if markdown_path is not None:
        _write_text(Path(markdown_path), render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Literature RAG 离线评测",
        "",
        "本报告由项目当前 corpus 与固定 gold paper/chunk IDs 实际计算生成，"
        "没有复用 MindBridge 或其他项目的数字。",
        "",
        "## 评测设置",
        "",
        "- Corpus：`{}`，{} 篇 paper / {} 个 chunk。".format(
            report["corpus"]["path"],
            report["corpus"]["paper_count"],
            report["corpus"]["chunk_count"],
        ),
        "- Eval cases：`{}`，{} 条，K={}。".format(
            report["eval_cases"]["path"], report["case_count"], report["k"]
        ),
        "- Vector backend：`{}`；production semantic embedding = `{}`。".format(
            report["vector"]["backend"],
            str(report["vector"]["production_semantic_embedding"]).lower(),
        ),
        "- 主指标按 gold chunk ID 严格匹配；另报告 document-level 指标。",
        "",
        "## 真实结果",
        "",
        "| 配置 | HitRate@K | MRR@K | Recall@K | Doc HitRate@K | Doc MRR@K |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("BM25", "Vector", "Hybrid", "Hybrid+Rerank"):
        metrics = report["configurations"][name]["metrics"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                name,
                metrics["hit_rate_at_k"],
                metrics["mrr_at_k"],
                metrics["recall_at_k"],
                metrics["document_hit_rate_at_k"],
                metrics["document_mrr_at_k"],
            )
        )
    lines.extend(
        [
            "",
            "## 可解释性与限制",
            "",
            report["vector"].get("warning", "") or "Vector provider 声明为生产语义模型。",
            "",
        ]
    )
    lines.extend("- {}".format(item) for item in report["limitations"])
    lines.extend(
        [
            "",
            "完整逐 case 排名、各阶段分数、source/identifier/page/neighbor provenance "
            "保存在 `reports/rag_eval.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _load_cases(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("RAG eval cases 无法读取") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("RAG eval cases 必须是非空数组")
    result = []
    seen = set()
    required_query = {
        "algorithm_names",
        "topic",
        "purpose",
        "reason",
        "max_results",
    }
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("eval case 必须是 object")
        case_id = str(item.get("case_id", "") or "")
        if not case_id or case_id in seen:
            raise ValueError("case_id 为空或重复：{}".format(case_id))
        seen.add(case_id)
        query = item.get("query")
        if not isinstance(query, dict) or set(query) != required_query:
            raise ValueError("{} query schema 无效".format(case_id))
        gold_papers = item.get("gold_paper_ids")
        gold_chunks = item.get("gold_chunk_ids")
        if not _nonempty_string_list(gold_papers) or not _nonempty_string_list(gold_chunks):
            raise ValueError("{} gold IDs 必须是非空字符串数组".format(case_id))
        result.append(
            {
                "case_id": case_id,
                "query": dict(query),
                "gold_paper_ids": list(gold_papers),
                "gold_chunk_ids": list(gold_chunks),
            }
        )
    return result


def _validate_gold(cases, retriever):
    paper_ids = {item["paper_id"] for item in retriever.bm25.papers}
    chunk_ids = set(retriever.bm25.chunks_by_id)
    for case in cases:
        missing_papers = set(case["gold_paper_ids"]) - paper_ids
        missing_chunks = set(case["gold_chunk_ids"]) - chunk_ids
        if missing_papers or missing_chunks:
            raise ValueError(
                "{} gold IDs 不存在：papers={} chunks={}".format(
                    case["case_id"], sorted(missing_papers), sorted(missing_chunks)
                )
            )


def _nonempty_string_list(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv=None):
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="运行 Literature RAG 离线评测")
    parser.add_argument(
        "--corpus", type=Path, default=project / "data" / "literature" / "corpus.json"
    )
    parser.add_argument(
        "--cases", type=Path, default=project / "data" / "literature" / "eval_cases.json"
    )
    parser.add_argument(
        "--output", type=Path, default=project / "reports" / "rag_eval.json"
    )
    parser.add_argument(
        "--markdown", type=Path, default=project / "docs" / "RAG_EVALUATION.md"
    )
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_evaluation(
        args.corpus,
        args.cases,
        k=args.k,
        output_path=args.output,
        markdown_path=args.markdown,
    )
    print(
        "RAG 离线评测完成：{} cases，报告 {}".format(
            report["case_count"], args.output
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
