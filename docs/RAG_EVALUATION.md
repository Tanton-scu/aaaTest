# Literature RAG 离线评测

本报告由项目当前 corpus 与固定 gold paper/chunk IDs 实际计算生成，没有复用 MindBridge 或其他项目的数字。

## 评测设置

- Corpus：`data/literature/corpus.json`，3 篇 paper / 6 个 chunk。
- Eval cases：`data/literature/eval_cases.json`，5 条，K=3。
- Vector backend：`deterministic-token-hashing-v1`；production semantic embedding = `false`。
- 主指标按 gold chunk ID 严格匹配；另报告 document-level 指标。

## 真实结果

| 配置 | HitRate@K | MRR@K | Recall@K | Doc HitRate@K | Doc MRR@K |
|---|---:|---:|---:|---:|---:|
| BM25 | 1.000000 | 0.900000 | 1.000000 | 1.000000 | 0.900000 |
| Vector | 1.000000 | 0.800000 | 1.000000 | 1.000000 | 0.800000 |
| Hybrid | 1.000000 | 0.900000 | 1.000000 | 1.000000 | 0.900000 |
| Hybrid+Rerank | 1.000000 | 0.900000 | 1.000000 | 1.000000 | 0.900000 |

## 可解释性与限制

当前 deterministic hashing vector 只提供离线、可解释、可复验的 lexical vector；它不是生产语义 embedding。

- 当前 curated corpus 只有少量论文和 section，指标方差很大，不能外推到生产规模。
- gold cases 由当前 corpus 人工固定，尚未经过多人标注一致性检验。
- 默认 Vector 是 deterministic lexical hashing，不是语义 embedding；生产部署应接入真实 embedding provider 后重跑同一评测。
- 当前评测只衡量 retrieval ranking，不衡量下游生成答案的事实正确性。

完整逐 case 排名、各阶段分数、source/identifier/page/neighbor provenance 保存在 `reports/rag_eval.json`。
