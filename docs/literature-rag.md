# 可选 Literature Evidence Tool

## 用例边界

只在 generation/reflection 出现陌生 optimizer/operator、机制说明不全、缺少原始论文、需说明假设/限制或 novelty/comparison 时调用。它返回带 provenance 的 `LiteratureEvidence`，不参与 FLA instance similarity，不产生 optimizer empirical rank。

## Corpus 与检索

`data/literature/corpus.json` 是小型 curated primary-paper corpus，保存 title/authors/year/DOI 或 source identifier/local source path/primary 标记/section。当前包含 Hyperband、DEHB、EoH 的人工核对摘要，不存放或输出长篇论文原文。

`LocalLiteratureBM25`：

- paper/section-aware chunk ID；
- overlap chunking 与 adjacent chunk IDs；
- BM25 + algorithm exact-name bonus + primary-source preference；
- `(paper_id, section)` 去重，最多 5 条；
- Evidence 保留 score、section/chunk/source/identifier。

v1 语料很小，BM25 足够；没有证据表明需要 embedding/vector DB。

## Tool policy

`LiteratureQuery` 要求 algorithm names/topic/purpose/reason/max results。`LiteratureSearchTool` 默认每 Run 最多 2 次、每次最多 3 条；通过已持久事件计数，重启后仍能执行预算。每次输出 `LITERATURE_EVIDENCE` artifact 和 `LITERATURE_SEARCHED` event。

MCP 未加入；若未来有外部 Agent 客户端，只能在同一 Tool 之上做 adapter，Core 不依赖 MCP。

## Evidence-aware reflection demo

```powershell
$env:PYTHONPATH='src'
python -m prievo_agent.cli.literature_demo
```

DEHB candidate 路径：`KNOWLEDGE_GAP_IDENTIFIED → LITERATURE_SEARCHED → LITERATURE_EVIDENCE_USED`。Top evidence 是 DEHB primary paper 的 mechanism/assumptions chunks；后续 `REFLECTION_DECISION` 明确写出论文 title/year/section/identifier 和需要验证的低保真相关性。第二次调用被 per-run budget 拒绝。

## 与 Prior Retrieval 的不可互换性

| 维度 | Instance-Specific Prior | Literature Tool |
|---|---|---|
| Query | 8 维 landscape metrics | algorithm/topic/purpose |
| Corpus | historical instances/optimizer ranks/operators | papers/sections/chunks |
| Ranking | 定向归一化 Euclidean + optional semantic refinement | BM25 + primary preference |
| Output | initial population empirical prior | generation/reflection explanation evidence |
| Failure | numeric fallback | empty evidence，不阻塞 evolution |
