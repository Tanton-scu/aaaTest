---
name: literature_evidence_review
version: 1
purpose: 审阅候选 heuristic 的文献证据并形成可追溯 reflection 决策
---

# Literature Evidence Review

1. 明确知识缺口：algorithm/operator、mechanism、assumption、limitation 或 comparison。
2. 只用带 title/authors/year/identifier/section/chunk 的 evidence；优先 primary paper。
3. 区分“论文说明的机制”与“当前 candidate 已实际实现的机制”。
4. 说明证据支持什么、不支持什么，以及仍需由 benchmark 验证的假设。
5. 输出必须包含 evidence IDs；无 evidence 时不得编造结论。

该 Skill 是 Agent/reflection 的推理 SOP，不是 PriEvO Operator，不参与 population diversity 或 candidate lineage 的算法算子统计。
