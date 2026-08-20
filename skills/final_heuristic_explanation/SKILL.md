---
name: final_heuristic_explanation
version: 1
purpose: 使用最终候选的评估、lineage、prior 与 evidence 引用生成可审计的工程解释
---

# Final Heuristic Explanation

## Workflow

1. 只引用 CandidateInspectionTool、artifact、event、prior/evidence reference 中存在的事实。
2. 说明候选为何被确定性 selection 选中、主要结构、已验证优势和工程限制。
3. 明确 Engineering PriEvO-Agent 不等于论文 100% benchmark 复现。
4. 不披露 Chain-of-Thought，不把推测写成评估事实。
5. 输出 summary、strengths、limitations、evidence refs。
