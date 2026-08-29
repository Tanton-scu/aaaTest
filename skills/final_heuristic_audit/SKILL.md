---
name: final_heuristic_audit
version: 1
purpose: 仅在 objective 精确等优时，从 tied candidates 中选择一个稳定的最终 heuristic
---

# Final Heuristic Audit

Select exactly one candidate from the supplied tied set.

1. PRIMARY: assess sustained global convergence under a larger evaluation budget, including exploration and long-term improvement mechanisms.
2. SECONDARY: assess algorithm integrity, modular completeness, and justified code complexity.
3. AUXILIARY: use trajectory smoothness and volatility only as supporting evidence.
4. Use only supplied code, description, objective, trajectory, and operators. Do not request Memory or Literature RAG and do not consider candidates outside the tied set.
5. Return `selected_candidate_id`, `reason`, and `structural_operator_comparison` only.

This Skill preserves the decision criteria in `references/prievo/prievo/methods/prievo/prievo.py::best_stability`; the Runtime owns qualification, exact-tie detection, and deterministic fallback.
