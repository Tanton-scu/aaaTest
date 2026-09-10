---
name: e1
version: 1
purpose: PriEvO e1 / Imitate，分析父代但使用完全不同的 operators 产生新 heuristic
---

# Imitate (e1)

Create one new heuristic after studying the supplied parents.

1. Analyze parent description, operators, code, objective, performance trajectory, used budget, stability, early-convergence risk, and randomness sensitivity.
2. Learn useful design lessons, but the new heuristic MUST use totally different operators. Do not reuse, modify, or extend any parent operator.
3. Reference operator components from the instance-specific PriEvO Prior may guide a structurally complete alternative.
4. Preserve the exact benchmark interface and categorical-value constraints in the task contract.
5. Return only the CandidateDraft schema. Never output claimed fitness or trajectory.

This is the PriEvO `e1` exploration strategy and normally receives two selected parents.
