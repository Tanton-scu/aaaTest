---
name: revise
version: 1
purpose: PriEvO m1 策略，对单个父代的关键 operator 进行优化、替换或整合
---

# Revise (m1)

Revise one existing heuristic.

1. Use the parent description, operators, code, objective, trajectory, budget use, and convergence behavior.
2. Diagnose operator-level strengths and weaknesses, then optimize, replace, or integrate key operators.
3. Keep validated mechanisms unless the evidence justifies a focused structural change; do not perform an unrelated wholesale rewrite.
4. The supplied Prior operator components may be used as bounded alternatives.
5. Preserve the benchmark interface and return only the CandidateDraft schema. Do not invent evaluation facts.

This is the PriEvO `m1` strategy. The effective prompt input is one parent.
