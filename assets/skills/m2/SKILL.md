---
name: m2
version: 1
purpose: PriEvO m2 / Fine-tune，在不改变 operator structure 的前提下调整内部参数
---

# Fine-tune (m2)

Fine-tune one existing heuristic.

1. Analyze the supplied parent objective, trajectory, budget use, convergence, code, and operators.
2. Adjust only modifiable internal algorithm parameters and parameter configurations.
3. MUST NOT change the operator structure or replace the parent with a different algorithm.
4. Preserve the exact benchmark interface and categorical-value constraints.
5. Return only the CandidateDraft schema. Do not invent fitness or trajectory.

This is the PriEvO `m2` strategy. The effective prompt input is one parent, and the original reference prompt does not inject the operator-reference section.
